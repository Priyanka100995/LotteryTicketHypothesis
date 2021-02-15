
import tensorflow as tf
import numpy as np
import math
import tensorflow_model_optimization as tfmot
import tensorflow_model_optimization
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning, prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep
import os
import pickle
import Prune
import matplotlib as pyplot
import matplotlib.pyplot as plt
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras import models, layers, datasets
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
from sklearn.metrics import accuracy_score, precision_score, recall_score


batch_size = 60
num_classes = 10
num_epochs = 50


# input image dimensions
img_rows, img_cols = 28, 28

# MNIST dataset-
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
    train_x = train_x.reshape(train_x.shape[0], 1, img_rows, img_cols)
    test_x = test_x.reshape(test_x.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_x = train_x.reshape(train_x.shape[0], img_rows, img_cols, 1)
    test_x = test_x.reshape(test_x.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("\n'input_shape' which will be used = {0}\n".format(input_shape))

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /= 255.0
test_x /= 255.0

train_y = tf.keras.utils.to_categorical(train_y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)


# Create training and testing datasets-
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

train_dataset = train_dataset.shuffle(buffer_size = 20000, reshuffle_each_iteration = True).batch(batch_size = batch_size, drop_remainder = False)

test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)

#Optimizer and loss function for training-
#loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
#optimizer = tf.keras.optimizers.Adam(lr = 0.0012)
loss_fn = tf.keras.losses.sparse_categorical_crossentropy
#optimizer = tf.keras.optimizers.Adam(lr = 0.0012)

#Accuracy
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'test_accuracy')

# Trained without pruning
epochs = num_epochs

num_train_samples = train_x.shape[0]

end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs

print("'end_step parameter' for this dataset =  {0}".format(end_step))

# Pruning parameters
unpruned = {
    'pruning_schedule': Prune.ConstantSparsity(
        target_sparsity=0.0, begin_step=0,
        end_step = end_step, frequency=100
    )
}

l = tf.keras.layers

def pruned_nn(pruning_params_conv, pruning_params_fc):

    pruned_model = Sequential()

    pruned_model.add(prune_low_magnitude(
        Conv2D(
            filters=6, kernel_size=(3, 3),
            activation='tanh', kernel_initializer=tf.initializers.GlorotUniform(),
            strides=(1, 1), padding='valid',
            input_shape=(28, 28, 1)
        ),
        **pruning_params_conv)
    )

    pruned_model.add(prune_low_magnitude(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ),
        **pruning_params_conv)
    )

    pruned_model.add(prune_low_magnitude(
        Conv2D(
            filters=16, kernel_size=(3, 3),
            activation='tanh', kernel_initializer=tf.initializers.GlorotUniform(),
            strides=(1, 1), padding='valid'
        ),
        **pruning_params_conv)
    )

    pruned_model.add(prune_low_magnitude(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ),
        **pruning_params_conv)
    )

    pruned_model.add(Flatten())

    pruned_model.add(prune_low_magnitude(
        Dense(
            units=300, activation='tanh',
            kernel_initializer=tf.initializers.GlorotUniform()
        ),
        **pruning_params_fc)
    )

    pruned_model.add(prune_low_magnitude(
        Dense(
            units=100, activation='tanh',
            kernel_initializer=tf.initializers.GlorotUniform()
        ),
        **pruning_params_fc)
    )
    pruned_model.add(prune_low_magnitude(
        Dense(
            units=10, activation='softmax'
        ),
        **pruning_params_fc)
    )

    # Compile pruned network
    pruned_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        # optimizer='adam',
        #optimizer=tf.keras.optimizers.Adam(lr=0.0012),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        metrics=['accuracy']
    )

    return pruned_model


callback = [
             UpdatePruningStep(),
             # sparsity.PruningSummaries(log_dir = logdir, profile_batch=0),
             tf.keras.callbacks.EarlyStopping(
                 monitor='val_loss', patience = 3,
                 min_delta=0.001
             )
]



# Normal unpruned model
orig_model = pruned_nn(unpruned, unpruned)

# Strip model of it's pruning parameters
orig_model_stripped = strip_pruning(orig_model)

# Save random wts-
orig_model.save_weights("LeNet_5_MNIST_Random_Weights.h5", overwrite=True)

# Save random wts-
orig_model.save_weights("LeNet_5_MNIST_Winning_Ticket.h5", overwrite=True)

#Network summary-
orig_model_stripped.summary()


orig_model.evaluate(test_x, test_y, verbose = 0)


# Number of convolutional parameters-
conv1 = 60
conv2 = 880

# Number of fully-connected dense parameters-
dense1 = 48120
dense2 = 10164
op_layer = 850


# Total number of parameters-
total_params = conv1 + conv2 + dense1 + dense2 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# Maximum pruning performed is till 0.5% of all parameters-
max_pruned_params = 0.005 * total_params

loc_tot_params = total_params
loc_conv1 = conv1
loc_conv2 = conv2
loc_dense1 = dense1
loc_dense2 = dense2
loc_op_layer = op_layer

# number of pruning rounds-
n = 0

# Lists to hold percentage of wts pruned in each round for all layers in CNN-
conv1_pruning = []
conv2_pruning = []
dense1_pruning = []
dense2_pruning = []
op_layer_pruning = []

while loc_tot_params >= max_pruned_params:
    loc_conv1 *= 0.8  # 20% wts are pruned
    loc_conv2 *= 0.8
    loc_dense1 *= 0.8
    loc_dense2 *= 0.8
    loc_op_layer *= 0.8

    conv1_pruning.append(((conv1 - loc_conv1) / conv1) * 100)
    conv2_pruning.append(((conv2 - loc_conv2) / conv2) * 100)
    dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
    dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
    op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

    loc_tot_params = loc_conv1 + loc_conv2 + loc_dense1 + loc_dense2 + loc_op_layer

    n += 1

    print("\nConv1 = {0:.3f}, Conv2 = {1:.3f}".format(loc_conv1, loc_conv2))
    print("Dense1 = {0:.3f}, Dense2 = {1:.3f} & O/p layer = {2:.3f}".format(
        loc_dense1, loc_dense2, loc_op_layer))
    print("Total number of parameters = {0:.3f}\n".format(loc_tot_params))


print("\nnumber of pruning rounds = {0}\n\n".format(n))


num_pruning_rounds = 10

# Convert from list to np.array-
conv1_pruning = np.array(conv1_pruning)
conv2_pruning = np.array(conv2_pruning)
dense1_pruning = np.array(dense1_pruning)
dense2_pruning = np.array(dense2_pruning)
op_layer_pruning = np.array(op_layer_pruning)


# Round off numpy arrays to 3 decimal digits-
conv1_pruning = np.round(conv1_pruning, decimals=3)
conv2_pruning = np.round(conv2_pruning, decimals=3)
dense1_pruning = np.round(dense1_pruning, decimals=3)
dense2_pruning = np.round(dense2_pruning, decimals=3)
op_layer_pruning = np.round(op_layer_pruning, decimals=3)

conv1_pruning = conv1_pruning / 100
conv2_pruning = conv2_pruning / 100
dense1_pruning = dense1_pruning / 100
dense2_pruning = dense2_pruning / 100
op_layer_pruning = op_layer_pruning / 100

dense1_pruning

# Instantiate a new neural network model for which a mask is to be created,
mask_model = pruned_nn(unpruned, unpruned)



# Strip the model of its pruning parameters-
mask_model_stripped = strip_pruning(mask_model)

# Assign all masks to one-

for wts in mask_model_stripped.trainable_weights:
    wts.assign(
        tf.ones_like(
            input = wts,
            dtype = tf.float32
        )

    )


print("Mask model metrics:")
print("Layer-wise number of nonzero parameters in each layer are: \n")

masked_sum_params = 0

for layer in mask_model_stripped.trainable_weights:
    print(tf.math.count_nonzero(layer, axis = None).numpy())
    masked_sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

print("\nTotal number of trainable parameters = {0}\n".format(masked_sum_params))


print("\nNumber of pruning rounds for LeNet-5 = {0} and number of epochs = {1}\n".format(num_pruning_rounds, num_epochs))


history_main = {}


for x in range(num_pruning_rounds):
    history = {}

    # CNN model, scalar metrics-
    history['accuracy'] = np.zeros(shape=num_epochs)
    history['val_accuracy'] = np.zeros(shape=num_epochs)
    history['loss'] = np.zeros(shape=num_epochs)
    history['val_loss'] = np.zeros(shape=num_epochs)

    # compute % of wts pruned at the end of each iterative pruning round-
    history['percentage_wts_pruned'] = 10

    # history['epoch_length'] = np.zeros(shape = num_epochs)

    history_main[x + 1] = history


history_main.keys()

history_main[10]['accuracy'].shape


orig_sum_params = 60074

# Parameters for Early Stopping-
minimum_delta = 0.001
patience = 3

best_val_loss = 100
loc_patience = 0

for i in range(1, num_pruning_rounds + 1):
    print("\n\n\nIterative pruning round: {0}\n\n".format(i))



    @tf.function
    def train_one_step(model, mask_model, optimizer, x, y):

        with tf.GradientTape() as tape:
            #  predictions using defined model-
            y_pred = model(x)

            #  loss-
            loss = loss_fn(y, y_pred)

        #  gradients wrt defined loss and wts and biases-
        grads = tape.gradient(loss, model.trainable_variables)

        # element-wise multiplication between- computed gradient and masks-
        grad_mask_mul = []

        for grad_layer, mask in zip(grads, mask_model.trainable_weights):
            grad_mask_mul.append(tf.math.multiply(grad_layer, mask))

        # Apply computed gradients to model's wts and biases-
        optimizer.apply_gradients(zip(grad_mask_mul, model.trainable_variables))

        # Compute accuracy-
        train_loss(loss)
        train_accuracy(y, y_pred)

        return None


    @tf.function
    def test_step(model, optimizer, data, labels):


        predictions = model(data)
        t_loss = loss_fn(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

        return None


    # Instantiate a model
    model_gt = pruned_nn(unpruned, unpruned)

    # Load winning ticket (from above)-
    model_gt.load_weights("LeNet_5_MNIST_Winning_Ticket.h5")

    # Strip model of pruning parameters-
    model_gt_stripped = strip_pruning(model_gt)

    # Initialize parameters for Early Stopping manual implementation-
    best_val_loss = 100
    loc_patience = 0
    for epoch in range(num_epochs):

        if loc_patience >= patience:
            print("\n'EarlyStopping' called!\n")
            break

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x, y in train_dataset:

            train_one_step(model_gt_stripped, mask_model_stripped, optimizer, x, y)

        for x_t, y_t in test_dataset:

            test_step(model_gt_stripped, optimizer, x_t, y_t)

        template = 'Epoch {0}, Loss: {1:.4f}, Accuracy: {2:.4f}, Test Loss: {3:.4f}, Test Accuracy: {4:4f}'

        # 'i' is the index for number of pruning rounds-
        history_main[i]['accuracy'][epoch] = train_accuracy.result() * 100
        history_main[i]['loss'][epoch] = train_loss.result()
        history_main[i]['val_loss'][epoch] = test_loss.result()
        history_main[i]['val_accuracy'][epoch] = test_accuracy.result() * 100

        print(template.format(epoch + 1,
                              train_loss.result(), train_accuracy.result() * 100,
                              test_loss.result(), test_accuracy.result() * 100))

        model_sum_params = 0

        for layer in model_gt_stripped.trainable_weights:

            model_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

        print("Total number of trainable parameters = {0}\n".format(model_sum_params))


        if np.abs(test_loss.result() < best_val_loss) >= minimum_delta:
            # update 'best_val_loss' variable to lowest loss encountered so far-
            best_val_loss = test_loss.result()

            # reset 'loc_patience' variable-
            loc_patience = 0

        else:  # there is no improvement in monitored metric 'val_loss'
            loc_patience += 1  # number of epochs without any improvement

    # Resize numpy arrays according to the epoch when 'EarlyStopping' was called-
    for metrics in history_main[i].keys():
        history_main[i][metrics] = np.resize(history_main[i][metrics], new_shape=epoch)


    # Save trained model wts-
    model_gt.save_weights("LeNet_5_MNIST_Trained_Weights.h5", overwrite=True)

    # Prune trained model:


    # Parameters fro Conv layer pruning-
    pruning_params_conv = {
        'pruning_schedule': Prune.ConstantSparsity(
            target_sparsity=conv1_pruning[i - 1], begin_step=1000,
            end_step=end_step, frequency=100
        )
    }

    # parameters for Fully-Connected layer pruning-
    pruning_params_fc = {
        'pruning_schedule': Prune.ConstantSparsity(
            target_sparsity=dense1_pruning[i - 1], begin_step=1000,
            end_step=end_step, frequency=100
        )
    }

    # Instantiate a Neural Network model to be pruned using parameters from above-
    pruned_model = pruned_nn(pruning_params_conv, pruning_params_fc)

    # Load wts from original trained and unpruned model-
    pruned_model.load_weights("LeNet_5_MNIST_Trained_Weights.h5")

    # Train pruned NN-
    history_pruned = pruned_model.fit(
        x=train_x, y=train_y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callback,
        validation_data=(test_x, test_y),
        shuffle=True
    )

    # Strip the pruning wrappers from pruned model-
    pruned_model_stripped = strip_pruning(pruned_model)

    pruned_sum_params = 0

    for layer in pruned_model_stripped.trainable_weights:

        pruned_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

    print("\nRound = {0}, total number of trainable parameters = {1}\n".format(i, pruned_sum_params))


    # 'i' is the index for number of pruning rounds-
    history_main[i]['percentage_wts_pruned'] = ((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100

    # Save weights of PRUNED and Trained model BEFORE stripping-
    pruned_model.save_weights("LeNet_5_MNIST_Pruned_Weights.h5", overwrite=True)



    # Mask
    mask_model = pruned_nn(unpruned, unpruned)

    # Load weights of PRUNED model-
    mask_model.load_weights("LeNet_5_MNIST_Pruned_Weights.h5")
    mask_model_stripped = strip_pruning(mask_model)

    # Reinitialize surviving wts to 1 -
    for wts in mask_model_stripped.trainable_weights:
        wts.assign(tf.where(tf.equal(wts, 0.), 0., 1.))

        # Instantiate a new neural network model for which, the weights are to be extracted-
    winning_ticket_model = pruned_nn(unpruned, unpruned)

        # Load weights of PRUNED model-
    winning_ticket_model.load_weights("LeNet_5_MNIST_Pruned_Weights.h5")

        # Strip the model of its pruning parameters-
    winning_ticket_model_stripped = strip_pruning(winning_ticket_model)

        # Reinitialize wts to the original value
    for orig_wts, pruned_wts in zip(orig_model_stripped.trainable_weights,
                                        winning_ticket_model_stripped.trainable_weights):
            pruned_wts.assign(tf.where(tf.equal(pruned_wts, 0), pruned_wts, orig_wts))

        # Save the weights (with pruning parameters) extracted to a file-
    winning_ticket_model.save_weights("LeNet_5_MNIST_Winning_Ticket.h5", overwrite=True)


print("Iterative-pruning complete.")

