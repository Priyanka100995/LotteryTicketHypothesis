import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow_model_optimization as tfmot
#from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning, prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep, callbacks
# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D, ReLU
from tensorflow.keras import models, layers, datasets
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
# import math
from sklearn.metrics import accuracy_score, precision_score, recall_score

import Prune

batch_size = 60
num_classes = 10
num_epochs = 50

# input image dimensions
img_rows, img_cols = 28, 28

# Load MNIST dataset-
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

# Convert datasets to floating point types-
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

# Normalize the training and testing datasets-
train_x /= 255.0
test_x /= 255.0


# convert class vectors/target to binary class matrices or one-hot encoded values-
train_y = tf.keras.utils.to_categorical(train_y, num_classes)
test_y = tf.keras.utils.to_categorical(test_y, num_classes)

# Reshape training and testing sets-
train_x = train_x.reshape(train_x.shape[0], 784)
test_x = test_x.reshape(test_x.shape[0], 784)


print("\nDimensions of training and testing sets are:")
print("train_x.shape = {0}, train_y.shape = {1}".format(train_x.shape, train_y.shape))
print("test_x.shape = {0}, y_test.shape = {1}".format(test_x.shape, test_y.shape))

# Create training and testing datasets-
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))


train_dataset = train_dataset.shuffle(buffer_size = 20000, reshuffle_each_iteration = True).batch(batch_size = batch_size, drop_remainder = False)

test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)

# Choose an optimizer and loss function for training-
loss_fn = tf.keras.losses.CategoricalCrossentropy()
#optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
optimizer = tf.keras.optimizers.Adam(lr = 0.0012)


# Select metrics to measure the error & accuracy of model.
# These metrics accumulate the values over epochs and then
# print the overall result-
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'test_accuracy')

# The model is first trained without any pruning for 'num_epochs' epochs-
epochs = num_epochs

num_train_samples = train_x.shape[0]

end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs

print("'end_step parameter' for this dataset =  {0}".format(end_step))

# Specify the parameters to be used for layer-wise pruning, NO PRUNING is done here:
pruning_params_unpruned = {
    'pruning_schedule': Prune.ConstantSparsity(
        target_sparsity=0.0, begin_step=0,
        end_step = end_step, frequency=100
    )
}


l = tf.keras.layers

def pruned_nn(pruning_params):


    pruned_model = Sequential()
    pruned_model.add(l.InputLayer(input_shape=(784,)))
    pruned_model.add(Flatten())
    pruned_model.add(prune_low_magnitude(
        Dense(units=300, activation='relu', kernel_initializer=tf.initializers.GlorotUniform()),
        **pruning_params))
    # pruned_model.add(l.Dropout(0.2))
    pruned_model.add(prune_low_magnitude(
        Dense(units=100, activation='relu', kernel_initializer=tf.initializers.GlorotUniform()),
        **pruning_params))
    # pruned_model.add(l.Dropout(0.1))
    pruned_model.add(prune_low_magnitude(
        Dense(units=num_classes, activation='softmax'),
        **pruning_params))

    # Compile pruned CNN-
    pruned_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(lr=0.0012),
        metrics=['accuracy'])

    return pruned_model


# Add a pruning step callback

callback = [
             UpdatePruningStep(),
             # sparsity.PruningSummaries(log_dir = logdir, profile_batch=0),
             tf.keras.callbacks.EarlyStopping(
                 monitor='val_loss', patience = 3,
                 min_delta=0.001
             )
]

# Initialize a CNN model-
orig_model = pruned_nn(pruning_params_unpruned)


# Strip model of it's pruning parameters
orig_model_stripped = strip_pruning(orig_model)

# Save random weights-
orig_model.save_weights("LeNet_MNIST_Ramdom_Weights.h5", overwrite=True)


# Save random weights-
orig_model.save_weights("LeNet_MNIST_Winning_Ticket.h5", overwrite=True)


# Get CNN summary-
orig_model_stripped.summary()

# number of fully-connected dense parameters-
dense1 = 235500
dense2 = 30100
op_layer = 1010


# total number of parameters-
total_params = dense1 + dense2 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# maximum pruning performed is till 0.5% of all parameters-
max_pruned_params = 0.002 * total_params


loc_tot_params = total_params
loc_dense1 = dense1
loc_dense2 = dense2
loc_op_layer = op_layer

# variable to count number of pruning rounds-
n = 0


# Lists to hold percentage of weights pruned in each round for all layers in CNN-
dense1_pruning = []
dense2_pruning = []
op_layer_pruning = []

while loc_tot_params >= max_pruned_params:
    loc_dense1 *= 0.8   # 20% weights are pruned
    loc_dense2 *= 0.8   # 20% weights are pruned
    loc_op_layer *= 0.8 # 20% weights are pruned

    dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
    dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
    op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

    loc_tot_params = loc_dense1 + loc_dense2 + loc_op_layer

    n += 1

    print("Dense1 = {0:.3f}, Dense2 = {1:.3f} & O/p layer = {2:.3f}".format(
        loc_dense1, loc_dense2, loc_op_layer))
    print("Total number of parameters = {0:.3f}\n".format(loc_tot_params))



print("\nnumber of pruning rounds = {0}\n\n".format(n))

num_pruning_rounds = 100


# Convert from list to np.array-
dense1_pruning = np.array(dense1_pruning)
dense2_pruning = np.array(dense2_pruning)
op_layer_pruning = np.array(op_layer_pruning)

# Round off numpy arrays to 3 decimal digits-
dense1_pruning = np.round(dense1_pruning, decimals=3)
dense2_pruning = np.round(dense2_pruning, decimals=3)
op_layer_pruning = np.round(op_layer_pruning, decimals=3)

dense1_pruning


dense1_pruning = dense1_pruning / 100
dense2_pruning = dense2_pruning / 100
op_layer_pruning = op_layer_pruning / 100

dense1_pruning

# Instantiate a new neural network model for which, the mask is to be created,
# according to the paper-
mask_model = pruned_nn(pruning_params_unpruned)

# Strip the model of its pruning parameters-
mask_model_stripped = strip_pruning(mask_model)


for wts in mask_model_stripped.trainable_weights:
    wts.assign(
        tf.ones_like(
            input = wts,
            dtype = tf.float32
        )

    )



print("\nMask model metrics:")
print("layer-wise number of nonzero parameters in each layer are: \n")

masked_sum_params = 0

for layer in mask_model_stripped.trainable_weights:
    print(tf.math.count_nonzero(layer, axis = None).numpy())
    masked_sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

print("\nTotal number of trainable parameters = {0}\n".format(masked_sum_params))

print("\nnumber of pruning rounds for LeNet NN = {0} and number of epochs = {1}\n".format(num_pruning_rounds, num_epochs))

history_main = {}

# for x in range(num_pruning_rounds + 1):
for x in range(num_pruning_rounds):
    history = {}

    # Neural Network model, scalar metrics-
    history['accuracy'] = np.zeros(shape=num_epochs)
    history['val_accuracy'] = np.zeros(shape=num_epochs)
    history['loss'] = np.zeros(shape=num_epochs)
    history['val_loss'] = np.zeros(shape=num_epochs)

    # compute % of weights pruned at the end of each iterative pruning round-
    history['percentage_wts_pruned'] = 90

    history_main[x + 1] = history

history_main.keys()

history_main[10]['accuracy'].shape

# User input parameters for Early Stopping in manual implementation-
minimum_delta = 0.001
patience = 3
best_val_loss = 100
loc_patience = 0
orig_sum_params = 266610

for i in range(1, num_pruning_rounds + 1):

    print("\n\n\nIterative pruning round: {0}\n\n".format(i))


    # Define 'train_one_step()' and 'test_step()' functions here-
    @tf.function
    def train_one_step(model, mask_model, optimizer, x, y):
        '''
        Function to compute one step of gradient descent optimization
        '''
        with tf.GradientTape() as tape:
            # Make predictions using defined model-
            y_pred = model(x)

            # Compute loss-
            loss = loss_fn(y, y_pred)

        # Compute gradients wrt defined loss and weights and biases-
        grads = tape.gradient(loss, model.trainable_variables)

        # type(grads)
        # list

        # List to hold element-wise multiplication between-
        # computed gradient and masks-
        grad_mask_mul = []

        # Perform element-wise multiplication between computed gradients and masks-
        for grad_layer, mask in zip(grads, mask_model.trainable_weights):
            grad_mask_mul.append(tf.math.multiply(grad_layer, mask))

        # Apply computed gradients to model's weights and biases-
        optimizer.apply_gradients(zip(grad_mask_mul, model.trainable_variables))

        # Compute accuracy-
        train_loss(loss)
        train_accuracy(y, y_pred)

        return None


    @tf.function
    def test_step(model, optimizer, data, labels):
        """
        Function to test model performance
        on testing dataset
        """

        predictions = model(data)
        t_loss = loss_fn(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

        return None


    # Instantiate a model
    model_gt = pruned_nn(pruning_params_unpruned)

    # Load winning ticket (from above)-
    model_gt.load_weights("LeNet_MNIST_Winning_Ticket.h5")

    # Strip model of pruning parameters-
    model_gt_stripped = strip_pruning(model_gt)

    # Train model using 'GradientTape'-

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
            # train_one_step(model_gt_stripped, mask_model, optimizer, x, y, grad_mask_mul)
            train_one_step(model_gt_stripped, mask_model_stripped, optimizer, x, y)

        for x_t, y_t in test_dataset:
            # test_step(x_t, y_t)
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

        # Count number of non-zero parameters in each layer and in total-
        # print("layer-wise manner model, number of nonzero parameters in each layer are: \n")

        model_sum_params = 0

        for layer in model_gt_stripped.trainable_weights:
            # print(tf.math.count_nonzero(layer, axis = None).numpy())
            model_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

        print("Total number of trainable parameters = {0}\n".format(model_sum_params))

        # Code for manual Early Stopping:
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
        # history[metrics] = np.resize(history[metrics], new_shape=epoch)

    # Save trained model weights-
    model_gt.save_weights("LeNet_MNIST_Trained_Weights.h5", overwrite=True)

    # Prune trained model:

    # print("\n% of weights to be pruned in round = {0} is: {1:.4f}\n".format(i, wts_np[i - 1]))

    # Specify the parameters to be used for layer-wise pruning, Fully-Connected layer pruning-
    pruning_params_fc = {
        'pruning_schedule': Prune.ConstantSparsity(
            target_sparsity=dense1_pruning[i - 1], begin_step=1000,
            end_step=end_step, frequency=100
        )
    }

    # Instantiate a Nueal Network model to be pruned using parameters from above-
    pruned_model = pruned_nn(pruning_params_fc)

    # Load weights from original trained and unpruned model-
    pruned_model.load_weights("LeNet_MNIST_Trained_Weights.h5")

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

    # print("\nIn pruned model, number of nonzero parameters in each layer are: \n")
    pruned_sum_params = 0

    for layer in pruned_model_stripped.trainable_weights:
        # print(tf.math.count_nonzero(layer, axis = None).numpy())
        pruned_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

    print("\nRound = {0}, total number of trainable parameters = {1}\n".format(i, pruned_sum_params))
    # print("\nTotal number of trainable parameters = {0}\n".format(pruned_sum_params))

    '''
    # Sanity-check: confirm that the weights are actually pruned away from the network-
    print("\nRound = {0}, % of weights pruned away = {1:.2f}%\n".format( \
                                                i, (orig_sum_params - pruned_sum_params) / orig_sum_params * 100))
    '''

    # 'i' is the index for number of pruning rounds-
    history_main[i]['percentage_wts_pruned'] = ((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100

    # Save weights of PRUNED and Trained model BEFORE stripping-
    pruned_model.save_weights("LeNet_MNIST_Pruned_Weights.h5", overwrite=True)

    # Create a mask:

    # Instantiate a new neural network model for which, the mask is to be created,
    mask_model = pruned_nn(pruning_params_unpruned)

    # Load weights of PRUNED model-
    mask_model.load_weights("LeNet_MNIST_Pruned_Weights.h5")

    # Strip the model of its pruning parameters-
    mask_model_stripped = strip_pruning(mask_model)

    # For each layer, for each weight which is 0, leave it, as is.
    # And for weights which survive the pruning,reinitialize it to ONE (1)-
    for wts in mask_model_stripped.trainable_weights:
        wts.assign(tf.where(tf.equal(wts, 0.), 0., 1.))

    # Extract Winning Ticket:

    # Instantiate a new neural network model for which, the weights are to be extracted-
    winning_ticket_model = pruned_nn(pruning_params_unpruned)

    # Load weights of PRUNED model-
    winning_ticket_model.load_weights("LeNet_MNIST_Pruned_Weights.h5")

    # Strip the model of its pruning parameters-
    winning_ticket_model_stripped = strip_pruning(winning_ticket_model)

    # For each layer, for each weight which is 0, leave it, as is. And for weights which survive the pruning,
    # reinitialize it to the value, the model received BEFORE it was trained and pruned-
    for orig_wts, pruned_wts in zip(orig_model_stripped.trainable_weights,
                                    winning_ticket_model_stripped.trainable_weights):
        pruned_wts.assign(tf.where(tf.equal(pruned_wts, 0), pruned_wts, orig_wts))

    # Save the weights (with pruning parameters) extracted to a file-
    winning_ticket_model.save_weights("LeNet_MNIST_Winning_Ticket.h5", overwrite=True)


import os
import pickle

os.getcwd()

with open("C:/Users/Admin/PycharmProjects/LeNet/Experiment_Results/LeNet-5_MNIST_history_main_Experiment_1.pkl", "wb") as f:
    pickle.dump(history_main, f)

with open("C:/Users/Admin/PycharmProjects/LeNet/Experiment_Results/LeNet-5_MNIST_history_main_Experiment_1.pkl", "rb") as f:
    history_main = pickle.load(f)

history_main[10]['percentage_wts_pruned']

plot_accuracy = {}
plot_test_accuracy = {}

for k in history_main.keys():
    epoch_length = len(history_main[k]['accuracy'])
    plot_accuracy[history_main[k]['percentage_wts_pruned']] = history_main[k]['accuracy'][epoch_length - 1]


# populate 'plot_test_accuracy'-
for k in history_main.keys():
    epoch_length = len(history_main[k]['accuracy'])
    plot_test_accuracy[history_main[k]['percentage_wts_pruned']] = history_main[k]['val_accuracy'][epoch_length - 1]


# Visualization of training and testing accuracy VS percentage of weights
# pruned-
fig=plt.figure(figsize=(9, 7), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_accuracy.keys()), list(plot_accuracy.values()), label = 'training_accuracy', marker='*')
plt.plot(list(plot_test_accuracy.keys()), list(plot_test_accuracy.values()), label = 'testing_accuracy', marker='*')

plt.title("Percentage of weights pruned VS Accuracy (at convergence)")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("Accuracy")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()

# Python 3 dict for training and testing loss visualization-
plot_loss = {}
plot_test_loss = {}

# populate 'plot_loss'-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_loss[history_main[k]['percentage_wts_pruned']] = history_main[k]['loss'][epoch_length - 1]


# populate 'plot_test_loss'-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_test_loss[history_main[k]['percentage_wts_pruned']] = history_main[k]['val_loss'][epoch_length - 1]

# Visualization of training and testing loss VS percentage of remaining weights-pruned-
fig=plt.figure(figsize=(9, 7), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_loss.keys()), list(plot_loss.values()), label = 'training_loss', marker='*')
plt.plot(list(plot_test_loss.keys()), list(plot_test_loss.values()), label = 'testing_loss', marker='*')

plt.title("Percentage of weights pruned VS Loss (at convergence)")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("Loss")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()

# Plot number of epochs (Early Stopping) VS percentage of weights pruned-

# Python 3 dict to hold number of epochs vs % of weights pruned-
plot_num_epochs = {}
plot_num_epochs_test = {}


# populate 'plot_num_epochs'-
for k in history_main.keys():
    num_epochs = len(history_main[k]['accuracy'])
    plot_num_epochs[history_main[k]['percentage_wts_pruned']] = num_epochs

# populate 'plot_num_epochs_test'-


# Visualize percentage of weights remaining VS number of epochs (Early Stopping)
fig=plt.figure(figsize=(9, 7), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_num_epochs.keys()), list(plot_num_epochs.values()), label = 'training set', marker='*')
#plt.plot(list(plot_num_epochs_test.keys()), list(plot_num_epochs_test.values()), label = 'testing set')

plt.title(" Percentage of weights pruned VS number of epochs (Early Stopping) until convergence")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("number of epochs")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()


# Python 3 dict for visualization-
plot_starting_accuracy = {}
plot_starting_test_accuracy = {}


# populate 'plot_starting_accuracy'-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_starting_accuracy[history_main[k]['percentage_wts_pruned']] = history_main[k]['accuracy'][0]

# populate 'plot_starting_test_accuracy'-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_starting_test_accuracy[history_main[k]['percentage_wts_pruned']] = history_main[k]['val_accuracy'][0]


# Visualize starting accuracy VS percentage of weights pruned-
fig=plt.figure(figsize=(9, 7), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_starting_accuracy.keys()), list(plot_starting_accuracy.values()), label = 'training starting accuracy', marker='*')
plt.plot(list(plot_starting_test_accuracy.keys()), list(plot_starting_test_accuracy.values()), label = 'testing starting accuracy', marker='*')

plt.title(" Percentage of weights pruned VS Starting Accuracy")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("Accuracy")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()

# Python 3 dict for visualization-
plot_starting_loss = {}
plot_starting_test_loss = {}

# Populate 'plot_starting_loss' Python 3 dict-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_starting_loss[history_main[k]['percentage_wts_pruned']] = history_main[k]['loss'][0]

# Populate 'plot_starting_test_loss' Python 3 dict-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_starting_test_loss[history_main[k]['percentage_wts_pruned']] = history_main[k]['val_loss'][0]

# Visualize Starting training & testing loss VS percentage of weights pruned-
fig=plt.figure(figsize=(9, 7), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_starting_loss.keys()), list(plot_starting_loss.values()), label = 'training starting loss',  marker='*')
plt.plot(list(plot_starting_test_loss.keys()), list(plot_starting_test_loss.values()), label = 'testing starting loss', marker='*')

plt.title("Percentage of weights pruned VS Starting Loss")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("Loss")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()

# Python 3 dict for visualization-
plot_best_accuracy = {}
plot_best_test_accuracy = {}


# Populate 'plot_best_accuracy' Python 3 dict-
for k in history_main.keys():
    epoch_length = len(history_main[k]['accuracy'])
    plot_best_accuracy[history_main[k]['percentage_wts_pruned']] = np.amax(history_main[k]['accuracy'])

# Populate 'plot_best_test_accuracy' Python 3 dict-
for k in history_main.keys():
    epoch_length = len(history_main[k]['accuracy'])
    plot_best_test_accuracy[history_main[k]['percentage_wts_pruned']] = np.amax(history_main[k]['val_accuracy'])


# Visualize best accuracy VS percentage of weights pruned-
fig=plt.figure(figsize=(10, 9), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_best_accuracy.keys()), list(plot_best_accuracy.values()), label = 'training best accuracy', marker='*')
plt.plot(list(plot_best_test_accuracy.keys()), list(plot_best_test_accuracy.values()), label = 'testing best accuracy', marker='*')

plt.title("Percentage of weights pruned VS Best Accuracy")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("Accuracy")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()

# Python 3 dict for visualization-
plot_best_loss = {}
plot_best_test_loss = {}

# Populate 'plot_best_loss' Python 3 dict-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_best_loss[history_main[k]['percentage_wts_pruned']] = np.amin(history_main[k]['loss'])

# Populate 'plot_best_test_loss' Python 3 dict-
for k in history_main.keys():
    epoch_length = len(history_main[k]['loss'])
    plot_best_test_loss[history_main[k]['percentage_wts_pruned']] = np.amin(history_main[k]['val_loss'])

# Visualize best loss VS percentage of weights pruned-
fig=plt.figure(figsize=(10, 9), dpi= 100, facecolor='w', edgecolor='k')

plt.plot(list(plot_best_loss.keys()), list(plot_best_loss.values()), label = 'training best loss', marker='*')
plt.plot(list(plot_best_test_loss.keys()), list(plot_best_test_loss.values()), label = 'testing best loss', marker='*')

plt.title("Percentage of weights pruned VS Best Loss")
plt.xlabel("percentage_wts_pruned")
plt.ylabel("Loss")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()

# Find highest training and testing accuracy for all iterative pruning rounds-
best_accuracy = 0
iterative_round = 0


for k in history_main.keys():
	epoch_length = len(history_main[k]['accuracy'])
	if history_main[k]['accuracy'][epoch_length - 1] > best_accuracy:
		best_accuracy = history_main[k]['accuracy'][epoch_length - 1]
		iterative_round = k


print("\nIterative round = {0} has highest training accuracy = {1:.4f}% at percentage of weights pruned = {2:.4f}%\n".format(
    iterative_round, best_accuracy, history_main[iterative_round]['percentage_wts_pruned']))


best_test_accuracy = 0
iterative_round = 0


for k in history_main.keys():
	epoch_length = len(history_main[k]['val_accuracy'])
	if history_main[k]['val_accuracy'][epoch_length - 1] > best_test_accuracy:
		best_test_accuracy = history_main[k]['val_accuracy'][epoch_length - 1]
		iterative_round = k


print("\nIterative round = {0} has highest testing accuracy = {1:.4f}% at percentage of weights pruned = {2:.4f}%\n".format(
    iterative_round, best_test_accuracy, history_main[iterative_round]['percentage_wts_pruned']))

fig=plt.figure(figsize=(10, 9), dpi= 100, facecolor='w', edgecolor='k')

for k in history_main.keys():
    plt.plot(history_main[k]['accuracy'], label = 'training_accuracy_epoch-{0}'.format(k))
    # plt.plot(history_main[k]['val_accuracy'], label = 'testing_accuracy_epoch-{0}'.format(k))

plt.title("Training Visualization [Scalar Metrics]")
plt.xlabel("number of epochs")
plt.ylabel("accuracy accuracy")
plt.legend(loc = 'best')
scale_factor = 2.5
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

plt.xlim(xmin * scale_factor, xmax * scale_factor)
plt.ylim(ymin * scale_factor, ymax * scale_factor)
plt.grid()
plt.show()