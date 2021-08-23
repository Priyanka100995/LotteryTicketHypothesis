import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential



batch_size = 60
num_classes = 10
num_epochs = 20

# Data preprocessing and cleaning:
num_classes = 10

    # input image dimensions
img_rows, img_cols = 28, 28

    # Load MNIST dataset-
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
       X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
       X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
       input_shape = (1, img_rows, img_cols)
else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

print("\n'input_shape' which will be used = {0}\n".format(input_shape))

    # Convert datasets to floating point types-
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

    # Normalize the training and testing datasets-
X_train /= 255.0
X_test /= 255.0

    # convert class vectors/target to binary class matrices or one-hot encoded values-
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Reshape training and testing sets-
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
print("\nDimensions of training and testing sets are:")
print("X_train.shape = {0}, y_train.shape = {1}".format(X_train.shape, y_train.shape))
print("X_test.shape = {0}, y_test.shape = {1}".format(X_test.shape, y_test.shape))

  # Create training and testing datasets-
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.shuffle(buffer_size = 20000, reshuffle_each_iteration = True).batch(batch_size = batch_size, drop_remainder = False)
test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)


    # Choose an optimizer and loss function for training-
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr = 0.0012)
#optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


    # These metrics accumulate the values over epochs and then print the overall result-
train_loss = tf.keras.metrics.Mean(name = 'train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')

test_loss = tf.keras.metrics.Mean(name = 'test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'test_accuracy')
epochs = num_epochs
num_train_samples = X_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print("'end_step parameter' for this dataset =  {0}".format(end_step))

def lenet():
    """
    Function to define the architecture of a neural network model
    following Lenet Fully connected architecture for MNIST dataset and using
    provided parameter which are used to prune the model.


    300, 100, 10  -- fully connected layers


    Output: Returns designed and compiled neural network model
    """

    pruned_model = Sequential()

    pruned_model.add(tf.keras.layers.InputLayer(input_shape=(784,)))
    pruned_model.add(Flatten())

    pruned_model.add(
        Dense(
            units=300, activation='tanh',
            kernel_initializer=tf.initializers.GlorotNormal()
        )
    )

    pruned_model.add(
        Dense(
            units=100, activation='tanh',
            kernel_initializer=tf.initializers.GlorotNormal()
        )
    )

    pruned_model.add(
        Dense(
            units=10, activation='softmax'
        )
    )

    # Compile pruned CNN-
    pruned_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(lr=0.0012),
        metrics=['accuracy']
    )

    return pruned_model


 # Add a pruning step callback to peg the pruning step to the optimizer's
    # step.
callback = [


        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3,
            min_delta=0.001
        )

    ]

# Initialize an FC model-
orig_model = lenet()

# Save random weights-
orig_model.save_weights("Lenet_MNIST_Random_Gaussian_Glorot_Weights.h5", overwrite=True)

# Get CNN summary-
# orig_model_stripped.summary()
orig_model.summary()


# number of fully-connected dense parameters-
dense1 = 235500
dense2 = 30100
op_layer = 1010


# total number of parameters-
total_params = dense1 + dense2 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# maximum pruning performed is till 0.5% of all parameters-
max_pruned_params = 0.005 * total_params

loc_tot_params = total_params

loc_dense1 = dense1
loc_dense2 = dense2
loc_op_layer = op_layer

# variable to count number of pruning rounds-
n = 0


dense1_pruning = []
dense2_pruning = []
op_layer_pruning = []

while loc_tot_params >= max_pruned_params:

    loc_dense1 *= 0.8  # 20% weights are pruned
    loc_dense2 *= 0.8  # 20% weights are pruned
    loc_op_layer *= 0.9  # 10% weights are pruned


    dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
    dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
    op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

    loc_tot_params =  loc_dense1 + loc_dense2 + loc_op_layer

    n += 1

print("\nnumber of pruning rounds = {0}\n\n".format(n))
num_pruning_rounds = n

# Round off numpy arrays to 3 decimal digits-

dense1_pruning = np.round(dense1_pruning, decimals=3)
dense2_pruning = np.round(dense2_pruning, decimals=3)
op_layer_pruning = np.round(op_layer_pruning, decimals=3)
dense1_pruning = np.insert(dense1_pruning,0,0)
op_layer_pruning = np.insert(op_layer_pruning,0,0)

print("Dense Layer Pruning Percent",dense1_pruning)
print("Output Layer Pruning Percent",op_layer_pruning)


# Instantiate a new neural network model for which, the mask is to be created,
# according to the paper-
mask_model = lenet()

# Assign all masks to one-

for wts in mask_model.trainable_weights:
# for wts in mask_model_stripped.trainable_weights:
    wts.assign(
        tf.ones_like(
            input = wts,
            dtype = tf.float32
        )

    )
    # wts.assign(1.)
    # wts.assign(tf.where(tf.equal(wts, 0.), 0., 1.))

print("\nMask model metrics:")
print("layer-wise number of nonzero parameters in each layer are: \n")

masked_sum_params = 0

for layer in mask_model.trainable_weights:
# for layer in mask_model_stripped.trainable_weights:
    # print(tf.math.count_nonzero(layer, axis = None).numpy())
    masked_sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

print("\nTotal number of trainable parameters = {0}\n".format(masked_sum_params))

#print("\nnumber of pruning rounds for Lenet FC = {0} and number of epochs = {1}\n".format(num_pruning_rounds, num_epochs))

History_data = {}
history_main = {}

num_trials = 5


# for x in range(num_pruning_rounds + 1):
''''for x in range(num_pruning_rounds):
    history = {}

    # Neural Network model, scalar metrics-
    history['accuracy'] = np.zeros(shape=num_epochs)
    history['val_accuracy'] = np.zeros(shape=num_epochs)
    history['loss'] = np.zeros(shape=num_epochs)
    history['val_loss'] = np.zeros(shape=num_epochs)

    # compute % of weights pruned at the end of each iterative pruning round-
    history['percentage_wts_pruned'] = 90

    history_main[x + 1] = history
    History_data [x + 1] = history


history_main.keys()'''''


def prune_lenet(model, pruning_params_fc, pruning_params_op):
    '''
    Function to prune top p% of trained weights using the provided parameters using
    magnitude-based weight pruning.

    Inputs:
    'model' is the TensorFlow 2.0 defined  neural network
    'pruning_params_fc' is the percentage of weights to prune for dense, fully-connected layer
    'pruning_params_op' is the percentage of weights to prune for output layer

    Returns:
    Python list containing pruned layers
    '''

    # List variable to hold magnitude-based pruned weights-
    pruned_weights = []

    for layer in model.trainable_weights:
        x = layer.numpy()

        if len(layer.shape) == 2 and layer.shape[1] != 10:
            # this is a fully-connected dense layer
            print("dense layer: {0}, pruning rate = {1}%".format(layer.shape, pruning_params_fc))

            # Compute absolute value of 'x'-
            x_abs = np.abs(x)

            # Mask values to zero which are less than 'p' in terms of magnitude-
            x_abs[x_abs < np.percentile(x_abs, pruning_params_fc)] = 0

            # Where 'x_abs' equals 0, keep 0, else, replace with values
            # of 'x'-
            # OR
            # If x_abs == 0 (condition) is True, use the value of 0, otherwise
            # use the value in 'x'
            x_mod = np.where(x_abs == 0, 0, x)

            pruned_weights.append(x_mod)

        elif len(layer.shape) == 2 and layer.shape[1] == 10:
            # this is the output layer
            print("op layer: {0}, pruning rate = {1}%".format(layer.shape, pruning_params_op))

            # Compute absolute value of 'x'-
            x_abs = np.abs(x)

            # Mask values to zero which are less than 'p' in terms of magnitude-
            x_abs[x_abs < np.percentile(x_abs, pruning_params_op)] = 0

            # Where 'x_abs' equals 0, keep 0, else, replace with values
            # of 'x'-
            # OR
            # If x_abs == 0 (condition) is True, use the value of 0, otherwise
            # use the value in 'x'
            x_mod = np.where(x_abs == 0, 0, x)

            pruned_weights.append(x_mod)

        elif len(layer.shape) == 1:
            # bias does not have to be pruned-
            # print("layer: {0}, pruning rate = {1}%".format(layer.shape, 0))
            pruned_weights.append(x)

    return pruned_weights

orig_model_pruned = prune_lenet(orig_model, dense1_pruning[0], op_layer_pruning[0])

# User input parameters for Early Stopping in manual implementation-
minimum_delta = 0.001
patience = 3

best_val_loss = 100
loc_patience = 0

orig_sum_params = total_params
print("\nTotal number of parameters in overparametrized, original, unpruned network = ", orig_sum_params)


winning_ticket_model = lenet()

# Use Randomly initialized weights-
winning_ticket_model.set_weights(orig_model.get_weights())

num_pruning_rounds=25

for j in range (1, num_trials+1):
    print("Trials", j)
    for i in range(1, num_pruning_rounds+1):
        history = {}

        # Neural Network model, scalar metrics-
        history['accuracy'] = np.zeros(shape=num_epochs)
        history['val_accuracy'] = np.zeros(shape=num_epochs)
        history['loss'] = np.zeros(shape=num_epochs)
        history['val_loss'] = np.zeros(shape=num_epochs)
        history['iterations'] = np.zeros(shape=num_epochs)  # updated

        # compute % of weights pruned at the end of each iterative pruning round-
        history['percentage_wts_pruned'] = np.zeros(shape=num_epochs)

        history_main[(i) ] = history
        history_main.keys()


    print("HISTORY keys", history_main.keys())




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

            # type(grads)c
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
        # model_gt = pruned_nn(pruning_params_unpruned)
        model_gt = lenet()

        # Load winning ticket (from above)-

        model_gt.set_weights(winning_ticket_model.get_weights())

        # Strip model of pruning parameters-
        # model_gt_stripped = sparsity.strip_pruning(model_gt)

        # Train model using 'GradientTape'-

        # Initialize parameters for Early Stopping manual implementation-
        best_val_loss = 100
        loc_patience = 0

        if i == 1:  # (For every 1st round of every trial set to initial wts)

            # Load winning ticket (from above)-
            model_gt.load_weights("Lenet_MNIST_Random_Gaussian_Glorot_Weights.h5")


        else:
            # Load winning ticket (from above)-
            model_gt.set_weights(winning_ticket_model.get_weights())


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
                # train_one_step(model_gt_stripped, mask_model_stripped, optimizer, x, y)
                train_one_step(model_gt, mask_model, optimizer, x, y)

            for x_t, y_t in test_dataset:
                # test_step(model_gt_stripped, optimizer, x_t, y_t)
                test_step(model_gt, optimizer, x_t, y_t)

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

            for layer in model_gt.trainable_weights:
                # for layer in model_gt_stripped.trainable_weights:
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


        # Prune trained model:

        # print("\n% of weights to be pruned in round = {0} is: {1:.4f}\n".format(i, wts_np[i - 1]))

        # Prune neural network-
        # pruned_weights = prune_lenet(model_gt, dense1_pruning[i - 1], op_layer_pruning[i - 1])
        pruned_weights = prune_lenet(
            model=model_gt,
            pruning_params_fc=dense1_pruning[i - 1],
            pruning_params_op=op_layer_pruning[i - 1]
        )

        # Instantiate a Neural Network model
        pruned_model = lenet()

        # Load pruned numpy weights-
        pruned_model.set_weights(pruned_weights)

        # print("\nIn pruned model, number of nonzero parameters in each layer are: \n")
        pruned_sum_params = 0

        for layer in pruned_model.trainable_weights:
            # for layer in pruned_model_stripped.trainable_weights:
            # print(tf.math.count_nonzero(layer, axis = None).numpy())
            pruned_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

        print("\nAfter Pruning Round = {0}, total number of surviving trainable parameters = {1}\n".format(i,
                                                                                                           pruned_sum_params))
        # print("\nTotal number of trainable parameters = {0}\n".format(pruned_sum_params))

        # 'i' is the index for number of pruning rounds-
        history_main[i]['percentage_wts_pruned'] = ((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100
        history_main[i]['percentage_wts_remaining'] = 100 - (((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100)

        # Save weights of PRUNED and Trained model-


        # Create a mask:

        # Instantiate a new neural network model for which, the mask is to be created,
        # mask_model = lenet_nn()
        mask_model = lenet()

        # Load weights of PRUNED model-

        mask_model.set_weights(pruned_model.get_weights())

        # Strip the model of its pruning parameters-
        # mask_model_stripped = sparsity.strip_pruning(mask_model)

        # For each layer, for each weight which is 0, leave it, as is.
        # And for weights which survive the pruning,reinitialize it to ONE (1)-
        # for wts in mask_model_stripped.trainable_weights:
        for wts in mask_model.trainable_weights:
            wts.assign(tf.where(tf.equal(wts, 0.), 0., 1.))

        # Extract Winning Ticket:

        # Instantiate a new neural network model for which, the weights are to be extracted-
        # winning_ticket_model = lenet_nn()
        winning_ticket_model = lenet()

        # Load weights of PRUNED model-
        # winning_ticket_model.load_weights("Lenet_FC_Pruned_Weights.h5")
        winning_ticket_model.set_weights(pruned_model.get_weights())

        # Strip the model of its pruning parameters-
        # winning_ticket_model_stripped = sparsity.strip_pruning(winning_ticket_model)

        # For each layer, for each weight which is 0, leave it, as is. And for weights which survive the pruning,
        # reinitialize it to the value, the model received BEFORE it was trained and pruned-
        for orig_wts, pruned_wts in zip(orig_model.trainable_weights,
                                        winning_ticket_model.trainable_weights):
            pruned_wts.assign(tf.where(tf.equal(pruned_wts, 0), pruned_wts, orig_wts))




        winning_ticket_model.save_weights("Lenet_MNIST_Magnitude_Based_Winning_Ticket_Distribution_{0}.h5".format(
            history_main[i]['percentage_wts_pruned']), overwrite=True)


    History_data[j] = history_main

    list_of_dict = []

    xlist = [History_data[j]]

for x in xlist:
    list_of_dict.append(History_data)

# Save winning ticket:
winning_ticket_model.save_weights("Lenet_FC_Magnitude_Based_Winning_Ticket_Distribution_{0}.h5".format(
    history_main[i]['percentage_wts_pruned']), overwrite = True)

print("\nIterative-pruning for Lenet CNN using Lottery Ticket Hypothesis & Magnitude-based weight pruning is now complete.\n")

import os
import pickle

os.getcwd()

with open("C:/Users/Admin/PycharmProjects/FC/LeNet_MNIST_history_main_Experiment_3.pkl", "wb") as f:
    pickle.dump(list_of_dict, f)

with open("C:/Users/Admin/PycharmProjects/FC/LeNet_MNIST_history_main_Experiment_3.pkl", "rb") as f:
    history_main = pickle.load(f)

for row in range(len(history_main)):
    for col in range(len(history_main[row])):
        print(history_main[row])
row=(len(history_main))
print("Rows", (len(history_main)))




#Accuracy Multiple trial
#rows = num of trials
plot_accuracy = {}
plot_test_accuracy = {}


for row in range(len(history_main)):
    for col in range(len(history_main[row])):
        length = len(history_main[row][col]['accuracy'])
        #print("Len", length)
        plot_accuracy[history_main[row][col]['percentage_wts_pruned']] = np.average(
            np.array((history_main[row][col]['accuracy'][length - 1])))
        plot_test_accuracy[history_main[row][col]['percentage_wts_pruned']] = np.average(
            np.array((history_main[row][col]['val_accuracy'][length - 1])))

Mean_Accuracy = np.mean(list(plot_accuracy.values()))
Stddev_Accuracy = np.std(list(plot_accuracy.values()))
Stddev_Test_Accuracy = np.std(list(plot_test_accuracy.values()))

x = np.array(list(plot_accuracy.keys()))
x1 = np.array(list(plot_test_accuracy.keys()))
fig=plt.figure(figsize=(9, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(x, list(plot_accuracy.values()), label='training_accuracy', marker='*')
plt.fill_between(x, list(plot_accuracy.values()) - Stddev_Accuracy, list(plot_accuracy.values()) + Stddev_Accuracy,
                     alpha=.2)
plt.plot(x1, list(plot_test_accuracy.values()), label='test_accuracy', marker='*')
plt.fill_between(x1, list(plot_test_accuracy.values()) - Stddev_Test_Accuracy,
                     list(plot_test_accuracy.values()) + Stddev_Test_Accuracy, alpha=.2)
plt.legend()
plt.grid()
plt.title("Training and Test Accuracy on LeNet-5 ")
plt.xlabel("Percentage of weights pruned")
plt.ylabel("Accuracy")
plt.savefig("Accuracy.png")
#plt.show()





#Epochs
plot_epoch_accuracy = {}
for row in range(len(history_main)):
    for col in range(len(history_main[row])):
        length = len(history_main[row][col]['val_accuracy'])
        plot_epoch_accuracy[history_main[row][col]['percentage_wts_pruned']] = np.average(np.array([length - 1])) * 1000

fig=plt.figure(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')

Mean_Epoch_Accuracy = np.mean(list(plot_epoch_accuracy.values()))
Stddev_Epoch_Accuracy = np.std(list(plot_epoch_accuracy.values()))

x2=np.array(list(plot_epoch_accuracy.keys()))

plt.plot(x2, list(plot_epoch_accuracy.values()),  label = 'Validation accuracy vs Early stopping', marker='*')
plt.fill_between(x2, list(plot_epoch_accuracy.values())-Stddev_Epoch_Accuracy, list(plot_epoch_accuracy.values())+Stddev_Epoch_Accuracy, alpha=.2)
plt.legend()
plt.grid()
plt.title("Percentage of weights pruned VS number of epochs (Early Stopping)")
plt.xlabel("Percentage of weights pruned")
plt.ylabel("Number of iterations. 1000 iterations = 1 Epoch")
plt.savefig("Val Accuracy vs Early stopping.png")
#plt.show()


#Percentages
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x11 = []
x21 = []
x31 = []
x41 = []
x51 = []
x61 = []
x71 = []

for row in range(len(history_main)):

                x1.append((history_main[row][0]['val_accuracy']))
                x11.extend((history_main[row][0]['val_accuracy']))
                x2.append((history_main[row][3]['val_accuracy']))
                x21.extend((history_main[row][3]['val_accuracy']))
                x3.append((history_main[row][7]['val_accuracy']))
                x31.extend((history_main[row][7]['val_accuracy']))
                x4.append((history_main[row][12]['val_accuracy']))
                x41.extend((history_main[row][12]['val_accuracy']))
                x5.append((history_main[row][15]['val_accuracy']))
                x51.extend((history_main[row][15]['val_accuracy']))
                x6.append((history_main[row][18]['val_accuracy']))
                x61.extend((history_main[row][18]['val_accuracy']))
                x7.append((history_main[row][23]['val_accuracy']))
                x71.extend((history_main[row][23]['val_accuracy']))

x11 = np.std(x11)
x21 = np.std(x21)
x31 = np.std(x31)
x41 = np.std(x41)
x51 = np.std(x51)
x61 = np.std(x61)
x71 = np.std(x71)

fig=plt.figure(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')
plt.plot((pd.DataFrame(x1).mean(axis = 0)).index.values, (pd.DataFrame(x1).mean(axis = 0)), label=str(np.around(history_main[1][0]['percentage_wts_remaining'], decimals=3)), color = 'blue')
plt.fill_between((pd.DataFrame(x1).mean(axis = 0)).index.values, pd.DataFrame(x1).mean(axis = 0)-x11, pd.DataFrame(x1).mean(axis = 0)+x11, alpha=.2, color = 'blue')

plt.plot((pd.DataFrame(x2).mean(axis = 0)).index.values, pd.DataFrame(x2).mean(axis = 0), label=str(np.around(history_main[1][3]['percentage_wts_remaining'], decimals=3)), color = 'orange')
plt.fill_between((pd.DataFrame(x2).mean(axis = 0)).index.values, pd.DataFrame(x2).mean(axis = 0)-x21, pd.DataFrame(x2).mean(axis = 0)+x21, alpha=.2, color = 'orange')

plt.plot((pd.DataFrame(x3).mean(axis = 0)).index.values, pd.DataFrame(x3).mean(axis = 0), label=str(np.around(history_main[1][7]['percentage_wts_remaining'], decimals=3)) , color = 'green')
plt.fill_between((pd.DataFrame(x3).mean(axis = 0)).index.values, pd.DataFrame(x3).mean(axis = 0)-x31, pd.DataFrame(x3).mean(axis = 0)+x31, alpha=.2, color = 'green')

plt.plot((pd.DataFrame(x4).mean(axis = 0)).index.values, pd.DataFrame(x4).mean(axis = 0), label=str(np.around(history_main[1][12]['percentage_wts_remaining'], decimals=3)), color = 'yellow')
plt.fill_between((pd.DataFrame(x4).mean(axis = 0)).index.values, pd.DataFrame(x4).mean(axis = 0)-x41, pd.DataFrame(x4).mean(axis = 0)+x41, alpha=.2, color = 'yellow')

plt.plot((pd.DataFrame(x5).mean(axis = 0)).index.values, pd.DataFrame(x5).mean(axis = 0), label=str(np.around(history_main[1][15]['percentage_wts_remaining'], decimals=3)), color = 'red')
plt.fill_between((pd.DataFrame(x5).mean(axis = 0)).index.values, pd.DataFrame(x5).mean(axis = 0)-x51, pd.DataFrame(x5).mean(axis = 0)+x51, alpha=.2, color = 'red')

plt.plot((pd.DataFrame(x6).mean(axis = 0)).index.values, pd.DataFrame(x6).mean(axis = 0), label=str(np.around(history_main[1][18]['percentage_wts_remaining'], decimals=3)), color = 'pink')
plt.fill_between((pd.DataFrame(x6).mean(axis = 0)).index.values, pd.DataFrame(x6).mean(axis = 0)-x61, pd.DataFrame(x6).mean(axis = 0)+x61, alpha=.2, color = 'pink')

plt.plot((pd.DataFrame(x7).mean(axis = 0)).index.values, pd.DataFrame(x7).mean(axis = 0), label=str(np.around(history_main[1][23]['percentage_wts_remaining'], decimals=3)), color = 'cyan')
plt.fill_between((pd.DataFrame(x7).mean(axis = 0)).index.values, pd.DataFrame(x7).mean(axis = 0)-x71, pd.DataFrame(x7).mean(axis = 0)+x71, alpha=.2, color = 'cyan')

plt.legend()
plt.grid()
plt.title("Percentage vs Test Accuracy")
plt.xlabel("Number of epochs. 1 Epoch = 1000 iterations")
plt.ylabel("Test accuracy")
plt.savefig('Percent.png')
plt.show()

