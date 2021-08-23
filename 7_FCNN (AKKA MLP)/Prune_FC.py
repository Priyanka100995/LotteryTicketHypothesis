import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import os
import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from m_architecture_FCNN_0_9 import r_squared
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

with open("C:/Users/Admin/PycharmProjects/FC/config_ANN_0_9_0.yml", "r") as stream:
    config = yaml.safe_load(stream)



if config["enable_gpu"]:
    physical_devices=tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs available", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

batch_size =config["batch_size"]
num_epochs = config["epochs"]





# read csv data with pandas
os.chdir(config["data_path"])
train_samples = pd.read_csv("train_samples.csv", delimiter=',')
train_targets = pd.read_csv("train_targets.csv", delimiter=',')
val_samples = pd.read_csv("val_samples.csv", delimiter=',')
val_targets = pd.read_csv("val_targets.csv", delimiter=',')
test_samples = pd.read_csv("test_samples.csv", delimiter=',')
test_targets = pd.read_csv("test_targets.csv", delimiter=',')

print("Train samples shape: ", train_samples.shape)
print("Val samples shape: ", val_samples.shape)
print("Train ground truth shape: ", train_targets.shape)
print("Val ground truth shape: ", val_targets.shape)

num_train_samples = train_samples.shape[0]

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr = config["learning_rate"])
#optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

train_dataset = tf.data.Dataset.from_tensor_slices((train_samples, train_targets))
test_dataset = tf.data.Dataset.from_tensor_slices((val_samples, val_targets))




    # These metrics accumulate the values over epochs and then print the overall result-

train_loss = tf.keras.metrics.MeanSquaredError(name = 'train_loss')
train_rsquared = tf.keras.metrics.MeanSquaredError(name = 'train_r_squared')

val_loss = tf.keras.metrics.MeanSquaredError(name = 'val_loss')
val_rsquared = tf.keras.metrics.MeanSquaredError(name = 'val_r_squared')

train_dataset = train_dataset.shuffle(buffer_size = 5000, reshuffle_each_iteration = True).batch(batch_size = batch_size, drop_remainder = False)
test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)


def build_model_architecture():
    #lr = model_architecture_params()
    lr = config["learning_rate"]

    # Initializer
    if config["kernel_init"] == "GlorotUniform":
        init = tf.keras.initializers.GlorotUniform(seed=config["seed"])
    elif config["kernel_init"] == "GlorotNormal":
        init = tf.keras.initializers.GlorotNormal(seed=config["seed"])
    elif config["kernel_init"] == "HeNormal":
        init = tf.keras.initializers.he_normal(seed=config["seed"])
    elif config["kernel_init"] == "HeUniform":
        init = tf.keras.initializers.he_uniform(seed=config["seed"])

    # Build sequential model
    model = Sequential()

    # Add Inputlayer & 2 Hidden Layer

    model.add(keras.Input(shape=(config["input_shape"],)))
    model.add(Dense(units=config["neuron_units"]["Dense0"], activation=config["activation_function"],
                    kernel_initializer=init, use_bias=config["use_bias"], bias_initializer=config["bias_init"]))
    model.add(Dense(units=config["neuron_units"]["Dense1"], activation=config["activation_function"],
                    kernel_initializer=init, use_bias=config["use_bias"], bias_initializer=config["bias_init"]))
    model.add(Dense(units=config["neuron_units"]["Dense2"], activation=config["activation_function"],
                    kernel_initializer=init, use_bias=config["use_bias"], bias_initializer=config["bias_init"]))
    # model.add(Dense(units=config["neuron_units"]["Dense3"], activation=config["dense_activations"]))

    # Add Outputlayer
    model.add(Dense(units=config['output_shape'], activation=config["output_activation"]))
    # model.add(Dense(units=config['output_shape']))

    #model.summary()
    print("Aktuelle Lernrate: ", lr)

    """Plot the architecture of the model"""
    # plot_model(model, to_file='model_layout.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mean_squared_error', metrics=[r_squared])
    return model



 # Add a pruning step callback to peg the pruning step to the optimizer's
    # step.
callback = [


        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=config["patience"],
            min_delta=config["min_delta"]
        )

    ]

def load_model():
    lr = config["learning_rate"]
    model_path = config["loading_model_path"] + config["loading_model_name"] + config["loading_model_format"]
    dependencies = {'r_squared': r_squared}

    model = tf.keras.models.load_model(model_path, custom_objects=dependencies)
    model.summary()

    print("Aktuelle Lernrate: ", lr)
    """Plot the architecture of the model"""
    # plot_model(model, to_file='model_layout.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mean_squared_error', metrics=[r_squared])

    return model

orig_model = build_model_architecture()

# Save random weights-
orig_model.save_weights("C:/Users/Admin/PycharmProjects/FC/Weights/Model_Random_Weights.h5", overwrite=True)

# number of fully-connected dense parameters-
dense1 = config["dense1"]
dense2 = config["dense2"]
dense3 = config["dense3"]
op_layer = config["op_layer"]


# total number of parameters-
total_params = dense1 + dense2 + dense3 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# maximum pruning performed is till 0.5% of all parameters-
max_pruned_params = 0.005 * total_params

loc_tot_params = total_params

loc_dense1 = dense1
loc_dense2 = dense2
loc_dense3 = dense3
loc_op_layer = op_layer


# variable to count number of pruning rounds-
n = 0


dense1_pruning = []
dense2_pruning = []
dense3_pruning = []
op_layer_pruning = []


while loc_tot_params >= max_pruned_params:

    loc_dense1 *= config["dense_pruning"]  # 20% weights are pruned
    loc_dense2 *= config["dense_pruning"]  # 20% weights are pruned
    loc_dense3 *= config["dense_pruning"]
    loc_op_layer *= config["output_pruning"]  # 10% weights are pruned


    dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
    dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
    dense3_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
    op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

    loc_tot_params =  loc_dense1 + loc_dense2 + loc_dense3 + loc_op_layer

    n += 1

print("\nnumber of pruning rounds = {0}\n\n".format(n))
num_pruning_rounds = n

# Round off numpy arrays to 3 decimal digits-

dense1_pruning = np.round(dense1_pruning, decimals=3)
dense2_pruning = np.round(dense2_pruning, decimals=3)
dense3_pruning = np.round(dense3_pruning, decimals=3)
op_layer_pruning = np.round(op_layer_pruning, decimals=3)
dense1_pruning = np.insert(dense1_pruning,0,0)
op_layer_pruning = np.insert(op_layer_pruning,0,0)

print("Dense Layer Pruning Percent",dense1_pruning)
print("Output Layer Pruning Percent",op_layer_pruning)

# Instantiate a new neural network model for which, the mask is to be created,
# according to the paper-
mask_model = build_model_architecture()

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

masked_sum_params = 0

for layer in mask_model.trainable_weights:
# for layer in mask_model_stripped.trainable_weights:
    # print(tf.math.count_nonzero(layer, axis = None).numpy())
    masked_sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

print("\nTotal number of trainable parameters = {0}\n".format(masked_sum_params))

History_data = {}
history_main = {}

num_trials = config["num_trials"]


def prune_model(model, pruning_params_fc, pruning_params_op):
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

        if len(layer.shape) == 2 and layer.shape[1] != 1400:
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

        elif len(layer.shape) == 2 and layer.shape[1] == 1400:
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

orig_model_pruned = prune_model(orig_model, dense1_pruning[0], op_layer_pruning[0])
minimum_delta = config["min_delta"]
patience = config["patience"]

best_val_loss = 100
loc_patience = 0

orig_sum_params = total_params
print("\nTotal number of parameters in overparametrized, original, unpruned network = ", orig_sum_params)

winning_ticket_model = build_model_architecture()

# Use Randomly initialized weights-
winning_ticket_model.set_weights(orig_model.get_weights())

History_data = {}
history_main = {}
history_main_2 = {}
list_of_dict = []


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

        history_main[(x)] = history
        history_main.keys()
        history_main_2[(x)] = history_main

    print("HISTORY keys", history_main.keys())
    dicts = []




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
            train_loss(y, y_pred)
            train_rsquared(y, y_pred)
            #r_squared(y, y_pred)

            return None


        @tf.function
        def test_step(model, optimizer, data, labels):
            """
            Function to evaluate model performance
            on validation dataset
            """

            predictions = model(data)
            t_loss = loss_fn(labels, predictions)

            val_loss(labels, predictions)
            val_rsquared(labels, predictions)
            #r_squared(labels, predictions)
            return None


        # Instantiate a model
        # model_gt = pruned_nn(pruning_params_unpruned)
        model_gt = build_model_architecture()

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
            model_gt.load_weights("C:/Users/Admin/PycharmProjects/FC/Weights/Model_Random_Weights.h5")


        else:
            # Load winning ticket (from above)-
            model_gt.set_weights(winning_ticket_model.get_weights())


        for epoch in range(num_epochs):

            if loc_patience >= patience:
                print("\n'EarlyStopping' called!\n")
                break

            # Reset the metrics at the start of the next epoch

            #reset_metrics_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            train_loss.reset_states()
            train_rsquared.reset_states()
            val_loss.reset_states()
            val_rsquared.reset_states()

            for x, y in train_dataset: #Check which is training data
                # train_one_step(model_gt_stripped, mask_model_stripped, optimizer, x, y)

                train_one_step(model_gt, mask_model, optimizer, x, y)

            for x_t, y_t in test_dataset: #check test data
                # test_step(model_gt_stripped, optimizer, x_t, y_t)
                test_step(model_gt, optimizer, x_t, y_t)

            template = 'Epoch {0}, Train Loss: {1:.7f}, Val Loss: {2:.7f}'

            # 'i' is the index for number of pruning rounds-
            history_main[i]['r_squared'][epoch] = train_rsquared.result()
            history_main[i]['train_loss'][epoch] = train_loss.result()
            history_main[i]['val_loss'][epoch] = val_loss.result()
            history_main[i]['val_r_squared'][epoch] = val_rsquared.result()

            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  val_loss.result()))
            # Count number of non-zero parameters in each layer and in total-
            # print("layer-wise manner model, number of nonzero parameters in each layer are: \n")

            model_sum_params = 0

            for layer in model_gt.trainable_weights:
                # for layer in model_gt_stripped.trainable_weights:
                # print(tf.math.count_nonzero(layer, axis = None).numpy())
                model_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

            print("Total number of trainable parameters = {0}\n".format(model_sum_params))

            # Code for manual Early Stopping:
            if np.abs(val_loss.result() < best_val_loss) >= minimum_delta:
                # update 'best_val_loss' variable to lowest loss encountered so far-
                best_val_loss = val_loss.result()

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
        pruned_weights = prune_model(
            model=model_gt,
            pruning_params_fc=dense1_pruning[i - 1],
            pruning_params_op=op_layer_pruning[i - 1]
        )

        # Instantiate a Neural Network model
        pruned_model = build_model_architecture()

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
        mask_model = build_model_architecture()

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
        winning_ticket_model = build_model_architecture()

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




        winning_ticket_model.save_weights("C:/Users/Admin/PycharmProjects/FC/Weights/Model_Magnitude_Based_Winning_Ticket_Distribution_{0}.h5".format(
            history_main[i]['percentage_wts_pruned']), overwrite=True)
        winning_ticket_model.save("C:/Users/Admin/PycharmProjects/FC/Pruned_Models/Magnitude_Based_Pruned_Model_{0}.h5".format(
            history_main[i]['percentage_wts_pruned']), overwrite=True)

		#Save metrics
        dicts.append(history_main[i])  # saves num_trials * num of rounds
        print("Dicts={0}".format(i), dicts)

    history_main_2[j] = dicts
    list_of_dict.append(history_main_2[j])
    


print("Iterative pruning of Fully Connected Neural Network Complete! ")

import os
import pickle

os.getcwd()

with open("C:/Users/Admin/PycharmProjects/FC/Pruned_Models_history_main_Experiment.pkl", "wb") as f:
    pickle.dump(list_of_dict, f)

with open("C:/Users/Admin/PycharmProjects/FC/Pruned_Models_history_main_Experiment.pkl", "rb") as f:
    history_main = pickle.load(f)


#Plot val_loss
plot_num_epochs= {}

for k in range(1,num_trials+1):
    for k1 in range(1,num_pruning_rounds+1):
        num_epochs = len(history_main[k][k1]['val_loss'])
        plot_num_epochs[history_main[k][k1]['percentage_wts_pruned']] = np.mean(np.array(num_epochs)) * 1000



Stddev_Accuracy_Epochs = (np.std(list(plot_num_epochs.values())))/ 100
print("Standard deviation for Epochs vs Percentages", Stddev_Accuracy_Epochs)

y=np.array(list(plot_num_epochs.keys()))

plt.plot(y, list(plot_num_epochs.values()), label= 'Epochs vs Percentags of weights pruned', marker='*')

plt.fill_between(y, list(plot_num_epochs.values()), (list(plot_num_epochs.values()))-Stddev_Accuracy_Epochs, (list(plot_num_epochs.values())+Stddev_Accuracy_Epochs), alpha=.2)


plt.title("Percentage of weights pruned VS number of epochs (Early Stopping)")
plt.xlabel("Percentage of weights pruned")
plt.ylabel("Number of iterations")
plt.savefig("Percentage of weights pruned VS number of epochs (Early Stopping).png")
#plt.show()

