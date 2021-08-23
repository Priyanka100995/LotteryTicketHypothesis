import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
import PruningPercent
import Dataset
import LenetModel
import Prune
import yaml

# Load yaml file
with open("C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r") as stream:
    config = yaml.safe_load(stream)

tf.random.set_seed(config["seed"])

if config["enable_gpu"]:
    physical_devices= tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


pruning_params_unpruned = {
    'pruning_schedule': Prune.ConstantSparsity(
        target_sparsity=0.0, begin_step=0,
        end_step = Dataset.Data.end_step, frequency=100
    )
}


# Initialize a FCNN model-
orig_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)

orig_model_stripped = strip_pruning(orig_model)
# Save random weights-
orig_model.save_weights("LeNet_MNIST_Random_Weights.h5", overwrite=True)

orig_model.save_weights("LeNet_MNIST_Winning_Ticket.h5", overwrite=True)

#FCNN summary-
orig_model_stripped.summary()

#Edit values after summary is printed in the output
#number of fully-connected dense parameters-
dense1 = config["dense1"]
dense2 = config["dense2"]
op_layer = config["op_layer"]
orig_sum_params = config["orig_sum_params"]

num_pruning_rounds = config["num_of_pruning_rounds"]
num_trials = config["num_trials"]
num_epochs = config["epochs"]

# User input parameters for Early Stopping in manual implementation-
minimum_delta = config["min_delta"]
patience = config["patience"]



model_gt = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)
mask_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)
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

print("\nMask model metrics:")
print("layer-wise number of nonzero parameters in each layer are: \n")

masked_sum_params = 0

for layer in mask_model_stripped.trainable_weights:
    print(tf.math.count_nonzero(layer, axis = None).numpy())
    masked_sum_params += tf.math.count_nonzero(layer, axis = None).numpy()

print("\nTotal number of trainable parameters = {0}\n".format(masked_sum_params))
History_data = {}
history_main = {}
history_main_2 = {}
list_of_dict = []


for j in range (1, num_trials+1):
    print("Trials", j)

    for x in range(1, num_pruning_rounds + 1):

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
                loss = Dataset.Data.loss_fn(y, y_pred)

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
            Dataset.Data.train_loss(loss)
            Dataset.Data.train_accuracy(y, y_pred)

            return None


        @tf.function
        def test_step(model, optimizer, data, labels):
            """
            Function to test model performance
            on testing dataset
            """

            predictions = model(data)
            t_loss = Dataset.Data.loss_fn(labels, predictions)

            Dataset.Data.test_loss(t_loss)
            Dataset.Data.test_accuracy(labels, predictions)

            return None


        # Instantiate a model
        model_gt = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)

        # Load winning ticket (from above)-
        model_gt.load_weights("LeNet_MNIST_Winning_Ticket.h5")

        # Strip model of pruning parameters-
        model_gt_stripped = strip_pruning(model_gt)

        # Initialize parameters for Early Stopping manual implementation-
        best_val_loss = config["best_val_loss"]
        loc_patience = config["loc_patience"]

        if i == 1:  # (For every 1st round of every trial set to initial wts)

            # Load winning ticket (from above)-
            model_gt.load_weights("LeNet_MNIST_Random_Weights.h5")

            # Strip model of pruning parameters-
            model_gt_stripped = strip_pruning(model_gt)

        else:
            # Load winning ticket (from above)-
            model_gt.load_weights("LeNet_MNIST_Winning_Ticket.h5")

            # Strip model of pruning parameters-
            model_gt_stripped = strip_pruning(model_gt)


        for epoch in range(num_epochs):

            if loc_patience >= patience:
                print("\n'EarlyStopping' called!\n")
                break

            # Reset the metrics at the start of the next epoch
            Dataset.Data.train_loss.reset_states()
            Dataset.Data.train_accuracy.reset_states()
            Dataset.Data.test_loss.reset_states()
            Dataset.Data.test_accuracy.reset_states()

            for x, y in Dataset.Data.train_dataset:

                train_one_step(model_gt_stripped, mask_model_stripped, Dataset.Data.optimizer, x, y)

            for x_t, y_t in Dataset.Data.test_dataset:

                test_step(model_gt_stripped, Dataset.Data.optimizer, x_t, y_t)

            template = 'Epoch {0}, Loss: {1:.4f}, Accuracy: {2:.4f}, Test Loss: {3:.4f}, Test Accuracy: {4:4f}'

            # 'i' is the index for number of pruning rounds-
            history_main[i]['accuracy'][epoch] = Dataset.Data.train_accuracy.result() * 100
            history_main[i]['loss'][epoch] = Dataset.Data.train_loss.result()
            history_main[i]['val_loss'][epoch] = Dataset.Data.test_loss.result()
            history_main[i]['val_accuracy'][epoch] = Dataset.Data.test_accuracy.result() * 100

            print(template.format(epoch + 1,
                                  Dataset.Data.train_loss.result(), Dataset.Data.train_accuracy.result() * 100,
                                  Dataset.Data.test_loss.result(), Dataset.Data.test_accuracy.result() * 100))

            # Count number of non-zero parameters in each layer and in total-
            # print("layer-wise manner model, number of nonzero parameters in each layer are: \n")

            model_sum_params = 0

            for layer in model_gt_stripped.trainable_weights:
                # print(tf.math.count_nonzero(layer, axis = None).numpy())
                model_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

            print("Total number of trainable parameters = {0}\n".format(model_sum_params))

            # Code for manual Early Stopping:
            if np.abs(Dataset.Data.test_loss.result() < best_val_loss) >= minimum_delta:
                # update 'best_val_loss' variable to lowest loss encountered so far-
                best_val_loss = Dataset.Data.test_loss.result()

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

        # Specify the parameters to be used for layer-wise pruning, Fully-Connected layer pruning-
        pruning_params_fc = {
            'pruning_schedule': Prune.ConstantSparsity(
                target_sparsity=PruningPercent.dense1_pruning[i - 1], begin_step=1000,
                end_step=Dataset.Data.end_step, frequency=100
            )
        }
        pruning_params_op = {
            'pruning_schedule': Prune.ConstantSparsity(
                target_sparsity=PruningPercent.op_layer_pruning[i - 1], begin_step=1000,
                end_step=Dataset.Data.end_step, frequency=100
            )
        }

        # Instantiate a Neural Network model to be pruned using parameters from above-
        pruned_model = LenetModel.pruned_nn.pruned_nn(pruning_params_fc, pruning_params_op)

        # Load weights from original trained and unpruned model-
        pruned_model.load_weights("LeNet_MNIST_Trained_Weights.h5")

        # Train pruned NN-
        history_pruned = pruned_model.fit(
            x=Dataset.Data.X_train, y=Dataset.Data.y_train,
            batch_size=config["batch_size"],
            epochs=num_epochs,
            verbose=1,
            callbacks=LenetModel.pruned_nn.callback,
            validation_data=(Dataset.Data.X_test, Dataset.Data.y_test),
            shuffle=config["shuffle"]
        )

        # Strip the pruning wrappers from pruned model-
        pruned_model_stripped = strip_pruning(pruned_model)

        # print("\nIn pruned model, number of nonzero parameters in each layer are: \n")
        pruned_sum_params = 0

        for layer in pruned_model_stripped.trainable_weights:
            # print(tf.math.count_nonzero(layer, axis = None).numpy())
            pruned_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

        print("\nRound = {0}, total number of trainable parameters = {1}\n".format(i, pruned_sum_params))


        '''
        # Check: confirm that the weights are actually pruned away from the network-
        print("\nRound = {0}, % of weights pruned away = {1:.2f}%\n".format( \
                                                    i, (orig_sum_params - pruned_sum_params) / orig_sum_params * 100))
        '''

        # 'i' is the index for number of pruning rounds-
        history_main[i]['percentage_wts_pruned'] = (((orig_sum_params - pruned_sum_params) / orig_sum_params)*100)
        history_main[i]['percentage_wts_remaining'] = 100- ((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100
        print("Percentage wts remaining after pruned **** Check", history_main[i]['percentage_wts_pruned'])



        # Save weights of PRUNED and Trained model BEFORE stripping (Weights trained and then pruned)
        pruned_model.save_weights("LeNet_MNIST_Pruned_Weights.h5", overwrite=True)


        model_gt_stripped = pruned_model_stripped

        # Create a mask:

        # Instantiate a new neural network model for which, the mask is to be created,

        mask_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)

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
        winning_ticket_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)

        # Load weights of PRUNED model-
        winning_ticket_model.load_weights("LeNet_MNIST_Pruned_Weights.h5")

        # Strip the model of its pruning parameters-
        winning_ticket_model_stripped = strip_pruning(winning_ticket_model)

        # For each layer, for each weight which is 0, leave it, as is. And for weights which survive the pruning,
        # reinitialize it to the value, the model received BEFORE it was trained and pruned-
        for orig_wts, pruned_wts in zip(orig_model_stripped.trainable_weights,
                                        winning_ticket_model_stripped.trainable_weights):
            pruned_wts.assign(tf.where(tf.equal(pruned_wts, 0), pruned_wts, orig_wts))


        winning_ticket_model.save_weights("Lenet_MNIST_Magnitude_Based_Winning_Ticket_Distribution_{0}.h5".format(
            history_main[i]['percentage_wts_pruned']), overwrite=True)

        winning_ticket_model.save("Pruned_Model_{0}.h5".format(
            history_main[i]['percentage_wts_pruned']), overwrite=True)
        #Save metrics
        dicts.append(history_main[i])  # saves num_trials * num of rounds
        print("Dicts={0}".format(i), dicts)

    history_main_2[j] = dicts
    list_of_dict.append(history_main_2[j])

#print("History 2", history_main_2[2])
#print ("Appended ", list_of_dict )

print("\nIterative-pruning for Lenet CNN using Lottery Ticket Hypothesis & Magnitude-based weight pruning is now complete.\n")


import os
import pickle
os.getcwd()


#Specify the Path where the pkl file is to be stored
with open("C:/Users/Admin/PycharmProjects/ModularCode_FC/LeNet_MNIST_history_main_Experiment.pkl", "wb") as f:
    pickle.dump(list_of_dict, f)
