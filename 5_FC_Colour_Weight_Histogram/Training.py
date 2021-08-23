import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
import PruningPercent
import Dataset
import LenetModel
import Prune
import matplotlib.pyplot as plt
import random
import matplotlib
import os
import pickle
import csv
os.getcwd()
import pandas

random.seed(10)

history_main = {}


""""writer = tf.summary.create_file_writer("Original_1")"""""
pruning_params_unpruned = {
    'pruning_schedule': Prune.ConstantSparsity(
        target_sparsity=0.0, begin_step=0,
        end_step = Dataset.Data.end_step, frequency=100
    )
}


# Initialize a CNN model-
orig_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned,pruning_params_unpruned)

orig_model_stripped = strip_pruning(orig_model)
# Save random weights-
orig_model.save_weights("LeNet_MNIST_Random_Weights.h5", overwrite=True)

def visualize_model_weights(sparse_model):
    """
    Visualize the weights of the layers of the sparse model.
    For weights with values of 0, they will be represented by the color white
    Args:
      sparse_model: a TF.Keras model
    """

    weights = sparse_model.get_weights()
    names = [weight.name for layer in sparse_model.layers for weight in layer.weights]
    my_cmap = matplotlib.cm.get_cmap('rainbow')

    # Iterate over all the weight matrices in the model and visualize them
    for i in range(len(weights)):
        weight_matrix = weights[i]

        layer_name = names[i]
        if weight_matrix.ndim == 1:  # If Bias or softmax
            weight_matrix = np.resize(weight_matrix,
                                      (1, weight_matrix.size))
            plt.imshow(np.abs(weight_matrix),
                       interpolation='none',
                       aspect="auto",
                       cmap=my_cmap);  # lower bound is set close to but not at 0
            plt.colorbar()
            plt.title(layer_name)
            plt.close()

        else:  # all other 2D matrices
            cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'darkblue', 'white',  'darkgreen', 'green'])
            fig, ax = plt.subplots()
            im = ax.imshow(weight_matrix, cmap=cmap, interpolation='nearest')
            fig.colorbar(im)
            plt.xlabel("Output")
            plt.ylabel("Input")
            if i == 0:
                plt.title('FC Layer, 300')
            elif i == 2:
                plt.title('FC Layer, 100')
            elif i == 4:
                   plt.title('FC Layer, 10')
            plt.close()

            plt.show()






#Save original weights and plot a histogram of he weights

for weight in (orig_model_stripped).get_weights():
    W = (orig_model_stripped.layers[1]).get_weights()
    W = np.array(W)
    W = np.ravel(W)
    W2 = (orig_model_stripped.layers[2]).get_weights()
    W2 = np.array(W2)
    W2 = np.ravel(W2)
    W3 = (orig_model_stripped.layers[3]).get_weights()
    W3 = np.array(W3)
    W3 = np.ravel(W3)

with open('Original Weights.txt', 'w') as f:
    f.write(str('FC Layer, 300 (Glorot Init.) Original_Weights'))
    f.write(str(W))
    f.write(str('FC Layer, 100 (Glorot Init.) Original_Weights'))
    f.write(str(W2))
    f.write(str('FC Layer, 10 (Softmax) Original_Weights'))
    f.write(str(W3))

print("Weights1", W.shape)
print("FC Layer, 300 (Glorot Init.) Original_Weights", W)
print("Weights2", W2.shape)
print("FC Layer, 100 (Glorot Init.) Original_Weights", W2)
print("Weights3", W3.shape)
print("FC Layer, 10 (Softmax) Original_Weights", W3)

fig = plt.figure()
plt.subplot(2, 2, 1)

plt.title("FC Layer 1, 300 (Glorot Init.)")
plt.xlabel('Final Weights')
plt.ylabel('Frequency')
axes = plt.gca()
axes.set_ylim([0, 40000])
plt.hist(W, bins=10, histtype='stepfilled')
plt.grid()
plt.subplot(2, 2, 2)
plt.title("FC Layer 2, 100 (Glorot Init.)")
plt.xlabel('Final Weights')
plt.ylabel('Frequency')
axes = plt.gca()
axes.set_ylim([0, 5000])
plt.hist(W2, bins=10, histtype='stepfilled')
plt.grid()
plt.subplot(2, 2, 3)
plt.title("FC Layer 3, 10 (Softmax)")
plt.xlabel('Final Weights')
plt.ylabel('Frequency')
axes = plt.gca()
axes.set_ylim([0, 150])
plt.hist(W3, bins=10, histtype='stepfilled')
plt.grid()
plt.tight_layout()
plt.savefig("Original_Weights.png")

# Save random weights-
orig_model.save_weights("LeNet_MNIST_Winning_Ticket.h5", overwrite=True)

# Get CNN summary-
orig_model_stripped.summary()


#Edit values after summary is printed in the output
#number of fully-connected dense parameters-
dense1 = 235500
dense2 = 30100
op_layer = 1010
#history_main1={}

# for x in range(num_pruning_rounds + 1):
for x in range(PruningPercent.num_pruning_rounds):
    history = {}

    # Neural Network model, scalar metrics-
    history['accuracy'] = np.zeros(shape=Dataset.Data.num_epochs)
    history['val_accuracy'] = np.zeros(shape=Dataset.Data.num_epochs)
    history['loss'] = np.zeros(shape=Dataset.Data.num_epochs)
    history['val_loss'] = np.zeros(shape=Dataset.Data.num_epochs)
    history['iterations'] = np.zeros(shape=Dataset.Data.num_epochs)#updated

    # compute % of weights pruned at the end of each iterative pruning round-
    history['percentage_wts_pruned'] = np.zeros(shape=Dataset.Data.num_epochs)

    #print("History of wts pruned", history['percentage_wts_pruned'] )

    history_main[x + 1] = history

history_main.keys()
print("HISTORY",history_main.keys())
# User input parameters for Early Stopping in manual implementation-
minimum_delta = 0.001
patience = 3

best_val_loss = 100
loc_patience = 0
orig_sum_params = 266610

# Instantiate a model
model_gt = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned,pruning_params_unpruned)

mask_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned,pruning_params_unpruned)
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

#for j in range (1,5):
#    print("\n\n\nTrials: {0}\n\n".format(j))

for i in range(1, PruningPercent.num_pruning_rounds+1):


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
        model_gt = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned,pruning_params_unpruned)

        # Load winning ticket (from above)-
        model_gt.load_weights("LeNet_MNIST_Winning_Ticket.h5")

        # Strip model of pruning parameters-
        model_gt_stripped = strip_pruning(model_gt)

        # Train model using 'GradientTape'-

        # Initialize parameters for Early Stopping manual implementation-
        best_val_loss = 100
        loc_patience = 0

        for epoch in range(20):


            if loc_patience >= patience:
                print("\n'EarlyStopping' called!\n")
                break

            # Reset the metrics at the start of the next epoch
            Dataset.Data.train_loss.reset_states()
            Dataset.Data.train_accuracy.reset_states()
            Dataset.Data.test_loss.reset_states()
            Dataset.Data.test_accuracy.reset_states()

            for x, y in Dataset.Data.train_dataset:
                # train_one_step(model_gt_stripped, mask_model, optimizer, x, y, grad_mask_mul)
                train_one_step(model_gt_stripped, mask_model_stripped, Dataset.Data.optimizer, x, y)

            for x_t, y_t in Dataset.Data.test_dataset:
                # test_step(x_t, y_t)
                test_step(model_gt_stripped, Dataset.Data.optimizer, x_t, y_t)

            template = 'Epoch {0}, Loss: {1:.4f}, Accuracy: {2:.4f}, Test Loss: {3:.4f}, Test Accuracy: {4:4f}'

            # 'i' is the index for number of pruning rounds-
            history_main[i]['accuracy'][epoch] = Dataset.Data.train_accuracy.result() * 100
            history_main[i]['loss'][epoch] = Dataset.Data.train_loss.result()
            history_main[i]['val_loss'][epoch] = Dataset.Data.test_loss.result()
            history_main[i]['val_accuracy'][epoch] = Dataset.Data.test_accuracy.result() * 100
            history_main[i]['iterations'][epoch] = epoch * 1000

            print(template.format(epoch + 1,
                                  Dataset.Data.train_loss.result(), Dataset.Data.train_accuracy.result() * 100,
                                  Dataset.Data.test_loss.result(), Dataset.Data.test_accuracy.result() * 100))

            # Count number of non-zero parameters in each layer and in total-
            # print("layer-wise manner model, number of nonzero parameters in each layer are: \n")

            model_sum_params = 0

            for layer in model_gt_stripped.trainable_weights:
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

        # Resize numpy arrays according to the epoch when 'EarlyStopping' was called- Stores metrics acc, val_acc, loss,etc
        for metrics in history_main[i].keys():
            history_main[i][metrics] = np.resize(history_main[i][metrics], new_shape = epoch)

            #print("history_main[i][metrics]", history_main[i][metrics])
            # history[metrics] = np.resize(history[metrics], new_shape=epoch)

        # Save trained model weights-
        model_gt.save_weights("LeNet_MNIST_Trained_Weights.h5", overwrite=True)

        #Pruned and Trained Model Weights Histogram

        for weight in (model_gt).get_weights():
            W = (model_gt.layers[1]).get_weights()
            W = np.array(W)
            W = np.ravel(W)
            W2 = (model_gt.layers[2]).get_weights()
            W2 = np.array(W2)
            W2 = np.ravel(W2)
            W3 = (model_gt.layers[3]).get_weights()
            W3 = np.array(W3)
            W3 = np.ravel(W3)

        with open('Pruned and Trained_Weights{y}.txt'.format(y=i), 'w') as f:
            f.write(str('FC Layer, 300 (Glorot Init.) Original_Weights'))
            f.write(str(W))
            f.write(str('FC Layer, 100 (Glorot Init.) Original_Weights'))
            f.write(str(W2))
            f.write(str('FC Layer, 10 (Softmax) Original_Weights'))
            f.write(str(W3))
        print("Weights1", W.shape)
        print("FC Layer, 300 (Glorot Init.) Pruned and Trained_Weights", W)
        print("Weights2", W2.shape)
        print("FC Layer, 100 (Glorot Init.) Pruned and Trained_Weights", W2)
        print("Weights3", W3.shape)
        print("FC Layer, 10 (Softmax) Pruned and Trained_Weights", W3)

        fig = plt.figure()
        #plt.title("Percentage of weights remaining after pruning", Weight)
        plt.subplot(2, 2, 1)
        plt.title("FC Layer, 300 (Glorot Init.)")
        plt.xlabel('Final Weights')
        plt.ylabel('Density')
        axes = plt.gca()
        axes.set_ylim([0, 300000])
        plt.ylim = (0, 200000)
        plt.hist(W, bins=15, histtype='stepfilled')
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.title("FC Layer, 100 (Glorot Init.)")

        plt.xlabel('Final Weights')
        plt.ylabel('Density')
        axes = plt.gca()
        axes.set_ylim([0, 40000])
        plt.hist(W2, bins=15, histtype='stepfilled')
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.title("FC Layer, 10 (Softmax)")

        plt.xlabel('Final Weights')
        plt.ylabel('Density')
        axes = plt.gca()
        axes.set_ylim([0, 900])
        plt.hist(W3, bins=15, histtype='stepfilled')
        plt.grid()
        plt.tight_layout()
        plt.savefig("Pruned and Trained_Weights{y}.png".format(y=i))



        # Prune trained model:

        # print("\n% of weights to be pruned in round = {0} is: {1:.4f}\n".format(i, wts_np[i - 1]))

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
        pruned_model = LenetModel.pruned_nn.pruned_nn(pruning_params_fc,pruning_params_op)

        # Load weights from original trained and unpruned model-
        pruned_model.load_weights("LeNet_MNIST_Trained_Weights.h5")

        # Train pruned NN-
        history_pruned = pruned_model.fit(
            x=Dataset.Data.X_train, y=Dataset.Data.y_train,
            batch_size=Dataset.Data.batch_size,
            epochs=Dataset.Data.epochs,
            verbose=1,
            callbacks=LenetModel.pruned_nn.callback,
            validation_data=(Dataset.Data.X_test, Dataset.Data.y_test),
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


        '''
        # check: confirm that the weights are actually pruned away from the network-
        print("\nRound = {0}, % of weights pruned away = {1:.2f}%\n".format( \
                                                    i, (orig_sum_params - pruned_sum_params) / orig_sum_params * 100))
        '''

        # 'i' is the index for number of pruning rounds-
        history_main[i]['percentage_wts_pruned'] = ((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100
        history_main[i]['percentage_wts_remaining'] = 100- (((orig_sum_params - pruned_sum_params) / orig_sum_params) * 100)
        #history_main[i]['percentage_wts_pruned'] = 100 - history_main[i]['percentage_wts_pruned'] #updated
        print("Percentage wts remaining after pruned **** Check", history_main[i]['percentage_wts_pruned'])
        Weight=history_main[i]['percentage_wts_pruned']





        # Save weights of PRUNED and Trained model BEFORE stripping (Weights trained and then pruned)
        pruned_model.save_weights("LeNet_MNIST_Pruned_Weights.h5", overwrite=True)

        # Plot Pruned Weights' graph for every round
        for weight in (pruned_model_stripped).get_weights():
            W = (pruned_model_stripped.layers[1]).get_weights()
            W = np.array(W)
            W = np.ravel(W)
            W2 = (pruned_model_stripped.layers[2]).get_weights()
            W2 = np.array(W2)
            W2 = np.ravel(W2)
            W3 = (pruned_model_stripped.layers[3]).get_weights()
            W3 = np.array(W3)
            W3 = np.ravel(W3)

        with open('Pruned_Weights{y}.txt'.format(y=i), 'w') as f:
            f.write(str('FC Layer, 300 (Glorot Init.) Original_Weights'))
            f.write(str(W))
            f.write(str('FC Layer, 100 (Glorot Init.) Original_Weights'))
            f.write(str(W2))
            f.write(str('FC Layer, 10 (Softmax) Original_Weights'))
            f.write(str(W3))
        print("Weights1", W.shape)
        print("FC Layer, 300 (Glorot Init.) Pruned_Weights", W)
        print("Weights2", W2.shape)
        print("FC Layer, 100 (Glorot Init.) Pruned_Weights", W2)
        print("Weights3", W3.shape)
        print("FC Layer, 10 (Softmax) Pruned_Weights", W3)

        fig = plt.figure()
        #plt.title("Percentage of weights remaining after pruning", Weight)
        plt.subplot(2, 2, 1)
        plt.title("FC Layer, 300 (Glorot Init.)")

        plt.xlabel('Final Weights')
        plt.ylabel('Frequency')
        axes = plt.gca()
        axes.set_ylim([0, 300000])
        plt.ylim = (0, 200000)
        plt.hist(W, bins=10, histtype='stepfilled')
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.title("FC Layer, 100 (Glorot Init.)")

        plt.xlabel('Final Weights')
        plt.ylabel('Frequency')
        axes = plt.gca()
        axes.set_ylim([0, 40000])
        plt.ylim = (0, 30000)
        plt.hist(W2, bins=10, histtype='stepfilled')
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.title("FC Layer, 10 (Softmax)")

        plt.xlabel('Final Weights')
        plt.ylabel('Frequency')
        axes = plt.gca()
        axes.set_ylim([0, 900])
        plt.ylim = (0, 800)
        plt.hist(W3, bins=10, histtype='stepfilled')
        plt.grid()
        plt.tight_layout()
        plt.savefig("Pruned_Weights{y}.png".format(y=i))




        model_gt_stripped = pruned_model_stripped

        # Create a mask:

        # Instantiate a new neural network model for which, the mask is to be created,

        mask_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned,pruning_params_unpruned)

        # Load weights of PRUNED model-
        mask_model.load_weights("LeNet_MNIST_Pruned_Weights.h5")

        #to plot weights

        # Strip the model of its pruning parameters-
        mask_model_stripped = strip_pruning(mask_model)

        # For each layer, for each weight which is 0, leave it, as is.
        # And for weights which survive the pruning,reinitialize it to ONE (1)-
        for wts in mask_model_stripped.trainable_weights:
            wts.assign(tf.where(tf.equal(wts, 0.), 0., 1.))

        # Extract Winning Ticket:

        # Instantiate a new neural network model for which, the weights are to be extracted-
        winning_ticket_model = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned,pruning_params_unpruned)

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

        for weight in (winning_ticket_model).get_weights():
            W = (winning_ticket_model.layers[1]).get_weights()
            W = np.array(W)
            W = np.ravel(W)
            W2 = (winning_ticket_model.layers[2]).get_weights()
            W2 = np.array(W2)
            W2 = np.ravel(W2)
            W3 = (winning_ticket_model.layers[3]).get_weights()
            W3 = np.array(W3)
            W3 = np.ravel(W3)

        with open('Final_Weights{y}.txt'.format(y=i), 'w') as f:
            f.write(str('FC Layer, 300 (Glorot Init.) Original_Weights'))
            f.write(str(W))
            f.write(str('FC Layer, 100 (Glorot Init.) Original_Weights'))
            f.write(str(W2))
            f.write(str('FC Layer, 10 (Softmax) Original_Weights'))
            f.write(str(W3))
        print("Weights1", W.shape)
        print("FC Layer, 300 (Glorot Init.) Final Weights", W)
        print("Weights2", W2.shape)
        print("FC Layer, 100 (Glorot Init.) Final Weights", W2)
        print("Weights3", W3.shape)
        print("FC Layer, 10 (Softmax) Final Weights", W3)

        fig = plt.figure()
        #plt.title("Percentage of weights remaining after pruning", Weight)
        plt.subplot(2, 2, 1)
        plt.title("FC Layer, 300 (Glorot Init.)")

        plt.xlabel('Final Weights')
        plt.ylabel('Frequency')
        axes = plt.gca()
        axes.set_ylim([0, 300000])
        plt.ylim = (0, 200000)
        plt.hist(W, bins=10, histtype='stepfilled')
        plt.grid()
        plt.subplot(2, 2, 2)
        plt.title("FC Layer, 100 (Glorot Init.)")

        plt.xlabel('Final Weights')
        plt.ylabel('Frequency')
        axes = plt.gca()
        axes.set_ylim([0, 40000])
        plt.ylim = (0, 30000)
        plt.hist(W2, bins=10, histtype='stepfilled')
        plt.grid()
        plt.subplot(2, 2, 3)
        plt.title("FC Layer, 10 (Softmax)")

        plt.xlabel('Final Weights')
        plt.ylabel('Frequency')
        axes = plt.gca()
        axes.set_ylim([0, 900])
        plt.ylim = (0, 800)
        plt.hist(W3, bins=10, histtype='stepfilled')
        plt.grid()
        plt.tight_layout()
        plt.savefig("Final_Weights{y}.png".format(y=i))

        #plt.show()

        #history_main1.update(history_main[i])

    #model_gt = LenetModel.pruned_nn.pruned_nn(pruning_params_unpruned, pruning_params_unpruned)

    # Load winning ticket (from above)-
    #model_gt.load_weights("LeNet_MNIST_Random_Weights.h5")

    # Strip model of pruning parameters-
    #model_gt_stripped = strip_pruning(model_gt)


        #hist_df = pd.DataFrame(history_main[i])
        #df = pd.DataFrame(list(history_main[i].items()))
        #hist_csv_file = 'history.csv'
        #with open(hist_csv_file, mode='w') as f:
        #    hist_df.to_csv(f)
        #df = pandas.read_csv('history.csv')
        #print("CSV", df



with open("C:/Users/Admin/PycharmProjects/Lenet_FC_Conv/LeNet_MNIST_history_main_Experiment.pkl","wb") as f:
          pickle.dump(history_main, f)








    #np.savez("C:/Users/Admin/PycharmProjects/Modularized_Lenet_FC_Convt/LeNet_MNIST_history_main_Experiment.npz", history_main=history_main)

    #with open("C:/Users/Admin/PycharmProjects/Lenet_FC_Conv/LeNet_MNIST_history_main_Experiment = {1}\n.pkl".format(i),"wb") as f:
          #pickle.dump(history_main, f)

