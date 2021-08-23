"""PruningPercent to calculate percentages of weights to be pruned in every pruning round.
   Extracted from yaml file:
   -Percent to be pruned (P = 0.8, in order to calculate 20% from every pruning round),
   -maximum pruning performed is till 0.2% (Extracted from yaml file) of all parameters,
   -maximum number of rounds for which pruning is to be carried out (until 0.2% of total weights)
   #Option between P%^1/n and P% from every pruning round

 """

import tensorflow as tf
import numpy as np
import Prune
import Dataset
import yaml

# Load yaml file
with open("C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r") as stream:
    config = yaml.safe_load(stream)

tf.random.set_seed(config["seed"])

if config["enable_gpu"]:
    physical_devices= tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


#Edit values after summary is printed in the output
#number of fully-connected dense parameters-
dense1 = config["dense1"]
dense2 = config["dense2"]
op_layer = config["op_layer"]

# total number of parameters-
total_params = dense1 + dense2 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# maximum pruning performed is till 0.2% of all parameters-
max_pruned_params = config["Max_Pruning"]
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


num_pruning_rounds = config["num_of_pruning_rounds"]



Pruning = config["Pruning_Option1"]
# Percent for pruning
if Pruning == '20% from every layer':

        while loc_tot_params >= max_pruned_params:
                loc_dense1 *= config["dense_pruning"]  # 20% weights are pruned
                loc_dense2 *= config["dense_pruning"]   # 20% weights are pruned
                loc_op_layer *= config["output_pruning"]   # 10% weights are pruned

                dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
                dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
                op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

                loc_tot_params = loc_dense1 + loc_dense2 + loc_op_layer

                n += 1

                print("Dense1 = {0:.3f}, Dense2 = {1:.3f} & O/p layer = {2:.3f}".format(
                        loc_dense1, loc_dense2, loc_op_layer))
                print("Total number of parameters = {0:.3f}\n".format(loc_tot_params))
                if n == num_pruning_rounds:
                        break


else:
        for i in range(1, num_pruning_rounds + 1):
                prune_percent = (2 ** (1 / i))

                print("i", i)
                print("Prune Percentage", prune_percent)

                loc_dense1 = loc_dense1 - (loc_dense1 * prune_percent)  # 20% weights are pruned
                loc_dense2 = loc_dense2 - (loc_dense2 * prune_percent)  # 20% weights are pruned
                loc_op_layer = loc_op_layer - (loc_op_layer * prune_percent)  # 20% weights are pruned

                dense1_pruning.append(prune_percent)
                dense2_pruning.append(prune_percent)
                op_layer_pruning.append(prune_percent)

                loc_tot_params = loc_dense1 + loc_dense2 + loc_op_layer
                n += 1

                print("Dense1 = {0:.3f}, Dense2 = {1:.3f} & O/p layer = {2:.3f}".format(
                        loc_dense1, loc_dense2, loc_op_layer))
                print("Total number of parameters = {0:.3f}\n".format(loc_tot_params))
                loc_tot_params = loc_dense1 + loc_dense2 + loc_op_layer

print("\nnumber of pruning rounds = {0}\n\n".format(num_pruning_rounds))


# Convert from list to np.array-
dense1_pruning = np.array(dense1_pruning)
dense2_pruning = np.array(dense2_pruning)
op_layer_pruning = np.array(op_layer_pruning)
dense1_pruning = np.insert(dense1_pruning,0,0)
dense1_pruning = np.insert(dense1_pruning,0,0)
dense2_pruning = np.insert(dense2_pruning,0,0)
op_layer_pruning = np.insert(op_layer_pruning,0,0)

# Round off numpy arrays to 3 decimal digits-
dense1_pruning = np.round(dense1_pruning, decimals=5)
dense2_pruning = np.round(dense2_pruning, decimals=5)
op_layer_pruning = np.round(op_layer_pruning, decimals=5)

dense1_pruning = dense1_pruning / 100
dense2_pruning = dense2_pruning / 100
op_layer_pruning = op_layer_pruning / 100

print("Prune Percent dense1_pruning", dense1_pruning)
print("\nNumber of pruning rounds for LeNet NN = {0}\n".format(num_pruning_rounds))