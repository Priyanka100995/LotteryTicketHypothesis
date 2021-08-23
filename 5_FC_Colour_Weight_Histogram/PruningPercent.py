"""PruningPercent to calculate percentages of weights to be pruned in every pruning round.
   Extracted from yaml file:
   -Percent to be pruned (P = 0.8, inorderto calculate 20% from every pruning round),
   -maximum pruning performed is till 0.2% (Extracted from yaml file) of all parameters,
   -maximum number of rounds for which pruning is to be carried out (until 0.2% of total weights)
   #Option between P%^1/n and P% from every pruning round

 """

import tensorflow as tf
import numpy as np
import Prune
import Dataset
import yaml


with open(r'C:\Users\Admin\PycharmProjects\Lenet_FC_Conv\Experiment.yaml') as file:
    doc = yaml.load(file, Loader=yaml.FullLoader)

    sort_file = yaml.dump(doc, sort_keys=True)



num_classes = 10
#Edit values after summary is printed in the output
#number of fully-connected dense parameters-
dense1 = 235500
dense2 = 30100
op_layer = 1010

# total number of parameters-
total_params = dense1 + dense2 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# maximum pruning performed is till 0.2% of all parameters-
for key, val in doc[5].items():
        value = val
        print(key, value)

max_pruned_params = value * total_params

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

# Number of pruning rounds from yaml file
for key, val in doc[3].items():
      value1 = val

num_pruning_rounds = value1

# Percent for pruning
for key, val in doc[0].items():
        value2 = val

Percent = value2

for key, val in doc[6].items():
        value3 = val

Pruning_Option1 = value3

for key, val in doc[7].items():
        value4 = val

Pruning_Option2 = value4

# Percent for pruning
if Pruning_Option1 == '20% from every layer':

        while loc_tot_params >= max_pruned_params:
                loc_dense1 *= 0.8  # 20% weights are pruned
                loc_dense2 *= 0.8  # 20% weights are pruned
                loc_op_layer *= 0.9  # 10% weights are pruned

                dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
                dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
                op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

                loc_tot_params = loc_dense1 + loc_dense2 + loc_op_layer

                n += 1

                print("Dense1 = {0:.3f}, Dense2 = {1:.3f} & O/p layer = {2:.3f}".format(
                        loc_dense1, loc_dense2, loc_op_layer))
                print("Total number of parameters = {0:.3f}\n".format(loc_tot_params))



if Pruning_Option2 == '20% from wts from previous layer':
        for i in range(1, num_pruning_rounds + 1):
                prune_percent = (0.2 ** (1 / i))

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
                break

print("\nnumber of pruning rounds = {0}\n\n".format(num_pruning_rounds))


# Convert from list to np.array-
dense1_pruning = np.array(dense1_pruning)
dense2_pruning = np.array(dense2_pruning)
op_layer_pruning = np.array(op_layer_pruning)
dense1_pruning = np.insert(dense1_pruning,0,0)
op_layer_pruning = np.insert(op_layer_pruning,0,0)

# Round off numpy arrays to 3 decimal digits-
dense1_pruning = np.round(dense1_pruning, decimals=5)
dense2_pruning = np.round(dense2_pruning, decimals=5)
op_layer_pruning = np.round(op_layer_pruning, decimals=5)

dense1_pruning = dense1_pruning / 100
dense2_pruning = dense2_pruning / 100
op_layer_pruning = op_layer_pruning / 100

print("Prune Percent dense1_pruning", dense1_pruning)
print("\nNumber of pruning rounds for LeNet NN = {0} and number of epochs = {1}\n".format(num_pruning_rounds, Dataset.Data.num_epochs))