"""PruningPercent to calculate percentages of weights to be pruned in every pruning round.
   Extracted from yaml file:
   -Percent to be pruned (P = 0.8, inorderto calculate 20% from every pruning round),
   -maximum pruning performed is till 0.2% (Extracted from yaml file) of all parameters,
   -maximum number of rounds for which pruning is to be carried out (until 0.2% of total weights)
   #Option between P%^1/n and P% from every pruning round

 """

import tensorflow as tf
import numpy as np
import yaml

# Load yaml file
with open("C:/Users/Admin/PycharmProjects/ModularCode_Conv6/Experiment_Conv6.yaml", "r") as stream:
    config = yaml.safe_load(stream)

tf.random.set_seed(config["seed"])

if config["enable_gpu"]:
    physical_devices= tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


#Edit values after summary is printed in the output
# number of convolutional parameters-
conv1 = config["conv1"]
conv2 = config["conv2"]
conv3 = config["conv3"]
conv4 = config["conv4"]
conv5 = config["conv5"]
conv6 = config["conv6"]
#number of fully-connected dense parameters-
dense1 = config["dense1"]
dense2 = config["dense2"]
op_layer = config["op_layer"]

# total number of parameters-
total_params = conv1 + conv2 + conv3 + conv4 + conv5 + conv6 + dense1 + dense2 + op_layer

print("\nTotal number of trainable parameters = {0}\n".format(total_params))

# maximum pruning performed is till 0.2% of all parameters-
max_pruned_params = config["Max_Pruning"]
loc_tot_params = total_params
loc_conv1 = conv1
loc_conv2 = conv2
loc_conv3 = conv3
loc_conv4 = conv4
loc_conv5 = conv5
loc_conv6 = conv6
loc_dense1 = dense1
loc_dense2 = dense2
loc_op_layer = op_layer

# variable to count number of pruning rounds-
n = 0

# Lists to hold percentage of weights pruned in each round for all layers in CNN-
conv1_pruning = []
conv2_pruning = []
conv3_pruning = []
conv4_pruning = []
conv5_pruning = []
conv6_pruning = []
dense1_pruning = []
dense2_pruning = []
op_layer_pruning = []


num_pruning_rounds = config["num_of_pruning_rounds"]



Pruning = config["Pruning_Option1"]
# Percent for pruning
if Pruning == '20% from every layer':

        while loc_tot_params >= max_pruned_params:
                loc_conv1 *= config["conv_pruning"]  # 15% weights are pruned
                loc_conv2 *= config["conv_pruning"]   # 15% weights are pruned
                loc_conv3 *= config["conv_pruning"]  # 15% weights are pruned
                loc_conv4 *= config["conv_pruning"]  # 15% weights are pruned
                loc_conv5 *= config["conv_pruning"] # 15% weights are pruned
                loc_conv6 *= config["conv_pruning"]  # 15% weights are pruned
                loc_dense1 *= config["dense_pruning"]  # 20% weights are pruned
                loc_dense2 *= config["dense_pruning"]   # 20% weights are pruned
                loc_op_layer *= config["output_pruning"]   # 10% weights are pruned

                conv1_pruning.append(((conv1 - loc_conv1) / conv1) * 100)
                conv2_pruning.append(((conv2 - loc_conv2) / conv2) * 100)
                conv3_pruning.append(((conv3 - loc_conv3) / conv3) * 100)
                conv4_pruning.append(((conv4 - loc_conv4) / conv4) * 100)
                conv5_pruning.append(((conv5 - loc_conv5) / conv5) * 100)
                conv6_pruning.append(((conv6 - loc_conv6) / conv6) * 100)
                dense1_pruning.append(((dense1 - loc_dense1) / dense1) * 100)
                dense2_pruning.append(((dense2 - loc_dense2) / dense2) * 100)
                op_layer_pruning.append(((op_layer - loc_op_layer) / op_layer) * 100)

                loc_tot_params = loc_conv1 + loc_conv2 + loc_conv3 + loc_conv4 + loc_conv5 + loc_conv6 + loc_dense1 + loc_dense2 + loc_op_layer

                n += 1

                print("\nConv1 = {0:.3f}, Conv2 = {1:.3f}, Conv3 = {2:.4f}".format(loc_conv1, loc_conv2, loc_conv3))
                print("Conv4 = {0:.3f}, Conv5 = {1:.3f} & Conv6 = {2:.3f}".format(loc_conv4, loc_conv5, loc_conv6))
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
                loc_conv1 = loc_conv1 - (loc_conv1 * prune_percent)
                loc_conv2 = loc_conv2 - (loc_conv2 * prune_percent)
                loc_conv3 = loc_conv3 - (loc_conv3 * prune_percent)
                loc_conv4 = loc_conv4 - (loc_conv4 * prune_percent)
                loc_conv5 = loc_conv5 - (loc_conv5 * prune_percent)
                loc_conv6 = loc_conv6 - (loc_conv6 * prune_percent)
                loc_dense1 = loc_dense1 - (loc_dense1 * prune_percent)  # 20% weights are pruned
                loc_dense2 = loc_dense2 - (loc_dense2 * prune_percent)  # 20% weights are pruned
                loc_op_layer = loc_op_layer - (loc_op_layer * prune_percent)  # 20% weights are pruned

                conv1_pruning.append(prune_percent)
                conv2_pruning.append(prune_percent)
                conv3_pruning.append(prune_percent)
                conv4_pruning.append(prune_percent)
                conv5_pruning.append(prune_percent)
                conv6_pruning.append(prune_percent)
                dense1_pruning.append(prune_percent)
                dense2_pruning.append(prune_percent)
                op_layer_pruning.append(prune_percent)

                loc_tot_params = loc_conv1 + loc_conv2 + loc_conv3 + loc_conv4 + loc_conv5 + loc_conv6 + loc_dense1 + loc_dense2 + loc_op_layer
                n += 1

                print("\nConv1 = {0:.3f}, Conv2 = {1:.3f}, Conv3 = {2:.4f}".format(loc_conv1, loc_conv2, loc_conv3))
                print("Conv4 = {0:.3f}, Conv5 = {1:.3f} & Conv6 = {2:.3f}".format(loc_conv4, loc_conv5, loc_conv6))
                print("Dense1 = {0:.3f}, Dense2 = {1:.3f} & O/p layer = {2:.3f}".format(
                        loc_dense1, loc_dense2, loc_op_layer))
                print("Total number of parameters = {0:.3f}\n".format(loc_tot_params))

print("\nnumber of pruning rounds = {0}\n\n".format(num_pruning_rounds))


# Convert from list to np.array-
conv1_pruning = np.array(conv1_pruning)
conv2_pruning = np.array(conv2_pruning)
conv3_pruning = np.array(conv3_pruning)
conv4_pruning = np.array(conv4_pruning)
conv5_pruning = np.array(conv5_pruning)
conv6_pruning = np.array(conv6_pruning)
dense1_pruning = np.array(dense1_pruning)
dense2_pruning = np.array(dense2_pruning)
op_layer_pruning = np.array(op_layer_pruning)

dense1_pruning = np.insert(dense1_pruning,0,0)
conv1_pruning = np.insert(conv1_pruning,0,0)
dense2_pruning = np.insert(dense2_pruning,0,0)
conv2_pruning = np.insert(conv2_pruning,0,0)
op_layer_pruning = np.insert(op_layer_pruning,0,0)

# Round off numpy arrays to 5 decimal digits-
conv1_pruning = np.round(conv1_pruning, decimals=5)
conv2_pruning = np.round(conv2_pruning, decimals=5)
conv3_pruning = np.round(conv3_pruning, decimals=5)
conv4_pruning = np.round(conv4_pruning, decimals=5)
conv5_pruning = np.round(conv5_pruning, decimals = 3)
conv6_pruning = np.round(conv6_pruning, decimals = 3)
dense1_pruning = np.round(dense1_pruning, decimals=5)
dense2_pruning = np.round(dense2_pruning, decimals=5)
op_layer_pruning = np.round(op_layer_pruning, decimals=5)

conv1_pruning = conv1_pruning / 100
conv2_pruning = conv2_pruning / 100
conv3_pruning = conv3_pruning / 100
conv4_pruning = conv4_pruning / 100
conv5_pruning = conv5_pruning / 100
conv6_pruning = conv6_pruning / 100
dense1_pruning = dense1_pruning / 100
dense2_pruning = dense2_pruning / 100
op_layer_pruning = op_layer_pruning / 100

#print(conv1_pruning)
print("\nNumber of pruning rounds for LeNet NN = {0}\n".format(num_pruning_rounds))