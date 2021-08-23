import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow_model_optimization as tfmot
#from tensorflow_model_optimization.sparsity import keras as sparsity
# from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from tensorflow.keras.layers import AveragePooling2D, Conv2D, MaxPooling2D, ReLU
from tensorflow.keras import models, layers, datasets
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.initializers import RandomNormal
# import math
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep
import PruningPercent
import Dataset
import LenetModel
import Prune
import Training
import os
import pickle
os.getcwd()

#history_main = np.load("C:/Users/Admin/PycharmProjects/ModularizedLenet/LeNet_MNIST_history_main_Experiment.npz", allow_pickle=True)['history_main']

with open("C:/Users/Admin/PycharmProjects/Lenet_FC_Conv/LeNet_MNIST_history_main_Experiment.pkl", "rb") as f:
    history_main = pickle.load(f)

print("Data from pkl files",history_main)

#print(list(list(history_main.values())[0].keys())[0])


plot_accuracy = {}
plot_test_accuracy = {}


for k in history_main.keys():
    epoch_length = len(history_main[k]['accuracy'])
    plot_accuracy[history_main[k]['percentage_wts_pruned']] = history_main[k]['accuracy'][epoch_length - 1]

for k in history_main.keys():
    epoch_length = len(history_main[k]['accuracy'])
    plot_test_accuracy[history_main[k]['percentage_wts_pruned']] = history_main[k]['val_accuracy'][epoch_length - 1]
fig=plt.figure(figsize=(9, 7), dpi= 80, facecolor='w', edgecolor='k')

x=np.array(list(plot_accuracy.keys()))
x1=np.array(list(plot_test_accuracy.keys()))
plt.plot(x, list(plot_accuracy.values()), label = 'training_accuracy', marker='*')
plt.plot(x1, list(plot_test_accuracy.values()), label = 'testing_accuracy', marker='*')

plt.legend()
plt.grid()
plt.title("Accuracy on LeNet-5 ")
plt.xlim(100,0,2)
plt.xlabel("Percentage of weights remaining")
plt.ylabel("Test Accuracy")
plt.savefig("Accuracy.png")
#plt.show()

fig=plt.figure(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')
plot_percent_accuracy = {}
for k in history_main.keys():
    epoch_length = len(history_main[k]['val_accuracy'])
    plot_percent_accuracy[history_main[k]['percentage_wts_pruned']] = np.amax(history_main[k]['val_accuracy'])
    x=np.array(list(plot_percent_accuracy.keys())) * 1000
plt.plot(np.array(history_main[1]['val_accuracy']), label=str(np.around(history_main[1]['percentage_wts_remaining'] , decimals=3)), marker='*')
plt.plot(history_main[4]['val_accuracy'], label=str(np.around(history_main[4]['percentage_wts_remaining'] , decimals=3)), marker='*')
plt.plot(history_main[8]['val_accuracy'], label=str(np.around(history_main[8]['percentage_wts_remaining'] , decimals=3)), marker='*')
plt.plot(history_main[13]['val_accuracy'], label=str(np.around(history_main[13]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[16]['val_accuracy'], label=str(np.around(history_main[16]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[19]['val_accuracy'], label=str(np.around(history_main[19]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[24]['val_accuracy'], label=str(np.around(history_main[8]['percentage_wts_remaining'] , decimals=3)), marker='*')
"""""plt.plot(history_main[40]['val_accuracy'], label=str(np.around(history_main[13]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[60]['val_accuracy'], label=str(np.around(history_main[16]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[80]['val_accuracy'], label=str(np.around(history_main[19]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[88]['val_accuracy'], label=str(np.around(history_main[8]['percentage_wts_remaining'] , decimals=3)), marker='*')
plt.plot(history_main[95]['val_accuracy'], label=str(np.around(history_main[13]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[100]['val_accuracy'], label=str(np.around(history_main[16]['percentage_wts_remaining'], decimals=3)), marker='*')
plt.plot(history_main[110]['val_accuracy'], label=str(np.around(history_main[19]['percentage_wts_remaining'], decimals=3)), marker='*')"""""

#plt.xticks([0, 2500, 5000, 7500,10000])
plt.legend()
plt.grid()
plt.title("Percentage")
plt.xlabel("Number of epochs")
plt.ylabel("Test accuracy")
plt.savefig('Percent.png')
#plt.show()

plot_num_epochs = {}
#plot_num_epochs_test = {}

for k in history_main.keys():
    num_epochs = len(history_main[k]['val_accuracy'])
    plot_num_epochs[history_main[k]['percentage_wts_pruned']] = np.array(num_epochs) * 1000

fig=plt.figure(figsize=(10, 9), dpi= 80, facecolor='w', edgecolor='k')

plt.plot(list(plot_num_epochs.keys()), list(plot_num_epochs.values()), marker='*')
#plt.xlim(100,0,2)
plt.grid()
plt.title("Percentage of weights pruned VS number of epochs (Early Stopping)")
plt.xlabel("Percentage of weights pruned")
plt.ylabel("Number of epochs. 1 Epoch = 1000 iterations")
plt.savefig("Epochs.png")
#plt.show()