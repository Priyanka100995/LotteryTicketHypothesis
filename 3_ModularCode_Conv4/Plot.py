import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import os
import pickle
os.getcwd()

# Load yaml file
with open("C:/Users/Admin/PycharmProjects/ModularCode_Conv4/Experiment_Conv4.yaml", "r") as stream:
    config = yaml.safe_load(stream)

tf.random.set_seed(config["seed"])

with open("C:/Users/Admin/PycharmProjects/ModularCode_Conv4/Conv4_CIFAR10.pkl", "rb") as f:
    history_main = pickle.load(f)

#print(history_main)

#Accuracy Multiple trial
#rows = num of trials

#Accuracy vs Percentages of weights pruned
plot_accuracy = {}
plot_test_accuracy = {}


for row in range(len(history_main)):
    for col in range(len(history_main[row])):
        length = len(history_main[row][col]['accuracy'])
        #print("Len", length)
        plot_accuracy[history_main[1][col]['percentage_wts_pruned']] = np.average(
            np.array((history_main[row][col]['accuracy'][length - 1])))
        plot_test_accuracy[history_main[1][col]['percentage_wts_pruned']] = np.average(
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
#plt.savefig("Accuracy.png")
plt.show()

#Epochs vs Percentages of weights pruned
plot_epoch_accuracy = {}
for row in range(len(history_main)):
    for col in range(len(history_main[row])):
        length = len(history_main[row][col]['val_accuracy'])
        plot_epoch_accuracy[history_main[1][col]['percentage_wts_pruned']] = np.average(np.array([length - 1])) * 1000

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
#plt.savefig("Val Accuracy vs Early stopping.png")
plt.show()


#Accuracy at various Percentages of weights pruned
Pruning_rate = config["Pruning_rate"]

if Pruning_rate == '20%':
        #20% pruning rate

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
                x2.append((history_main[row][6]['val_accuracy']))
                x21.extend((history_main[row][6]['val_accuracy']))
                x3.append((history_main[row][15]['val_accuracy']))
                x31.extend((history_main[row][15]['val_accuracy']))
                x4.append((history_main[row][25]['val_accuracy']))
                x41.extend((history_main[row][25]['val_accuracy']))
                x5.append((history_main[row][31]['val_accuracy']))
                x51.extend((history_main[row][31]['val_accuracy']))
                x6.append((history_main[row][38]['val_accuracy']))
                x61.extend((history_main[row][38]['val_accuracy']))
                x7.append((history_main[row][45]['val_accuracy']))
                x71.extend((history_main[row][45]['val_accuracy']))

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

        plt.plot((pd.DataFrame(x2).mean(axis = 0)).index.values, pd.DataFrame(x2).mean(axis = 0), label=str(np.around(history_main[1][6]['percentage_wts_remaining'], decimals=3)), color = 'orange')
        plt.fill_between((pd.DataFrame(x2).mean(axis = 0)).index.values, pd.DataFrame(x2).mean(axis = 0)-x21, pd.DataFrame(x2).mean(axis = 0)+x21, alpha=.2, color = 'orange')

        plt.plot((pd.DataFrame(x3).mean(axis = 0)).index.values, pd.DataFrame(x3).mean(axis = 0), label=str(np.around(history_main[1][15]['percentage_wts_remaining'], decimals=3)) , color = 'green')
        plt.fill_between((pd.DataFrame(x3).mean(axis = 0)).index.values, pd.DataFrame(x3).mean(axis = 0)-x31, pd.DataFrame(x3).mean(axis = 0)+x31, alpha=.2, color = 'green')

        plt.plot((pd.DataFrame(x4).mean(axis = 0)).index.values, pd.DataFrame(x4).mean(axis = 0), label=str(np.around(history_main[1][25]['percentage_wts_remaining'], decimals=3)), color = 'yellow')
        plt.fill_between((pd.DataFrame(x4).mean(axis = 0)).index.values, pd.DataFrame(x4).mean(axis = 0)-x41, pd.DataFrame(x4).mean(axis = 0)+x41, alpha=.2, color = 'yellow')

        plt.plot((pd.DataFrame(x5).mean(axis = 0)).index.values, pd.DataFrame(x5).mean(axis = 0), label=str(np.around(history_main[1][31]['percentage_wts_remaining'], decimals=3)), color = 'red')
        plt.fill_between((pd.DataFrame(x5).mean(axis = 0)).index.values, pd.DataFrame(x5).mean(axis = 0)-x51, pd.DataFrame(x5).mean(axis = 0)+x51, alpha=.2, color = 'red')

        plt.plot((pd.DataFrame(x6).mean(axis = 0)).index.values, pd.DataFrame(x6).mean(axis = 0), label=str(np.around(history_main[1][38]['percentage_wts_remaining'], decimals=3)), color = 'pink')
        plt.fill_between((pd.DataFrame(x6).mean(axis = 0)).index.values, pd.DataFrame(x6).mean(axis = 0)-x61, pd.DataFrame(x6).mean(axis = 0)+x61, alpha=.2, color = 'pink')

        plt.plot((pd.DataFrame(x7).mean(axis = 0)).index.values, pd.DataFrame(x7).mean(axis = 0), label=str(np.around(history_main[1][45]['percentage_wts_remaining'], decimals=3)), color = 'cyan')
        plt.fill_between((pd.DataFrame(x7).mean(axis = 0)).index.values, pd.DataFrame(x7).mean(axis = 0)-x71, pd.DataFrame(x7).mean(axis = 0)+x71, alpha=.2, color = 'cyan')

        plt.legend()
        plt.grid()
        plt.title("Percentage vs Test Accuracy")
        plt.xlabel("Number of epochs. 1 Epoch = 1000 iterations")
        plt.ylabel("Test accuracy")
        plt.savefig('Percent.png')
        plt.show()


#Percentages
elif Pruning_rate == '40%':
        #40% pruning rate

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

        #x1 = (np.asarray(x1))
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
        #plt.savefig('Percent.png')
        plt.show()

elif Pruning_rate == '60%':
        #60 % pruning rate

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
                x2.append((history_main[row][2]['val_accuracy']))
                x21.extend((history_main[row][2]['val_accuracy']))
                x3.append((history_main[row][4]['val_accuracy']))
                x31.extend((history_main[row][4]['val_accuracy']))
                x4.append((history_main[row][7]['val_accuracy']))
                x41.extend((history_main[row][7]['val_accuracy']))
                x5.append((history_main[row][9]['val_accuracy']))
                x51.extend((history_main[row][9]['val_accuracy']))
                x6.append((history_main[row][11]['val_accuracy']))
                x61.extend((history_main[row][11]['val_accuracy']))
                x7.append((history_main[row][15]['val_accuracy']))
                x71.extend((history_main[row][15]['val_accuracy']))

        #x1 = (np.asarray(x1))
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

        plt.plot((pd.DataFrame(x2).mean(axis = 0)).index.values, pd.DataFrame(x2).mean(axis = 0), label=str(np.around(history_main[1][2]['percentage_wts_remaining'], decimals=3)), color = 'orange')
        plt.fill_between((pd.DataFrame(x2).mean(axis = 0)).index.values, pd.DataFrame(x2).mean(axis = 0)-x21, pd.DataFrame(x2).mean(axis = 0)+x21, alpha=.2, color = 'orange')

        plt.plot((pd.DataFrame(x3).mean(axis = 0)).index.values, pd.DataFrame(x3).mean(axis = 0), label=str(np.around(history_main[1][4]['percentage_wts_remaining'], decimals=3)) , color = 'green')
        plt.fill_between((pd.DataFrame(x3).mean(axis = 0)).index.values, pd.DataFrame(x3).mean(axis = 0)-x31, pd.DataFrame(x3).mean(axis = 0)+x31, alpha=.2, color = 'green')

        plt.plot((pd.DataFrame(x4).mean(axis = 0)).index.values, pd.DataFrame(x4).mean(axis = 0), label=str(np.around(history_main[1][7]['percentage_wts_remaining'], decimals=3)), color = 'yellow')
        plt.fill_between((pd.DataFrame(x4).mean(axis = 0)).index.values, pd.DataFrame(x4).mean(axis = 0)-x41, pd.DataFrame(x4).mean(axis = 0)+x41, alpha=.2, color = 'yellow')

        plt.plot((pd.DataFrame(x5).mean(axis = 0)).index.values, pd.DataFrame(x5).mean(axis = 0), label=str(np.around(history_main[1][9]['percentage_wts_remaining'], decimals=3)), color = 'red')
        plt.fill_between((pd.DataFrame(x5).mean(axis = 0)).index.values, pd.DataFrame(x5).mean(axis = 0)-x51, pd.DataFrame(x5).mean(axis = 0)+x51, alpha=.2, color = 'red')

        plt.plot((pd.DataFrame(x6).mean(axis = 0)).index.values, pd.DataFrame(x6).mean(axis = 0), label=str(np.around(history_main[1][11]['percentage_wts_remaining'], decimals=3)), color = 'pink')
        plt.fill_between((pd.DataFrame(x6).mean(axis = 0)).index.values, pd.DataFrame(x6).mean(axis = 0)-x61, pd.DataFrame(x6).mean(axis = 0)+x61, alpha=.2, color = 'pink')

        plt.plot((pd.DataFrame(x7).mean(axis = 0)).index.values, pd.DataFrame(x7).mean(axis = 0), label=str(np.around(history_main[1][15]['percentage_wts_remaining'], decimals=3)), color = 'cyan')
        plt.fill_between((pd.DataFrame(x7).mean(axis = 0)).index.values, pd.DataFrame(x7).mean(axis = 0)-x71, pd.DataFrame(x7).mean(axis = 0)+x71, alpha=.2, color = 'cyan')

        plt.legend()
        plt.grid()
        plt.title("Percentage vs Test Accuracy")
        plt.xlabel("Number of epochs. 1 Epoch = 1000 iterations")
        plt.ylabel("Test accuracy")
        #plt.savefig('Percent.png')
        plt.show()

elif Pruning_rate == '80%':
        # 80 % pruning rate

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
            x2.append((history_main[row][1]['val_accuracy']))
            x21.extend((history_main[row][1]['val_accuracy']))
            x3.append((history_main[row][2]['val_accuracy']))
            x31.extend((history_main[row][2]['val_accuracy']))
            x4.append((history_main[row][3]['val_accuracy']))
            x41.extend((history_main[row][3]['val_accuracy']))
            x5.append((history_main[row][4]['val_accuracy']))
            x51.extend((history_main[row][4]['val_accuracy']))
            x6.append((history_main[row][5]['val_accuracy']))
            x61.extend((history_main[row][5]['val_accuracy']))
            x7.append((history_main[row][6]['val_accuracy']))
            x71.extend((history_main[row][6]['val_accuracy']))

        # x1 = (np.asarray(x1))
        x11 = np.std(x11)
        x21 = np.std(x21)
        x31 = np.std(x31)
        x41 = np.std(x41)
        x51 = np.std(x51)
        x61 = np.std(x61)
        x71 = np.std(x71)

        fig = plt.figure(figsize=(10, 9), dpi=80, facecolor='w', edgecolor='k')
        plt.plot((pd.DataFrame(x1).mean(axis=0)).index.values, (pd.DataFrame(x1).mean(axis=0)),
                 label=str(np.around(history_main[1][0]['percentage_wts_remaining'], decimals=3)), color='blue')
        plt.fill_between((pd.DataFrame(x1).mean(axis=0)).index.values, pd.DataFrame(x1).mean(axis=0) - x11,
                         pd.DataFrame(x1).mean(axis=0) + x11, alpha=.2, color='blue')

        plt.plot((pd.DataFrame(x2).mean(axis=0)).index.values, pd.DataFrame(x2).mean(axis=0),
                 label=str(np.around(history_main[1][1]['percentage_wts_remaining'], decimals=3)), color='orange')
        plt.fill_between((pd.DataFrame(x2).mean(axis=0)).index.values, pd.DataFrame(x2).mean(axis=0) - x21,
                         pd.DataFrame(x2).mean(axis=0) + x21, alpha=.2, color='orange')

        plt.plot((pd.DataFrame(x3).mean(axis=0)).index.values, pd.DataFrame(x3).mean(axis=0),
                 label=str(np.around(history_main[1][2]['percentage_wts_remaining'], decimals=3)), color='green')
        plt.fill_between((pd.DataFrame(x3).mean(axis=0)).index.values, pd.DataFrame(x3).mean(axis=0) - x31,
                         pd.DataFrame(x3).mean(axis=0) + x31, alpha=.2, color='green')

        plt.plot((pd.DataFrame(x4).mean(axis=0)).index.values, pd.DataFrame(x4).mean(axis=0),
                 label=str(np.around(history_main[1][3]['percentage_wts_remaining'], decimals=3)), color='yellow')
        plt.fill_between((pd.DataFrame(x4).mean(axis=0)).index.values, pd.DataFrame(x4).mean(axis=0) - x41,
                         pd.DataFrame(x4).mean(axis=0) + x41, alpha=.2, color='yellow')

        plt.plot((pd.DataFrame(x5).mean(axis=0)).index.values, pd.DataFrame(x5).mean(axis=0),
                 label=str(np.around(history_main[1][4]['percentage_wts_remaining'], decimals=3)), color='red')
        plt.fill_between((pd.DataFrame(x5).mean(axis=0)).index.values, pd.DataFrame(x5).mean(axis=0) - x51,
                         pd.DataFrame(x5).mean(axis=0) + x51, alpha=.2, color='red')

        plt.plot((pd.DataFrame(x6).mean(axis=0)).index.values, pd.DataFrame(x6).mean(axis=0),
                 label=str(np.around(history_main[1][5]['percentage_wts_remaining'], decimals=3)), color='pink')
        plt.fill_between((pd.DataFrame(x6).mean(axis=0)).index.values, pd.DataFrame(x6).mean(axis=0) - x61,
                         pd.DataFrame(x6).mean(axis=0) + x61, alpha=.2, color='pink')

        plt.plot((pd.DataFrame(x7).mean(axis=0)).index.values, pd.DataFrame(x7).mean(axis=0),
                 label=str(np.around(history_main[1][6]['percentage_wts_remaining'], decimals=3)), color='cyan')
        plt.fill_between((pd.DataFrame(x7).mean(axis=0)).index.values, pd.DataFrame(x7).mean(axis=0) - x71,
                         pd.DataFrame(x7).mean(axis=0) + x71, alpha=.2, color='cyan')

        plt.legend()
        plt.grid()
        plt.title("Percentage vs Test Accuracy")
        plt.xlabel("Number of epochs. 1 Epoch = 1000 iterations")
        plt.ylabel("Test accuracy")
        # plt.savefig('Percent.png')
        plt.show()
