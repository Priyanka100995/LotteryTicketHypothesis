# LotteryTicketHypothesis

CONTENTS of ModularCodes FC - Fully Connected Lenet Architecture/ Conv2,Conv4,Conv6 - Scaled down variants of VGG16: 

1.Experiment_FC/Conv2/Conv4/Conv6.yaml : consists of Hyperparameters to be modified before running rest of the code

	a.Batchsize/epochs/num_classes/img_rows/img_cols/learning_rate/shuffle/num_of_pruning_rounds/num_trials
	b.Neuron units (300,100,10 – Lenet/ 256,256,10-Conv2/4/6)
	c.Optimizer/Initializer/Activation function
	d.Convolution & fully-connected dense parameters 
	e.Early Stopping
	f.Pruning rate
		i.Pruning_option1 (default): 20% weights from every round 
		ii.Pruning_option2 : 20% weights from the surviving weights from previous round

2.Prune.py: imports tensorflow model optimization functions (ConstantSparsity)

3.PruningPercent.py: Calculates pruning percentages to be used on every layer of neural network (i.e., Dense layers / Conv layers/ Output Layers)

4.Dataset.py (CIFAR - Conv/ MNIST – FC): CIFAR/MNIST downloaded and prepared

5.Lenet/Conv2/Conv4/Conv6Model.py: Definition of neural network models (Number of dense and Convolutional layer parameters to be entered into .yaml file after running this program)

6.Training.py : Layerwise iterative pruning and training for the chosen number of trials and number of pruning rounds. Saves metrics (Accuracies/Losses/ Percentages of weights pruned from every pruning round) into a .pkl file

7.Plot.py : Plot Accuracy, Epochs required for training in every round and Percentages vs Percentage pruned Graphs from the .pkl file




Steps to run the LTH Experiments:
 
1.Make the following changes in the .yaml file : 

	a.enable_gpu: true/false
	b.learning_rate: 0.0012 – Adam , 0.01/0.0012 - SGD
	c.num_of_pruning_rounds: 25 – Lenet/ 45 – Conv2/4/6
	d.num_trials: 5
	e.neuron_units: FC(300/100/10) CONV(256/256/10)
  			Dense1: 300 
  			Dense2: 100
  			Output: 10
	f.optimizer: AdaM
	g.kernel_init: GlorotNormaL
	h.number of convolution & fully-connected dense parameters : TO BE CHANGED AFTER RUNNING RESPECTIVE MODEL PROG
	i.Early Stopping: change min_delta and patience 
	j.Pruning_rate: !!str 20% #Pruning_rate must correspond to the below rates
	k.Pruning rates : conv_pruning: 0.9 #10% pruning/ 15% for CONV6
                          dense_pruning: 0.8 #20% pruning  
	                  output_pruning: 0.9 #10% pruning 

                      
2.Run Dataset.py : 

	a.Change the path to the location of .yaml file "C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r"
	b.Change optimizer on line 76
		i.optimizer = tf.keras.optimizers.Adam(lr)	
		ii.#optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

3.Run Lenet/Conv2/Conv4/Conv6Model.py : 

	a.Change the path to the location of .yaml file "C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r"
	b.Change optimizer on line 52/53
		i.optimizer = tf.keras.optimizers.Adam(lr)	
		ii.#optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
	c.Change number of convolution & fully-connected dense parameters from model summary. (Refer : Experiment.yaml 1. h.)

4.Run PruningPercent.py : 

	a.Change the path to the location of .yaml file "C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r"
	b.Change pruning rate Experiment.yaml (Refer: Experiment.yaml 1.j.)

5.Run Training.py:

	a.Change the path to the location of .yaml file "C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r"
	b.Change the path to the location of .pkl file "C:/Users/Admin/PycharmProjects/ModularCode_FC/LeNet_MNIST_history_main_Experiment.pkl"

6.Run Plot.py : 

	a.Change the path to the location of .pkl file "C:/Users/Admin/PycharmProjects/ModularCode_FC/LeNet_MNIST_history_main_Experiment.pkl"


Example:

Lenet Architecture/ 20% Pruning/ 25 pruning rounds/ Num of trials-5/ Adam-GlorotNormal: 

1.Set following from Experiment_FC.yaml

	a.num_of_pruning_rounds: 25
	b.num_trials: 5
	c.optimizer: Adam
	d.kernel_init: GlorotNormal
	e.use_bias: !!bool True
	f.activation_function: !!str tanh   # tanh
	g.output_activation: !!str softmax
	h.early_stopping: !!bool true
	i.patience: !!int 3
	j.min_delta: 0.001
	k.Pruning_Option1: !!str 20% from every layer
	l.Pruning_rate: !!str 20% 
	m.dense_pruning: 0.8 
	n.output_pruning: 0.9

2.Run Dataset.py

	a.Change optimizer on line 76
		i.optimizer = tf.keras.optimizers.Adam(lr)	
		ii.#optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

3.Run LenetModel.py

	a.Change optimizer on line 52/53
		i.optimizer = tf.keras.optimizers.Adam(lr)	
		ii.#optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

4.Change number of convolution & fully-connected dense parameters from Experiment_FC.yaml file:

	a.dense1: 235500
	b.dense2: 30100
	c.op_layer: 1010
	d.orig_sum_params: 266610
	
5.Run PruningPercent.py

6.Run Training.py

7.Run Plot.py (For Resultant Graphs)

