""" LenetModel for initialization of Lenet Fullyconnect Architecture """

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep
import Dataset
import Prune
from tensorflow.keras.callbacks import TensorBoard

class pruned_nn():

    def pruned_nn(pruning_params_fc, pruning_params_op):
        """
        Function to define the architecture of a neural network model
        following LeNet-5 architecture for MNIST dataset and using
        provided parameter which are used to prune the model.


        Input: 'pruning_params' Python 3 dictionary containing parameters which are used for pruning
        Output: Returns designed and compiled neural network model
        """



        pruned_model = tf.keras.Sequential()
        pruned_model.add(tf.keras.layers.InputLayer(input_shape=(784,)))


        pruned_model.add(Flatten())

        pruned_model.add(prune_low_magnitude(
            Dense(
                units=300, activation='tanh',
                kernel_initializer=tf.initializers.GlorotNormal()
            ),
            **pruning_params_fc)
        )

        pruned_model.add(prune_low_magnitude(
            Dense(
                units=100, activation='tanh',
                kernel_initializer=tf.initializers.GlorotNormal()
            ),
            **pruning_params_fc)
        )

        pruned_model.add(prune_low_magnitude(
            Dense(
                units=10, activation='softmax'
            ),
            **pruning_params_op)
        )

        # Compile pruned CNN-
        pruned_model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            # optimizer='adam',
            optimizer=tf.keras.optimizers.Adam(lr=0.0012),
            #optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
            metrics=['accuracy']
        )

        return pruned_model

    # Add a pruning step callback to peg the pruning step to the optimizer's
    # step.
    callback = [
        UpdatePruningStep(),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3,
            min_delta=0.001
        )

    ]


    pruning_params_unpruned = {
        'pruning_schedule': Prune.ConstantSparsity(
            target_sparsity=0.0, begin_step=0,
            end_step=Dataset.Data.end_step, frequency=100
        )
    }

    # Initialize a CNN model-
    orig_model = pruned_nn(pruning_params_unpruned, pruning_params_unpruned)
    #len(orig_model.trainable_weights)
    #print("Len", len(orig_model.trainable_weights))

    orig_model.summary()
    #orig_model= strip_pruning(orig_model)
    #plot_model(orig_model, to_file='model.png')