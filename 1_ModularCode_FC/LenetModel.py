""" LenetModel for initialization of Lenet Fullyconnect Architecture """

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude, strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep
import yaml
import Dataset
import Prune

# Load yaml file
with open("C:/Users/Admin/PycharmProjects/ModularCode_FC/Experiment_FC.yaml", "r") as stream:
    config = yaml.safe_load(stream)

tf.random.set_seed(config["seed"])

if config["enable_gpu"]:
    physical_devices= tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class pruned_nn():

    def pruned_nn(pruning_params_fc, pruning_params_op ):
        lr = config["learning_rate"]
        # Initializer
        if config["kernel_init"] == "GlorotUniform":
            init = tf.keras.initializers.GlorotUniform(seed=config["seed"])
        elif config["kernel_init"] == "GlorotNormal":
            init = tf.keras.initializers.GlorotNormal(seed=config["seed"])

        pruned_model = Sequential()
        pruned_model.add(tf.keras.layers.InputLayer(input_shape=(784,)))
        pruned_model.add(Flatten())
        pruned_model.add(prune_low_magnitude(
            Dense(units=config["neuron_units"]["Dense1"], activation=config["activation_function"], kernel_initializer=init),
            **pruning_params_fc))
        # pruned_model.add(l.Dropout(0.2))
        pruned_model.add(prune_low_magnitude(
            Dense(units=config["neuron_units"]["Dense2"], activation=config["activation_function"], kernel_initializer=init),
            **pruning_params_fc))
        # pruned_model.add(l.Dropout(0.1))
        pruned_model.add(prune_low_magnitude(
            Dense(units=config["neuron_units"]["Output"], activation=config["output_activation"]),
            **pruning_params_op))

        # Compile pruned CNN-
        pruned_model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(lr),
            #optimizer=tf.keras.optimizers.Adam(lr),
            metrics=['accuracy'])

        return pruned_model

    # Add a pruning step callback to peg the pruning step to the optimizer's
    # step.
    callback = [
        UpdatePruningStep(),

        tf.keras.callbacks.EarlyStopping(
            monitor=config["monitor"], patience=config["patience"],
            min_delta=config["min_delta"]
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
    orig_model.summary()
