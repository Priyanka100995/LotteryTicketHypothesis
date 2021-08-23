# Import tensorflow & keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from keras.utils.vis_utils import plot_model
import os
import yaml

# Load yaml file
# Load yaml file
with open("C:/Users/Admin/PycharmProjects/FC/config_ANN_0_9_0.yml", "r") as stream:
    config = yaml.safe_load(stream)

# Metrik für Regression
def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator)) # 0 schlecht, 1 gut
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


def build_model_architecture():
    #lr = model_architecture_params()
    lr = config["learning_rate"]

    # Initializer
    if config["kernel_init"] == "GlorotUniform":
        init = tf.keras.initializers.GlorotUniform(seed=config["seed"])
    elif config["kernel_init"] == "GlorotNormal":
        init = tf.keras.initializers.GlorotNormal(seed=config["seed"])
    elif config["kernel_init"] == "HeNormal":
        init = tf.keras.initializers.he_normal(seed=config["seed"])
    elif config["kernel_init"] == "HeUniform":
        init = tf.keras.initializers.he_uniform(seed=config["seed"])

    # Build sequential model
    model = Sequential()

    # Add Inputlayer & 2 Hidden Layer
    model.add(keras.Input(shape=config["input_shape"]))
    model.add(Dense(units=config["neuron_units"]["Dense0"], activation=config["activation_function"],
                    kernel_initializer=init, use_bias=config["use_bias"], bias_initializer=config["bias_init"]))
    model.add(Dense(units=config["neuron_units"]["Dense1"], activation=config["activation_function"],
                    kernel_initializer=init, use_bias=config["use_bias"], bias_initializer=config["bias_init"]))
    model.add(Dense(units=config["neuron_units"]["Dense2"], activation=config["activation_function"],
                    kernel_initializer=init, use_bias=config["use_bias"], bias_initializer=config["bias_init"]))
    # model.add(Dense(units=config["neuron_units"]["Dense3"], activation=config["dense_activations"]))

    # Add Outputlayer
    model.add(Dense(units=config['output_shape'], activation=config["output_activation"]))
    # model.add(Dense(units=config['output_shape']))

    model.summary()
    print("Aktuelle Lernrate: ", lr)

    """Plot the architecture of the model"""
    # plot_model(model, to_file='model_layout.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mean_squared_error', metrics=[r_squared])
    return model


def load_model():
    lr = config["learning_rate"]
    model_path = config["loading_model_path"] + config["loading_model_name"] + config["loading_model_format"]
    dependencies = {'r_squared': r_squared}

    model = tf.keras.models.load_model(model_path, custom_objects=dependencies)
    model.summary()

    print("Aktuelle Lernrate: ", lr)
    """Plot the architecture of the model"""
    # plot_model(model, to_file='model_layout.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='mean_squared_error', metrics=[r_squared])

    return model

