# Import tensorflow & keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
# from keras.utils.vis_utils import plot_model
import os
import yaml

# Load yaml file
with open("C:/Users/Priyanka-Sanjeev.BHO/PycharmProjects/CNN/config_CNN_0_9_0.yml", "r") as stream:
    config = yaml.safe_load(stream)

# Metrik für Regression (R2-Score)
def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator)) # 0 schlecht, 1 gut
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


def build_model_architecture():
    #lr = model_architecture_params()
    # arch: 09_05_21
    lr = config["learning_rate"]
    input_shape = (config["input_shape"], 1)

    # Build sequential model
    model = Sequential()
    # Add Inputlayer
    model.add(Conv1D(96, kernel_size=15, activation=config["activation_function"], strides=6, padding='valid',
                     input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(3, padding='valid', strides=2))
    # Add Conv Layers
    model.add(Conv1D(96, kernel_size=8, strides=2, padding="same", activation=config["activation_function"]))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2, padding='valid', strides=2))

    # Add Flatten layer
    model.add(Flatten())

    # Add Dense-Layer
    model.add(Dense(1024, activation=config["activation_function"]))
    # model.add(Dropout(config["dropout_rate"]))

    # Add Outputlayer
    model.add(Dense(units=config['output_shape'], activation=config["output_activation"]))

    model.summary()
    print("Aktuelle Lernrate: ", lr)

    """Plot the architecture of the model"""
    # plot_model(model, to_file='cnn_model_layout.png', show_shapes=True, show_layer_names=True)

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

    # CNN 1. Modell --> AlexNet
    # # Build sequential model
    # model = Sequential()
    # # Add Inputlayer
    # model.add(Conv1D(config["filters"]["Conv0"], kernel_size=11, activation=config["activation_function"], strides=4, padding='valid',
    #                  input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(3, padding='valid', strides=2))
    # # Add Conv Layers
    # model.add(Conv1D(config["filters"]["Conv1"], kernel_size=5, strides=1, padding="same", activation=config["activation_function"]))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(3, padding='valid', strides=2))
    #
    # model.add(Conv1D(config["filters"]["Conv2"], kernel_size=3, strides=1, padding="same", activation=config["activation_function"]))
    # model.add(BatchNormalization())
    # model.add(Conv1D(config["filters"]["Conv3"], kernel_size=1, strides=1, padding="same", activation=config["activation_function"]))
    # model.add(BatchNormalization())
    # model.add(Conv1D(config["filters"]["Conv4"], kernel_size=1, strides=1, padding="same", activation=config["activation_function"]))
    # model.add(BatchNormalization())
    #
    # model.add(MaxPooling1D(3, strides=2))
    #
    # # Add Flatteninglayer
    # model.add(Flatten())
    #
    # # Add Dense-Layer
    # model.add(Dense(config["dense_neurons"][0], activation=config["activation_function"]))
    # model.add(Dropout(config["dropout_rate"]))
    # model.add(Dense(config["dense_neurons"][0], activation=config["activation_function"]))
    # model.add(Dropout(config["dropout_rate"]))
    #
    # # Add Outputlayer
    # model.add(Dense(units=config['output_shape'], activation=config["output_activation"]))