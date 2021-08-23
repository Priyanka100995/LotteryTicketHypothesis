import math
import os
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from kerassurgeon import identify, Surgeon
from tensorflow.keras.layers import Dense
from kerassurgeon.identify import get_apoz
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from kerassurgeon.operations import delete_channels
with open("C:/Users/Admin/PycharmProjects/Keras Surgeon/config_ANN_0_9_0.yml", "r") as stream:
    config = yaml.safe_load(stream)
tf_xla_enable_xla_devices = True
def to_onehot(a, n):
    b = np.zeros((a.size, n+1))
    b[np.arange(a.size), a] = 1
    return b

lr = config["learning_rate"]
# Download data if needed and import.
batch_size =config["batch_size"]
epochs = config["epochs"]
percent_pruning = 4
total_percent_pruning = 80
# read csv data with pandas
os.chdir(config["data_path"])
path = config["data_path"]
data_len = np.array(pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "train_samples.csv", delimiter=',', usecols=[0]))

data_len_val = np.array(pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "val_samples.csv", delimiter=',', usecols=[0]))

data_len_test = np.array(pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "test_samples.csv", delimiter=',', usecols=[0]))

nrows_train = np.rint(data_len.shape[0] * config["train"]["nrows"]).astype(np.int32)
skiprows_train = np.rint(data_len.shape[0] * config["train"]["skiprows"]).astype(np.int32)
nrows_val = np.rint(data_len_val.shape[0] * config["val"]["nrows"]).astype(np.int32)
skiprows_val = np.rint(data_len_val.shape[0] * config["val"]["skiprows"]).astype(np.int32)
nrows_test = np.rint(data_len_test.shape[0] * config["test"]["nrows"]).astype(np.int32)
skiprows_test = np.rint(data_len_test.shape[0] * config["test"]["skiprows"]).astype(np.int32)

train_samples = pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "train_samples.csv", delimiter=',', nrows=nrows_train, skiprows=skiprows_train)
train_targets = pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "train_targets.csv", delimiter=',', nrows=nrows_train, skiprows=skiprows_train)
val_samples = pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "val_samples.csv", delimiter=',', nrows=nrows_val, skiprows=skiprows_val)
val_targets = pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "val_targets.csv", delimiter=',', nrows=nrows_val, skiprows=skiprows_val)
test_samples = pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "test_samples.csv", delimiter=',', nrows=nrows_test, skiprows=skiprows_test)
test_targets = pd.read_csv("C:/Users/Admin/PycharmProjects/FC/Preprocessed_Gesamtdatensatz_Var2_skaliert_no_balancing_Splitted_data_shuffled_gear_zeros_droprows/" + "test_targets.csv", delimiter=',', nrows=nrows_test, skiprows=skiprows_test)

print("Train samples shape: ", train_samples.shape)
print("Val samples shape: ", val_samples.shape)
print("Train ground truth shape: ", train_targets.shape)
print("Val ground truth shape: ", val_targets.shape)

num_train_samples = train_samples.shape[0]

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr)

train_loss = tf.keras.metrics.MeanSquaredError(name = 'train_loss')
train_rsquared = tf.keras.metrics.MeanSquaredError(name = 'train_r_squared')

val_loss = tf.keras.metrics.MeanSquaredError(name = 'val_loss')
val_rsquared = tf.keras.metrics.MeanSquaredError(name = 'val_r_squared')



def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator)) # 0 schlecht, 1 gut
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


dependencies = {'r_squared': r_squared}
HeUniform = tf.keras.initializers.he_uniform()
model = tf.keras.models.load_model('C:/Users/Admin/PycharmProjects/Keras Surgeon/Unstructured Pruned/Magnitude_Based_Pruned_Model_74.1578047404731.h5',custom_objects={'HeUniform': HeUniform}, compile=False)
model.summary()
model.compile(optimizer=Adam(lr ),
                  loss='mean_squared_error', metrics=[r_squared])
model.fit(x=train_samples, y=train_targets, validation_data=(val_samples,
                           val_targets), batch_size=config["batch_size"], epochs=config["epochs"],
                            shuffle=config["shuffle"], verbose=1)

for layer in model.layers:
    layers = layer.name
    layer = layers
    print("Layers", layer)


def get_model_apoz(model, generator):
    # Get APoZ
    start = None
    end = None
    apoz = []

    for layer in model.layers:
            apoz.extend([(layer.name, i, value) for (i, value)
                         in enumerate(get_apoz(model, layer, generator))])

    layer_name, index, apoz_value = zip(*apoz)
    apoz_df = pd.DataFrame({'layer': layer_name, 'index': index,
                            'apoz': apoz_value})
    apoz_df = apoz_df.set_index('layer')
    print("APOZ", apoz_df)
    return apoz_df



get_model_apoz(model, val_samples)

masked_sum_params = 0

for layer in model.trainable_weights:
            # for layer in mask_model_stripped.trainable_weights:
            # print(tf.math.count_nonzero(layer, axis = None).numpy())
            masked_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

print("\nTotal number of trainable parameters in original model = {0}\n".format(masked_sum_params))

for layer in model.layers:

            if layer.output_shape[1] != 1400:
                layers = layer.name
                layer = model.get_layer(name=layers)
                apoz = identify.get_apoz(model, layer, val_samples)
                high_apoz_channels = identify.high_apoz(apoz, "both")
                model = delete_channels(model, layer, high_apoz_channels)


masked_sum_params = 0

for layer in model.trainable_weights:
            # for layer in mask_model_stripped.trainable_weights:
            # print(tf.math.count_nonzero(layer, axis = None).numpy())
            masked_sum_params += tf.math.count_nonzero(layer, axis=None).numpy()

print("\nTotal number of trainable parameters in pruned model = {0}\n".format(masked_sum_params))


model.compile(optimizer=Adam(learning_rate=lr),
              loss='mean_squared_error', metrics=[r_squared])
#loss = model.evaluate(test_samples, test_targets, verbose=0)
#print('Pruned model loss before training: ', loss)
model.save('C:/Users/Admin/PycharmProjects/Keras Surgeon/Structured Pruned/Structured Pruned & Compiled Dense Magnitude_Based_Pruned_Model_74.1578047404731.h5')


model.fit(x=train_samples, y=train_targets, validation_data=(val_samples,val_targets),
                  batch_size=config["batch_size"], epochs=config["epochs"],
                  shuffle=config["shuffle"], verbose=1)


    # Evaluate the final model performance
loss = model.evaluate(test_samples, test_targets, verbose=0)
print('Pruned and Trained model loss: ', loss)


