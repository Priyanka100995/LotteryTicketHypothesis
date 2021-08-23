import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import yaml
from timeit import default_timer as timer
import pandas as pd
import numpy as np
from keras.utils import CustomObjectScope
from keras.initializers import he_uniform

with open("C:/Users/Priyanka-Sanjeev.BHO/PycharmProjects/Test/config_eval_compare_jetson (1).yml", "r") as stream:
    config = yaml.safe_load(stream)


# Metrik für Regression
def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))  # 0 schlecht, 1 gut
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


# main
# load data
num_samples = 1000
path = config["data_path"]
file_name = config["data_names"]["test_samples"]
loading_time = timer()
test_samples = np.array(pd.read_csv(path + file_name + ".csv", delimiter=',', nrows=num_samples))
print("Loading time: ", timer() - loading_time, " s")

# load model
model_path = config["loading_model_path"] + config["loading_model_name"] + config["loading_model_format"]
dependencies = {'r_squared': r_squared, 'HeUniform': he_uniform()}

# with CustomObjectScope({'HeUniform': he_uniform()}):
model = load_model(model_path, custom_objects=dependencies)

lr = config["learning_rate"]
model.compile(optimizer=Adam(learning_rate=lr),
              loss='mean_squared_error', metrics=[r_squared])
model.summary()

# Starte Evaluierung
start_timer = timer()
predictions = model.predict(test_samples)
inference_time = timer() - start_timer
print("Inference time: ", inference_time, " s")
print("Inference time pro sample: ", inference_time / num_samples, " ms")

# Score
file_name = config["data_names"]["test_ground_truth"]
test_targets = np.array(pd.read_csv(path + file_name + ".csv", delimiter=',', nrows=num_samples))
score = model.evaluate(test_samples, test_targets, verbose=0)
print("Ergebnis des Testdatensatzes: ")
print("Score: ", score)

# Speichern des scores (alle Metriken) als .csv-file
# score = np.array(score).reshape(1, 5)
# score_df = pd.DataFrame(data=score, columns=config["metrik_names"])
# score_df.to_csv(score_path + "_scores.csv", encoding='utf-8', index=False)
