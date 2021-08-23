import pandas as pd
import numpy as np
import sys
import os
import joblib
from time import time, ctime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import yaml
from tensorflow.keras.callbacks import *
import tensorflow as tf
#-------------------------------------------------------------------------------------------------------------------
from m_architecture_CNN_0_9_1 import *
from model_resources import *

"""Beschreibung: 
Hier ensteht das erste Convolutional Neural Net im Rahmen des AITM Projektes für Module: MdlPwr, MdlBatPwr, Construct, BatConstruct
Inputdaten: 809
Outputdaten: 1400

"""

# working directory: /home/ase/Dokumente/eh_basics/masterarbeit_eh
# Auführen Tensorboard im Terminal: TODO: tensorboard --logdir="/home/ase/Dokumente/eh_basics/masterarbeit_eh/05_neural_nets/CNN/Train/logs/"

sys.path.append(os.getcwd())
print(os.getcwd())

# Load yaml file
with open("C:/Users/Priyanka-Sanjeev.BHO/PycharmProjects/CNN/config_CNN_0_9_0.yml", "r") as stream:
    config = yaml.safe_load(stream)

# Setze Seed
tf.random.set_seed(config["seed"])

# Konfigurationen für das Speichern der logs & Modelle
model_name = config["model_name"] + ".h5"
log_name = config["model_name"]
model_cb_name = config["model_name"] + "_best_checkpoint" + ".h5"

model_path = path_for_model(model_name)   # evtl in die init
log_dir_model1 = path_for_logs(log_name)  # evtl in die init
score_path = path_for_scores(log_name)
act_time = ctime(time()).replace(":", "_")
act_time = act_time.replace(" ", "_")

''''if config["enable_gpu"]:
    physical_devices=tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)'''''

def main():
    # Instanz der Klasse Model
    model = Model()

    # Lade Trainings- und Testdatensatz
    model.load_data()

    # Das Neuronale Netz wird erstellt (mit Architektur) und Gewichte ggfs. geladen
    model_architecture = model.generate_model()

    # Das Neuronale Netz wird trainiert
    history = model.train(model_architecture)

    # Trainings- und Validierungsloss werden über Epochen geplottet
    # model.plot_history(history)

    # Das trainierte Netz wird mit Testdaten evaluiert
    model.eval(model_architecture)

    # Berechnete Werte (prediction) und ground_truth (target) werden zurücknormiert
    # model.rescaling_predicted_data()  # TODO: Hier werden alle Daten gespeichert für ANN_0_8_1_eval.py

    # Das Modell/Neuronale Netz wird gespeichert (nur die Gewichte oder gesammtes Modell)
    # model.save(model_architecture)


class Model():
    def __init__(self):
        self.r2_clipped = 0
        self.input_shape = config['input_shape']
        self.output_shape = config['output_shape']
        self.model_path = path_for_model(model_name)
        self.model_callback_path = path_for_model(model_cb_name)

        self.train_samples = []
        self.train_targets = []
        self.val_samples = []
        self.val_targets = []
        self.test_samples = []
        self.test_targets = []
        self.predictions = []

    def generate_model(self):
        """Build the model or build and load weights"""
        print(os.getcwd())
        if config['use_trained_model']:
            if os.path.isfile(config["loading_model_path"] + config["loading_model_name"] + config["loading_model_format"]):
                model_architecture = load_model()  # load complete model instead of just the weights

                header("Lade bereits trainiertes Modell!")
                print("Geladenes Modell: " + config["loading_model_name"])
            else:
                print("Trainiertes Modell: " + config["loading_model_name"] + " konnte nicht gefunden werden! \n Es wird ein neues Modell erstellt!")
                model_architecture = build_model_architecture()
        else:
            model_architecture = build_model_architecture()
        return model_architecture

    def load_data(self):
        # read csv data with pandas
        data_path = config["data_path"]
        self.train_samples = pd.read_csv(data_path + "train_samples.csv", delimiter=',')
        self.train_targets = pd.read_csv(data_path + "train_targets.csv", delimiter=',')
        self.val_samples = pd.read_csv(data_path + "val_samples.csv", delimiter=',')
        self.val_targets = pd.read_csv(data_path + "val_targets.csv", delimiter=',')
        self.test_samples = pd.read_csv(data_path + "test_samples.csv", delimiter=',')
        self.test_targets = pd.read_csv(data_path + "test_targets.csv", delimiter=',')  # TODO: Änderung kurzfristig
        # self.test_targets = pd.read_csv("ground_truth.csv", delimiter=',', nrows=50000, skiprows=40000)  # TODO: Änderung kurzfristig

        # Reshape data for 1D-CNN (3-Dim Input)
        self.train_samples = np.array(self.train_samples)
        self.train_samples = self.train_samples.reshape(self.train_samples.shape[0], self.train_samples.shape[1], 1)

        self.val_samples = np.array(self.val_samples)
        self.val_samples = self.val_samples.reshape(self.val_samples.shape[0], self.val_samples.shape[1], 1)

        self.test_samples = np.array(self.test_samples)
        self.test_samples = self.test_samples.reshape(self.test_samples.shape[0], self.test_samples.shape[1], 1)

        print("Train samples shape: ", self.train_samples.shape)
        print("Validation samples shape: ", self.val_samples.shape)
        print("Test samples shape: ", self.test_samples.shape)
        print("Train ground truth shape: ", self.train_targets.shape)
        print("Validation ground truth shape: ", self.val_targets.shape)
        print("Test ground truth shape: ", self.test_targets.shape)

    def train(self, model):
        # Activate Tensorboard for Visualisation
        tf.keras.backend.clear_session()

        tb = TensorBoard(log_dir_model1, histogram_freq=1, write_graph=True)
        print("\nUm TensorBoard aufzurufen, führe das Folgende aus:")
        print('tensorboard --logdir="C:/Users/Priyanka-Sanjeev.BHO/PycharmProjects/CNN/logs/"')
        if config["early_stopping"]:
            es = EarlyStopping(monitor=config["monitor"], mode=config["mode"],
                               verbose=1, patience=config["patience"])

        # lrs = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        if config["model_checkpoint"]:
            mc = ModelCheckpoint(self.model_callback_path, monitor=config["monitor"],
                                 save_best_only=config["save_best_only"], verbose=1)
        cb_list = [tb, mc, es]  # lrs

        # Train the model
        history = model.fit(x=self.train_samples, y=self.train_targets, validation_data=(self.val_samples, self.val_targets), batch_size=config["batch_size"], epochs=config["epochs"], shuffle=config["shuffle"], verbose=1, callbacks=cb_list)

        return history

    def eval(self, model):
        # Evaluate the ANN on Testdata
        score = model.evaluate(self.test_samples, self.test_targets, verbose=0)
        print("Ergebnis des Testdatensatzes: ")
        print("Score: ", score)

        # Speichern des scores (alle Metriken) als .csv-file
        # score = np.array(score).reshape(1, 5)
        # score_df = pd.DataFrame(data=score, columns=config["metrik_names"])
        # score_df.to_csv(score_path + "_scores.csv", encoding='utf-8', index=False)

        # Predict targets based on unknown data (test data)
        self.predictions = model.predict(self.test_samples)


    def rescaling_predicted_data(self):
        # Renormieren der predictions und des groundtruth bevor dem Speichern der Daten als .csv
        predictions_df = pd.DataFrame(self.predictions)
        ground_truth_df = pd.DataFrame(self.test_targets)

        # feature range of normalization
        X_feature_max = config["normalization"]["feature_range"]["max"]
        X_feature_min = config["normalization"]["feature_range"]["min"]

        # TODO: Verbessern als for schleife!
        # 1. CurBat
        print("CurBat")
        X_max = config["normalization"]["target"]["CurBat"]["Max"]
        X_min = config["normalization"]["target"]["CurBat"]["Min"]
        i = 0
        j = 200
        print(ground_truth_df.iloc[:, i:j].describe())
        predictions_df.iloc[:, i:j].where(predictions_df.iloc[:, i:j] < 0.001,
                                          normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                     predictions_df.iloc[:, i:j]),
                                          inplace=True)
        ground_truth_df.iloc[:, i:j].where(ground_truth_df.iloc[:, i:j] < 0.001,
                                           normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                      ground_truth_df.iloc[:, i:j]),
                                           inplace=True)
        # predictions_df.iloc[:, i:j].loc[predictions_df.iloc[:, i:j] < 0] = 0
        print(ground_truth_df.iloc[:, i:j].describe())

        # 2. GrSt # TODO: Wird nicht renormiert, da nicht normiert
        print("GrSt")
        i = 400
        j = 600
        X_max = config["normalization"]["target"]["GrSt"]["Max"]
        X_min = config["normalization"]["target"]["GrSt"]["Min"]
        print(ground_truth_df.iloc[:, i:j].describe())
        # TODO: np.around(decimal=0) --> Ganze Zahlen
        predictions_df.iloc[:, i:j].where(predictions_df.iloc[:, i:j] < 0.001,
                                          np.rint(normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                             predictions_df.iloc[:, i:j])),
                                          inplace=True)
        ground_truth_df.iloc[:, i:j].where(ground_truth_df.iloc[:, i:j] < 0.001,
                                           np.rint(
                                               normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                          ground_truth_df.iloc[:, i:j])),
                                           inplace=True)
        # predictions_df.iloc[:, i:j] = np.rint(predictions_df.iloc[:, i:j])
        print(predictions_df.iloc[:, i:j].describe())

        # 3. NEng
        print("NEng")
        i = 600
        j = 800
        X_max = config["normalization"]["target"]["NEng"]["Max"]
        X_min = config["normalization"]["target"]["NEng"]["Min"]
        print(ground_truth_df.iloc[:, i:j].describe())
        predictions_df.iloc[:, i:j].where(predictions_df.iloc[:, i:j] < 0.001,
                                          normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                     predictions_df.iloc[:, i:j]),
                                          inplace=True)
        ground_truth_df.iloc[:, i:j].where(ground_truth_df.iloc[:, i:j] < 0.001,
                                           normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                      ground_truth_df.iloc[:, i:j]),
                                           inplace=True)

        print(ground_truth_df.iloc[:, i:j].describe())

        # 4. VVeh
        print("VVeh")
        i = 800
        j = 1000
        X_max = config["normalization"]["target"]["VVeh"]["Max"]
        X_min = config["normalization"]["target"]["VVeh"]["Min"]
        print(ground_truth_df.iloc[:, i:j].describe())
        predictions_df.iloc[:, i:j].where(predictions_df.iloc[:, i:j] < 0.001,
                                          normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                     predictions_df.iloc[:, i:j]),
                                          inplace=True)
        ground_truth_df.iloc[:, i:j].where(ground_truth_df.iloc[:, i:j] < 0.001,
                                           normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                      ground_truth_df.iloc[:, i:j]),
                                           inplace=True)
        print(ground_truth_df.iloc[:, i:j].describe())

        # 5. TqEng
        print("TqEng")
        i = 1000
        j = 1200
        X_max = config["normalization"]["target"]["TqEng"]["Max"]
        X_min = config["normalization"]["target"]["TqEng"]["Min"]
        print(ground_truth_df.iloc[:, i:j].describe())
        predictions_df.iloc[:, i:j].where(predictions_df.iloc[:, i:j] < 0.001,
                                          normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                     predictions_df.iloc[:, i:j]),
                                          inplace=True)
        ground_truth_df.iloc[:, i:j].where(ground_truth_df.iloc[:, i:j] < 0.001,
                                           normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                      ground_truth_df.iloc[:, i:j]),
                                           inplace=True)
        print(ground_truth_df.iloc[:, i:j].describe())

        # 6. TAry
        print("TAry")
        i = 1200
        j = 1400
        X_max = config["normalization"]["target"]["TAry"]["Max"]
        X_min = config["normalization"]["target"]["TAry"]["Min"]
        print(ground_truth_df.iloc[:, i:j].describe())
        predictions_df.iloc[:, i:j].where(predictions_df.iloc[:, i:j] < 0.001,
                                          normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                     predictions_df.iloc[:, i:j]),
                                          inplace=True)
        ground_truth_df.iloc[:, i:j].where(ground_truth_df.iloc[:, i:j] < 0.001,
                                           normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min,
                                                                      ground_truth_df.iloc[:, i:j]),
                                           inplace=True)
        print(ground_truth_df.iloc[:, i:j].describe())

        # Wechsele in neuen Ordner und speichere die Berechnungen als .csv ab
        os.chdir('C:/Users/Priyanka-Sanjeev.BHO/PycharmProjects/CNN/predictions/')
        predictions = os.path.abspath(act_time)
        if not os.path.exists(predictions):
            os.mkdir(predictions)
        os.chdir(predictions)

        ground_truth_df.to_csv("b_" + log_name + "_ground_truth.csv", header=True, index=False)
        predictions_df.to_csv("a_" + log_name + "_predictions.csv", header=True, index=False)

    def plot_history(self, history):
        """ Plots for R2 & Loss for training/validation set as function of the epochs
        :param history:
        :return:
        """
        # fig, axs = plt.subplot(111)
        plt.subplot(211)
        plt.plot(history.history['r_squared'], label="train_r_squared")
        plt.plot(history.history['val_r_squared'], label="val_r_squared")
        plt.ylabel('R_squared')
        plt.legend(loc='best')
        plt.title('R_squared eval')

        plt.subplot(212)
        plt.plot(history.history['loss'], label="train_loss")
        plt.plot(history.history['val_loss'], label="val_loss")
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.title('Loss eval')

        plt.show()

    def save(self, model):
        if config["save_weights"]:
            model.save_weights(filepath=self.model_path)
        else:
            model.save(filepath=self.model_path)

if __name__ == "__main__":
    main()





"""
CNN_14_04_21_5e_0.001_bs256
Ergebnis des Testdatensatzes: 
Score:  [0.0002157841216529535, 0.99510723]

CNN_14_04_21_10e_0.001_bs256_t1
Ergebnis des Testdatensatzes: 
Score:  [8.698198816785042e-05, 0.99802464]

10_et2
Ergebnis des Testdatensatzes: 
Score:  [6.32946164215433e-05, 0.9985592]
10e_es_t3
Ergebnis des Testdatensatzes: 
Score:  [5.710157677706583e-05, 0.9987055]

10e_t4_lr0.0001
Ergebnis des Testdatensatzes: 
Score:  [4.1316051222393754e-05, 0.9990633]
"""