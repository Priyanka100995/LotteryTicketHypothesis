import tensorflow as tf
import os

def path_for_model(model_name):
    # Sichere Pfad # TODO Implementiere Tensorboard & model.h5 für Speicherung der Gewichte
    dir_path = os.path.abspath("C:/Users/Admin/PycharmProjects/FC")
    if not os.path.exists(dir_path):
        os.mkdir((dir_path))  # neuen Ordner erstellen
    model_path = os.path.join(dir_path, model_name)  # Ändern

    return model_path


def path_for_scores(log_name):
    dir_path = os.path.abspath("C:/Users/Admin/PycharmProjects/FC")
    if not os.path.exists(dir_path):
        os.mkdir((dir_path))
    score_path = os.path.join(dir_path, log_name)
    return score_path


def path_for_logs(log_name):
    # Für Tensorboard
    log_dir = os.path.abspath("C:/Users/Admin/PycharmProjects/FC")
    if not os.path.exists(log_dir):
        os.mkdir((log_dir))

    log_dir_model1 = os.path.join(log_dir, log_name)
    if not os.path.exists(log_dir_model1):
        os.mkdir((log_dir_model1))

    return log_dir_model1

def header(msg): # Formatvorlage für Printbefehle zur besseren Übersicht in der Console
    print('-' * 50)
    print(' [ ' + msg + ' ]')


def lr_scheduler(epoch, lr):
    lr1 = 0.001
    if epoch <= 10:
        return lr
    else:
        lr = lr1
        return lr

def normalization_func_inverse(X_feature_max, X_feature_min, X_max, X_min, X_norm):
    X = ((X_norm - X_feature_min) * (X_max - X_min)) / (X_feature_max - X_feature_min)
    return X