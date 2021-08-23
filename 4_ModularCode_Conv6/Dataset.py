"""Dataset : Create testing and Training data.
Consists of the Batchsize, number of epochs to be trained for(Exported from the yaml files) """

import tensorflow as tf
import numpy as np
import yaml

#Specify the Path where the yaml file is stored

# Load yaml file
with open("C:/Users/Admin/PycharmProjects/ModularCode_Conv6/Experiment_Conv6.yaml", "r") as stream:
    config = yaml.safe_load(stream)

tf.random.set_seed(config["seed"])

if config["enable_gpu"]:
    physical_devices= tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Data():
    # Hyperparamters
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    num_classes = config["num_classes"]

    # input image dimensions
    img_rows = config["img_rows"]
    img_cols = config["img_cols"]


    # Load CIFAR-10 dataset-
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    print("\n'input_shape' which will be used = {0}\n".format(input_shape))

    # Convert datasets to floating point types-
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize the training and testing datasets-
    X_train /= 255.0
    X_test /= 255.0

    # convert class vectors/target to binary class matrices or one-hot encoded values-
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    print("\nDimensions of training and testing sets are:")
    print("X_train.shape = {0}, y_train.shape = {1}".format(X_train.shape, y_train.shape))
    print("X_test.shape = {0}, y_test.shape = {1}".format(X_test.shape, y_test.shape))

    # Create training and testing datasets-
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=20000, reshuffle_each_iteration=True).batch(batch_size=batch_size,drop_remainder=False)

    test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)

    # Choose an optimizer and loss function for training-
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    lr = config["learning_rate"]
    optimizer = tf.keras.optimizers.Adam(lr)

    # Select metrics to measure the error & accuracy of model.
    # These metrics accumulate the values over epochs and then
    # print the overall result-
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # The model is first trained without any pruning for 'num_epochs' epochs-
    epochs = num_epochs
    num_train_samples = X_train.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
    print("'end_step parameter' for this dataset =  {0}".format(end_step))
