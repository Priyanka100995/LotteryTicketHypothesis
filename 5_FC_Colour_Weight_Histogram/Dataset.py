"""Dataset : Create testing and Training data.
Consists of the Batchsize, number of epochs to be trained for(Exported from the yaml files) """

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Prune
import yaml

with open(r'C:\Users\Admin\PycharmProjects\Lenet_FC_Conv\Experiment.yaml') as file:
    doc = yaml.load(file, Loader=yaml.FullLoader)

    sort_file = yaml.dump(doc, sort_keys=True)

class Data():
    # Batch_size
    for key, val in doc[2].items():
        value = val
        print(key, value)
    batch_size = value
    # num_of_epochs
    for key, val in doc[1].items():
        value1 = val
        print(key, value1)
    num_epochs = value1

    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # Load MNIST dataset-
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

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

    # Reshape training and testing sets-
    X_train = X_train.reshape(X_train.shape[0], 784)
    X_test = X_test.reshape(X_test.shape[0], 784)
    print("\nDimensions of training and testing sets are:")
    print("X_train.shape = {0}, y_train.shape = {1}".format(X_train.shape, y_train.shape))
    print("X_test.shape = {0}, y_test.shape = {1}".format(X_test.shape, y_test.shape))

    # Create training and testing datasets-
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size = 20000, reshuffle_each_iteration = True).batch(batch_size = batch_size, drop_remainder = False)
    test_dataset = test_dataset.batch(batch_size=batch_size, drop_remainder=False)


    # Choose an optimizer and loss function for training-
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(lr = 0.0012)
    #optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


    # These metrics accumulate the values over epochs and then print the overall result-
    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'train_accuracy')

    test_loss = tf.keras.metrics.Mean(name = 'test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'test_accuracy')

    epochs = num_epochs
    num_train_samples = X_train.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
    print("'end_step parameter' for this dataset =  {0}".format(end_step))
