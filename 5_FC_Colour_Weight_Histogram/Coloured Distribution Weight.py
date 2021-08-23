import sys
import matplotlib
import matplotlib as plt
import tensorflow as tf
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import yaml

batch_size = 60
num_classes = 10
num_epochs = 20

# Data preprocessing and cleaning:
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

def lenet():
    """
    Function to define the architecture of a neural network model
    following Lenet Fully connected architecture for MNIST dataset and using
    provided parameter which are used to prune the model.


    300, 100, 10  -- fully connected layers


    Output: Returns designed and compiled neural network model
    """

    pruned_model = Sequential()

    pruned_model.add(tf.keras.layers.InputLayer(input_shape=(784,)))
    pruned_model.add(Flatten())

    pruned_model.add(
        Dense(
            units=300, activation='tanh',
            kernel_initializer=tf.initializers.GlorotNormal()
        )
    )

    pruned_model.add(
        Dense(
            units=100, activation='tanh',
            kernel_initializer=tf.initializers.GlorotNormal()
        )
    )

    pruned_model.add(
        Dense(
            units=10, activation='softmax'
        )
    )

    # Compile pruned CNN-
    pruned_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        # optimizer='adam',
        optimizer=tf.keras.optimizers.Adam(lr=0.0012),
        metrics=['accuracy']
    )

    return pruned_model


 # Add a pruning step callback to peg the pruning step to the optimizer's
    # step.
callback = [


        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3,
            min_delta=0.001
        )

    ]

# Initialize an FC model-
orig_model = lenet()

# Save random weights-
wts = orig_model.save_weights("Lenet_MNIST_Random_Gaussian_Glorot_Weights.h5", overwrite=True)

# Get CNN summary-
# orig_model_stripped.summary()
orig_model.summary()


def visualize_model_weights(sparse_model):
    """
    Visualize the weights of the layers of the sparse model.
    For weights with values of 0, they will be represented by the color white
    Args:
      sparse_model: a TF.Keras model
    """

    weights = sparse_model.get_weights()
    names = [weight.name for layer in sparse_model.layers for weight in layer.weights]
    my_cmap = matplotlib.cm.get_cmap('rainbow')

    # Iterate over all the weight matrices in the model and visualize them
    for i in range(len(weights)):
        weight_matrix = weights[i]

        layer_name = names[i]
        if weight_matrix.ndim == 1:  # If Bias or softmax
            weight_matrix = np.resize(weight_matrix,
                                      (1, weight_matrix.size))
            plt.imshow(np.abs(weight_matrix),
                       interpolation='none',
                       aspect="auto",
                       cmap=my_cmap);  # lower bound is set close to but not at 0
            plt.colorbar()
            plt.title(layer_name)
            plt.close()

        else:  # all other 2D matrices
            cmap = LinearSegmentedColormap.from_list('mycmap', ['blue', 'darkblue', 'white',  'darkgreen', 'green'])
            fig, ax = plt.subplots()
            im = ax.imshow(weight_matrix, cmap=cmap, interpolation='nearest')
            fig.colorbar(im)
            plt.xlabel("Output")
            plt.ylabel("Input")
            if i == 0:
                plt.title('FC Layer, 300')
            elif i == 2:
                plt.title('FC Layer, 100')
            elif i == 4:
                   plt.title('FC Layer, 10')
            plt.close()

            plt.show()

