"""Common utilities.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from keras.datasets import cifar10
from keras.utils import np_utils


def load_cifar10():
    """Load CIFAR-10 using Keras builtin function.
    Keras 2 note: dimension has been handled by system.
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    nb_classes = 10

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.
    X_test /= 255.

    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
              "horse", "ship", "truck"]

    return X_train, Y_train, X_test, Y_test, labels
