import keras
# MNIST database of handwritten digits (grayscale, 28x28=784 pixels)
from keras.datasets import mnist

# x - image, y - label
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert the images to 784-dimension vectors with pixel grayscale values
# shape[0] = size of dataset, shape[1],[2] - pixel values
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# convert labels to one-hot encoded vectors
# (e.g. 1 = [0,1,0,0,0,0,0,0,0,0])
num_digits = 10
y_train = keras.utils.to_categorical(y_train, num_digits)
y_test = keras.utils.to_categorical(y_test, num_digits)