import keras
# MNIST database of handwritten digits (grayscale, 28x28=784 pixels)
from keras.datasets import mnist

# x - image, y - label
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert the images to 784-dimension vectors with pixel grayscale values
# shape[0] = size of dataset, shape[1],[2] - pixel values
input_dim = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], input_dim)
x_test = x_test.reshape(x_test.shape[0], input_dim)

# convert labels to one-hot encoded vectors
# (e.g. 1 = [0,1,0,0,0,0,0,0,0,0])
num_labels = 10
y_train = keras.utils.to_categorical(y_train, num_labels)
y_test = keras.utils.to_categorical(y_test, num_labels)

# create the network model
model = keras.models.Sequential()

# hidden layers
model.add(keras.layers.Dense(128, activation='sigmoid', input_dim=input_dim))
# output layer
model.add(keras.layers.Dense(num_labels, activation='softmax'))
model.summary()

# configure, train and test the model
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, verbose=1, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)