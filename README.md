# digit_recognition

Handwritten digit recognition using a neural network - the *Hello World* of machine learning :)  

The project is made in Python 3.7 with Keras 2.3.1.  

## The data

This program uses the well-known [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/), which contains 60 000 grayscale 28x28 pixel images.  

![Picture source: Wikipedia](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)  

## Accuracy

The accuracy of the model's predictions is around 90% after 10 epochs of training.  

## Neural network architecture

The neural network consists of two fully-connected layers.  
The first one takes as input a 784-dimensional (28x28) vector of pixel values and uses the sigmoid activation function (e<sup>x</sup> / (e<sup>x</sup> + 1)).  

The output layer uses the [softmax](https://en.wikipedia.org/wiki/Softmax_function) activation function and returns a one-hot encoded vector (e.g. 1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0].  