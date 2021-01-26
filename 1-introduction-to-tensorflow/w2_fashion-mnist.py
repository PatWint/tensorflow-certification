import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


display_image = 1

# get the fashion mnsit dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#print otptions
np.set_printoptions(linewidth=200)


#print()
#plt.imshow(training_images[display_image])
#plt.show()


print("example training_images" , training_images[display_image])
print("label of example image" , training_labels[display_image])

# Normalize values
# All values in the training image are  between 0 and 255.
# ML works better when numbers are 'normalized' (between 0 and 1 for example)

# training_images  = training_images / 255.0
# test_images = test_images / 255.0

##########################################################################################
# Remember
#
# Sequential: That defines a SEQUENCE of layers in the neural network
#
# Flatten: Remember images are a square:
# flatten just takes that square and turns it into a 1 dimensional set.
#
# Dense: Adds a layer of neurons
#
# Each layer of neurons need an activation function to tell them what to do.
# There's lots of options, but just use these for now.
#
#  Relu effectively means "If X>0 return X, else return 0" --
#  so what it does it it only passes values 0 or greater to the next layer in the
#  network.
#
#  Softmax takes a set of values, and effectively picks the biggest one
#########


# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# 1: Layer: first layer in your network should be the same shape as your data.
# 2: Layer:  dense layer
# 3: Layer: the number of neurons in the last layer should match the number of classes you are classifying

# build the model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train the model:
# by calling model.fit() it  fits the training data to the training labels
model.fit(training_images, training_labels, epochs=5)


# evaulate the model with the test data
# or in other words: test the model with unseen data (test data set)
#
print("Evaluate the Model:")
model.evaluate(test_images, test_labels)

# Result:
# As expected it probably would not do as well with unseen data as it did with data it was trained on