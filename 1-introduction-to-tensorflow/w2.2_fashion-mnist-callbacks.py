import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


display_image = 1

# get the fashion mnsit dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#print options
np.set_printoptions(linewidth=200)


print("example training_images" , training_images[display_image])
print("label of example image" , training_labels[display_image])



# Callback Function:
# Stop the training if the desired accuracy is achieved.
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()



# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


# build the model
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


# train the model:
#model.fit(training_images, training_labels, epochs=5)
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])



# evaulate the model with the test data
print("Evaluate the Model:")
model.evaluate(test_images, test_labels)
