import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# data
x = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
print(x)


# Define and Compile the Neural Network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training the Neural Network
model.fit(x, y, epochs=500)

# Predict
predictions = (model.predict( (x)))
print(predictions)

#plt.plot(x,y-1)
#plt.plot(x, predictions)
