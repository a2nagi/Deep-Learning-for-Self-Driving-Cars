import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

# Build model

# Using Sequential (Feed Forward NN)
model = tf.keras.models.Sequential()

#Input Layer - a flattened network.
model.add(tf.keras.layers.Flatten())

# 2 Hidden layers (with 128 nuerons in layer with Activation function - Rectified Linear)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Output Layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy']
             )

model.fit(x_train, y_train, epochs = 3)
