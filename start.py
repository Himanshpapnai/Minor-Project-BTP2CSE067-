import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and normalize data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) # Completed line

# Train the model (standard addition to run the code)
model.fit(x_train, y_train, epochs=3)

# Evaluate the model (standard addition)
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")
