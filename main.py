"""
main.py

This script trains a neural network to recognize handwritten digits using the MNIST dataset. 
It includes steps for data loading, preprocessing, augmentation, model creation, training, and evaluation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist

# ===================================
# Load and Preprocess Data
# ===================================
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Add a channel dimension (MNIST images are grayscale, so the channel dimension is 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# ===================================
# Data Augmentation
# ===================================
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# ===================================
# Model Creation
# ===================================
# Create the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ===================================
# Model Training
# ===================================
# Train the model on the augmented dataset
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))

# ===================================
# Model Evaluation
# ===================================
# Evaluate the model's performance
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy}')
print(f'Loss: {loss}')


