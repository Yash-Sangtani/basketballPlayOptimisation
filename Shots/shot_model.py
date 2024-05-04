import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split


# Function to load shot_features1 and labels
def load_data():
    features = []

    # Load shot_features1 from all files in the directory

    with open('./data/labels/labels.txt', 'r') as f:
        labels = f.readline().strip().split(', ')

    for filename in os.listdir('data/shot_features1'):
        with open(os.path.join('data/shot_features1', filename), 'rb') as f:
            data = pickle.load(f)
            features.extend(data)

    return np.array(features), np.array(labels)


# Load shot_features1 and labels
features, labels = load_data()

# Split the data into training and testing sets based on the labels
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                            random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(test_features, test_labels)
print("Model Loss:", loss)
print("Model Accuracy:", accuracy)
