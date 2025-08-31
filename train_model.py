# train_model.py
import numpy as np, csv, tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
features, labels = [], []
with open("datasets/labels.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        features.append([float(row["min_all"]),
                         float(row["min_left"]),
                         float(row["min_center"]),
                         float(row["min_right"])])
        labels.append(int(row["label"]))

X, Y = np.array(features), np.array(labels)

# Train model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, Y, epochs=50, batch_size=8, validation_split=0.2)

# Export TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("guidance_model.tflite", "wb").write(tflite_model)

print("✅ Trained model saved as guidance_model.tflite")
