import numpy as np
import tensorflow as tf

# Example training dataset
X = np.array([
    [5.0, 4.0, 0],   # safe ahead
    [2.0, 1.5, 0],   # danger center
    [2.5, 2.0, -1],  # obstacle left
    [2.5, 2.0, 1],   # obstacle right
    [1.0, 0.5, 0],   # very close center
], dtype=np.float32)

y = np.array([0, 1, 3, 2, 4])  # labels (see mapping above)

# Build tiny NN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),    # dmin, TTC, obstacle_pos
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")  # 5 advice categories
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=100, verbose=0)

# Save standard TF model
model.save("guidance_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("guidance_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved TFLite model: guidance_model.tflite")
