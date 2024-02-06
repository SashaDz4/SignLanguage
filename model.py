import numpy as np
import tensorflow as tf

from operator import itemgetter as ig
from utils import num_classes, classes, extract_feature, module


class Letter:
    def __init__(self, name, confidence):
        self.name = name
        self.confidence = confidence

    def __repr__(self):
        return f"{self.name} {self.confidence * 100:.0f}%"


class LetterDetector(tf.keras.Model):
    def __init__(self, shape):
        super(LetterDetector, self).__init__()
        self.model = self.build(shape)
        self.compile()

    def build(self, input_shape):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu",
                                   input_shape=input_shape),
            tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, padding="causal", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(rate=0.2),
            # Flatten the results to feed into a DNN
            tf.keras.layers.Flatten(),
            # 512 neuron hidden layer
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')])

    def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")

    def summary(self):
        self.model.summary()

    def predict_classes(self, x):
        if len(x.shape) == 3:
            x = self.preprocess(x)
        if not np.max(x):
            return Letter("", 0)
        prediction = self.predict(x)
        letter = ig(*np.argmax(prediction, axis=1).astype(int))([*classes.keys()])
        confidence = np.max(prediction)
        return Letter(letter, confidence)

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def preprocess(self, x):
        return np.array([extract_feature(x)[0]]).reshape(-1, 63, 1)
