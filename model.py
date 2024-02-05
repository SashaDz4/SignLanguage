import numpy as np
import tensorflow as tf

from operator import itemgetter as ig
from utils import num_classes, classes, extract_feature


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
        # if x is image
        if len(x.shape) == 3:
            x = self.preprocess(x)
        return ig(*np.argmax(self.predict(x), axis=1).astype(int))([*classes.keys()])

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def preprocess(self, img):
        (wristX, wristY, wristZ,
         thumb_CmcX, thumb_CmcY, thumb_CmcZ,
         thumb_McpX, thumb_McpY, thumb_McpZ,
         thumb_IpX, thumb_IpY, thumb_IpZ,
         thumb_TipX, thumb_TipY, thumb_TipZ,
         index_McpX, index_McpY, index_McpZ,
         index_PipX, index_PipY, index_PipZ,
         index_DipX, index_DipY, index_DipZ,
         index_TipX, index_TipY, index_TipZ,
         middle_McpX, middle_McpY, middle_McpZ,
         middle_PipX, middle_PipY, middle_PipZ,
         middle_DipX, middle_DipY, middle_DipZ,
         middle_TipX, middle_TipY, middle_TipZ,
         ring_McpX, ring_McpY, ring_McpZ,
         ring_PipX, ring_PipY, ring_PipZ,
         ring_DipX, ring_DipY, ring_DipZ,
         ring_TipX, ring_TipY, ring_TipZ,
         pinky_McpX, pinky_McpY, pinky_McpZ,
         pinky_PipX, pinky_PipY, pinky_PipZ,
         pinky_DipX, pinky_DipY, pinky_DipZ,
         pinky_TipX, pinky_TipY, pinky_TipZ,
         output_IMG) = extract_feature(img)

        return np.array([[[wristX], [wristY], [wristZ],
                               [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                               [thumb_McpX], [thumb_McpY], [thumb_McpZ],
                               [thumb_IpX], [thumb_IpY], [thumb_IpZ],
                               [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                               [index_McpX], [index_McpY], [index_McpZ],
                               [index_PipX], [index_PipY], [index_PipZ],
                               [index_DipX], [index_DipY], [index_DipZ],
                               [index_TipX], [index_TipY], [index_TipZ],
                               [middle_McpX], [middle_McpY], [middle_McpZ],
                               [middle_PipX], [middle_PipY], [middle_PipZ],
                               [middle_DipX], [middle_DipY], [middle_DipZ],
                               [middle_TipX], [middle_TipY], [middle_TipZ],
                               [ring_McpX], [ring_McpY], [ring_McpZ],
                               [ring_PipX], [ring_PipY], [ring_PipZ],
                               [ring_DipX], [ring_DipY], [ring_DipZ],
                               [ring_TipX], [ring_TipY], [ring_TipZ],
                               [pinky_McpX], [pinky_McpY], [pinky_McpZ],
                               [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                               [pinky_DipX], [pinky_DipY], [pinky_DipZ],
                               [pinky_TipX], [pinky_TipY], [pinky_TipZ]]])

