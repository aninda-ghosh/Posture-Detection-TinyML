import tensorflow as tf
from tensorflow import keras


class PostureDetectionModel(keras.Model):
    def __init__(self, features, sequence_per_feature, classes):
        super().__init__()
        self.hidden_dimension = 80

        self.sens_seq_model =   keras.Sequential([
                                    keras.layers.GRU(self.hidden_dimension, input_shape=(sequence_per_feature, features)),
                                    keras.layers.Dense(40, activation='relu'),
                                    keras.layers.Dropout(0.65),
                                    keras.layers.Dense(classes)
                                ])

    def call(self, x, training=False):
        x = self.sens_seq_model(x, training=training)
        x = tf.sigmoid(x)

        return x