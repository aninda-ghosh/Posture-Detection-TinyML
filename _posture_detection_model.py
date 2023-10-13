import tensorflow as tf
from tensorflow import keras


class PostureDetectionModel_GRU(keras.Model):
    def __init__(self, features, sequence_per_feature, classes):
        super().__init__()
        self.hidden_dimension = 80

        self.input0_0 = keras.Input(shape=(200, 3, ))

        self.gru1_0  = keras.layers.GRU(self.hidden_dimension, return_sequences=True)(self.input0_0[:, :, 0])
        self.gru1_1  = keras.layers.GRU(self.hidden_dimension, return_sequences=True)(self.input0_0[:, :, 1])
        self.gru1_2  = keras.layers.GRU(self.hidden_dimension, return_sequences=True)(self.input0_0[:, :, 2])

        self.linear2_0  = keras.layers.Dense(60, activation='tanh')(tf.concat([self.gru1_0, self.gru1_1, self.gru1_2], axis=-1))

        self.dropout3_0 = keras.layers.Dropout(0.30)(self.linear2_0)

        self.linear4_0  = keras.layers.Dense(classes)(self.dropout3_0)

        self.sigmoid5_0 = keras.layers.Activation('sigmoid')(self.linear4_0)

        self.model      = keras.models.Model(inputs=self.input0_0, outputs=self.sigmoid5_0)

    def call(self, x, training=False):
        # Set the training mode for the dropout layers based on the 'training' argument
        self.dropout3_0.training = training

        x = self.model(x, training=training)
        return x
    
class PostureDetectionModel_LSTM(keras.Model):
    def __init__(self, features, sequence_per_feature, classes):
        super().__init__()
        self.hidden_dimension = 80

        self.input0_0 = keras.Input(shape=(200, 3, ))

        self.lstm1_0  = keras.layers.LSTM(self.hidden_dimension, return_sequences=True)(self.input0_0[:, :, 0])
        self.lstm1_1  = keras.layers.LSTM(self.hidden_dimension, return_sequences=True)(self.input0_0[:, :, 1])
        self.lstm1_2  = keras.layers.LSTM(self.hidden_dimension, return_sequences=True)(self.input0_0[:, :, 2])

        self.linear2_0  = keras.layers.Dense(60, activation='tanh')(tf.concat([self.lstm1_0, self.lstm1_1, self.lstm1_2], axis=-1))

        self.dropout3_0 = keras.layers.Dropout(0.30)(self.linear2_0)

        self.linear4_0  = keras.layers.Dense(classes)(self.dropout3_0)

        self.sigmoid5_0 = keras.layers.Activation('sigmoid')(self.linear4_0)

        self.model      = keras.models.Model(inputs=self.input0_0, outputs=self.sigmoid5_0)

    def call(self, x, training=False):
        # Set the training mode for the dropout layers based on the 'training' argument
        self.dropout3_0.training = training

        x = self.model(x, training=training)
        return x
    

class PostureDetectionModel_Dense(keras.Model):
    def __init__(self, classes):
        super().__init__()
        self.hidden_dimension = 80
        
        self.input0_0 = keras.Input(shape=(200, 3, ))
        
        self.linear1_0  = keras.layers.Dense(40, activation='relu')(self.input0_0[:, :, 0])
        self.dropout1_0 = keras.layers.Dropout(0.20)(self.linear1_0)

        self.linear1_1  = keras.layers.Dense(40, activation='relu')(self.input0_0[:, :, 1])
        self.dropout1_1 = keras.layers.Dropout(0.20)(self.linear1_1)

        self.linear1_2  = keras.layers.Dense(40, activation='relu')(self.input0_0[:, :, 2])
        self.dropout1_2 = keras.layers.Dropout(0.20)(self.linear1_2)

        self.linear2_0  = keras.layers.Dense(60, activation='tanh')(tf.concat([self.dropout1_0, self.dropout1_1, self.dropout1_2], axis=-1))
        
        self.dropout3_0 = keras.layers.Dropout(0.30)(self.linear2_0)
        
        self.linear4_0  = keras.layers.Dense(classes)(self.dropout3_0)
        
        self.sigmoid5_0 = keras.layers.Activation('sigmoid')(self.linear4_0)
        
        self.model      = keras.models.Model(inputs=self.input0_0, outputs=self.sigmoid5_0)

    def call(self, x, training=False):
        # Set the training mode for the dropout layers based on the 'training' argument
        self.dropout1_0.training = training
        self.dropout1_1.training = training
        self.dropout1_2.training = training
        self.dropout3_0.training = training

        x = self.model(x, training=training)
        return x