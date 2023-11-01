# Posture-Detection-TinyML

## Scope of Work
- Collection of IMU data (Labeled)
- Pre-Process IMU data to produce window of 200 to create contextual window for the time series data.
- Produce classification result (Online + Offline)

## Workflow

### Step 1: Capturing Data
Data capture is done using an arduino based mcu (BLE Sense Nano), Used onboard IMU sensor to capture at 50dps (snchronized) using IMU Frame buffer.

Classes Involved:
- Supine
- Prone
- Side
- Sitting
- Unknown

### Step 2: Data Pre-Processing
- Data is bifurcated in a sequence of 200 data points to create window of 4 seconds.
- Sometimes we might not get exact 200 data points, hence used bilinear interpolation to create a sequence of 200 data points.

files involved
```
1. data/raw
2. data/processed
```

### Step 3: Model Architecture
- Online Model (To be run on a laptop)
    - We are looking at a sequence modeling problem, hence GRU based models will perform very well in this regard. This model involves subgraphs deeper than a single layer, hence not suitable for tensorflow lite environment (arduino).
        ```python
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
        ```

- Offline Model (To be run on an embedded device)
    - For deploying in a arduino based environment we have to create a model with only single layer subgraph.
        ```python
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

        ```

### Step 4: Conversion of PyTorch/Tensorflow to Tensorflow Lite
- Training configuration
    - Optimizer: `RMSProp`
    - Loss Function: `CategoricalCrossEntropy`
![Losses](./assets/losses%20-%20relu%20activated.png?raw=true)

- Conversion of Tensorflow to Tflite format
    ```python
    # Save the full blown model in .h5 format 
    model.save_weights(Model_Name + '.h5')
    # Convert the model into tflite using builtin converter, Choose the supported ops for the conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    # Save the model.
    with open(Model_Name + '.tflite', 'wb') as f:
        f.write(tflite_model)
    ```
### Step 5: Arduino Deployment
The tflite model is converted into hex encoded format to be loaded by the arduino program as a hex array.

```bash
xxd -i posture_detection_model.tflite > posture_detection_model.cc
```

See the classify_sleep_posture folder for arduino implementation