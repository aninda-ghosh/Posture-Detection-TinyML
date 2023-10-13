#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "iirfilter.h"
#include "model.h"

IIRFILTER butterworth_ax;
IIRFILTER butterworth_ay;
IIRFILTER butterworth_az;
IIRFILTER butterworth_gx;
IIRFILTER butterworth_gy;
IIRFILTER butterworth_gz;
IIRFILTER butterworth_mx;
IIRFILTER butterworth_my;
IIRFILTER butterworth_mz;

float ax, ay, az, gx, gy, gz, mx, my, mz;
float filt_ax, filt_ay, filt_az, filt_gx, filt_gy, filt_gz, filt_mx, filt_my, filt_mz;
int window_start = 0;
int print_time = 0;
char sensor2use = 'a';

// Array to map gesture index to a name
const char* GESTURES[] = {
  "Supine", "Prone", "Side", "Sitting", "Unknown"
};


//==============================================================================
// Capture variables
//==============================================================================
#define NUM_SAMPLES 200
#define MOTION_THRESHOLD 0.0
#define CAPTURE_DELAY 10  // This is now in milliseconds
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

bool isPredicting = false;

// Num samples read from the IMU sensors
// "Full" by default to start in idle
int numSamplesRead = 0;

//==============================================================================
// TensorFlow variables
//==============================================================================

// Global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// Auto resolve all the TensorFlow Lite for MicroInterpreters ops, for reduced memory-footprint change this to only
// include the op's you need.
tflite::AllOpsResolver tflOpsResolver;

// Setup model
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TensorFlow Lite for MicroInterpreters, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 30 * 1024;
byte tensorArena[tensorArenaSize];


void setup() {
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(9600);
  while (!Serial)
    ;

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1)
      ;
  }

  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.print("Magnetic field sample rate = ");
  Serial.print(IMU.magneticFieldSampleRate());
  Serial.println(" Hz");

  // Get the TFL representation of the model byte array
  tflModel = tflite::GetModel(posture_detection_model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1)
      ;
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("\nSelect Sensor Menu");
  Serial.println("a -> accelerometer");
  Serial.println("g -> Gyroscope");
  Serial.println("m -> Magnetometer");
}

void loop() {
  if (!isPredicting) {
    if (Serial.available()) {
      sensor2use = Serial.read();
      Serial.print("\nSelecting Mode: ");
      switch (sensor2use) {
        case 'a':
          {
            Serial.println("Accelerometer");
            break;
          }
        case 'g':
          {
            Serial.println("Gyroscope");
            break;
          }
        case 'm':
          {
            Serial.println("Magnetometer");
            break;
          }
      }
      isPredicting = true;
      numSamplesRead = 0;  // Reset the window index
      Serial.print("Capturing Data: ");
    }
  }

  if (isPredicting) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      IMU.readMagneticField(mx, my, mz);

      filt_ax = butterworth_ax.filter(ax);
      filt_ay = butterworth_ay.filter(ay);
      filt_az = butterworth_az.filter(az);

      filt_gx = butterworth_gx.filter(gx);
      filt_gy = butterworth_gy.filter(gy);
      filt_gz = butterworth_gz.filter(gz);

      filt_mx = butterworth_mx.filter(mx);
      filt_my = butterworth_my.filter(my);
      filt_mz = butterworth_mz.filter(mz);

      Serial.print(".");

      switch (sensor2use) {
        case 'a':
          {
            tflInputTensor->data.f[numSamplesRead * 3 + 0] = filt_ax / 4.0;
            tflInputTensor->data.f[numSamplesRead * 3 + 1] = filt_ay / 4.0;
            tflInputTensor->data.f[numSamplesRead * 3 + 2] = filt_az / 4.0;
            break;
          }
        case 'g':
          {
            tflInputTensor->data.f[numSamplesRead * 3 + 0] = filt_gx / 2000.0;
            tflInputTensor->data.f[numSamplesRead * 3 + 1] = filt_gy / 2000.0;
            tflInputTensor->data.f[numSamplesRead * 3 + 2] = filt_gz / 2000.0;
            break;
          }
        case 'm':
          {
            tflInputTensor->data.f[numSamplesRead * 3 + 0] = filt_mx / 400.0;
            tflInputTensor->data.f[numSamplesRead * 3 + 1] = filt_my / 400.0;
            tflInputTensor->data.f[numSamplesRead * 3 + 2] = filt_mz / 400.0;
            break;
          }
      }

      numSamplesRead++;
    }

    // Do we have the samples we need?
    if (numSamplesRead == NUM_SAMPLES) {
      digitalWrite(LED_BUILTIN, HIGH);
      Serial.println("\nPrediction Probabilities");

      // Run inference
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Error: Invoke failed!");
        while (1)
          ;
        return;
      }

      // Loop through the output tensor values from the model
      int maxIndex = 0;
      float maxValue = 0;
      for (int i = 0; i < NUM_GESTURES; i++) {
        float _value = tflOutputTensor->data.f[i];
        if (_value > maxValue) {
          maxValue = _value;
          maxIndex = i;
        }
        Serial.print(GESTURES[i]);
        Serial.print("\t: ");
        Serial.println(tflOutputTensor->data.f[i], 6);
      }

      Serial.println("*******************************");
      Serial.print("Ultimate Posture: ");
      Serial.print(GESTURES[maxIndex]);
      Serial.println();
      Serial.println("*******************************");
      digitalWrite(LED_BUILTIN, LOW);
      numSamplesRead = 0;
      isPredicting = false;
    }
  }

  // Add delay to not double trigger
  delay(1);
}