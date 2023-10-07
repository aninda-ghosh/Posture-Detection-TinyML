#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include "iirfilter.h"

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
constexpr int tensorArenaSize = 100 * 1024;
byte tensorArena[tensorArenaSize];


void setup() {
  Serial.begin(9600);
  while (!Serial)
    ;
  Serial.println("Started");

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
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  delay(3000);
}

void loop() {
  if(Serial.available()){
    sensor2use = Serial.read();
  }

  if (IMU.accelerationAvailable()) { 
    IMU.readAcceleration(ax, ay, az);
    filt_ax = butterworth_ax.filter(ax);
    filt_ay = butterworth_ay.filter(ay);
    filt_az = butterworth_az.filter(az); 
  }

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx, gy, gz);
    filt_gx = butterworth_gx.filter(gx);
    filt_gy = butterworth_gy.filter(gy);
    filt_gz = butterworth_gz.filter(gz); 
  }

  if (IMU.magneticFieldAvailable()) {
    IMU.readMagneticField(mx, my, mz);
    filt_mx = butterworth_mx.filter(mx);
    filt_my = butterworth_my.filter(my);
    filt_mz = butterworth_mz.filter(mz); 
  }

  switch (sensor2use) {
    case 'a' : {
      Serial.print(filt_ax);
      Serial.print(',');
      Serial.print(filt_ay);
      Serial.print(',');
      Serial.println(filt_az);
      break;
    }

    case 'g' : {
      Serial.print(filt_gx);
      Serial.print(',');
      Serial.print(filt_gy);
      Serial.print(',');
      Serial.println(filt_gz);
      break;
    }
    
    case 'm' : {
      Serial.print(filt_mx);
      Serial.print(',');
      Serial.print(filt_my);
      Serial.print(',');
      Serial.println(filt_mz);
      break;
    }
  }

  delay(10);
}