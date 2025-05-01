#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>

// This demo is for OV2640_MINI_2MP_PLUS. Ensure that your memorysaver.h is set accordingly.
#if !(defined OV2640_MINI_2MP_PLUS)
#error Please select the hardware platform and camera module in memorysaver.h file
#endif

#define my_CS 7  // Custom name for chip select to avoid conflicts

// Button input pin and last known state (using internal pull-up, HIGH when unpressed)
const int buttonPin = 2;
bool lastButtonState = HIGH;

// Pins controlling the 74HC595 shift register for LED indicators
const int vccPin = 3;
const int dataPin = 4;
const int outputEnablePin = 5;
const int clockPin = 6;

// Buzzer output pin for tones
const int buzzerPositivePin = 9;

// Counters for each drink type (coffee, water, fruit punch)
int coffee_count = 0;
int water_count = 0;
int fruit_punch_count = 0;

// Timestamp of last photo capture for periodic triggering
static unsigned long lastCaptureTime = 0;

// Initialize ArduCAM object for OV2640 module using defined CS pin
ArduCAM myCAM(OV2640, my_CS);

// Buffer and flag for receiving serial commands (max 32 chars including terminator)
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

// ---------------------------------------------------------------------------
// Low-level shift register routines to send bits/bytes to the LED driver
// ---------------------------------------------------------------------------

// Shift a single bit into the shift register (drive data pin, toggle clock)
void shift_bit(boolean bit) {
  digitalWrite(dataPin, bit ? HIGH : LOW);
  digitalWrite(clockPin, HIGH);
  delayMicroseconds(1);
  digitalWrite(clockPin, LOW);
}

// Send one or more '1' bits in sequence
void shift_one(int times = 1) {
  for (int i = 0; i < times; i++) shift_bit(true);
}

// Send one or more '0' bits in sequence
void shift_zero(int times = 1) {
  for (int i = 0; i < times; i++) shift_bit(false);
}

// ---------------------------------------------------------------------------
// LED indicator update function: lights up three indicators
// based on provided counts (range 0–4 elements each).
// ---------------------------------------------------------------------------
boolean indicate(int indicator_1, int indicator_2, int indicator_3) {

  // Validate input ranges
  if (indicator_1 < 0 || indicator_1 > 4 ||
      indicator_2 < 0 || indicator_2 > 4 ||
      indicator_3 < 0 || indicator_3 > 4) {
    return false;
  }

  // Configure shift register control pins as outputs
  pinMode(vccPin, OUTPUT);
  pinMode(outputEnablePin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(clockPin, OUTPUT);

  // Pulse Vcc pin to reset shift register storage
  digitalWrite(vccPin, LOW);
  delay(1);
  digitalWrite(vccPin, HIGH);
  // Set outputEnable pin to LOW to enable output
  digitalWrite(outputEnablePin, LOW);

  // Reset shift registers
  shift_zero(16);
  shift_zero(16);

  // Load first indicator value
  shift_zero(3 - indicator_1);
  shift_one(indicator_1);

  // Load second indicator value, pad 4 unused bits, then third
  shift_zero(4 - indicator_2);
  shift_one(indicator_2);
  shift_one(4);  // those outputs are unused on the PCB
  shift_one(indicator_3);
  shift_zero(4 - indicator_3);

  // Final workaround bit for full-scale first indicator
  shift_bit(indicator_1 == 4);
  return true;
}

// ---------------------------------------------------------------------------
// Serial reception helper: read until newline, store in receivedChars
// ---------------------------------------------------------------------------
void recvWithEndMarker() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;

  // Continue reading while data is available and no full command yet
  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (rc != endMarker) {
      receivedChars[ndx] = rc;
      ndx++;
      if (ndx >= numChars) {
        ndx = numChars - 1;
      }
    }
    else {
      // End marker found: terminate string and flag newData
      receivedChars[ndx] = '\0';
      ndx = 0;
      newData = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Parse received serial command and take action:
//  - "TEST": flash all LEDs via blinkAllLEDs()
//  - "<drink>,<count>": update counters and call indicate()
// ---------------------------------------------------------------------------
void parseSerialData() {
  if (newData) {
    String input = String(receivedChars);
    input.trim();

    if (input == "TEST") {
      blinkAllLEDs();
    } else {
      int commaIndex = input.indexOf(',');
      if (commaIndex > 0) {
        String drink = input.substring(0, commaIndex);
        int count = input.substring(commaIndex + 1).toInt();

        // Update the corresponding drink count
        if (drink == "coffee") coffee_count = count;
        else if (drink == "water") water_count = count;
        else if (drink == "fruit punch") fruit_punch_count = count;

        // Refresh LED indicators
        indicate(coffee_count, water_count, fruit_punch_count);
      }
    }
    newData = false;
  }
}

// ---------------------------------------------------------------------------
// Blink all LEDs on and off repeatedly to indicate TEST or initialization.
// ---------------------------------------------------------------------------
void blinkAllLEDs() {
  for (int i = 0; i < 10; i++) {
    indicate(4, 4, 4);  // all indicators on
    delay(500);
    indicate(0, 0, 0);  // all indicators off
    delay(500);
  }
}

// ---------------------------------------------------------------------------
// Play a short melody sequence on the buzzer (e.g., Zoom notification tone)
// ---------------------------------------------------------------------------
void playToneSequence() {
  const int buzzerPin = 9;
  int melody[] = {
    294, 330, 392, 330, 494, 494, 440,    // D E G E B B A "Never gonna give you up"
    // 294, 330, 392, 330, 440, 440, 392,    // D E G E A A G "Never gonna let you down"
    // 294, 330, 392, 330, 392, 440, 370, 330, 294, 294, 294, 440, 392 // "Never gonna run around and desert you"
    //D    E    G   E    G    A    F#    E    D   D   D     A    G
  };
  int noteDurations[] = {
    100, 100, 100, 100, 300, 300, 600,
    //100, 100, 100, 100, 300, 300, 600,
    //100, 100, 100, 100, 400, 200, 300, 100, 200, 200, 200, 400, 400
  };
  int notes = sizeof(melody) / sizeof(melody[0]);

  for (int i = 0; i < notes; i++) {
    tone(buzzerPin, melody[i], noteDurations[i]);
    delay(noteDurations[i] + 50);
  }

  noTone(buzzerPin); // Turn off the buzzer
}

// ---------------------------------------------------------------------------
// Simple chime played when the physical button is pressed.
// ---------------------------------------------------------------------------
void chime() {
  const int buzzerPin = 9;
  int melody[] = { 440, 294 };
  int noteDurations[] = { 200, 200 };
  int notes = sizeof(melody) / sizeof(melody[0]);

  for (int i = 0; i < notes; i++) {
    tone(buzzerPin, melody[i], noteDurations[i]);
    delay(noteDurations[i] + 50);
  }
  noTone(buzzerPin);
}

// ---------------------------------------------------------------------------
// setup(): runs once at startup, configures hardware, initializes camera
// Much of the backbone code is referenced from: https://github.com/ArduCAM/Arduino/blob/master/ArduCAM/examples/mini/ArduCAM_Mini_2MP_OV2640_functions/ArduCAM_Mini_2MP_OV2640_functions.ino 
// ---------------------------------------------------------------------------
void setup() {
  uint8_t vid, pid, temp;

  // Initialize high-speed serial port for debug and image transfer
  Serial.begin(921600);
  while (!Serial);  // wait for serial connection

  // Initialize I2C and SPI buses
  Wire.begin();
  SPI.begin();

  // Configure chip-select and button pins
  pinMode(my_CS, OUTPUT);
  digitalWrite(my_CS, HIGH);
  pinMode(buttonPin, INPUT_PULLUP);

  // Configure shift-register and buzzer pins
  pinMode(vccPin, OUTPUT);
  pinMode(outputEnablePin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(clockPin, OUTPUT);
  pinMode(buzzerPositivePin, OUTPUT);
  digitalWrite(buzzerPositivePin, LOW);

  Serial.println(F("ArduCAM Auto Capture Demo"));

  // Camera module reset sequence
  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);

  // Verify SPI interface with camera
  while (1) {
    myCAM.write_reg(ARDUCHIP_TEST1, 0x55);
    temp = myCAM.read_reg(ARDUCHIP_TEST1);
    if (temp != 0x55) {
      Serial.println(F("SPI interface Error!"));
      delay(1000);
      continue;
    } else {
      Serial.println(F("SPI interface OK."));
      break;
    }
  }

  // Detect OV2640 camera ID
  while (1) {
    myCAM.wrSensorReg8_8(0xff, 0x01);
    myCAM.rdSensorReg8_8(OV2640_CHIPID_HIGH, &vid);
    myCAM.rdSensorReg8_8(OV2640_CHIPID_LOW, &pid);
    if ((vid != 0x26) && ((pid != 0x41) && (pid != 0x42))) {
      Serial.println(F("Can't find OV2640 module!"));
      delay(1000);
      continue;
    } else {
      Serial.println(F("OV2640 detected."));
      break;
    }
  }

  // Configure camera to JPEG at 320x240, lock exposure and white balance
  myCAM.set_format(JPEG);
  myCAM.InitCAM();
  lockExposureAndWB();
  myCAM.OV2640_set_JPEG_size(OV2640_320x240);

  // Apply additional camera sensor register tweaks for colour balancing
  // Reference GitHub: https://github.com/ArduCAM/Arduino/blob/master/ArduCAM/ArduCAM.cpp
  // around line 1040 for information about changing registers for various lighting conditions
  myCAM.wrSensorReg8_8(0xff, 0x01);
  myCAM.wrSensorReg8_8(0x24, 0x20);
  myCAM.wrSensorReg8_8(0x2a, 0x00);
  myCAM.wrSensorReg8_8(0x2b, 0x00);
  myCAM.wrSensorReg8_8(0xff, 0x00);
  myCAM.wrSensorReg8_8(0x7c, 0x00);
  myCAM.wrSensorReg8_8(0x7d, 0x08);
  myCAM.wrSensorReg8_8(0xff, 0x00);

  delay(1000);
  myCAM.clear_fifo_flag();  // ready for first capture
}

// ---------------------------------------------------------------------------
// lockExposureAndWB(): freeze current exposure and white-balance settings
// Reference GitHub: https://github.com/ArduCAM/Arduino/blob/master/ArduCAM/ArduCAM.cpp
// around line 1040 for information about changing registers for various lighting conditions
// ---------------------------------------------------------------------------
void lockExposureAndWB() {
  myCAM.wrSensorReg8_8(0xff, 0x01);

  uint8_t r13;
  myCAM.rdSensorReg8_8(0x13, &r13);           
  r13 &= ~0x07;
  myCAM.wrSensorReg8_8(0x13, r13);

  myCAM.wrSensorReg8_8(0x45, 0x00);
  myCAM.wrSensorReg8_8(0x10, 0x08);
  myCAM.wrSensorReg8_8(0x04, 0x00);
  myCAM.wrSensorReg8_8(0x01, 0x48);
  myCAM.wrSensorReg8_8(0x02, 0x40);
  myCAM.wrSensorReg8_8(0x03, 0x40);
  myCAM.wrSensorReg8_8(0x14, 0x38);
  myCAM.wrSensorReg8_8(0x15, 0x00);
  myCAM.wrSensorReg8_8(0xff, 0x00);
}

static unsigned long startTime = millis();

// ---------------------------------------------------------------------------
// loop(): main runtime loop handling serial commands, button press,
// zoom-simulated tone, and periodic photo capture & transmission.
// ---------------------------------------------------------------------------
void loop() {
  // Handle incoming serial commands
  recvWithEndMarker();
  parseSerialData();

  //---------------------------------------- Button handling ----------------------------------------
  int currentButtonState = digitalRead(buttonPin);
  unsigned long currentTime = millis();

  // Detect falling edge (unpressed to pressed)
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    Serial.write("BUTTON_PRESSED");
    chime();  // play confirmation tone
  }
  lastButtonState = currentButtonState;  // update state

  //---------------------------------------- Simulate Zoom invite tone ----------------------------------------
  static bool tonePlayed = false;
  static unsigned long randomInterval = random(60000, 120001); 
  if (!tonePlayed && millis() - startTime >= randomInterval) {
    playToneSequence();
    tonePlayed = true;
  }

  //---------------------------------------- Periodic photo capture ----------------------------------------
  if (currentTime - lastCaptureTime >= 10000) {  // every 10 seconds
    lastCaptureTime = currentTime;


    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();

    Serial.println(F("Starting picture capture..."));
    myCAM.start_capture();
    while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) delay(1);
    Serial.println(F("Picture capture complete."));

    // Read length and check for errors
    uint32_t length = myCAM.read_fifo_length();
    Serial.print(F("Image length: "));
    Serial.println(length, DEC);
    if ((length >= MAX_FIFO_SIZE) || (length == 0)) {
      Serial.println(F("Image size error."));
      myCAM.clear_fifo_flag();
      return;
    }

    // Stream image bytes over serial
    Serial.println(F("Sending image data..."));
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();
    for (uint32_t i = 0; i < length; i++) {
      uint8_t data = SPI.transfer(0x00);
      Serial.write(data);
    }
    myCAM.CS_HIGH();
    Serial.println(F("Image data sent."));

    myCAM.clear_fifo_flag();  // clear for next capture
  }
}
