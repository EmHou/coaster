#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>

// This demo is for OV2640_MINI_2MP_PLUS. Ensure that your memorysaver.h is set accordingly.
#if !(defined OV2640_MINI_2MP_PLUS)
#error Please select the hardware platform and camera module in memorysaver.h file
#endif

#define my_CS 7  // Custom name for chip select to avoid conflicts

// Define button pin
const int buttonPin = 2;
bool lastButtonState = HIGH;  // Because of pull-up, "unpressed" is HIGH

// Pins for LED indicators (74HC595 shift register)
const int vccPin = 3;
const int dataPin = 4;
const int outputEnablePin = 5;
const int clockPin = 6;

// beep pin
const int buzzerPositivePin = 9;


// Drink counts
int coffee_count = 0;
int water_count = 0;
int fruit_punch_count = 0;

static unsigned long lastCaptureTime = 0;

// Create an ArduCAM object for OV2640
ArduCAM myCAM(OV2640, my_CS);

// Serial input handling
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

void shift_bit(boolean bit) {
  digitalWrite(dataPin, bit ? HIGH : LOW);
  digitalWrite(clockPin, HIGH);
  delayMicroseconds(1);
  digitalWrite(clockPin, LOW);
}

void shift_one(int times = 1) {
  for (int i = 0; i < times; i++) shift_bit(true);
}

void shift_zero(int times = 1) {
  for (int i = 0; i < times; i++) shift_bit(false);
}

boolean indicate(int indicator_1, int indicator_2, int indicator_3) {

  // Sanity check inputs
  if (indicator_1 < 0 || indicator_1 > 4 ||
      indicator_2 < 0 || indicator_2 > 4 ||
      indicator_3 < 0 || indicator_3 > 4) {
    return false;
  }

  pinMode(vccPin, OUTPUT);
  pinMode(outputEnablePin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(clockPin, OUTPUT);

  // Reset flash memory of shift registers
  digitalWrite(vccPin, LOW);
  delay(1);
  digitalWrite(vccPin, HIGH);
  // Set outputEnable pin to LOW to enable output
  digitalWrite(outputEnablePin, LOW);

  // Reset shift registers
  shift_zero(16);
  shift_zero(16);

  // Workaround pt 1
  shift_zero(3 - indicator_1);
  shift_one(indicator_1);

  shift_zero(4 - indicator_2);
  shift_one(indicator_2);
  shift_one(4);  // those outputs are unused on the PCB
  shift_one(indicator_3);
  shift_zero(4 - indicator_3);

  // Workaround pt 2  
  shift_bit(indicator_1 == 4);
  return true;
}

void recvWithEndMarker() {
  static byte ndx = 0;
  char endMarker = '\n';
  char rc;

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
      receivedChars[ndx] = '\0'; // terminate the string
      ndx = 0;
      newData = true;
    }
  }
}




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

        if (drink == "coffee") coffee_count = count;
        else if (drink == "water") water_count = count;
        else if (drink == "fruit punch") fruit_punch_count = count;

        indicate(coffee_count, water_count, fruit_punch_count);
      }
    }
    newData = false;
  }
}

// this is to show that it is initialising the ceiling photo (empty.jpg)
void blinkAllLEDs() {
  for (int i = 0; i < 10; i++) {
    indicate(4, 4, 4);  // Turn all LEDs on
    delay(500);         // Wait 500ms
    indicate(0, 0, 0);  // Turn all LEDs off
    delay(500);         // Wait 500ms
  }
}


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
    delay(noteDurations[i] + 50); // Slight pause between notes
  }

  noTone(buzzerPin); // Turn off the buzzer
}

// any time the button is pressed
void chime() {
  const int buzzerPin = 9;
  int melody[] = {
    440, 294
  };
  int noteDurations[] = {
    200, 200
  };
  int notes = sizeof(melody) / sizeof(melody[0]);

  for (int i = 0; i < notes; i++) {
    tone(buzzerPin, melody[i], noteDurations[i]);
    delay(noteDurations[i] + 50); // Slight pause between notes
  }

  noTone(buzzerPin); // Turn off the buzzer
}

void setup() {
  uint8_t vid, pid, temp;

  Serial.begin(921600);
  while (!Serial);

  Wire.begin();
  SPI.begin();

  pinMode(my_CS, OUTPUT);
  digitalWrite(my_CS, HIGH);

  pinMode(buttonPin, INPUT_PULLUP);

  pinMode(vccPin, OUTPUT);
  pinMode(outputEnablePin, OUTPUT);
  pinMode(dataPin, OUTPUT);
  pinMode(clockPin, OUTPUT);

  pinMode(buzzerPositivePin, OUTPUT);
  digitalWrite(buzzerPositivePin, LOW);

  Serial.println(F("ArduCAM Auto Capture Demo"));

  myCAM.write_reg(0x07, 0x80);
  delay(100);
  myCAM.write_reg(0x07, 0x00);
  delay(100);

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

  myCAM.set_format(JPEG);
  myCAM.InitCAM();

  lockExposureAndWB(); 

  myCAM.OV2640_set_JPEG_size(OV2640_320x240);

  myCAM.wrSensorReg8_8(0xff, 0x01);
  myCAM.wrSensorReg8_8(0x24, 0x20);
  myCAM.wrSensorReg8_8(0x2a, 0x00);
  myCAM.wrSensorReg8_8(0x2b, 0x00);

  myCAM.wrSensorReg8_8(0xff, 0x00);
  myCAM.wrSensorReg8_8(0x7c, 0x00);
  myCAM.wrSensorReg8_8(0x7d, 0x08);

  myCAM.wrSensorReg8_8(0xff, 0x00);

  delay(1000);
  myCAM.clear_fifo_flag();
}

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

void loop() {
  recvWithEndMarker();
  parseSerialData();



  //---------------------------------------- button ---------------------------------------- // 
  int currentButtonState = digitalRead(buttonPin);
  unsigned long currentTime = millis();

  // Check if the button changed from not-pressed to pressed
  if (lastButtonState == HIGH && currentButtonState == LOW) {
    // Button was just pressed
    Serial.write("BUTTON_PRESSED");
    chime();
  }

  // Update the stored button state
  lastButtonState = currentButtonState;

  //---------------------------------------- simulate zoom invite ---------------------------------------- // 
  static bool tonePlayed = false;
  static unsigned long randomInterval = random(60000, 120001); // Random interval between 1 min (60000 ms) and 2 min (120000 ms)

  if (!tonePlayed && millis() - startTime >= randomInterval) {
    playToneSequence();
    tonePlayed = true;
  }


  //---------------------------------------- taking photo ---------------------------------------- // 
  if (currentTime - lastCaptureTime >= 10000) {  // Changed interval to 10 seconds
    lastCaptureTime = currentTime;

    myCAM.flush_fifo();
    myCAM.clear_fifo_flag();

    Serial.println(F("Starting picture capture..."));
    myCAM.start_capture();

    while (!myCAM.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) delay(1);
    Serial.println(F("Picture capture complete."));

    uint32_t length = myCAM.read_fifo_length();
    Serial.print(F("Image length: "));
    Serial.println(length, DEC);

    if ((length >= MAX_FIFO_SIZE) || (length == 0)) {
      Serial.println(F("Image size error."));
      myCAM.clear_fifo_flag();
      return;
    }

    Serial.println(F("Sending image data..."));
    myCAM.CS_LOW();
    myCAM.set_fifo_burst();
    for (uint32_t i = 0; i < length; i++) {
      uint8_t data = SPI.transfer(0x00);
      Serial.write(data);
    }
    myCAM.CS_HIGH();
    Serial.println(F("Image data sent."));

    myCAM.clear_fifo_flag();
  }
}
