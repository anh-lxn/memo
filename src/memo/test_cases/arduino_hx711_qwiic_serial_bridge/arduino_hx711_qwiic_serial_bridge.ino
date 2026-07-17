#include <Wire.h>

const unsigned long SERIAL_BAUDRATE = 115200;
const unsigned long SAMPLE_INTERVAL_MS = 100;
const unsigned long I2C_CLOCK_HZ = 50000;

const byte SET_GAIN_32 = 1;
const byte SET_GAIN_64 = 2;
const byte SET_GAIN_128 = 3;
const byte SET_SLEEP_OFF = 5;
const byte HX711_GAIN_COMMAND = SET_GAIN_64;

const byte HX711_ADDRESSES[] = {
  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37
};
const byte HX711_COUNT = sizeof(HX711_ADDRESSES) / sizeof(HX711_ADDRESSES[0]);

unsigned long lastSampleMs = 0;

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);

  while (!Serial) {
    ; // Wait for native USB boards. On Uno/Nano this exits immediately.
  }

  delay(1000);
  Serial.println("HX711 Qwiic serial bridge ready");
  Serial.println("Format: HX711,<millis>,0x30=<raw>,...,0x37=<raw>");

  configureHx711Modules();
}

void loop() {
  unsigned long nowMs = millis();
  if (nowMs - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSampleMs = nowMs;
  sendHx711Frame(nowMs);
}

void configureHx711Modules() {
  for (byte i = 0; i < HX711_COUNT; i++) {
    byte address = HX711_ADDRESSES[i];
    writeCommand(address, SET_SLEEP_OFF);
    delay(20);
    writeCommand(address, HX711_GAIN_COMMAND);
    delay(20);
  }
}

void sendHx711Frame(unsigned long timestampMs) {
  Serial.print("HX711,");
  Serial.print(timestampMs);

  for (byte i = 0; i < HX711_COUNT; i++) {
    byte address = HX711_ADDRESSES[i];
    int32_t rawValue = 0;
    byte error = readRaw(address, rawValue);

    Serial.print(",0x");
    if (address < 16) {
      Serial.print("0");
    }
    Serial.print(address, HEX);
    Serial.print("=");

    if (error == 0) {
      Serial.print(rawValue);
    } else {
      Serial.print("ERR");
      Serial.print(error);
    }
  }

  Serial.println();
}

void writeCommand(byte address, byte command) {
  Wire.beginTransmission(address);
  Wire.write(command);
  Wire.endTransmission();
}

byte readRaw(byte address, int32_t &rawValue) {
  byte received = Wire.requestFrom(address, (byte)4);
  if (received != 4) {
    while (Wire.available()) {
      Wire.read();
    }
    return received == 0 ? 1 : 2;
  }

  uint32_t b0 = Wire.read();
  uint32_t b1 = Wire.read();
  uint32_t b2 = Wire.read();
  uint32_t b3 = Wire.read();

  uint32_t packedValue = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
  rawValue = (int32_t)packedValue;
  return 0;
}
