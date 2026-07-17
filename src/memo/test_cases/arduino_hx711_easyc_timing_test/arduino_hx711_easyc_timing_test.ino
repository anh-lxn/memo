#include <Wire.h>
#include "HX711-SOLDERED.h"

const unsigned long SERIAL_BAUDRATE = 115200;
const unsigned long I2C_CLOCK_HZ = 50000;
const uint8_t HX711_GAIN = GAIN_64;

const byte HX711_ADDRESSES[] = {0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37};
const byte HX711_COUNT = sizeof(HX711_ADDRESSES) / sizeof(HX711_ADDRESSES[0]);

HX711 hx711Sensors[HX711_COUNT];
unsigned long lastFrameStartMs = 0;

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);

  while (!Serial) {
    ; // Wait for native USB boards. On Uno/Nano this exits immediately.
  }

  delay(1000);
  for (byte i = 0; i < HX711_COUNT; i++) {
    hx711Sensors[i].begin(HX711_ADDRESSES[i]);
    hx711Sensors[i].setDeepSleep(false);
    delay(50);
    hx711Sensors[i].setGain(HX711_GAIN);
    delay(50);
  }

  delay(200);

  Serial.println("HX711 easyC timing test - 8 sensors");
  Serial.print("addresses=");
  for (byte i = 0; i < HX711_COUNT; i++) {
    if (i > 0) {
      Serial.print(",");
    }
    printHexAddress(HX711_ADDRESSES[i]);
  }
  Serial.print(" gain=");
  Serial.println(HX711_GAIN);
  Serial.println("Format: FRAME,dt_ms=<frame_delta>,total_read_ms=<duration>,0x30=<raw>;read_ms=<duration>,...");
}

void loop() {
  unsigned long frameStartMs = millis();

  Serial.print("FRAME,dt_ms=");
  if (lastFrameStartMs == 0) {
    Serial.print("-");
  } else {
    Serial.print(frameStartMs - lastFrameStartMs);
  }

  long rawValues[HX711_COUNT];
  unsigned long readDurationsMs[HX711_COUNT];

  for (byte i = 0; i < HX711_COUNT; i++) {
    unsigned long readStartMs = millis();
    rawValues[i] = normalizeHx711Raw(hx711Sensors[i].getRawReading());
    readDurationsMs[i] = millis() - readStartMs;
  }

  unsigned long frameEndMs = millis();
  Serial.print(",total_read_ms=");
  Serial.print(frameEndMs - frameStartMs);

  for (byte i = 0; i < HX711_COUNT; i++) {
    Serial.print(",");
    printHexAddress(HX711_ADDRESSES[i]);
    Serial.print("=");
    Serial.print(rawValues[i]);
    Serial.print(";read_ms=");
    Serial.print(readDurationsMs[i]);
  }
  Serial.println();

  lastFrameStartMs = frameStartMs;
}

long normalizeHx711Raw(long rawValue) {
  uint32_t raw24 = ((uint32_t)rawValue) & 0x00FFFFFF;
  if (raw24 & 0x00800000) {
    raw24 |= 0xFF000000;
  }
  return (long)((int32_t)raw24);
}

void printHexAddress(byte address) {
  Serial.print("0x");
  if (address < 16) {
    Serial.print("0");
  }
  Serial.print(address, HEX);
}
