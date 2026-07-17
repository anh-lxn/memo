#include <Wire.h>
#include "HX711-SOLDERED.h"

const unsigned long SERIAL_BAUDRATE = 230400;
const unsigned long SAMPLE_INTERVAL_MS = 0;
const unsigned long I2C_CLOCK_HZ = 50000;

const byte HX711_ADDRESS = 0x31;
const uint8_t HX711_GAIN = GAIN_64;

unsigned long lastSampleMs = 0;
HX711 hx711;

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);

  while (!Serial) {
    ; // Wait for native USB boards. On Uno/Nano this exits immediately.
  }

  delay(1000);
  hx711.begin(HX711_ADDRESS);
  hx711.setDeepSleep(false);
  delay(50);
  hx711.setGain(HX711_GAIN);
  delay(200);

  Serial.println("HX711 Qwiic serial bridge ready");
  Serial.println("Backend: SOLDERED HX711 Arduino Library");
  Serial.println("Format: HX711,<millis>,0x31=<raw>");
}

void loop() {
  unsigned long nowMs = millis();
  if (nowMs - lastSampleMs < SAMPLE_INTERVAL_MS) {
    return;
  }

  lastSampleMs = nowMs;
  long rawValue = normalizeHx711Raw(hx711.getRawReading());

  Serial.print("HX711,");
  Serial.print(nowMs);
  Serial.print(",0x");
  if (HX711_ADDRESS < 16) {
    Serial.print("0");
  }
  Serial.print(HX711_ADDRESS, HEX);
  Serial.print("=");
  Serial.println(rawValue);
}

long normalizeHx711Raw(long rawValue) {
  uint32_t raw24 = ((uint32_t)rawValue) & 0x00FFFFFF;
  if (raw24 & 0x00800000) {
    raw24 |= 0xFF000000;
  }
  return (long)((int32_t)raw24);
}
