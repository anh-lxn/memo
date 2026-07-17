#include <Wire.h>

const unsigned long SERIAL_BAUDRATE = 115200;
const unsigned long SCAN_INTERVAL_MS = 5000;

unsigned long lastScanMs = 0;

void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  Wire.begin();
  Wire.setClock(10000);

  while (!Serial) {
    ; // Wait for native USB boards. On Uno/Nano this exits immediately.
  }

  delay(1000);
  Serial.println("Arduino I2C scanner ready");
  Serial.println("Scanning every 5 seconds. Send SCAN for an extra scan.");
  scanI2CBus();
  lastScanMs = millis();
}

void loop() {
  unsigned long nowMs = millis();
  if (nowMs - lastScanMs >= SCAN_INTERVAL_MS) {
    scanI2CBus();
    lastScanMs = nowMs;
  }

  if (!Serial.available()) {
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();
  command.toUpperCase();

  if (command == "SCAN") {
    scanI2CBus();
    lastScanMs = millis();
  } else if (command.length() > 0) {
    Serial.print("Unknown command: ");
    Serial.println(command);
    Serial.println("Use: SCAN");
  }
}

void scanI2CBus() {
  byte error;
  byte foundCount = 0;

  Serial.println("Scanning I2C bus...");

  for (byte address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("I2C device found at 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.println(address, HEX);
      foundCount++;
    } else if (error == 4) {
      Serial.print("Unknown I2C error at 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.println(address, HEX);
    }

    delay(2);
  }

  if (foundCount == 0) {
    Serial.println("No I2C devices found");
  } else {
    Serial.print("Found ");
    Serial.print(foundCount);
    Serial.println(" I2C device(s)");
  }

  Serial.println("Scan done");
}
