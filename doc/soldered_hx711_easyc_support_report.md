# Soldered HX711 easyC/Qwiic Support Report

## Setup

- Host: Raspberry Pi 5
- OS: Debian GNU/Linux 13 (trixie), aarch64
- I2C bus: `/dev/i2c-1`
- I2C config:
  - `dtparam=i2c_arm=on`
  - `dtparam=i2c_arm_baudrate=10000`
- Sensors: Soldered HX711 easyC/Qwiic boards
- Wiring: 3.3V, GND, SDA, SCL
- Arduino test board: Arduino Uno

## Observed Behavior

Some HX711 easyC/Qwiic boards are detected by the Raspberry Pi 5, while others are not detected even when connected individually.

When a problematic board is connected to the Pi I2C bus, `i2cdetect` becomes very slow and appears to scan cell by cell. If the Qwiic connector is unplugged during the scan, `i2cdetect` immediately continues normally.

Kernel log during problematic scans:

```text
i2c_designware 1f00074000.i2c: controller timed out
```

This looks like the board or its easyC controller may be holding the I2C bus busy or pulling SDA/SCL low.

## Raspberry Pi Direct I2C Tests

Command:

```bash
i2cdetect -y -r 1 0x30 0x37
```

Example result with one working board:

```text
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:
10:
20:
30: -- -- 32 -- -- -- -- --
40:
50:
60:
70:
```

The board was detected at `0x32`.

Direct read timing on the Pi for address `0x32`:

```text
Adresse 0x32 antwortet auf Bus 1.
raw=   8388607 read_ms=    5.0 dt_ms=-
raw=   8388607 read_ms=    4.9 dt_ms=5.0
raw=   8388607 read_ms=    4.8 dt_ms=4.9
```

After changing gain:

```text
Adresse 0x32 antwortet auf Bus 1.
raw=   6677714 read_ms=    5.1 dt_ms=-
raw=   6677714 read_ms=    4.8 dt_ms=5.0
```

So at least one board responds quickly on the Raspberry Pi: about `5 ms` per read.

## Arduino Uno Tests

Using the official Soldered HX711 Arduino library with the easyC constructor:

```cpp
#include <Wire.h>
#include "HX711-SOLDERED.h"

HX711 hx711;

void setup() {
  Serial.begin(115200);
  Wire.begin();
  hx711.begin(0x31);
  hx711.setDeepSleep(false);
  hx711.setGain(GAIN_64);
}

void loop() {
  long reading = hx711.getRawReading();
  Serial.println(reading);
}
```

Measured timing with one sensor:

```text
SOLDERED LIB TEST 10 ms
frames 18 valid 18 errs 0
all dt mean=410.5ms min=409ms max=412ms
valid dt mean=410.5ms min=409ms max=412ms
```

Measured timing with 8 sensors:

```text
dt_ms approximately 2000 ms per frame
```

This is much slower than the direct Raspberry Pi I2C test with a board that is detected.

## Questions

1. Is the HX711 easyC/Qwiic board fully compatible with Raspberry Pi 5 I2C?
2. Can the onboard ATTINY404 easyC controller hold SDA or SCL low during startup, address detection, or readout?
3. Are there known issues with Raspberry Pi 5 / Linux `i2c_designware` and this board?
4. Is there a firmware update for the onboard ATTINY404 easyC controller?
5. How can we verify or reflash the ATTINY404 firmware?
6. Are the SW1/SW2/SW3 address switches read only at boot, or dynamically?
7. What pullup configuration is recommended when using multiple HX711 easyC boards on one Qwiic bus?
8. What is the expected maximum readout rate over easyC/I2C?
9. Why would Arduino Uno detect/read boards that the Raspberry Pi 5 cannot detect individually?

## Current Hypothesis

The HX711 ADC itself is probably not the root problem. The issue appears to be related to the easyC/Qwiic I2C bridge, address handling, bus state, or compatibility with Raspberry Pi 5 I2C.

The direct Raspberry Pi read speed can be fast when a board is detected, but several boards are not reliably detected and can block the Pi I2C bus.
