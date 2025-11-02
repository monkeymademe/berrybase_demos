# Arduino Uno R4 NeoPixel Rainbow Demo

This is the Arduino version of the Raspberry Pi Pico NeoPixel rainbow demo.

## Hardware Setup

- **Arduino Uno R4** (WiFi or Minima) or similar
- **NeoPixel strip** (WS2812 or compatible, 10 pixels)
- Connect NeoPixel data pin to **Digital Pin 6** (you can change `STRIP_PIN` if needed)
- Connect NeoPixel VCC to **5V** (Arduino Uno R4 has 5V output)
- Connect NeoPixel GND to **GND**

## Software Setup

1. Install the **Arduino IDE** (version 2.0 or later recommended)
2. Install the **Adafruit NeoPixel library**:
   - Go to **Sketch → Include Library → Manage Libraries**
   - Search for "Adafruit NeoPixel"
   - Install the library by Adafruit
3. Select your board:
   - Go to **Tools → Board → Arduino UNO R4 Boards → Arduino UNO R4 WiFi** (or **Arduino UNO R4 Minima**)
4. Upload the sketch:
   - Connect your Arduino Uno R4 via USB
   - Click the **Upload** button


