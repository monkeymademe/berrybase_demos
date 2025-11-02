#include <Adafruit_NeoPixel.h>

#define NUM_PIXELS 10
#define STRIP_PIN 6  // Digital pin 6 (you can change this to any available digital pin)

Adafruit_NeoPixel strip(NUM_PIXELS, STRIP_PIN, NEO_GRB + NEO_KHZ800);

// Generate rainbow colors across 0-255 positions.
uint32_t wheel(uint8_t pos) {
    if (pos < 85) {
        return strip.Color(255 - pos * 3, pos * 3, 0);
    } else if (pos < 170) {
        pos -= 85;
        return strip.Color(0, 255 - pos * 3, pos * 3);
    } else {
        pos -= 170;
        return strip.Color(pos * 3, 0, 255 - pos * 3);
    }
}

void setup() {
    strip.begin();
    strip.show(); // Initialize all pixels to 'off'
}

void loop() {
    static uint8_t offset = 0;
    
    for (int i = 0; i < NUM_PIXELS; i++) {
        strip.setPixelColor(i, wheel((i * 256 / NUM_PIXELS + offset) % 256));
    }
    
    strip.show();
    offset = (offset + 1) % 256;
    delay(10);
}

