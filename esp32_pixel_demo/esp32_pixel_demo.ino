#include <Adafruit_NeoPixel.h>

// Configuration
#define NUM_PIXELS 24
#define BRIGHTNESS 0.3  // Adjust brightness (0.0 to 1.0, where 1.0 is full brightness)
#define STRIP_PIN 2      // ESP32 GPIO pin (common pins: 2, 4, 5, 12, 13, 14, 15, 18, 19, 21, 22, 23)
#define ANIMATION_DURATION 10000  // Duration in milliseconds for each animation

// Primary color RGB (205, 60, 101) = #cd3c65
#define PRIMARY_R 205
#define PRIMARY_G 60
#define PRIMARY_B 101

// Create NeoPixel strip object
Adafruit_NeoPixel strip(NUM_PIXELS, STRIP_PIN, NEO_GRB + NEO_KHZ800);

// Forward declarations for animation functions
void animation_rainbow(unsigned long duration);
void animation_circle_chase(unsigned long duration);
void animation_checkerboard(unsigned long duration);
void animation_trail(unsigned long duration);
void animation_sparkle(unsigned long duration);
void animation_color_wipe(unsigned long duration);
void animation_larson_scanner(unsigned long duration);
void animation_fade_fill(unsigned long duration);

// Animation function pointers
typedef void (*AnimationFunc)(unsigned long duration);
AnimationFunc animations[] = {
  animation_sparkle,
  animation_trail,
  animation_circle_chase,
  animation_checkerboard,
  animation_rainbow
};
const int num_animations = sizeof(animations) / sizeof(animations[0]);

void setup() {
  strip.begin();
  strip.setBrightness(255);  // Set to max, we'll control brightness in code
  strip.show();  // Initialize all pixels to 'off'
  randomSeed(analogRead(0));  // Seed random number generator
}

void loop() {
  // Randomly select an animation
  int selected = random(0, num_animations);
  animations[selected](ANIMATION_DURATION);
}

// Generate rainbow colors across 0-255 positions
uint32_t wheel(byte pos) {
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

// Scale RGB color by brightness factor
uint32_t applyBrightness(uint8_t r, uint8_t g, uint8_t b, float brightness) {
  return strip.Color((uint8_t)(r * brightness), (uint8_t)(g * brightness), (uint8_t)(b * brightness));
}

uint32_t applyBrightnessColor(uint32_t color, float brightness) {
  uint8_t r = (uint8_t)((color >> 16) * brightness);
  uint8_t g = (uint8_t)((color >> 8) * brightness);
  uint8_t b = (uint8_t)(color * brightness);
  return strip.Color(r, g, b);
}

// Rotating rainbow animation across all LEDs
void animation_rainbow(unsigned long duration) {
  int offset = 0;
  unsigned long startTime = millis();
  
  while (millis() - startTime < duration) {
    for (int i = 0; i < NUM_PIXELS; i++) {
      uint32_t color = wheel((i * 256 / NUM_PIXELS + offset) % 256);
      // Properly extract RGB components with masking
      uint8_t r = ((color >> 16) & 0xFF);
      uint8_t g = ((color >> 8) & 0xFF);
      uint8_t b = (color & 0xFF);
      // Apply brightness
      r = (uint8_t)(r * BRIGHTNESS);
      g = (uint8_t)(g * BRIGHTNESS);
      b = (uint8_t)(b * BRIGHTNESS);
      strip.setPixelColor(i, strip.Color(r, g, b));
    }
    strip.show();
    offset = (offset + 1) % 256;
    delay(30);  // Increased delay for smoother, slower animation
  }
}

// Circular chase animation - lights up LEDs one by one, then turns them off
void animation_circle_chase(unsigned long duration) {
  uint32_t color = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, BRIGHTNESS);
  uint32_t offColor = strip.Color(0, 0, 0);
  unsigned long startTime = millis();
  
  while (millis() - startTime < duration) {
    // Turn on LEDs one by one
    strip.fill(offColor);
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (millis() - startTime >= duration) break;
      strip.setPixelColor(i, color);
      strip.show();
      delay(50);
    }
    
    // Turn off LEDs one by one
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (millis() - startTime >= duration) break;
      strip.setPixelColor(i, offColor);
      strip.show();
      delay(50);
    }
  }
}

// Checkerboard pattern animation - alternates every second LED, then flips the pattern
void animation_checkerboard(unsigned long duration) {
  uint32_t color = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, BRIGHTNESS);
  uint32_t offColor = strip.Color(0, 0, 0);
  unsigned long startTime = millis();
  
  while (millis() - startTime < duration) {
    // Pattern 1: Even indices on, odd indices off
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (i % 2 == 0) {
        strip.setPixelColor(i, color);
      } else {
        strip.setPixelColor(i, offColor);
      }
    }
    strip.show();
    delay(500);
    
    if (millis() - startTime >= duration) break;
    
    // Pattern 2: Even indices off, odd indices on (flipped)
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (i % 2 == 0) {
        strip.setPixelColor(i, offColor);
      } else {
        strip.setPixelColor(i, color);
      }
    }
    strip.show();
    delay(500);
  }
}

// Trail animation - one bright LED moves around the ring with dimmer trailing LEDs
void animation_trail(unsigned long duration) {
  uint32_t offColor = strip.Color(0, 0, 0);
  int trailLength = 3;  // Number of trailing LEDs
  unsigned long startTime = millis();
  
  while (millis() - startTime < duration) {
    for (int pos = 0; pos < NUM_PIXELS; pos++) {
      if (millis() - startTime >= duration) break;
      
      // Clear all LEDs
      strip.fill(offColor);
      
      // Main LED at full brightness
      uint32_t mainColor = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, BRIGHTNESS);
      strip.setPixelColor(pos, mainColor);
      
      // Trailing LEDs with decreasing brightness
      for (int trail = 1; trail <= trailLength; trail++) {
        int trailPos = (pos - trail + NUM_PIXELS) % NUM_PIXELS;
        // Decreasing brightness: 0.6, 0.3, 0.15 for 3 trails
        float trailBrightness = BRIGHTNESS * (0.6 / trail);
        uint32_t trailColor = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, trailBrightness);
        strip.setPixelColor(trailPos, trailColor);
      }
      
      strip.show();
      delay(50);
    }
  }
}

// Sparkle/Twinkle animation - random LEDs briefly flash on and off
void animation_sparkle(unsigned long duration) {
  uint32_t offColor = strip.Color(0, 0, 0);
  uint32_t sparkleColor = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, BRIGHTNESS);
  unsigned long startTime = millis();
  
  while (millis() - startTime < duration) {
    // Keep most LEDs off, randomly sparkle a few
    strip.fill(offColor);
    
    // Randomly select 2-4 LEDs to sparkle
    int numSparkles = random(2, 5);
    bool used[NUM_PIXELS] = {false};
    int sparklePositions[4];
    int count = 0;
    
    while (count < numSparkles) {
      int pos = random(0, NUM_PIXELS);
      if (!used[pos]) {
        used[pos] = true;
        sparklePositions[count] = pos;
        count++;
      }
    }
    
    for (int i = 0; i < numSparkles; i++) {
      strip.setPixelColor(sparklePositions[i], sparkleColor);
    }
    
    strip.show();
    delay(200);  // Brief flash
    
    // Turn off the sparkles
    for (int i = 0; i < numSparkles; i++) {
      strip.setPixelColor(sparklePositions[i], offColor);
    }
    
    strip.show();
    delay(150);  // Brief pause before next sparkle
  }
}

// Color Wipe animation - a color sweeps across all LEDs in one direction
void animation_color_wipe(unsigned long duration) {
  uint32_t offColor = strip.Color(0, 0, 0);
  uint32_t color = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, BRIGHTNESS);
  unsigned long startTime = millis();
  
  while (millis() - startTime < duration) {
    // Wipe forward
    strip.fill(offColor);
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (millis() - startTime >= duration) break;
      strip.setPixelColor(i, color);
      strip.show();
      delay(50);
    }
    
    if (millis() - startTime >= duration) break;
    
    // Wipe backward
    for (int i = NUM_PIXELS - 1; i >= 0; i--) {
      if (millis() - startTime >= duration) break;
      strip.setPixelColor(i, offColor);
      strip.show();
      delay(50);
    }
  }
}

// Larson Scanner animation - a few LEDs move back and forth (Knight Rider style)
void animation_larson_scanner(unsigned long duration) {
  uint32_t offColor = strip.Color(0, 0, 0);
  int scannerWidth = 3;  // Number of LEDs in the scanner
  unsigned long startTime = millis();
  int direction = 1;
  int pos = 0;
  
  while (millis() - startTime < duration) {
    // Clear all LEDs
    strip.fill(offColor);
    
    // Draw scanner with fade effect
    for (int i = 0; i < scannerWidth; i++) {
      int ledPos = (pos + i) % NUM_PIXELS;
      // Fade brightness from center to edges
      float fade = 1.0f - (i * 0.3f);
      float scannerBrightness = BRIGHTNESS * max(0.2f, fade);
      uint32_t scannerColor = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, scannerBrightness);
      strip.setPixelColor(ledPos, scannerColor);
    }
    
    strip.show();
    delay(80);
    
    // Move scanner
    pos += direction;
    
    // Reverse direction at ends
    if (pos >= NUM_PIXELS) {
      pos = NUM_PIXELS - 1;
      direction = -1;
    } else if (pos < 0) {
      pos = 0;
      direction = 1;
    }
  }
}

// Fade Fill animation - LEDs fade in one by one, then fade out in reverse
void animation_fade_fill(unsigned long duration) {
  uint32_t offColor = strip.Color(0, 0, 0);
  unsigned long startTime = millis();
  int fadeSteps = 20;  // Number of brightness steps for fade
  
  while (millis() - startTime < duration) {
    // Fade in one by one
    for (int i = 0; i < NUM_PIXELS; i++) {
      if (millis() - startTime >= duration) break;
      for (int step = 0; step < fadeSteps; step++) {
        if (millis() - startTime >= duration) break;
        float fadeBrightness = BRIGHTNESS * (step / (float)fadeSteps);
        uint32_t fadeColor = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, fadeBrightness);
        strip.setPixelColor(i, fadeColor);
        strip.show();
        delay(10);
      }
    }
    
    if (millis() - startTime >= duration) break;
    
    // Fade out in reverse
    for (int i = NUM_PIXELS - 1; i >= 0; i--) {
      if (millis() - startTime >= duration) break;
      for (int step = fadeSteps - 1; step >= 0; step--) {
        if (millis() - startTime >= duration) break;
        float fadeBrightness = BRIGHTNESS * (step / (float)fadeSteps);
        uint32_t fadeColor = applyBrightness(PRIMARY_R, PRIMARY_G, PRIMARY_B, fadeBrightness);
        strip.setPixelColor(i, fadeColor);
        strip.show();
        delay(10);
      }
    }
  }
}

