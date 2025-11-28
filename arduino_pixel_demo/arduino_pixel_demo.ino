#include <Adafruit_NeoPixel.h>

#define NUM_PIXELS 24
#define STRIP_PIN 6  // Digital pin 6 (you can change this to any available digital pin)

const float brightness = 0.3;  // Adjust brightness (0.0 to 1.0, where 1.0 is full brightness)
const uint8_t primary_color_r = 62;   // RGB color #3e91be
const uint8_t primary_color_g = 145;
const uint8_t primary_color_b = 190;
const unsigned long animation_duration = 10000;  // Duration in milliseconds (10 seconds)

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

// Generate rainbow colors with brightness applied.
uint32_t wheel_with_brightness(uint8_t pos, float brightness) {
    uint8_t r, g, b;
    if (pos < 85) {
        r = 255 - pos * 3;
        g = pos * 3;
        b = 0;
    } else if (pos < 170) {
        pos -= 85;
        r = 0;
        g = 255 - pos * 3;
        b = pos * 3;
    } else {
        pos -= 170;
        r = pos * 3;
        g = 0;
        b = 255 - pos * 3;
    }
    return apply_brightness(r, g, b, brightness);
}

// Scale RGB color by brightness factor.
uint32_t apply_brightness(uint8_t r, uint8_t g, uint8_t b, float brightness) {
    return strip.Color((uint8_t)(r * brightness), (uint8_t)(g * brightness), (uint8_t)(b * brightness));
}

// Rotating rainbow animation across all LEDs.
void animation_rainbow(unsigned long duration) {
    uint8_t offset = 0;
    unsigned long start_time = millis();
    
    while (millis() - start_time < duration) {
        for (int i = 0; i < NUM_PIXELS; i++) {
            uint8_t pos = (i * 256 / NUM_PIXELS + offset) % 256;
            strip.setPixelColor(i, wheel_with_brightness(pos, brightness));
        }
        strip.show();
        offset = (offset + 1) % 256;
        delay(10);
    }
}

// Circular chase animation - lights up LEDs one by one, then turns them off.
void animation_circle_chase(unsigned long duration) {
    uint32_t color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, brightness);
    uint32_t off_color = strip.Color(0, 0, 0);
    unsigned long start_time = millis();
    
    while (millis() - start_time < duration) {
        // Turn on LEDs one by one (keep previous LEDs on)
        strip.fill(off_color);
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (millis() - start_time >= duration) {
                break;
            }
            strip.setPixelColor(i, color);
            strip.show();
            delay(50);
        }
        
        // Turn off LEDs one by one (keep previous LEDs off)
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (millis() - start_time >= duration) {
                break;
            }
            strip.setPixelColor(i, off_color);
            strip.show();
            delay(50);
        }
    }
}

// Checkerboard pattern animation - alternates every second LED, then flips the pattern.
void animation_checkerboard(unsigned long duration) {
    uint32_t color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, brightness);
    uint32_t off_color = strip.Color(0, 0, 0);
    unsigned long start_time = millis();
    
    while (millis() - start_time < duration) {
        // Pattern 1: Even indices on, odd indices off
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (i % 2 == 0) {
                strip.setPixelColor(i, color);
            } else {
                strip.setPixelColor(i, off_color);
            }
        }
        strip.show();
        delay(500);
        
        if (millis() - start_time >= duration) {
            break;
        }
        
        // Pattern 2: Even indices off, odd indices on (flipped)
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (i % 2 == 0) {
                strip.setPixelColor(i, off_color);
            } else {
                strip.setPixelColor(i, color);
            }
        }
        strip.show();
        delay(500);
    }
}

// Trail animation - one bright LED moves around the ring with dimmer trailing LEDs.
void animation_trail(unsigned long duration) {
    uint32_t off_color = strip.Color(0, 0, 0);
    int trail_length = 3;  // Number of trailing LEDs
    unsigned long start_time = millis();
    
    while (millis() - start_time < duration) {
        for (int pos = 0; pos < NUM_PIXELS; pos++) {
            if (millis() - start_time >= duration) {
                break;
            }
            
            // Clear all LEDs
            strip.fill(off_color);
            
            // Main LED at full brightness
            uint32_t main_color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, brightness);
            strip.setPixelColor(pos, main_color);
            
            // Trailing LEDs with decreasing brightness
            for (int trail = 1; trail <= trail_length; trail++) {
                int trail_pos = (pos - trail + NUM_PIXELS) % NUM_PIXELS;
                // Decreasing brightness: 0.6, 0.3, 0.15 for 3 trails
                float trail_brightness = brightness * (0.6 / trail);
                uint32_t trail_color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, trail_brightness);
                strip.setPixelColor(trail_pos, trail_color);
            }
            
            strip.show();
            delay(50);
        }
    }
}

// Sparkle/Twinkle animation - random LEDs briefly flash on and off.
void animation_sparkle(unsigned long duration) {
    uint32_t off_color = strip.Color(0, 0, 0);
    uint32_t sparkle_color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, brightness);
    unsigned long start_time = millis();
    
    while (millis() - start_time < duration) {
        // Keep most LEDs off, randomly sparkle a few
        strip.fill(off_color);
        
        // Randomly select 2-4 LEDs to sparkle
        int num_sparkles = random(2, 5);  // random(2, 5) gives 2-4
        int sparkle_positions[4];
        int count = 0;
        
        while (count < num_sparkles) {
            int pos = random(0, NUM_PIXELS);
            bool duplicate = false;
            for (int i = 0; i < count; i++) {
                if (sparkle_positions[i] == pos) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                sparkle_positions[count] = pos;
                count++;
            }
        }
        
        for (int i = 0; i < num_sparkles; i++) {
            strip.setPixelColor(sparkle_positions[i], sparkle_color);
        }
        
        strip.show();
        delay(200);  // Brief flash
        
        // Turn off the sparkles
        for (int i = 0; i < num_sparkles; i++) {
            strip.setPixelColor(sparkle_positions[i], off_color);
        }
        
        strip.show();
        delay(150);  // Brief pause before next sparkle
    }
}

// Color Wipe animation - a color sweeps across all LEDs in one direction.
void animation_color_wipe(unsigned long duration) {
    uint32_t off_color = strip.Color(0, 0, 0);
    uint32_t color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, brightness);
    unsigned long start_time = millis();
    
    while (millis() - start_time < duration) {
        // Wipe forward
        strip.fill(off_color);
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (millis() - start_time >= duration) {
                break;
            }
            strip.setPixelColor(i, color);
            strip.show();
            delay(50);
        }
        
        if (millis() - start_time >= duration) {
            break;
        }
        
        // Wipe backward
        for (int i = NUM_PIXELS - 1; i >= 0; i--) {
            if (millis() - start_time >= duration) {
                break;
            }
            strip.setPixelColor(i, off_color);
            strip.show();
            delay(50);
        }
    }
}

// Larson Scanner animation - a few LEDs move back and forth (Knight Rider style).
void animation_larson_scanner(unsigned long duration) {
    uint32_t off_color = strip.Color(0, 0, 0);
    int scanner_width = 3;  // Number of LEDs in the scanner
    unsigned long start_time = millis();
    int direction = 1;
    int pos = 0;
    
    while (millis() - start_time < duration) {
        // Clear all LEDs
        strip.fill(off_color);
        
        // Draw scanner with fade effect
        for (int i = 0; i < scanner_width; i++) {
            int led_pos = (pos + i) % NUM_PIXELS;
            // Fade brightness from center to edges
            float fade = 1.0 - (i * 0.3);
            float scanner_brightness = brightness * max(0.2, fade);
            uint32_t scanner_color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, scanner_brightness);
            strip.setPixelColor(led_pos, scanner_color);
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

// Fade Fill animation - LEDs fade in one by one, then fade out in reverse.
void animation_fade_fill(unsigned long duration) {
    uint32_t off_color = strip.Color(0, 0, 0);
    unsigned long start_time = millis();
    int fade_steps = 20;  // Number of brightness steps for fade
    
    while (millis() - start_time < duration) {
        // Fade in one by one
        for (int i = 0; i < NUM_PIXELS; i++) {
            if (millis() - start_time >= duration) {
                break;
            }
            for (int step = 0; step < fade_steps; step++) {
                if (millis() - start_time >= duration) {
                    break;
                }
                float fade_brightness = brightness * (step / (float)fade_steps);
                uint32_t fade_color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, fade_brightness);
                strip.setPixelColor(i, fade_color);
                strip.show();
                delay(10);
            }
        }
        
        if (millis() - start_time >= duration) {
            break;
        }
        
        // Fade out in reverse
        for (int i = NUM_PIXELS - 1; i >= 0; i--) {
            if (millis() - start_time >= duration) {
                break;
            }
            for (int step = fade_steps - 1; step >= 0; step--) {
                if (millis() - start_time >= duration) {
                    break;
                }
                float fade_brightness = brightness * (step / (float)fade_steps);
                uint32_t fade_color = apply_brightness(primary_color_r, primary_color_g, primary_color_b, fade_brightness);
                strip.setPixelColor(i, fade_color);
                strip.show();
                delay(10);
            }
        }
    }
}

void setup() {
    strip.begin();
    strip.show(); // Initialize all pixels to 'off'
    randomSeed(analogRead(0));  // Seed random number generator
}

void loop() {
    // Array of function pointers for animations
    void (*animations[])(unsigned long) = {
        animation_sparkle,
        animation_trail,
        animation_circle_chase,
        animation_checkerboard,
        animation_rainbow,
    };
    
    int num_animations = sizeof(animations) / sizeof(animations[0]);
    
    // Randomly select an animation
    int selected_index = random(0, num_animations);
    animations[selected_index](animation_duration);
}
