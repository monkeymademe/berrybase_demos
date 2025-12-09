# Calliope mini NeoPixel Demo
# Adapted from Raspberry Pi Pico version

# Calliope mini specific imports
from calliopemini import *
import neopixel
import random

# Time functions - try different import methods for compatibility
# Note: calliopemini import * might already provide sleep and time
try:
    # Try standard time module
    from time import sleep
    try:
        from time import time
    except ImportError:
        # If time() doesn't exist, try ticks_ms() and convert
        try:
            from time import ticks_ms
            def time():
                return ticks_ms() / 1000.0
        except ImportError:
            # Fallback: create simple time counter using ticks
            import utime
            def time():
                return utime.ticks_ms() / 1000.0
except ImportError:
    try:
        # Try utime module (some MicroPython versions)
        import utime
        sleep = utime.sleep
        try:
            time = utime.time
        except AttributeError:
            # Use ticks_ms if time() not available
            def time():
                return utime.ticks_ms() / 1000.0
    except ImportError:
        # If both fail, sleep and time should be available from calliopemini import *
        # They should already be in the namespace from line 5
        pass

num_pixels = 24
brightness = 1  # Adjust brightness (0.0 to 1.0, where 1.0 is full brightness)
primary_color = (205, 60, 101)  # RGB color #cd3c65
animation_duration = 10  # Duration in seconds for each animation

# Calliope mini pin configuration
# Use pin0, pin1, pin2, etc. from calliopemini module
# Change pin0 to pin1, pin2, etc. based on your wiring
# Try to initialize with brightness parameter if supported
try:
    # Some NeoPixel implementations support brightness in constructor
    my_strip = neopixel.NeoPixel(pin0, num_pixels, brightness=int(brightness * 255))
except TypeError:
    # If brightness parameter not supported, initialize without it
    my_strip = neopixel.NeoPixel(pin0, num_pixels)
    # Try setting brightness attribute if it exists
    if hasattr(my_strip, 'brightness'):
        my_strip.brightness = brightness

# Compatibility helper for different NeoPixel implementations
# Some use .show(), others use .write()
def update_strip():
    """Update the NeoPixel strip - tries .show() first, then .write()."""
    if hasattr(my_strip, 'show'):
        my_strip.show()
    else:
        my_strip.write()

def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    else:
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)

def apply_brightness(color, brightness):
    """Scale RGB color by brightness factor."""
    # Ensure brightness is between 0.0 and 1.0
    brightness = max(0.0, min(1.0, float(brightness)))
    # Scale each color component and ensure it's an integer between 0-255
    return tuple(max(0, min(255, int(c * brightness))) for c in color)

def animation_rainbow(duration):
    """Rotating rainbow animation across all LEDs."""
    offset = 0
    start_time = time()
    while time() - start_time < duration:
        for i in range(num_pixels):
            color = wheel((i * 256 // num_pixels + offset) % 256)
            my_strip[i] = apply_brightness(color, brightness)
        update_strip()
        offset = (offset + 1) % 256
        sleep(0.01)

def animation_circle_chase(duration):
    """Circular chase animation - lights up LEDs one by one, then turns them off."""
    color = apply_brightness(primary_color, brightness)
    off_color = (0, 0, 0)
    start_time = time()
    
    while time() - start_time < duration:
        # Turn on LEDs one by one (keep previous LEDs on)
        my_strip.fill(off_color)
        for i in range(num_pixels):
            if time() - start_time >= duration:
                break
            my_strip[i] = color
            update_strip()
            sleep(0.05)
        
        # Turn off LEDs one by one (keep previous LEDs off)
        for i in range(num_pixels):
            if time() - start_time >= duration:
                break
            my_strip[i] = off_color
            update_strip()
            sleep(0.05)

def animation_checkerboard(duration):
    """Checkerboard pattern animation - alternates every second LED, then flips the pattern."""
    color = apply_brightness(primary_color, brightness)
    off_color = (0, 0, 0)
    start_time = time()
    
    while time() - start_time < duration:
        # Pattern 1: Even indices on, odd indices off
        for i in range(num_pixels):
            if i % 2 == 0:
                my_strip[i] = color
            else:
                my_strip[i] = off_color
        update_strip()
        sleep(0.5)
        
        if time() - start_time >= duration:
            break
        
        # Pattern 2: Even indices off, odd indices on (flipped)
        for i in range(num_pixels):
            if i % 2 == 0:
                my_strip[i] = off_color
            else:
                my_strip[i] = color
        update_strip()
        sleep(0.5)

def animation_trail(duration):
    """Trail animation - one bright LED moves around the ring with dimmer trailing LEDs."""
    off_color = (0, 0, 0)
    trail_length = 3  # Number of trailing LEDs
    start_time = time()
    
    while time() - start_time < duration:
        for pos in range(num_pixels):
            if time() - start_time >= duration:
                break
            
            # Clear all LEDs
            my_strip.fill(off_color)
            
            # Main LED at full brightness
            main_color = apply_brightness(primary_color, brightness)
            my_strip[pos] = main_color
            
            # Trailing LEDs with decreasing brightness
            for trail in range(1, trail_length + 1):
                trail_pos = (pos - trail) % num_pixels
                # Decreasing brightness: 0.6, 0.3, 0.15 for 3 trails
                trail_brightness = brightness * (0.6 / trail)
                trail_color = apply_brightness(primary_color, trail_brightness)
                my_strip[trail_pos] = trail_color
            
            update_strip()
            sleep(0.05)

def animation_sparkle(duration):
    """Sparkle/Twinkle animation - random LEDs briefly flash on and off."""
    off_color = (0, 0, 0)
    sparkle_color = apply_brightness(primary_color, brightness)
    start_time = time()
    
    while time() - start_time < duration:
        # Keep most LEDs off, randomly sparkle a few
        my_strip.fill(off_color)
        
        # Randomly select 2-4 LEDs to sparkle (MicroPython compatible)
        num_sparkles = random.randint(2, 4)
        sparkle_positions = []
        while len(sparkle_positions) < num_sparkles:
            pos = random.randint(0, num_pixels - 1)
            if pos not in sparkle_positions:
                sparkle_positions.append(pos)
        
        for pos in sparkle_positions:
            my_strip[pos] = sparkle_color
        
        update_strip()
        sleep(0.2)  # Brief flash
        
        # Turn off the sparkles
        for pos in sparkle_positions:
            my_strip[pos] = off_color
        
        update_strip()
        sleep(0.15)  # Brief pause before next sparkle

def animation_color_wipe(duration):
    """Color Wipe animation - a color sweeps across all LEDs in one direction."""
    off_color = (0, 0, 0)
    color = apply_brightness(primary_color, brightness)
    start_time = time()
    
    while time() - start_time < duration:
        # Wipe forward
        my_strip.fill(off_color)
        for i in range(num_pixels):
            if time() - start_time >= duration:
                break
            my_strip[i] = color
            update_strip()
            sleep(0.05)
        
        if time() - start_time >= duration:
            break
        
        # Wipe backward
        for i in range(num_pixels - 1, -1, -1):
            if time() - start_time >= duration:
                break
            my_strip[i] = off_color
            update_strip()
            sleep(0.05)

def animation_larson_scanner(duration):
    """Larson Scanner animation - a few LEDs move back and forth (Knight Rider style)."""
    off_color = (0, 0, 0)
    color = apply_brightness(primary_color, brightness)
    scanner_width = 3  # Number of LEDs in the scanner
    start_time = time()
    direction = 1
    pos = 0
    
    while time() - start_time < duration:
        # Clear all LEDs
        my_strip.fill(off_color)
        
        # Draw scanner with fade effect
        for i in range(scanner_width):
            led_pos = (pos + i) % num_pixels
            # Fade brightness from center to edges
            fade = 1.0 - (i * 0.3)
            scanner_brightness = brightness * max(0.2, fade)
            scanner_color = apply_brightness(primary_color, scanner_brightness)
            my_strip[led_pos] = scanner_color
        
        update_strip()
        sleep(0.08)
        
        # Move scanner
        pos += direction
        
        # Reverse direction at ends
        if pos >= num_pixels:
            pos = num_pixels - 1
            direction = -1
        elif pos < 0:
            pos = 0
            direction = 1

def animation_fade_fill(duration):
    """Fade Fill animation - LEDs fade in one by one, then fade out in reverse."""
    off_color = (0, 0, 0)
    start_time = time()
    fade_steps = 20  # Number of brightness steps for fade
    
    while time() - start_time < duration:
        # Fade in one by one
        for i in range(num_pixels):
            if time() - start_time >= duration:
                break
            for step in range(fade_steps):
                if time() - start_time >= duration:
                    break
                fade_brightness = brightness * (step / fade_steps)
                fade_color = apply_brightness(primary_color, fade_brightness)
                my_strip[i] = fade_color
                update_strip()
                sleep(0.01)
        
        if time() - start_time >= duration:
            break
        
        # Fade out in reverse
        for i in range(num_pixels - 1, -1, -1):
            if time() - start_time >= duration:
                break
            for step in range(fade_steps - 1, -1, -1):
                if time() - start_time >= duration:
                    break
                fade_brightness = brightness * (step / fade_steps)
                fade_color = apply_brightness(primary_color, fade_brightness)
                my_strip[i] = fade_color
                update_strip()
                sleep(0.01)

# Main loop - randomly selects animations
if __name__ == "__main__":
    animations = [
        animation_sparkle,
        animation_trail,
        animation_circle_chase,
        animation_checkerboard,
        animation_rainbow,
    ]
    
    while True:
        # Randomly select an animation
        selected_animation = animations[random.randint(0, len(animations) - 1)]
        selected_animation(animation_duration)

