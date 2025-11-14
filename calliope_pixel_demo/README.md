# Calliope mini NeoPixel Demo

A collection of animated LED patterns for the Calliope mini microcontroller with NeoPixel (WS2812B) LED strips. Adapted from the Raspberry Pi Pico version.

## Overview

This demo features 5 different LED animations that randomly cycle on a NeoPixel strip:
- **Rainbow**: Rotating rainbow colors across all LEDs
- **Circle Chase**: LEDs light up one by one in sequence, then turn off
- **Checkerboard**: Alternating pattern that flips between even/odd LEDs
- **Trail**: A bright LED moves around the ring with trailing dimmer LEDs
- **Sparkle**: Random LEDs briefly flash on and off

## Hardware Requirements

- Calliope mini board
- NeoPixel (WS2812B) LED strip (24 LEDs recommended, but configurable)
- Jumper wires for connections
- Power supply (depending on strip length and LED count)

## Wiring Instructions

### NeoPixel Strip Connections

Connect your NeoPixel strip to the Calliope mini as follows:

| NeoPixel Strip | Calliope mini Pin | Description |
|----------------|-------------------|-------------|
| VCC (Red/+) | 3.3V or 5V* | Power supply |
| GND (Black/-) | GND | Ground |
| DIN (Data/Green) | P0, P1, or P2 | Data signal |

\* *Use 5V if your strip requires it, but ensure proper power supply capacity. For longer strips (24+ LEDs at full brightness), you may need an external power supply.*

### Pin Selection

The default pin is **Pin 0 (P0)**. You can change this in the code:
- Edit `strip_pin = Pin(0, Pin.OUT)` in `calliope_pixel_demo.py`
- Common options: `Pin(0)`, `Pin(1)`, `Pin(2)`, `Pin.P0`, `Pin.P1`, `Pin.P2`

## Software Setup

### Prerequisites

- Calliope mini flashed with MicroPython firmware
- Access to upload Python files to the board (via USB or wireless)

### Installation

1. Flash your Calliope mini with MicroPython firmware (if not already done)
2. Upload `calliope_pixel_demo.py` to your Calliope mini
3. Ensure the `neopixel` module is available in your MicroPython installation

### Running the Demo

1. Connect the hardware as described above
2. Power on the Calliope mini
3. Run the script:
   ```python
   exec(open('calliope_pixel_demo.py').read())
   ```
   Or use your IDE/editor to execute the file

The animations will start immediately and cycle randomly.

## Configuration

### Adjustable Parameters

Edit the variables at the top of `calliope_pixel_demo.py`:

```python
num_pixels = 24              # Number of LEDs in your strip
brightness = 0.3             # Brightness level (0.0 to 1.0)
primary_color = (62, 145, 190)  # RGB color for animations (0-255 each)
animation_duration = 10      # Duration in seconds for each animation
```

### Pin Configuration

**Option 1: Standard MicroPython (Default)**
```python
from machine import Pin
strip_pin = Pin(0, Pin.OUT)
my_strip = NeoPixel(strip_pin, num_pixels)
```

**Option 2: Using calliopemini Module**
If Option 1 doesn't work, uncomment and use:
```python
from calliopemini import *
import neopixel
my_strip = neopixel.NeoPixel(pin0, num_pixels)  # or pin1, pin2, etc.
```

### Method Differences

Some NeoPixel implementations use `.show()` instead of `.write()`. If you encounter issues, replace all `.write()` calls with `.show()` in the code.

## Available Animations

### 1. Rainbow (`animation_rainbow`)
Rotating rainbow colors smoothly sweep across all LEDs.

### 2. Circle Chase (`animation_circle_chase`)
LEDs light up one by one in sequence, then turn off one by one.

### 3. Checkerboard (`animation_checkerboard`)
Alternates every second LED, then flips the pattern every 0.5 seconds.

### 4. Trail (`animation_trail`)
A bright LED moves around the ring with 3 trailing LEDs that fade in brightness.

### 5. Sparkle (`animation_sparkle`)
Random LEDs (2-4 at a time) briefly flash on and off, creating a twinkling effect.

### Additional Animations (Available but not in default rotation)

The code includes additional animations that you can add to the rotation:
- `animation_color_wipe`: Color sweeps across LEDs in one direction, then reverses
- `animation_larson_scanner`: Knight Rider-style scanner effect (back and forth)
- `animation_fade_fill`: LEDs fade in one by one, then fade out in reverse

To add these to the rotation, edit the `animations` list in the main loop.

## Troubleshooting

### LEDs Not Lighting Up

1. **Check wiring**: Verify all connections (VCC, GND, DIN)
2. **Verify pin number**: Make sure the pin in code matches your wiring
3. **Check power**: Ensure adequate power supply for your LED count
4. **Try different pin**: Some pins may not work; try P1 or P2

### Import Errors

- **`neopixel` module not found**: Ensure your MicroPython firmware includes the NeoPixel library
- **`machine.Pin` error**: Try the `calliopemini` module alternative (see Configuration)

### Wrong Colors or Patterns

- **Method error**: If `.write()` doesn't work, try `.show()` instead
- **Pin timing issues**: Some pins may have timing constraints; try a different pin

### Performance Issues

- **Reduce brightness**: Lower the `brightness` value
- **Reduce LED count**: If `num_pixels` is too high, reduce it
- **Increase sleep delays**: Add longer delays in animation functions

## Power Considerations

- **3.3V vs 5V**: Check your NeoPixel strip specifications
- **Current draw**: Each LED can draw up to 60mA at full white brightness
- **External power**: For strips with 24+ LEDs or high brightness, use an external 5V power supply
- **Power injection**: For longer strips, inject power at both ends

## Customization

### Adding Your Own Animation

Create a new function following this pattern:

```python
def animation_your_name(duration):
    """Description of your animation."""
    color = apply_brightness(primary_color, brightness)
    off_color = (0, 0, 0)
    start_time = time()
    
    while time() - start_time < duration:
        # Your animation code here
        my_strip.write()
        sleep(0.05)
```

Then add it to the `animations` list in the main loop.

### Changing Animation Selection

To manually select animations or change the order, modify the `animations` list:

```python
animations = [
    animation_rainbow,
    animation_sparkle,
    # Add or remove as desired
]
```

## License

This code is adapted from the Pico NeoPixel demo for use with Calliope mini.

## Resources

- [Calliope mini Official Site](https://calliope.cc/)
- [Calliope mini NeoPixel Documentation](https://calliope.cc/en/calliope-mini/accessories/sensoren/neopixel)
- [NeoPixel Best Practices](https://learn.adafruit.com/adafruit-neopixel-uberguide)

## Credits

Adapted from the Raspberry Pi Pico NeoPixel demo for the Calliope mini platform.

