# GIF Player for RP2040 LCD 1.28"

This project displays a GIF animation on a Waveshare LCD 1.28" round display with RP2040 microcontroller.

## Features

- Memory-efficient: Only loads one frame at a time into the LCD buffer
- Uses RGB565 format for optimal performance
- Loops animation continuously
- Works with Waveshare LCD 1.28" round display

## Files Structure

- `main.py` - Main application that plays the GIF animation
- `frames/` - Directory containing RGB565 frame files (frame000.rgb565, frame001.rgb565, etc.)
- `hamster.gif` - Original GIF file
- `convert_gif.py` - Script to convert GIF to RGB565 frames (run on PC, not on RP2040)

## Setup Instructions

### 1. Convert GIF to Frames (On PC/Mac)

#### The scripts function

Takes the GIF path, output folder, and number of frames as arguments and converts the 240x240 gif into frames in RGB565 format (big-endian).

#### Useage

Run the conversion script on your PC (requires PIL/Pillow):
```bash
python3 convert_gif.py <input_gif> <output_folder> <num_frames>
```
#### More examples

```bash
Convert hamster2.gif to 21 frames in frames2_opt21/
python3 convert_gif.py hamster2.gif frames2_opt21 21

# Convert hamster.gif to 20 frames in frames_opt/
python3 convert_gif.py hamster.gif frames_opt 20

# Convert any GIF to specified number of frames
python3 convert_gif.py my_animation.gif my_frames 15
```

### 2. Upload Files to RP2040

Upload the following to your RP2040:
- `main.py`
- `frames/` directory with all `.rgb565` files

### 3. Run the Application

Once uploaded, the script will run automatically on boot (if `main.py` is set as the main script).

To manually run:
```python
import main
```

## How It Works

The `main.py` file:

1. Initializes the LCD with the proper driver configuration
2. Loads frames one at a time from the `frames/` directory
3. Each frame is loaded directly into the LCD's buffer (only one frame in memory at a time)
4. Displays each frame for a configurable delay
5. Loops the animation continuously

## Customization

In `main.py`, you can adjust:

- `delay=0.06` in `play_animation()` call - Change animation speed (lower = faster)
- `frames_dir='frames'` - Change frames directory if needed
- Backlight brightness in `lcd.set_bl_pwm(65535)` - Range: 0-65535

