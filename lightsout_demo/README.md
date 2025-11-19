# Lights Out Game for Pimoroni Pico RGB Keypad

A classic Lights Out puzzle game implementation for the Pimoroni Pico RGB Keypad (4x4 grid).

## Game Rules

- The goal is to turn off all the lights on the 4x4 grid
- Pressing a button toggles that button's light and the lights of its adjacent neighbors (up, down, left, right)
- Find the correct sequence of button presses to solve the puzzle!

## Hardware Requirements

- Raspberry Pi Pico
- Pimoroni Pico RGB Keypad Base

## Installation

1. Install Pimoroni's MicroPython firmware on your Raspberry Pi Pico
   - Download from [Pimoroni's releases](https://github.com/pimoroni/pimoroni-pico/releases)
   - Flash the firmware to your Pico

2. The `picokeypad` library is included with Pimoroni's MicroPython firmware

3. Copy `lightsout.py` to your Pico using Thonny or another MicroPython IDE

## Usage

Run the game:
```python
python lightsout.py
```

Or upload `lightsout.py` to your Pico and run it using Thonny or another MicroPython IDE.

## Features

- Random puzzle generation with adjustable difficulty
- Visual feedback when buttons are pressed
- Win animation when puzzle is solved
- Automatic new game after solving
- Debounced button inputs for reliable gameplay

## Controls

- Press any button on the 4x4 grid to toggle lights
- The game automatically starts a new puzzle after you solve one

## Customization

You can adjust the difficulty by changing the `difficulty` parameter in the `randomize_grid()` function call. Higher values create more complex puzzles.

You can also adjust the brightness by changing the value in `keypad.set_brightness(1.0)` (range 0.0 to 1.0).

## Color Scheme

- **Yellow/White**: Lights that are ON
- **Black**: Lights that are OFF
- **Blue**: Brief flash when a button is pressed
- **Green**: Win animation

Enjoy playing Lights Out!

