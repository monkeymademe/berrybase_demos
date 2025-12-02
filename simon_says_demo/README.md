# Simon Says Game for Calliope mini v3

A classic Simon Says memory game implementation using the Calliope mini v3 and 4 RGB LED arcade buttons with NeoPixel LEDs.

## Overview

This project implements a Simon Says game where:
- The game displays a sequence of colored lights
- The player must repeat the sequence by pressing the corresponding buttons
- Each correct round adds one more color to the sequence
- The game continues until the player makes a mistake

## Hardware Requirements

- **Calliope mini v3** microcontroller board
- **4 RGB LED Arcade Buttons** with NeoPixel LEDs (one LED per button)
- **Jumper wires** for connections
- **Power supply** for Calliope mini v3

## Pinout Configuration

### NeoPixel LED Connections (Daisy-Chained)

The 4 NeoPixel LEDs are daisy-chained together, requiring only **one data pin** on the Calliope mini:

| Connection | Pin/Value | Description |
|------------|-----------|-------------|
| Data In (First LED) | P0 | Connect to Calliope mini P0 |
| Data Out (LED 1) → Data In (LED 2) | Chain | Connect LED 1's Data Out to LED 2's Data In |
| Data Out (LED 2) → Data In (LED 3) | Chain | Connect LED 2's Data Out to LED 3's Data In |
| Data Out (LED 3) → Data In (LED 4) | Chain | Connect LED 3's Data Out to LED 4's Data In |
| VCC (All LEDs) | 3.3V | Connect all NeoPixel VCC terminals to 3.3V |
| GND (All LEDs) | GND | Connect all NeoPixel GND terminals to GND |

**LED Order in Chain:**
- **LED 0** (Red button) - First in chain, connects to P0
- **LED 1** (Yellow button) - Second in chain
- **LED 2** (Blue button) - Third in chain
- **LED 3** (Green button) - Last in chain

**Important Notes:**
- All NeoPixels share the same **3.3V** power rail
- All NeoPixels share the same **GND** (ground) rail
- Only **one data pin** (P0) is needed for all 4 LEDs
- The LEDs must be connected in sequence: Data Out of one LED connects to Data In of the next
- The order matters! LED 0 = Red, LED 1 = Yellow, LED 2 = Blue, LED 3 = Green

### Button Input Connections

Each arcade button has 2 terminals for the switch:

| Button Color | Button Input Pin | Common Terminal |
|-------------|------------------|-----------------|
| Red         | C8               | GND             |
| Yellow      | C9               | GND             |
| Blue        | C13              | GND             |
| Green       | C14              | GND             |

**Important Notes:**
- Buttons should be configured as **active LOW** (pressed = LOW, released = HIGH)
- One terminal of each button connects to its respective input pin
- The other terminal of each button connects to **GND**
- The code uses internal pull-up resistors, so buttons read HIGH when not pressed

### Complete Pinout Summary

```
Calliope mini v3 Pinout:

Power:
- 3.3V  → All NeoPixel VCC terminals
- GND   → All NeoPixel GND terminals + All button common terminals

NeoPixel Data Pin (Daisy-Chained):
- P0    → First NeoPixel Data In (Red button LED)
  └─→ LED 0 Data Out → LED 1 Data In (Yellow button LED)
      └─→ LED 1 Data Out → LED 2 Data In (Blue button LED)
          └─→ LED 2 Data Out → LED 3 Data In (Green button LED)

Button Input Pins:
- C8    → Red button input
- C9    → Yellow button input
- C13   → Blue button input
- C14   → Green button input
```

## Wiring Diagram

```
                    Calliope mini v3
                    ┌──────────────┐
                    │              │
     Red Button ────┤ C8           │
     Yellow Btn ────┤ C9           │
     Blue Button ───┤ C13          │
     Green Btn ────┤ C14          │
                    │              │
     NeoPixel Chain ┤ P0           │
     (Data In)      │              │
                    │              │
     All VCC ───────┤ 3.3V         │
     All GND ───────┤ GND          │
                    └──────────────┘

NeoPixel Daisy Chain:
P0 → [LED 0: Red] → [LED 1: Yellow] → [LED 2: Blue] → [LED 3: Green]
     (Data In)     (Data Out→In)      (Data Out→In)    (Data Out→In)
```

## Software Setup

### Option 1: MakeCode (Recommended for Beginners)

**Prerequisites:**

1. **MakeCode for Calliope mini**
   - Visit: https://makecode.calliope.cc/
   - Or use the MakeCode desktop app

2. **NeoPixel Extension**
   - In MakeCode, click the gear icon (⚙️)
   - Select "Extensions"
   - Search for "neopixel" and add the extension

**Installation:**

1. Open the `simon_says.ts` file in MakeCode for Calliope mini
2. The code will automatically convert to blocks if needed
3. Connect your Calliope mini v3 via USB
4. Click "Download" to flash the program to your board

### Option 2: MicroPython

**Prerequisites:**

1. **MicroPython on Calliope mini**
   - Your Calliope mini should already have MicroPython installed
   - If not, follow the official Calliope mini MicroPython installation guide

2. **Python Editor/IDE**
   - Thonny IDE (recommended): https://thonny.org/
   - Or any editor that supports MicroPython

**Installation:**

1. Connect your Calliope mini v3 via USB
2. Open Thonny IDE and select "MicroPython (Calliope mini)" as the interpreter
3. Open the `simon_says.py` file
4. Click "Run" to execute, or save as `main.py` to run automatically on boot

**Note:** If you get pin number errors, you may need to adjust the pin numbers in `simon_says.py` to match your Calliope mini's pin mapping. Check the Calliope mini pinout documentation for your specific version.

## How It Works

### Game Flow

1. **Initialization**
   - On startup or when Button A is pressed, the game resets
   - All LEDs are cleared
   - "READY" message is displayed

2. **Game Start**
   - Press any button to start the game
   - The game generates the first color in the sequence
   - The sequence is played back (LEDs light up in order)

3. **Player Turn**
   - The player must press buttons in the same order as the sequence
   - Each button press lights up the corresponding LED
   - The game checks if the sequence is correct after each press

4. **Level Progression**
   - If the sequence is correct, a new color is added
   - The new sequence is played back
   - The game continues to the next level

5. **Game Over**
   - If the player makes a mistake, all LEDs flash red
   - The current level is displayed
   - Press Button A to start a new game

### Code Structure

- **State Variables**: Track game state, sequences, and level
- **LED Control**: Functions to light up and clear NeoPixel LEDs
- **Sequence Management**: Generate and play back color sequences
- **Input Handling**: Detect button presses and validate sequences
- **Game Logic**: Check correctness and manage game flow

### Key Functions

- `initGame()`: Resets the game to initial state
- `lightUpButton()`: Lights up a specific button's LED
- `playSequence()`: Plays back the current sequence
- `handleButtonPress()`: Processes player button input
- `checkSequence()`: Validates player input against game sequence
- `gameOver()`: Handles game over sequence and reset

## Usage

1. **Start a New Game**
   - Press **Button A** on the Calliope mini to reset and start
   - Or press any arcade button to start after initialization

2. **Play**
   - Watch the sequence of colored lights
   - Press the buttons in the same order
   - Continue until you make a mistake

3. **Restart**
   - After game over, press **Button A** to start a new game

## Troubleshooting

### LEDs Not Lighting Up

- **Check power connections**: Ensure all NeoPixel VCC pins are connected to 3.3V
- **Check ground connections**: Ensure all GND connections are secure
- **Check data pin**: Verify the first NeoPixel's Data In is connected to P0
- **Check daisy-chain connections**: Verify Data Out of each LED connects to Data In of the next LED
- **Check NeoPixel orientation**: Ensure data flows in the correct direction (Data In → Data Out)
- **Check chain order**: Verify LEDs are connected in order: Red (0) → Yellow (1) → Blue (2) → Green (3)
- **Test individual LEDs**: If only some LEDs work, check the chain connections between them

### Buttons Not Responding

- **Check button connections**: Verify buttons are connected to correct pins (C8, C9, C13, C14)
- **Check button type**: Ensure buttons are configured as active LOW (connect one terminal to GND)
- **Test continuity**: Use a multimeter to verify button switches work correctly
- **Check pull-up resistors**: The code uses internal pull-ups; buttons should read HIGH when not pressed

### Game Not Starting

- **Press Button A**: Use Button A on the Calliope mini to start/reset the game
- **Check code upload**: Ensure the program was successfully flashed to the board
- **Check serial output**: Use MakeCode's serial monitor to debug if available

### Sequence Too Fast/Slow

- Adjust the `duration` parameter in `lightUpButton()` function
- Modify the pause delays in `playSequence()` function

## Customization

### Change Colors

Edit the `COLORS` array:
```typescript
const COLORS = [
    NeoPixelColors.Red,      // Change to any NeoPixel color
    NeoPixelColors.Yellow,
    NeoPixelColors.Blue,
    NeoPixelColors.Green
]
```

### Change Timing

- **LED flash duration**: Modify `duration` in `lightUpButton()` calls
- **Sequence playback speed**: Adjust pause values in `playSequence()`
- **Button debounce**: Modify the debounce delay in the main loop

### Change Pin Assignments

Update the `LED_DATA_PIN` constant to use a different pin:
```typescript
const LED_DATA_PIN = DigitalPin.P0  // Change to your desired pin
```

Update the `BUTTON_PINS` array to use different pins:
```typescript
const BUTTON_PINS = [
    DigitalPin.C8,   // Change to your desired pin
    // ...
]
```

## Technical Notes

- **NeoPixel Library**: Uses the MakeCode NeoPixel extension for LED control
- **Daisy-Chained NeoPixels**: All 4 NeoPixel LEDs are connected in a chain, requiring only one data pin
- **LED Strip**: The code creates a single NeoPixel strip with 4 pixels (one per button)
- **LED Indexing**: LEDs are indexed 0-3: 0=Red, 1=Yellow, 2=Blue, 3=Green
- **Active LOW Buttons**: Buttons are configured to read LOW when pressed
- **Pull-up Resistors**: Internal pull-ups are used (buttons read HIGH when not pressed)
- **Debouncing**: Button debouncing is handled with a 200ms delay
- **Data Flow**: Data flows from P0 → LED 0 → LED 1 → LED 2 → LED 3

## License

This project is provided as-is for educational and demonstration purposes.

## Resources

- [Calliope mini Documentation](https://calliope.cc/)
- [MakeCode for Calliope mini](https://makecode.calliope.cc/)
- [NeoPixel Guide](https://calliope.cc/en/calliope-mini/accessories/sensoren/neopixel)

