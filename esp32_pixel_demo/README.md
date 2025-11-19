# ESP32 NeoPixel LED Ring Animation

A collection of animated patterns for a NeoPixel LED ring controlled by an ESP32 microcontroller. This project features 8 different animation modes that randomly cycle through various visual effects.

## Features

- **8 Animation Modes:**
  - 🌈 Rainbow - Rotating rainbow colors across all LEDs
  - ⭕ Circle Chase - LEDs light up sequentially in a circle
  - 🔲 Checkerboard - Alternating pattern that flips
  - ✨ Sparkle - Random LEDs flash on and off
  - 🌊 Trail - Moving LED with trailing effect
  - 🎨 Color Wipe - Color sweeps across the ring
  - 🚗 Larson Scanner - Knight Rider style back-and-forth effect
  - 💫 Fade Fill - LEDs fade in/out sequentially

- **Configurable Settings:**
  - Adjustable brightness (0.0 to 1.0)
  - Customizable primary color (RGB)
  - Animation duration per cycle
  - Number of LEDs

## Hardware Requirements

- **ESP32 Development Board** (ESP32 DevKit, ESP32-WROOM, etc.)
- **NeoPixel LED Ring** (WS2812B or compatible)
  - Tested with 24-pixel ring, but works with any count
- **Power Supply:**
  - For small rings (< 30 LEDs): Can power from ESP32's 5V pin
  - For larger rings: External 5V power supply recommended
- **Jumper Wires** (3 wires minimum)
- **USB Cable** (for programming and power)

## Wiring Instructions

### Basic Wiring

1. **Data Line (DIN):**
   - Connect NeoPixel ring's **DIN** (Data In) to **ESP32 GPIO 2**
   - You can use other GPIO pins (4, 5, 12, 13, 14, 15, 18, 19, 21, 22, 23)
   - If using a different pin, update `strip_pin = Pin(2, Pin.OUT)` in `main.py`

2. **Power:**
   - Connect NeoPixel ring's **VCC** (5V) to ESP32's **5V** pin
   - Connect NeoPixel ring's **GND** to ESP32's **GND** pin

3. **External Power (Recommended for larger rings):**
   - Connect external 5V power supply **positive** to NeoPixel ring's **VCC**
   - Connect external 5V power supply **ground** to both ESP32 **GND** and NeoPixel ring's **GND**
   - **Important:** Keep grounds connected even with external power

### Wiring Diagram

```
ESP32                    NeoPixel Ring
------                   -------------
GPIO 2  ───────────────>  DIN (Data In)
5V      ───────────────>  VCC (5V)
GND     ───────────────>  GND
```

**Note:** For rings with more than ~30 LEDs, use an external 5V power supply and connect:
- External 5V+ → Ring VCC
- External GND → Ring GND + ESP32 GND
- ESP32 GPIO 2 → Ring DIN

## Software Installation

### Step 1: Install MicroPython on ESP32

1. **Download MicroPython for ESP32:**
   - Visit: https://micropython.org/download/esp32/
   - Download the latest stable `.bin` file for your ESP32 variant

2. **Flash MicroPython using esptool:**
   ```bash
   # Install esptool if needed
   pip3 install esptool
   # OR if that doesn't work:
   # python3 -m pip install esptool
   
   # On macOS, use python3 -m esptool (most reliable method):
   # Replace /dev/tty.usbserial-* with your actual port (see below)
   python3 -m esptool --chip esp32 --port /dev/tty.usbserial-* erase_flash
   python3 -m esptool --chip esp32 --port /dev/tty.usbserial-* write_flash -z 0x1000 firmware.bin
   
   # Alternative methods (if python3 -m doesn't work):
   # Option 1: esptool (if executable is in PATH)
   # esptool --chip esp32 --port /dev/tty.usbserial-* erase_flash
   
   # Option 2: esptool.py (older versions)
   # esptool.py --chip esp32 --port /dev/tty.usbserial-* erase_flash
   ```
   
   **Note:** On macOS, if `esptool` command is not found after installation, use `python3 -m esptool` instead. This works regardless of PATH configuration.

   **Finding your port on macOS:**
   ```bash
   # List available serial ports
   ls /dev/tty.usb* /dev/cu.usb* /dev/tty.SLAB* /dev/cu.SLAB* 2>/dev/null
   
   # Common macOS port names:
   # /dev/tty.usbserial-* or /dev/cu.usbserial-*
   # /dev/tty.SLAB_USBtoUART or /dev/cu.SLAB_USBtoUART
   # /dev/tty.wchusbserial* or /dev/cu.wchusbserial*
   ```
   
   **On Linux:** Port is typically `/dev/ttyUSB0`  
   **On Windows:** Port is typically `COM3`, `COM4`, etc.

### Step 2: Install NeoPixel Library

1. **Connect to ESP32 REPL:**
   - Use a serial terminal (PuTTY, screen, minicom, or Thonny IDE)
   - Connect at 115200 baud

2. **Install neopixel library:**
   ```python
   import upip
   upip.install('micropython-neopixel')
   ```

   **Alternative:** If `upip` doesn't work, download manually:
   - Visit: https://github.com/micropython/micropython-lib
   - Copy the `neopixel` module to your ESP32

### Step 3: Upload Code

**Option A: Using Thonny IDE (Recommended)**
1. Download and install [Thonny IDE](https://thonny.org/)
2. Open Thonny and select **Tools → Options → Interpreter**
3. Choose **MicroPython (ESP32)** and select your COM port
4. Open `main.py` in Thonny
5. Click **File → Save As** and save to **"MicroPython device"**
6. The code will run automatically on boot

**Option B: Using ampy or mpremote**
```bash
# Install ampy
pip install adafruit-ampy

# Upload main.py
ampy --port /dev/ttyUSB0 put main.py
```

**Option C: Using mpremote (MicroPython's official tool)**
```bash
# Install mpremote
pip install mpremote

# Upload main.py
mpremote cp main.py :main.py
```

## Configuration

Edit the following variables in `main.py` to customize:

```python
num_pixels = 24              # Number of LEDs in your ring
brightness = 0.3             # Brightness (0.0 to 1.0)
primary_color = (62, 145, 190)  # RGB color tuple
animation_duration = 10      # Seconds per animation
strip_pin = Pin(2, Pin.OUT)  # GPIO pin for data line
```

### Changing the Pin

If you need to use a different GPIO pin, change:
```python
strip_pin = Pin(2, Pin.OUT)  # Change 2 to your desired pin
```

**Common ESP32 GPIO pins:** 2, 4, 5, 12, 13, 14, 15, 18, 19, 21, 22, 23

**Note:** Avoid pins 0, 1, 3, 6, 7, 8, 9, 10, 11 (used for flash/PSRAM)

## How It Works

### Main Loop
The program randomly selects from the available animations and runs each for the specified duration before switching to the next.

### Animation Functions

Each animation function:
1. Takes a `duration` parameter (seconds)
2. Uses `time()` to track elapsed time
3. Updates LED colors in a loop
4. Calls `my_strip.write()` to push colors to the LEDs
5. Exits when duration is reached

### Color System

- **Primary Color:** Used by most animations (configurable RGB tuple)
- **Brightness Control:** All colors are scaled by the brightness factor
- **Rainbow Colors:** Generated using the `wheel()` function (HSV-like color wheel)

## Troubleshooting

### LEDs Not Lighting Up
- ✅ Check wiring connections (especially GND)
- ✅ Verify GPIO pin number matches your wiring
- ✅ Ensure power supply can handle current draw
- ✅ Try a different GPIO pin
- ✅ Check that neopixel library is installed

### Colors Look Wrong
- ✅ Verify RGB color tuple format: `(R, G, B)` where values are 0-255
- ✅ Check brightness setting (too low = dim, too high = washed out)

### Code Won't Upload
- ✅ Check COM port selection
- ✅ Try holding BOOT button while connecting
- ✅ Verify MicroPython is installed correctly
- ✅ Check USB cable (some are power-only)

### Animations Too Fast/Slow
- ✅ Adjust `animation_duration` for longer/shorter cycles
- ✅ Modify `sleep()` values in individual animation functions

### ESP32 Keeps Resetting
- ✅ Power supply issue - use external 5V supply
- ✅ Too many LEDs drawing too much current
- ✅ Add a capacitor (1000µF) between 5V and GND near the ring

## Power Consumption

**Approximate current draw:**
- Each LED at full brightness: ~60mA
- 24 LEDs at 30% brightness: ~430mA
- ESP32: ~80-240mA (depending on WiFi usage)

**Recommendations:**
- < 30 LEDs: Can use ESP32's 5V pin
- 30+ LEDs: Use external 5V power supply (2A+ recommended)
- Always connect grounds together

## License

This code is provided as-is for educational and personal use.

## Credits

Adapted from Raspberry Pi Pico version for ESP32 compatibility.

