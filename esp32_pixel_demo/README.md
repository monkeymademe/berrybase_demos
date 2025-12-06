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
   - If using a different pin:
     - **Arduino IDE:** Update `#define STRIP_PIN 2` in `esp32_pixel_demo.ino`
     - **MicroPython:** Update `strip_pin = Pin(2, Pin.OUT)` in `main.py`

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

This project supports two development environments:

- **Arduino IDE** (Recommended for better performance) - See below
- **MicroPython** - See [MicroPython Installation](#micropython-installation) section

### Arduino IDE Installation (Recommended)

Arduino IDE provides better performance and is easier to use if you're already familiar with Arduino.

#### Step 1: Install Arduino IDE and ESP32 Support

1. **Download and install Arduino IDE:**
   - Visit: https://www.arduino.cc/en/software
   - Download and install Arduino IDE (version 1.8.x or 2.x)

2. **Add ESP32 Board Support:**
   - Open Arduino IDE
   - Go to **File → Preferences**
   - In "Additional Board Manager URLs", add:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Click **OK**

3. **Install ESP32 Board:**
   - Go to **Tools → Board → Boards Manager**
   - Search for "ESP32"
   - Install "esp32" by Espressif Systems
   - Wait for installation to complete

4. **Select Your Board:**
   - Go to **Tools → Board → ESP32 Arduino**
   - Select your ESP32 board:
     - **LOLIN32**: Select "LOLIN32" or "WEMOS LOLIN32"
     - **ESP32 DevKit**: Select "ESP32 Dev Module"
     - **Other ESP32 boards**: Select the appropriate board from the list
   - **Note:** The code works with all ESP32 boards - just select the correct board type

#### Step 2: Install Adafruit NeoPixel Library

1. **Install Library:**
   - Go to **Sketch → Include Library → Manage Libraries**
   - Search for "Adafruit NeoPixel"
   - Install "Adafruit NeoPixel" by Adafruit

#### Step 3: Upload Code

1. **Open the sketch:**
   - Open `esp32_pixel_demo.ino` in Arduino IDE

2. **Configure settings (optional):**
   - Edit the constants at the top if needed:
     ```cpp
     #define NUM_PIXELS 24
     #define BRIGHTNESS 0.3
     #define STRIP_PIN 2
     #define ANIMATION_DURATION 10000
     ```

3. **Select Port:**
   - Go to **Tools → Port**
   - Select your ESP32's COM port (on Mac: `/dev/cu.usbserial-*` or `/dev/cu.SLAB_USBtoUART`)

4. **Upload:**
   - Click the **Upload** button (→) or press `Ctrl+U` (Windows/Linux) / `Cmd+U` (Mac)
   - Wait for compilation and upload to complete
   - The code will run automatically!

**Advantages of Arduino IDE:**
- ✅ Better performance (faster animations)
- ✅ More memory efficient
- ✅ Easier to use if familiar with Arduino
- ✅ Better real-time performance
- ✅ No need to install MicroPython

---

### MicroPython Installation

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

### Arduino IDE Configuration

Edit the following constants at the top of `esp32_pixel_demo.ino`:

```cpp
#define NUM_PIXELS 24              // Number of LEDs in your ring
#define BRIGHTNESS 0.3             // Brightness (0.0 to 1.0)
#define STRIP_PIN 2                // GPIO pin for data line
#define ANIMATION_DURATION 10000   // Duration in milliseconds

// Primary color RGB values
#define PRIMARY_R 62               // Red component (0-255)
#define PRIMARY_G 145              // Green component (0-255)
#define PRIMARY_B 190              // Blue component (0-255)
```

### MicroPython Configuration

Edit the following variables in `main.py`:

```python
num_pixels = 24              # Number of LEDs in your ring
brightness = 0.3             # Brightness (0.0 to 1.0)
primary_color = (62, 145, 190)  # RGB color tuple
animation_duration = 10      # Seconds per animation
strip_pin = Pin(2, Pin.OUT)  # GPIO pin for data line
```

### Changing the Pin

If you need to use a different GPIO pin:

**Arduino IDE:**
```cpp
#define STRIP_PIN 2  // Change 2 to your desired pin
```

**MicroPython:**
```python
strip_pin = Pin(2, Pin.OUT)  # Change 2 to your desired pin
```

**Common ESP32 GPIO pins:** 2, 4, 5, 12, 13, 14, 15, 18, 19, 21, 22, 23

**Note:** Avoid pins 0, 1, 3, 6, 7, 8, 9, 10, 11 (used for flash/PSRAM)

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

### Code Won't Upload (Arduino IDE)

**Error: `termios.error: (22, 'Invalid argument')` or Serial Port Issues**

This is a common issue on macOS. Try these solutions in order:

1. **Manual Upload Mode (Most Reliable):**
   - Hold the **BOOT** button on your ESP32
   - While holding BOOT, press and release the **RESET** button
   - Release the BOOT button
   - Click **Upload** in Arduino IDE
   - Wait for "Connecting..." message, then release buttons if needed

2. **Check Port Permissions (macOS):**
   ```bash
   # Check if you can access the port
   ls -l /dev/cu.usbserial-*
   
   # If permission denied, you may need to add your user to dialout group
   # Or use sudo (not recommended, but works)
   ```

3. **Close Other Programs:**
   - Close any serial monitors, terminal programs, or other IDEs using the port
   - Close Thonny, PlatformIO, or any other ESP32 tools

4. **Try Different Port:**
   - Unplug and replug USB cable
   - Check **Tools → Port** for updated port name
   - Try `/dev/cu.usbserial-*` instead of `/dev/tty.usbserial-*` (use `cu` not `tty`)

5. **Lower Upload Speed:**
   - Go to **Tools → Upload Speed**
   - Try **115200** or **921600** (lower speeds are more reliable)

6. **Check USB Cable:**
   - Use a data-capable USB cable (not power-only)
   - Try a different USB port (preferably USB 2.0, not USB 3.0)

7. **Driver Issues:**
   - Some ESP32 boards need CH340 or CP2102 drivers
   - Check if your board appears in System Information → USB
   - Download drivers from manufacturer if needed

8. **Arduino IDE Settings:**
   - Go to **Tools → Board** and ensure correct ESP32 board is selected
   - Try **Tools → Erase All Flash Before Sketch Upload** → **Enabled**

**General Upload Tips:**
- ✅ Always use `/dev/cu.*` ports on macOS (not `/dev/tty.*`)
- ✅ Close Serial Monitor before uploading
- ✅ Some boards need BOOT button held during entire upload
- ✅ Try uploading immediately after connecting (before auto-reset)

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

