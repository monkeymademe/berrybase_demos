# Raspberry Pi Pico NeoPixel Rainbow Demo

This is the Raspberry Pi Pico version of the NeoPixel rainbow demo.

## Hardware Setup

- **Raspberry Pi Pico** (or Pico W)
- **NeoPixel strip** (WS2812 or compatible, 10 pixels)
- Connect NeoPixel data pin to **GPIO 28** (you can change `strip_pin` if needed)
- Connect NeoPixel VCC to **3.3V** (Raspberry Pi Pico uses 3.3V logic)
- Connect NeoPixel GND to **GND**

## Software Setup

1. Install **MicroPython** on your Raspberry Pi Pico:
   - Download the latest MicroPython UF2 file from the [official Raspberry Pi Pico website](https://www.raspberrypi.com/documentation/microcontrollers/micropython.html)
   - Hold the **BOOTSEL** button while connecting the Pico to your computer via USB
   - Drag and drop the UF2 file onto the mounted drive
2. Install a code editor (optional but recommended):
   - **Thonny** (simple IDE with built-in MicroPython support)
   - **VS Code** with the Pico-W-Go extension
   - Any text editor if using command-line tools
3. Transfer the code:
   - Using **Thonny**: Open `main.py`, then go to **File → Save as → Raspberry Pi Pico** and save it as `main.py`
   - Using **VS Code**: Use the Pico-W-Go extension to upload files
   - Using command line: Use tools like `ampy` or `mpremote` to upload `main.py`
4. Run the script:
   - In **Thonny**: Click the **Run** button (or press F5)
   - The script will automatically run on boot if saved as `main.py` on the Pico

