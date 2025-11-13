# Cosmic Unicorn Camera Stream

This demo streams live video from a Raspberry Pi Zero to a Pimoroni Cosmic Unicorn display over WiFi.

## Hardware

- Raspberry Pi Zero / Zero 2 W with camera module
- Pimoroni Cosmic Unicorn (RP2040) with WiFi capability
- Separate power supplies for both devices (WiFi requires more power than USB can provide)

## Raspberry Pi Setup

1. Update packages and install dependencies:

   ```bash
   sudo apt update
   sudo apt install python3-picamera2 python3-pil -y
   ```

2. Configure the Pico's IP address:
   - Edit `stream_to_pico.py` and update `PICO_IP` with your Cosmic Unicorn's IP address
   - You'll get this IP after the Pico connects to WiFi (see below)

3. Run the streamer:

   ```bash
   python3 stream_to_pico.py
   ```

## Cosmic Unicorn Setup

1. Flash Pimoroni MicroPython firmware with WiFi support if needed.

2. Configure WiFi credentials:
   - Copy `pico/wifi_config.example.py` to `pico/wifi_config.py`
   - Edit `pico/wifi_config.py` and update:
     - `WIFI_SSID` with your WiFi network name
     - `WIFI_PASSWORD` with your WiFi password
   - Note: `wifi_config.py` is gitignored and won't be committed to the repository

3. Copy both `pico/main.py` and `pico/wifi_config.py` to the board (e.g. with Thonny).

4. Power the Cosmic Unicorn (use a proper power supply, not USB from Pi Zero).

5. The script will:
   - Connect to WiFi
   - Print its IP address (check serial console or Thonny)
   - Start listening for video frames

6. Update `PICO_IP` in `stream_to_pico.py` with the IP address shown.

## Operation

1. Power both devices separately (Cosmic Unicorn needs adequate power for WiFi).
2. Wait for the Cosmic Unicorn to connect to WiFi and display its IP address.
3. Update `PICO_IP` in `stream_to_pico.py` if needed.
4. Start the Pi streamer: `python3 stream_to_pico.py`
5. The display will show the live camera feed.

### Finding the Pico's IP Address

- Connect via serial/Thonny to see the print output
- Or check your router's connected devices list
- The script prints: `Connected to WiFi. IP: x.x.x.x`

### Adjustments

- Tweak `FPS_LIMIT` in `stream_to_pico.py` for performance (lower = less network traffic).
- Modify `pico/main.py` to add effects or gamma correction.
- If frames are dropped, try lowering `FPS_LIMIT` or ensure both devices have strong WiFi signal.
