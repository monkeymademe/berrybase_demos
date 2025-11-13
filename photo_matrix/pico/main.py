import time
import struct
import socket
import network
from picographics import PicoGraphics, DISPLAY_COSMIC_UNICORN
from cosmic import CosmicUnicorn

# Import WiFi configuration
try:
    from wifi_config import WIFI_SSID, WIFI_PASSWORD
except ImportError:
    raise ImportError(
        "wifi_config.py not found! Copy wifi_config.example.py to wifi_config.py "
        "and update with your WiFi credentials."
    )

PORT = 5000
MAGIC = b"CUF0"

cu = CosmicUnicorn()
graphics = PicoGraphics(display=DISPLAY_COSMIC_UNICORN)
WIDTH, HEIGHT = graphics.get_bounds()
EXPECTED_PAYLOAD = WIDTH * HEIGHT * 3


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    max_wait = 10
    while max_wait > 0:
        if wlan.status() < 0 or wlan.status() >= 3:
            break
        max_wait -= 1
        time.sleep(1)
    
    if wlan.status() != 3:
        raise RuntimeError("WiFi connection failed")
    
    print(f"Connected to WiFi. IP: {wlan.ifconfig()[0]}")
    return wlan


# Connect to WiFi
wlan = connect_wifi()

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(0.1)  # Non-blocking with timeout

print(f"Listening on port {PORT}")

while True:
    try:
        data, addr = sock.recvfrom(EXPECTED_PAYLOAD + 6)
        
        if len(data) < 6:
            continue
        
        header = data[:6]
        magic, count = struct.unpack(">4sH", header)
        
        if magic != MAGIC or count != EXPECTED_PAYLOAD:
            continue
        
        if len(data) < 6 + count:
            continue
        
        payload = data[6:6+count]
        
        index = 0
        for y in range(HEIGHT):
            for x in range(WIDTH):
                r = payload[index]
                g = payload[index + 1]
                b = payload[index + 2]
                pen = graphics.create_pen(r, g, b)
                graphics.set_pen(pen)
                graphics.pixel(x, y)
                index += 3
        
        cu.update(graphics)
    except OSError:
        # Timeout - continue waiting
        time.sleep(0.01)
        continue
