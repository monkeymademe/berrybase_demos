#!/usr/bin/env python3
import socket
import struct
import time
from picamera2 import Picamera2
import numpy as np

# WiFi configuration - update these for your network
PICO_IP = "192.168.1.100"  # Update with your Pico's IP address
PICO_PORT = 5000
FRAME_WIDTH = 32
FRAME_HEIGHT = 32
FPS_LIMIT = 10


def main() -> None:
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    ))
    picam2.start()

    try:
        last = 0.0
        while True:
            # Capture frame (already resized to 32x32 RGB)
            frame = picam2.capture_array()
            
            # Ensure it's the right shape and type
            if frame.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
                # Fallback resize if needed
                from PIL import Image
                img = Image.fromarray(frame)
                frame = np.array(img.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.Resampling.LANCZOS))
            
            # Convert to bytes (frame is already RGB)
            payload = frame.tobytes()

            # Send frame over UDP
            header = struct.pack(">4sH", b"CUF0", len(payload))
            packet = header + payload
            sock.sendto(packet, (PICO_IP, PICO_PORT))

            elapsed = time.time() - last
            wait = max(0.0, (1 / FPS_LIMIT) - elapsed)
            time.sleep(wait)
            last = time.time()
    finally:
        picam2.stop()
        sock.close()


if __name__ == "__main__":
    main()

