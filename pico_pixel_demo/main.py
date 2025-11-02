from machine import Pin
from neopixel import NeoPixel
from time import sleep

num_pixels = 10
strip_pin = Pin(28, Pin.OUT)

my_strip = NeoPixel(strip_pin, num_pixels)

def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    else:
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)

offset = 0
while True:
    for i in range(num_pixels):
        my_strip[i] = wheel((i * 256 // num_pixels + offset) % 256)
    my_strip.write()
    offset = (offset + 1) % 256
    sleep(0.01)