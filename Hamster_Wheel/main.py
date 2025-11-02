from machine import Pin, SPI, PWM
import framebuf
import time
import os

# Pin definitions
DC = 8
CS = 9
SCK = 10
MOSI = 11
MISO = 12
RST = 13
BL = 25

# LCD Driver
class LCD_1inch28(framebuf.FrameBuffer):
    def __init__(self):
        self.width = 240
        self.height = 240
        
        self.cs = Pin(CS, Pin.OUT)
        self.rst = Pin(RST, Pin.OUT)
        
        self.cs(1)
        self.spi = SPI(1, 100_000_000, polarity=0, phase=0, bits=8, sck=Pin(SCK), mosi=Pin(MOSI), miso=None)
        self.dc = Pin(DC, Pin.OUT)
        self.dc(1)
        self.buffer = bytearray(self.height * self.width * 2)
        super().__init__(self.buffer, self.width, self.height, framebuf.RGB565)
        self.init_display()
        
        # Define colors
        self.red   = 0x07E0
        self.green = 0x001f
        self.blue  = 0xf800
        self.white = 0xffff
        self.black = 0x0000
        
        self.fill(self.white)
        self.show()
        
        self.pwm = PWM(Pin(BL))
        self.pwm.freq(5000)
        
    def write_cmd(self, cmd):
        self.cs(1)
        self.dc(0)
        self.cs(0)
        self.spi.write(bytearray([cmd]))
        self.cs(1)
    
    def write_data(self, buf):
        self.cs(1)
        self.dc(1)
        self.cs(0)
        self.spi.write(bytearray([buf]))
        self.cs(1)
        
    def set_bl_pwm(self, duty):
        self.pwm.duty_u16(duty)
        
    def init_display(self):
        self.rst(1)
        time.sleep(0.01)
        self.rst(0)
        time.sleep(0.01)
        self.rst(1)
        time.sleep(0.05)
        
        self.write_cmd(0xEF)
        self.write_cmd(0xEB)
        self.write_data(0x14)
        
        self.write_cmd(0xFE)
        self.write_cmd(0xEF)
        
        self.write_cmd(0xEB)
        self.write_data(0x14)
        
        self.write_cmd(0x84)
        self.write_data(0x40)
        
        self.write_cmd(0x85)
        self.write_data(0xFF)
        
        self.write_cmd(0x86)
        self.write_data(0xFF)
        
        self.write_cmd(0x87)
        self.write_data(0xFF)
        
        self.write_cmd(0x88)
        self.write_data(0x0A)
        
        self.write_cmd(0x89)
        self.write_data(0x21)
        
        self.write_cmd(0x8A)
        self.write_data(0x00)
        
        self.write_cmd(0x8B)
        self.write_data(0x80)
        
        self.write_cmd(0x8C)
        self.write_data(0x01)
        
        self.write_cmd(0x8D)
        self.write_data(0x01)
        
        self.write_cmd(0x8E)
        self.write_data(0xFF)
        
        self.write_cmd(0x8F)
        self.write_data(0xFF)
        
        self.write_cmd(0xB6)
        self.write_data(0x00)
        self.write_data(0x20)
        
        self.write_cmd(0x36)
        self.write_data(0x98)
        
        self.write_cmd(0x3A)
        self.write_data(0x05)
        
        self.write_cmd(0x90)
        self.write_data(0x08)
        self.write_data(0x08)
        self.write_data(0x08)
        self.write_data(0x08)
        
        self.write_cmd(0xBD)
        self.write_data(0x06)
        
        self.write_cmd(0xBC)
        self.write_data(0x00)
        
        self.write_cmd(0xFF)
        self.write_data(0x60)
        self.write_data(0x01)
        self.write_data(0x04)
        
        self.write_cmd(0xC3)
        self.write_data(0x13)
        self.write_cmd(0xC4)
        self.write_data(0x13)
        
        self.write_cmd(0xC9)
        self.write_data(0x22)
        
        self.write_cmd(0xBE)
        self.write_data(0x11)
        
        self.write_cmd(0xE1)
        self.write_data(0x10)
        self.write_data(0x0E)
        
        self.write_cmd(0xDF)
        self.write_data(0x21)
        self.write_data(0x0c)
        self.write_data(0x02)
        
        self.write_cmd(0xF0)
        self.write_data(0x45)
        self.write_data(0x09)
        self.write_data(0x08)
        self.write_data(0x08)
        self.write_data(0x26)
        self.write_data(0x2A)
        
        self.write_cmd(0xF1)
        self.write_data(0x43)
        self.write_data(0x70)
        self.write_data(0x72)
        self.write_data(0x36)
        self.write_data(0x37)
        self.write_data(0x6F)
        
        self.write_cmd(0xF2)
        self.write_data(0x45)
        self.write_data(0x09)
        self.write_data(0x08)
        self.write_data(0x08)
        self.write_data(0x26)
        self.write_data(0x2A)
        
        self.write_cmd(0xF3)
        self.write_data(0x43)
        self.write_data(0x70)
        self.write_data(0x72)
        self.write_data(0x36)
        self.write_data(0x37)
        self.write_data(0x6F)
        
        self.write_cmd(0xED)
        self.write_data(0x1B)
        self.write_data(0x0B)
        
        self.write_cmd(0xAE)
        self.write_data(0x77)
        
        self.write_cmd(0xCD)
        self.write_data(0x63)
        
        self.write_cmd(0x70)
        self.write_data(0x07)
        self.write_data(0x07)
        self.write_data(0x04)
        self.write_data(0x0E)
        self.write_data(0x0F)
        self.write_data(0x09)
        self.write_data(0x07)
        self.write_data(0x08)
        self.write_data(0x03)
        
        self.write_cmd(0xE8)
        self.write_data(0x34)
        
        self.write_cmd(0x62)
        self.write_data(0x18)
        self.write_data(0x0D)
        self.write_data(0x71)
        self.write_data(0xED)
        self.write_data(0x70)
        self.write_data(0x70)
        self.write_data(0x18)
        self.write_data(0x0F)
        self.write_data(0x71)
        self.write_data(0xEF)
        self.write_data(0x70)
        self.write_data(0x70)
        
        self.write_cmd(0x63)
        self.write_data(0x18)
        self.write_data(0x11)
        self.write_data(0x71)
        self.write_data(0xF1)
        self.write_data(0x70)
        self.write_data(0x70)
        self.write_data(0x18)
        self.write_data(0x13)
        self.write_data(0x71)
        self.write_data(0xF3)
        self.write_data(0x70)
        self.write_data(0x70)
        
        self.write_cmd(0x64)
        self.write_data(0x28)
        self.write_data(0x29)
        self.write_data(0xF1)
        self.write_data(0x01)
        self.write_data(0xF1)
        self.write_data(0x00)
        self.write_data(0x07)
        
        self.write_cmd(0x66)
        self.write_data(0x3C)
        self.write_data(0x00)
        self.write_data(0xCD)
        self.write_data(0x67)
        self.write_data(0x45)
        self.write_data(0x45)
        self.write_data(0x10)
        self.write_data(0x00)
        self.write_data(0x00)
        self.write_data(0x00)
        
        self.write_cmd(0x67)
        self.write_data(0x00)
        self.write_data(0x3C)
        self.write_data(0x00)
        self.write_data(0x00)
        self.write_data(0x00)
        self.write_data(0x01)
        self.write_data(0x54)
        self.write_data(0x10)
        self.write_data(0x32)
        self.write_data(0x98)
        
        self.write_cmd(0x74)
        self.write_data(0x10)
        self.write_data(0x85)
        self.write_data(0x80)
        self.write_data(0x00)
        self.write_data(0x00)
        self.write_data(0x4E)
        self.write_data(0x00)
        
        self.write_cmd(0x98)
        self.write_data(0x3e)
        self.write_data(0x07)
        
        self.write_cmd(0x35)
        self.write_cmd(0x21)
        
        self.write_cmd(0x11)
        
        self.write_cmd(0x29)
    
    def set_windows(self, Xstart, Ystart, Xend, Yend):
        self.write_cmd(0x2A)
        self.write_data(0x00)
        self.write_data(Xstart)
        self.write_data(0x00)
        self.write_data(Xend-1)
        
        self.write_cmd(0x2B)
        self.write_data(0x00)
        self.write_data(Ystart)
        self.write_data(0x00)
        self.write_data(Yend-1)
        
        self.write_cmd(0x2C)
     
    def show(self):
        self.set_windows(0, 0, self.width, self.height)
        
        self.cs(1)
        self.dc(1)
        self.cs(0)
        self.spi.write(self.buffer)
        self.cs(1)


def load_frame(lcd, frame_path):
    """Load a frame file directly into the LCD buffer without extra memory allocation."""
    try:
        with open(frame_path, 'rb') as f:
            # Read directly into LCD buffer - no temporary memory allocation!
            bytes_read = f.readinto(lcd.buffer)
            if bytes_read != len(lcd.buffer):
                print(f"Frame size mismatch: expected {len(lcd.buffer)}, got {bytes_read}")
                return False
        return True
    except Exception as e:
        print(f"Error loading frame {frame_path}: {e}")
        return False


def play_animation(lcd, frames_dir='frames', frame_pattern='.rgb565', delay=0.08):
    """Play the animation by loading frames one at a time."""
    # Get list of frame files
    try:
        files = sorted([f for f in os.listdir(frames_dir) if frame_pattern in f])
    except OSError:
        print(f"Error: Could not find frames directory: {frames_dir}")
        return
    
    if not files:
        print("No frame files found")
        return
    
    print(f"Playing {len(files)} frames from {frames_dir}")
    
    # Play animation in loop
    frame_count = 0
    while True:
        for i, filename in enumerate(files):
            frame_path = f"{frames_dir}/{filename}"
            
            if load_frame(lcd, frame_path):
                lcd.show()
                frame_count += 1
                # Don't delay after last frame to avoid pause on loop
                if i < len(files) - 1:
                    time.sleep(delay)
            else:
                print(f"Failed to load {filename}")


if __name__ == '__main__':
    # Initialize LCD
    print("Initializing LCD...")
    lcd = LCD_1inch28()
    lcd.set_bl_pwm(65535)
    
    print("Starting animation...")
    # Play animation - uses frames/ directory by default
    # For optimized (reduced frame) version, use: frames_dir='frames_opt'
    # Adjust delay to change speed: 0.04 = ~25fps, 0.08 = ~12.5fps
    play_animation(lcd, frames_dir='frames_opt', frame_pattern='.rgb565', delay=0.06)