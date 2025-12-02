"""
Simon Says Game for Calliope mini v3 (MicroPython) - Simplified Version
Uses 4 RGB LED arcade buttons with NeoPixel LEDs
Colors: Red, Yellow, Blue, Green

Run check_modules.py first to see what's available, then adjust this code accordingly.
"""

import time
import random

# Try to import threading if available
try:
    import _thread
    THREADING_AVAILABLE = True
except ImportError:
    THREADING_AVAILABLE = False

# Try to import display for LED matrix
try:
    from microbit import display
    DISPLAY_AVAILABLE = True
except ImportError:
    try:
        from calliopemini import display
        DISPLAY_AVAILABLE = True
    except ImportError:
        print("Display not available")
        DISPLAY_AVAILABLE = False

# Try to import neopixel
try:
    import neopixel
except ImportError:
    print("Error: neopixel module not found")
    raise

# Try different ways to access pins
pin0 = None
buttons = []

# Method 1: Try calliopemini module
try:
    from calliopemini import *
    print("Using calliopemini module")
    # After import *, try to use pins
    # Check if pin0 exists (it should be imported)
    try:
        # Try to use pin0 - if it was imported, it will work
        pin0_test = pin0
        pin0 = pin0_test
    except NameError:
        # pin0 not in calliopemini, try pin_RGB
        try:
            pin0 = pin_RGB
        except NameError:
            pin0 = None
    # Try to get button pins
    try:
        pin8_test = pin8  # Test if pin8 exists
        buttons = [pin8, pin9, pin13, pin14]
        print("Using calliopemini pin objects")
        # Configure pins as digital inputs if needed
        for btn in buttons:
            if hasattr(btn, 'set_pull'):
                btn.set_pull(btn.PULL_UP)
            elif hasattr(btn, 'set_mode'):
                try:
                    btn.set_mode(btn.IN, btn.PULL_UP)
                except:
                    pass
    except NameError:
        # Pins not available
        buttons = []
except ImportError:
    print("calliopemini module not available, trying machine module...")
    
    # Method 2: Try machine module
    try:
        from machine import Pin
        print("Using machine.Pin")
        pin0 = Pin(0, Pin.OUT)  # P0 for NeoPixel data
        buttons = [
            Pin(8, Pin.IN, Pin.PULL_UP),   # C8
            Pin(9, Pin.IN, Pin.PULL_UP),   # C9
            Pin(13, Pin.IN, Pin.PULL_UP),  # C13
            Pin(14, Pin.IN, Pin.PULL_UP)   # C14
        ]
    except ImportError:
        print("machine.Pin not available")
        # Method 3: Try direct pin numbers
        print("Trying direct pin numbers...")
        pin0 = 0
        buttons = [8, 9, 13, 14]

# Initialize NeoPixel strip
NUM_LEDS = 4
try:
    if pin0 is not None:
        np = neopixel.NeoPixel(pin0, NUM_LEDS)
        print("NeoPixel initialized on pin", pin0)
    else:
        print("Error: Could not determine pin0")
        raise RuntimeError("Pin0 not available")
except Exception as e:
    print("Error initializing NeoPixel:", e)
    print("You may need to adjust the pin number or import method")
    raise

# Game state variables
sequence = []
player_sequence = []
game_started = False
is_playing_sequence = False
current_level = 0

# NeoPixel color values (RGB tuples)
COLORS = [
    (255, 0, 0),      # LED 0 - Red button
    (255, 255, 0),    # LED 1 - Yellow button
    (0, 0, 255),      # LED 2 - Blue button
    (0, 255, 0)       # LED 3 - Green button
]

def wheel(pos):
    """
    Generate rainbow colors across 0-255 positions.
    Returns RGB tuple.
    """
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    else:
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)

def clear_all_leds():
    """Clear all NeoPixel LEDs"""
    np.fill((0, 0, 0))
    np.write()

def light_up_button(index, duration_ms=500):
    """Light up a specific button LED"""
    if index < 0 or index >= NUM_LEDS:
        return
    np[index] = COLORS[index]
    np.write()
    time.sleep_ms(duration_ms)
    np[index] = (0, 0, 0)
    np.write()

def play_sequence():
    """Play the current sequence"""
    global is_playing_sequence
    is_playing_sequence = True
    time.sleep_ms(500)
    for color_index in sequence:
        light_up_button(color_index, 600)
        time.sleep_ms(200)
    is_playing_sequence = False

def add_to_sequence():
    """Add a new color to the sequence"""
    global current_level
    new_color = random.randint(0, 3)
    sequence.append(new_color)
    current_level = len(sequence)

def check_sequence():
    """Check if player sequence matches the game sequence"""
    if len(player_sequence) > len(sequence):
        return False
    for i in range(len(player_sequence)):
        if player_sequence[i] != sequence[i]:
            return False
    return True

def start_game():
    """Start the game - flash white and begin"""
    global game_started
    # Flash all LEDs white
    np.fill((255, 255, 255))
    np.write()
    time.sleep_ms(1000)  # Wait 1 second
    clear_all_leds()
    
    # Clear display
    if DISPLAY_AVAILABLE:
        try:
            display.clear()
        except:
            pass
    
    # Start the game
    game_started = True
    add_to_sequence()
    play_sequence()

def handle_button_press(button_index):
    """Handle button press"""
    global game_started, player_sequence
    if is_playing_sequence:
        return
    if not game_started:
        # Button pressed to start game
        start_game()
        return
    player_sequence.append(button_index)
    light_up_button(button_index, 300)
    if not check_sequence():
        game_over()
        return
    if len(player_sequence) == len(sequence):
        player_sequence = []
        # Flash green for success
        for i in range(2):
            np.fill((0, 255, 0))
            np.write()
            time.sleep_ms(200)
            clear_all_leds()
            time.sleep_ms(200)
        
        # Show level number on display (just the number)
        if DISPLAY_AVAILABLE:
            show_text(str(current_level + 1), delay=100)
        
        add_to_sequence()
        play_sequence()

def game_over():
    """Game over sequence"""
    global sequence, player_sequence
    
    # Flash LEDs red
    for i in range(3):
        for j in range(NUM_LEDS):
            np[j] = (255, 0, 0)
        np.write()
        time.sleep_ms(200)
        clear_all_leds()
        time.sleep_ms(200)
    
    # Flash level number
    for i in range(current_level):
        np.fill((255, 255, 255))
        np.write()
        time.sleep_ms(300)
        clear_all_leds()
        time.sleep_ms(300)
    
    time.sleep_ms(1000)
    init_game()

def show_text(text, delay=100, loop=False):
    """Show scrolling text on LED matrix"""
    if DISPLAY_AVAILABLE:
        try:
            if loop:
                # For continuous scrolling, we'll handle it in the main loop
                return text
            else:
                display.scroll(text, delay=delay)
        except:
            # If scroll doesn't work, try show
            try:
                display.show(text)
            except:
                pass
    return text

def scroll_text_character_by_character(text, delay_ms=80):
    """
    Scroll text character by character, checking buttons between each character.
    This allows immediate button response.
    Returns True if button was pressed, False otherwise.
    """
    if not DISPLAY_AVAILABLE:
        return False
    
    global rainbow_counter
    
    # Add padding spaces for smooth scroll
    padded_text = "     " + text + "     "
    
    # Scroll through each character position
    for i in range(len(padded_text)):
        # Check for button press BEFORE showing each character
        for btn_idx in range(len(buttons)):
            if read_button(btn_idx):
                try:
                    display.clear()
                except:
                    pass
                return True  # Button pressed!
        
        # Update rainbow LEDs
        rainbow_counter = (rainbow_counter + 1) % 256
        for j in range(NUM_LEDS):
            pixel_index = (j * 256 // NUM_LEDS) + rainbow_counter
            np[j] = wheel(pixel_index & 255)
        np.write()
        
        # Show current character
        try:
            if i < len(padded_text):
                display.show(padded_text[i])
        except:
            pass
        
        # Small delay, but check buttons during delay
        # Break delay into smaller chunks to check buttons frequently
        delay_chunks = delay_ms // 10  # Check every 10ms
        for _ in range(delay_chunks):
            # Check buttons during delay
            for btn_idx in range(len(buttons)):
                if read_button(btn_idx):
                    try:
                        display.clear()
                    except:
                        pass
                    return True  # Button pressed!
            time.sleep_ms(10)
        
        # Handle remainder of delay
        remainder = delay_ms % 10
        if remainder > 0:
            time.sleep_ms(remainder)
            # Final button check
            for btn_idx in range(len(buttons)):
                if read_button(btn_idx):
                    try:
                        display.clear()
                    except:
                        pass
                    return True
    
    # Clear display at end
    try:
        display.clear()
    except:
        pass
    
    return False

def rainbow_cycle(wait_ms=10):
    """Cycle through rainbow colors on all LEDs"""
    for j in range(256):  # Full rainbow cycle
        for i in range(NUM_LEDS):
            # Offset each LED by a different amount for rainbow effect
            pixel_index = (i * 256 // NUM_LEDS) + j
            np[i] = wheel(pixel_index & 255)
        np.write()
        time.sleep_ms(wait_ms)

def init_game():
    """Initialize the game"""
    global sequence, player_sequence, game_started, is_playing_sequence, current_level
    sequence = []
    player_sequence = []
    game_started = False
    is_playing_sequence = False
    current_level = 0
    clear_all_leds()

def read_button(button_index):
    """Read button state - handles different button types"""
    try:
        btn = buttons[button_index]
        
        # Try different methods to read the pin
        # Method 1: Try read_digital() (common for MicroBit pins)
        if hasattr(btn, 'read_digital'):
            val = btn.read_digital()
            return val == 0  # Active LOW
        
        # Method 2: Try value() method
        if hasattr(btn, 'value'):
            val = btn.value()
            return val == 0  # Active LOW
        
        # Method 3: Try is_touched() for touch pins
        if hasattr(btn, 'is_touched'):
            return btn.is_touched()
        
        # Method 4: Try direct read
        try:
            val = btn.read()
            return val == 0
        except:
            pass
            
        return False
    except Exception as e:
        # Debug: uncomment to see errors
        # print("Button read error:", e)
        return False

# Initialize game
init_game()
print("Simon Says Game Started!")
print("Press any button to begin...")

# Clear display initially
if DISPLAY_AVAILABLE:
    try:
        display.clear()
    except:
        pass

# Main loop
button_pressed = False
rainbow_counter = 0

try:
    while True:
        # If game not started, show rainbow LEDs and wait for button
        if not game_started:
            # Update rainbow LEDs continuously
            rainbow_counter = (rainbow_counter + 1) % 256
            for i in range(NUM_LEDS):
                pixel_index = (i * 256 // NUM_LEDS) + rainbow_counter
                np[i] = wheel(pixel_index & 255)
            np.write()
            
            # Check for button press
            for i in range(len(buttons)):
                if read_button(i):
                    if not button_pressed:
                        print("Button", i, "pressed!")
                        handle_button_press(i)
                        button_pressed = True
                        break
            
            time.sleep_ms(20)  # Small delay for rainbow animation
        else:
            # Game is started - normal game loop
            for i in range(len(buttons)):
                if read_button(i):
                    if not button_pressed:
                        print("Button", i, "pressed!")
                        handle_button_press(i)
                        button_pressed = True
                        time.sleep_ms(200)  # Debounce
            
            if not any([read_button(j) for j in range(len(buttons))]):
                button_pressed = False  # Reset when all buttons released
            
            time.sleep_ms(50)
        
except KeyboardInterrupt:
    print("\nGame stopped")
    clear_all_leds()
    if DISPLAY_AVAILABLE:
        try:
            display.clear()
        except:
            pass

