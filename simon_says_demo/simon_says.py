"""
Simon Says Game for Calliope mini v3 (MicroPython)
Uses 4 RGB LED arcade buttons with NeoPixel LEDs
Colors: Red, Yellow, Blue, Green

Hardware:
- NeoPixel Data In connected to P0
- Buttons connected to C8, C9, C13, C14 (active LOW)
"""

try:
    # Try Calliope mini specific module first
    from calliopemini import *
    import neopixel
    CALLIOPE_MINI = True
except ImportError:
    # Fall back to standard MicroPython
    try:
        from machine import Pin
        import neopixel
        CALLIOPE_MINI = False
    except ImportError:
        # If neither works, try alternative imports
        import neopixel
        CALLIOPE_MINI = None

import time
import random

# Game state variables
sequence = []
player_sequence = []
game_started = False
is_playing_sequence = False
current_level = 0

# NeoPixel strip configuration
NUM_LEDS = 4  # One LED per button

# Initialize NeoPixel strip
if CALLIOPE_MINI:
    # Use Calliope mini pin_RGB or create pin for P0
    try:
        # Try using pin_RGB if available, otherwise create pin for P0
        np = neopixel.NeoPixel(pin0, NUM_LEDS)
    except:
        # Fallback: try to create pin manually
        try:
            from calliopemini import pin0
            np = neopixel.NeoPixel(pin0, NUM_LEDS)
        except:
            # Last resort: use pin_RGB if it exists
            np = neopixel.NeoPixel(pin_RGB, NUM_LEDS)
elif CALLIOPE_MINI is False:
    LED_DATA_PIN = 0  # P0 pin number
    np = neopixel.NeoPixel(Pin(LED_DATA_PIN), NUM_LEDS)
else:
    # Try without Pin class - might have direct pin access
    np = neopixel.NeoPixel(0, NUM_LEDS)

# Button input pins (one per button)
# Note: Calliope mini pin numbers - adjust if needed
BUTTON_PINS = [
    8,   # C8 - Red button
    9,   # C9 - Yellow button
    13,  # C13 - Blue button
    14   # C14 - Green button
]

# Initialize button pins as inputs with pull-up resistors
buttons = []
if CALLIOPE_MINI:
    # Use Calliope mini pin access
    try:
        buttons = [pin8, pin9, pin13, pin14]
    except:
        # If named pins don't exist, try creating them
        try:
            from calliopemini import pin8, pin9, pin13, pin14
            buttons = [pin8, pin9, pin13, pin14]
        except:
            # Fallback: create pins manually if possible
            buttons = []
            for i, pin_num in enumerate(BUTTON_PINS):
                try:
                    pin = eval(f"pin{pin_num}")
                    buttons.append(pin)
                except:
                    pass
elif CALLIOPE_MINI is False:
    for pin_num in BUTTON_PINS:
        buttons.append(Pin(pin_num, Pin.IN, Pin.PULL_UP))
else:
    # Fallback: try to create buttons without Pin class
    buttons = []

# NeoPixel color values (RGB tuples)
COLORS = [
    (255, 0, 0),      # LED 0 - Red button
    (255, 255, 0),    # LED 1 - Yellow button
    (0, 0, 255),      # LED 2 - Blue button
    (0, 255, 0)       # LED 3 - Green button
]

# Color names for display
COLOR_NAMES = ["Red", "Yellow", "Blue", "Green"]

def init_game():
    """Initialize the game"""
    global sequence, player_sequence, game_started, is_playing_sequence, current_level
    
    sequence = []
    player_sequence = []
    game_started = False
    is_playing_sequence = False
    current_level = 0
    
    # Turn off all LEDs
    clear_all_leds()
    
    # Show ready message (using LED pattern since we don't have display)
    # Flash all LEDs white briefly
    for i in range(3):
        np.fill((255, 255, 255))
        np.write()
        time.sleep_ms(100)
        clear_all_leds()
        time.sleep_ms(100)
    
    time.sleep_ms(500)

def clear_all_leds():
    """Clear all NeoPixel LEDs"""
    np.fill((0, 0, 0))
    np.write()

def light_up_button(index, duration_ms=500):
    """
    Light up a specific button LED
    index: Button/LED index (0-3)
    duration_ms: How long to keep it lit in milliseconds
    """
    if index < 0 or index >= NUM_LEDS:
        return
    
    # Set the specific pixel in the chain
    np[index] = COLORS[index]
    np.write()
    time.sleep_ms(duration_ms)
    np[index] = (0, 0, 0)  # Turn off this pixel
    np.write()

def play_sequence():
    """Play the current sequence"""
    global is_playing_sequence
    
    is_playing_sequence = True
    
    # Small delay before starting
    time.sleep_ms(500)
    
    for color_index in sequence:
        light_up_button(color_index, 600)
        time.sleep_ms(200)  # Pause between flashes
    
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

def handle_button_press(button_index):
    """Handle button press"""
    global game_started, player_sequence
    
    # Ignore button presses while sequence is playing
    if is_playing_sequence:
        return
    
    # Start game on first button press if not started
    if not game_started:
        game_started = True
        add_to_sequence()
        play_sequence()
        return
    
    # Add to player sequence
    player_sequence.append(button_index)
    
    # Light up the button
    light_up_button(button_index, 300)
    
    # Check if sequence is correct so far
    if not check_sequence():
        # Wrong sequence - game over
        game_over()
        return
    
    # Check if player completed the sequence
    if len(player_sequence) == len(sequence):
        # Correct! Move to next level
        player_sequence = []
        
        # Flash all LEDs green to indicate success
        for i in range(2):
            np.fill((0, 255, 0))
            np.write()
            time.sleep_ms(200)
            clear_all_leds()
            time.sleep_ms(200)
        
        # Add new color and play sequence
        add_to_sequence()
        play_sequence()

def game_over():
    """Game over sequence"""
    global sequence, player_sequence
    
    # Flash all LEDs red
    for i in range(3):
        # Set all pixels to red
        for j in range(NUM_LEDS):
            np[j] = (255, 0, 0)
        np.write()
        time.sleep_ms(200)
        clear_all_leds()
        time.sleep_ms(200)
    
    # Flash level number using LED pattern
    # Flash all LEDs white for each level
    for i in range(current_level):
        np.fill((255, 255, 255))
        np.write()
        time.sleep_ms(300)
        clear_all_leds()
        time.sleep_ms(300)
    
    time.sleep_ms(1000)
    
    # Reset game
    init_game()

def check_buttons():
    """Check button states"""
    for i in range(len(buttons)):
        # Read button state (active LOW - pressed = 0, not pressed = 1)
        if buttons[i].value() == 0:
            handle_button_press(i)
            # Debounce delay
            time.sleep_ms(200)

# Initialize game on startup
init_game()

# Main loop
print("Simon Says Game Started!")
print("Press any button to begin...")

try:
    while True:
        check_buttons()
        # Small delay to prevent excessive CPU usage
        time.sleep_ms(50)
except KeyboardInterrupt:
    print("\nGame stopped by user")
    clear_all_leds()

