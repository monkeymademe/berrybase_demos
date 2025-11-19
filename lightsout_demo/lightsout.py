"""
Lights Out Game for Pimoroni Pico RGB Keypad
A classic puzzle game where you need to turn off all the lights.
Pressing a button toggles that button and its adjacent neighbors.
"""

import time
import random
import picokeypad

# Initialize the keypad
keypad = picokeypad.PicoKeypad()
keypad.set_brightness(1.0)

NUM_PADS = keypad.get_num_pads()
GRID_SIZE = 4

# Game grid (4x4) - True means light is ON, False means OFF
# Grid is indexed as grid[y][x] where (0,0) is top-left
grid = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# Color definitions (RGB values 0-255)
COLOR_ON = (255, 255, 50)      # Bright yellow for lights ON
COLOR_OFF = (5, 5, 5)          # Dim for lights OFF (not completely black for visibility)
COLOR_WIN = (0, 255, 0)        # Green for win animation
COLOR_PRESSED = (100, 100, 255)  # Blue when button is pressed

def xy_to_index(x, y):
    """Convert (x, y) coordinates to linear index."""
    return y * GRID_SIZE + x

def index_to_xy(index):
    """Convert linear index to (x, y) coordinates."""
    return index % GRID_SIZE, index // GRID_SIZE

def toggle_lights(x, y):
    """Toggle the light at (x, y) and its adjacent neighbors."""
    # Toggle positions: center, up, down, left, right
    positions = [
        (x, y),      # Center
        (x-1, y),    # Left
        (x+1, y),    # Right
        (x, y-1),    # Up
        (x, y+1),    # Down
    ]
    
    for px, py in positions:
        if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
            grid[py][px] = not grid[py][px]

def update_leds():
    """Update the keypad LEDs to reflect the current grid state."""
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            index = xy_to_index(x, y)
            if grid[y][x]:
                keypad.illuminate(index, COLOR_ON[0], COLOR_ON[1], COLOR_ON[2])
            else:
                keypad.illuminate(index, COLOR_OFF[0], COLOR_OFF[1], COLOR_OFF[2])
    keypad.update()

def check_win():
    """Check if all lights are off (win condition)."""
    for row in grid:
        for cell in row:
            if cell:
                return False
    return True

def randomize_grid(difficulty=15):
    """Create a random solvable starting state by making random moves."""
    # Start with all lights off
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            grid[y][x] = False
    
    # Make random moves to create a puzzle
    for _ in range(difficulty):
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        toggle_lights(x, y)
    
    update_leds()

def win_animation():
    """Play a win animation when the player solves the puzzle."""
    # Flash green a few times
    for _ in range(3):
        for i in range(NUM_PADS):
            keypad.illuminate(i, COLOR_WIN[0], COLOR_WIN[1], COLOR_WIN[2])
        keypad.update()
        time.sleep(0.2)
        
        for i in range(NUM_PADS):
            keypad.illuminate(i, COLOR_OFF[0], COLOR_OFF[1], COLOR_OFF[2])
        keypad.update()
        time.sleep(0.2)

def main():
    """Main game loop."""
    print("Lights Out Game - Turn off all the lights!")
    print("Press any button to start a new game")
    
    # Initialize with a random puzzle
    randomize_grid(difficulty=15)
    
    last_button_states = 0
    debounce_delay = 0.2  # 200ms debounce
    last_press_time = {}
    
    while True:
        keypad.update()
        button_states = keypad.get_button_states()
        
        # Check if button states have changed
        if last_button_states != button_states and button_states > 0:
            current_time = time.time()
            
            # Check each button to see which one was pressed
            for button in range(NUM_PADS):
                # Check if this specific button is pressed
                if button_states & (1 << button):
                    # Check if no other buttons are pressed (single button press)
                    if not (button_states & (~(1 << button))):
                        button_id = button
                        
                        # Debounce check
                        if button_id not in last_press_time or \
                           (current_time - last_press_time[button_id]) > debounce_delay:
                            
                            last_press_time[button_id] = current_time
                            
                            # Convert index to coordinates
                            x, y = index_to_xy(button)
                            
                            # Visual feedback - briefly show pressed color
                            keypad.illuminate(button, COLOR_PRESSED[0], COLOR_PRESSED[1], COLOR_PRESSED[2])
                            keypad.update()
                            time.sleep(0.05)
                            
                            # Toggle the lights
                            toggle_lights(x, y)
                            update_leds()
                            
                            # Check for win condition
                            if check_win():
                                print("Congratulations! You solved it!")
                                win_animation()
                                time.sleep(1)
                                # Start a new game
                                randomize_grid(difficulty=15)
                            
                            break
        
        last_button_states = button_states
        time.sleep(0.01)  # Small delay to prevent CPU spinning

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Turn off all LEDs on exit
        for i in range(NUM_PADS):
            keypad.illuminate(i, 0, 0, 0)
        keypad.update()
        print("\nGame stopped.")

