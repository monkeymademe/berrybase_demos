/**
 * Simon Says Game for Calliope mini v3
 * Uses 4 RGB LED arcade buttons with NeoPixel LEDs
 * Colors: Red, Yellow, Blue, Green
 * 
 * NOTE: This code is designed for MakeCode for Calliope mini
 * The linting errors are expected - MakeCode provides these APIs at runtime:
 * - basic, input, pins, neopixel, DigitalPin, NeoPixelColors, NeoPixelMode, IconNames, Button
 * Upload this code to MakeCode (https://makecode.calliope.cc/) to compile and flash
 */

// Game state variables
let sequence: number[] = []
let playerSequence: number[] = []
let gameStarted = false
let isPlayingSequence = false
let currentLevel = 0

// NeoPixel strip configuration
// Single data pin for daisy-chained NeoPixels (4 LEDs in chain)
const LED_DATA_PIN = DigitalPin.P0
const NUM_LEDS = 4  // One LED per button

// Create NeoPixel strip (will be initialized in initGame)
let ledStrip: neopixel.Strip = null

// Button input pins (one per button)
const BUTTON_PINS = [
    DigitalPin.C8,   // Red button
    DigitalPin.C9,   // Yellow button
    DigitalPin.C13,  // Blue button
    DigitalPin.C14   // Green button
]

// NeoPixel color values (one per button/LED position)
const COLORS = [
    NeoPixelColors.Red,    // LED 0 - Red button
    NeoPixelColors.Yellow,  // LED 1 - Yellow button
    NeoPixelColors.Blue,    // LED 2 - Blue button
    NeoPixelColors.Green    // LED 3 - Green button
]

// Color names for display
const COLOR_NAMES = ["Red", "Yellow", "Blue", "Green"]

/**
 * Initialize the game
 */
function initGame() {
    sequence = []
    playerSequence = []
    gameStarted = false
    isPlayingSequence = false
    currentLevel = 0
    
    // Initialize NeoPixel strip if not already done
    if (ledStrip == null) {
        ledStrip = neopixel.create(LED_DATA_PIN, NUM_LEDS, NeoPixelMode.RGB)
    }
    
    // Turn off all LEDs
    clearAllLEDs()
    
    // Show ready message
    basic.showString("READY", 100)
    basic.pause(500)
    basic.clearScreen()
}

/**
 * Clear all NeoPixel LEDs
 */
function clearAllLEDs() {
    if (ledStrip != null) {
        ledStrip.clear()
        ledStrip.show()
    }
}

/**
 * Light up a specific button LED
 * @param index - Button/LED index (0-3)
 * @param duration - How long to keep it lit in ms
 */
function lightUpButton(index: number, duration: number = 500) {
    if (ledStrip == null || index < 0 || index >= NUM_LEDS) {
        return
    }
    
    // Set the specific pixel in the chain
    ledStrip.setPixelColor(index, COLORS[index])
    ledStrip.show()
    basic.pause(duration)
    ledStrip.setPixelColor(index, NeoPixelColors.Black) // Turn off this pixel
    ledStrip.show()
}

/**
 * Play the current sequence
 */
function playSequence() {
    isPlayingSequence = true
    
    // Small delay before starting
    basic.pause(500)
    
    for (let i = 0; i < sequence.length; i++) {
        let colorIndex = sequence[i]
        lightUpButton(colorIndex, 600)
        basic.pause(200) // Pause between flashes
    }
    
    isPlayingSequence = false
}

/**
 * Add a new color to the sequence
 */
function addToSequence() {
    let newColor = Math.randomRange(0, 3)
    sequence.push(newColor)
    currentLevel = sequence.length
}

/**
 * Check if player sequence matches the game sequence
 */
function checkSequence(): boolean {
    if (playerSequence.length > sequence.length) {
        return false
    }
    
    for (let i = 0; i < playerSequence.length; i++) {
        if (playerSequence[i] != sequence[i]) {
            return false
        }
    }
    
    return true
}

/**
 * Handle button press
 * @param buttonIndex - Index of the pressed button (0-3)
 */
function handleButtonPress(buttonIndex: number) {
    // Ignore button presses while sequence is playing
    if (isPlayingSequence) {
        return
    }
    
    // Start game on first button press if not started
    if (!gameStarted) {
        gameStarted = true
        addToSequence()
        playSequence()
        return
    }
    
    // Add to player sequence
    playerSequence.push(buttonIndex)
    
    // Light up the button
    lightUpButton(buttonIndex, 300)
    
    // Check if sequence is correct so far
    if (!checkSequence()) {
        // Wrong sequence - game over
        gameOver()
        return
    }
    
    // Check if player completed the sequence
    if (playerSequence.length == sequence.length) {
        // Correct! Move to next level
        playerSequence = []
        basic.showIcon(IconNames.Happy)
        basic.pause(1000)
        basic.clearScreen()
        
        // Add new color and play sequence
        addToSequence()
        playSequence()
    }
}

/**
 * Game over sequence
 */
function gameOver() {
    // Flash all LEDs red
    for (let i = 0; i < 3; i++) {
        // Set all pixels to red
        for (let j = 0; j < NUM_LEDS; j++) {
            ledStrip.setPixelColor(j, NeoPixelColors.Red)
        }
        ledStrip.show()
        basic.pause(200)
        clearAllLEDs()
        basic.pause(200)
    }
    
    // Show score
    basic.showString("LVL" + currentLevel, 100)
    basic.pause(1000)
    basic.showIcon(IconNames.Sad)
    basic.pause(2000)
    
    // Reset game
    initGame()
}

/**
 * Check button states continuously
 */
basic.forever(function () {
    // Check each button
    for (let i = 0; i < BUTTON_PINS.length; i++) {
        // Read button state (assuming buttons are active LOW with pull-up)
        // Adjust logic if your buttons are active HIGH
        if (pins.digitalReadPin(BUTTON_PINS[i]) == 0) {
            handleButtonPress(i)
            // Debounce delay
            basic.pause(200)
        }
    }
    
    // Small delay to prevent excessive CPU usage
    basic.pause(50)
})

/**
 * Start button (Button A) - Start new game
 */
input.onButtonPressed(Button.A, function () {
    initGame()
})

/**
 * Initialize on startup
 */
initGame()

