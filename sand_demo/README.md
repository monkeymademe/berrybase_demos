# Sand Simulation for Raspberry Pi Sense HAT

A physics-based sand simulation that responds to gyroscope rotation on the Raspberry Pi Sense HAT.

## Hardware Requirements

- Raspberry Pi A+ (or any Raspberry Pi with GPIO)
- Sense HAT add-on board

## Installation

1. Install the Sense HAT library:
```bash
sudo apt-get update
sudo apt-get install sense-hat
```

Or using pip:
```bash
pip3 install -r requirements.txt
```

## Usage

Run the simulation:
```bash
python3 sand_simulation.py
```

## How It Works

- The simulation creates 30 sand particles on the 8x8 LED matrix
- Particles respond to gravity and rotation forces from the gyroscope
- When you tilt/rotate the Raspberry Pi, the sand particles move accordingly
- Particles bounce off the edges of the display
- The sand is rendered in a warm yellow/orange color with a glow effect

## Controls

- **Tilt the Pi**: Move the sand particles around
- **Press Ctrl+C**: Exit the simulation

## Customization

You can modify the following parameters in `sand_simulation.py`:

- `num_particles`: Number of sand particles (default: 30)
- `gravity`: Gravity strength (default: 0.3)
- `friction`: Friction coefficient (default: 0.85)
- `bounce`: Bounce coefficient (default: 0.3)

## Notes

- The simulation runs at approximately 20 FPS
- The gyroscope data is scaled for smooth movement
- Particles start in the upper third of the display when initialized



