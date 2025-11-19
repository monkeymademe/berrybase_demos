#!/usr/bin/env python3
"""
Sand Simulation for Raspberry Pi Sense HAT
Simulates sand particles that respond to gyroscope rotation
"""

from sense_hat import SenseHat
import time
import math
import random

class SandSimulation:
    def __init__(self):
        self.sense = SenseHat()
        self.sense.clear()
        
        # Matrix dimensions (Sense HAT is 8x8)
        self.width = 8
        self.height = 8
        
        # Sand particle storage: list of (x, y, vx, vy) tuples
        # x, y are positions (0-7), vx, vy are velocities
        self.particles = []
        
        # Number of sand particles
        self.num_particles = 30
        
        # Physics constants
        self.gravity = 0.3
        self.friction = 0.85
        self.bounce = 0.3
        
        # Initialize sand particles randomly
        self.reset_sand()
        
        # Calibrate gyroscope
        self.sense.set_imu_config(True, True, True)  # Enable gyro, accel, mag
        time.sleep(0.1)
        
    def reset_sand(self):
        """Initialize sand particles at the top of the screen"""
        self.particles = []
        for _ in range(self.num_particles):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height * 0.3)  # Start in upper third
            vx = random.uniform(-0.5, 0.5)
            vy = random.uniform(0, 0.5)
            self.particles.append([x, y, vx, vy])
    
    def get_rotation_vector(self):
        """Get rotation vector from gyroscope"""
        orientation = self.sense.get_gyroscope_raw()
        # Convert to rotation vector (x, y, z)
        # We'll use x and y for horizontal movement, z for vertical
        return {
            'x': orientation['x'] / 100.0,  # Scale down for smoother movement
            'y': orientation['y'] / 100.0,
            'z': orientation['z'] / 100.0
        }
    
    def update_particles(self, rotation):
        """Update particle positions based on rotation and physics"""
        # Apply rotation forces
        force_x = rotation['y']  # Pitch affects x movement
        force_y = rotation['x']  # Roll affects y movement
        force_z = rotation['z']  # Yaw affects both
        
        for particle in self.particles:
            x, y, vx, vy = particle
            
            # Apply rotation forces
            vx += force_x + force_z * 0.3
            vy += force_y + force_z * 0.3
            
            # Apply gravity (adjusted by rotation)
            vy += self.gravity + abs(rotation['x']) * 0.2
            
            # Apply friction
            vx *= self.friction
            vy *= self.friction
            
            # Update position
            x += vx
            y += vy
            
            # Boundary collision detection
            if x < 0:
                x = 0
                vx *= -self.bounce
            elif x >= self.width:
                x = self.width - 0.1
                vx *= -self.bounce
            
            if y < 0:
                y = 0
                vy *= -self.bounce
            elif y >= self.height:
                y = self.height - 0.1
                vy *= -self.bounce
            
            # Update particle
            particle[0] = x
            particle[1] = y
            particle[2] = vx
            particle[3] = vy
    
    def render(self):
        """Render particles on the LED matrix"""
        # Create a brightness map for the display
        display = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Add particles to display with falloff
        for particle in self.particles:
            x, y = int(particle[0]), int(particle[1])
            
            # Ensure within bounds
            if 0 <= x < self.width and 0 <= y < self.height:
                # Add brightness at particle location
                display[y][x] = min(255, display[y][x] + 200)
                
                # Add some glow to nearby pixels
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            distance = math.sqrt(dx*dx + dy*dy)
                            if distance > 0:
                                glow = int(100 / (distance + 1))
                                display[ny][nx] = min(255, display[ny][nx] + glow)
        
        # Convert to Sense HAT pixel format and render
        pixels = []
        for y in range(self.height):
            for x in range(self.width):
                brightness = display[y][x]
                # Sand color: warm yellow/orange
                r = min(255, int(brightness * 0.9))
                g = min(255, int(brightness * 0.7))
                b = min(255, int(brightness * 0.3))
                pixels.append([r, g, b])
        
        self.sense.set_pixels(pixels)
    
    def run(self):
        """Main simulation loop"""
        print("Sand Simulation Started!")
        print("Tilt the Raspberry Pi to move the sand around")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                # Get rotation data
                rotation = self.get_rotation_vector()
                
                # Update physics
                self.update_particles(rotation)
                
                # Render
                self.render()
                
                # Control frame rate
                time.sleep(0.05)  # ~20 FPS
                
        except KeyboardInterrupt:
            print("\nExiting...")
            self.sense.clear()

def main():
    sim = SandSimulation()
    sim.run()

if __name__ == "__main__":
    main()


