#!/usr/bin/env python3
"""
Sand Simulation for Raspberry Pi Sense HAT
Simulates sand particles that respond to accelerometer tilt (like Adafruit PixelDust)
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
        
        # Number of sand particles - balanced for 8x8 display
        self.num_particles = 45  # Reduced slightly to prevent overcrowding
        
        # Physics constants
        self.gravity = 0.3
        self.friction = 0.95  # Very low friction for fluid movement
        self.bounce = 0.4  # Slightly more bounce for livelier movement
        
        # Shake detection
        self.prev_accel = {'x': 0, 'y': 0, 'z': 0}
        self.shake_threshold = 2.0  # Threshold for detecting shake
        self.shake_decay = 0.95  # How quickly shake effect decays
        self.shake_active = False  # Track if currently shaking
        
        # Initialize sand particles randomly
        self.reset_sand()
        
        # Calibrate IMU (enable accelerometer for tilt detection)
        self.sense.set_imu_config(True, True, True)  # Enable gyro, accel, mag
        time.sleep(0.1)
        
        # Test accelerometer reading (measures tilt/gravity direction)
        test_accel = self.sense.get_accelerometer_raw()
        print(f"Sensor initialized. Test accelerometer reading: x={test_accel['x']:.2f}, y={test_accel['y']:.2f}, z={test_accel['z']:.2f}")
        print("(Values should change when you tilt the device)")
        
    def reset_sand(self):
        """Initialize sand particles at the top of the screen (like hourglass)
        Ensures particles start spread out to prevent initial clustering"""
        self.particles = []
        # Use grid-based initial placement to ensure even distribution
        particles_per_row = int(math.sqrt(self.num_particles)) + 1
        
        for i in range(self.num_particles):
            # Distribute more evenly across the top portion
            row = (i // particles_per_row) % int(self.height * 0.5)
            col = i % particles_per_row
            
            # Spread across width
            x = (col / particles_per_row) * self.width + random.uniform(-0.2, 0.2)
            y = row * 0.5 + random.uniform(0, 0.3)  # Start in upper portion
            
            # Ensure within bounds
            x = max(0.1, min(self.width - 0.1, x))
            y = max(0.1, min(self.height * 0.4, y))
            
            vx = random.uniform(-0.2, 0.2)
            vy = random.uniform(0, 0.2)
            self.particles.append([x, y, vx, vy])
    
    def get_acceleration_vector(self, debug=False):
        """Get acceleration vector from accelerometer (like PixelDust)
        Accelerometer measures tilt/gravity direction - perfect for sand simulation"""
        accel = self.sense.get_accelerometer_raw()
        
        # Detect shake by measuring sudden changes in acceleration
        accel_change_x = abs(accel['x'] - self.prev_accel['x'])
        accel_change_y = abs(accel['y'] - self.prev_accel['y'])
        accel_change_z = abs(accel['z'] - self.prev_accel['z'])
        
        # Total change magnitude
        shake_magnitude = math.sqrt(accel_change_x**2 + accel_change_y**2 + accel_change_z**2)
        is_shaking = shake_magnitude > self.shake_threshold
        
        # Update previous values
        self.prev_accel = accel.copy()
        
        # Accelerometer values represent gravity direction
        # When flat: z ≈ 1.0, x ≈ 0, y ≈ 0
        # When tilted: x and y values indicate tilt direction
        # These values directly represent the force direction for sand
        
        # Scale and invert for intuitive movement (tilt right = sand moves right)
        # PixelDust uses accelerometer X and Y directly as forces
        acceleration = {
            'x': accel['x'] * 0.5,  # Tilt left/right affects horizontal movement
            'y': accel['y'] * 0.5,  # Tilt forward/back affects vertical movement  
            'z': accel['z'],        # Z component (gravity strength)
            'shake': shake_magnitude if is_shaking else 0.0  # Shake intensity
        }
        
        if debug and is_shaking:
            print(f"SHAKE DETECTED! Magnitude: {shake_magnitude:.2f}")
        
        if debug:
            print(f"Raw accel: x={accel['x']:.2f}, y={accel['y']:.2f}, z={accel['z']:.2f}")
            print(f"Acceleration forces: x={acceleration['x']:.4f}, y={acceleration['y']:.4f}, z={acceleration['z']:.4f}")
        
        return acceleration
    
    def check_collision(self, x, y, exclude_index=-1):
        """Check if a position collides with another particle
        Returns True if collision detected, False otherwise"""
        collision_threshold = 0.55  # Allow particles to get closer before collision
        
        # Only check nearby particles to reduce computation and prevent over-clustering
        for i, other_particle in enumerate(self.particles):
            if i == exclude_index:
                continue
            
            ox, oy = other_particle[0], other_particle[1]
            # Quick distance check first
            dx = abs(x - ox)
            dy = abs(y - oy)
            if dx > collision_threshold or dy > collision_threshold:
                continue  # Skip if clearly far away
            
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < collision_threshold:
                return True, (ox, oy)
        
        return False, None
    
    def resolve_collision(self, x, y, vx, vy, other_pos):
        """Resolve collision by pushing particles apart smoothly to prevent clustering"""
        ox, oy = other_pos
        
        # Calculate direction from other particle to this one
        dx = x - ox
        dy = y - oy
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 0.01:  # Too close, add moderate random offset
            dx = random.uniform(-0.6, 0.6)
            dy = random.uniform(-0.6, 0.6)
            distance = math.sqrt(dx**2 + dy**2)
        
        # Normalize direction
        if distance > 0:
            dx /= distance
            dy /= distance
        
        # Gentle push apart to prevent clustering
        push_strength = 0.2  # Reduced further
        x += dx * push_strength
        y += dy * push_strength
        
        # Gentle repulsion velocity
        repulsion_strength = 0.25  # Reduced further
        vx += dx * repulsion_strength
        vy += dy * repulsion_strength
        
        # Hourglass behavior: particles slide off each other smoothly
        if abs(dy) > abs(dx):  # More vertical collision
            if dy < -0.2:  # This particle is above the other
                # Horizontal sliding (sand falling over)
                vx += dx * 0.3  # Reduced from 0.6
                # Reduce vertical velocity (hitting something below)
                vy *= 0.7  # Less aggressive
            else:  # This particle is below the other
                # Can slide horizontally
                vx += dx * 0.25
                vy *= 0.8
        else:  # More horizontal collision
            # Push apart horizontally
            vx += dx * 0.4  # Reduced from 0.8
            # Allow vertical movement
            vy += dy * 0.2  # Reduced from 0.4
        
        return x, y, vx, vy
    
    def update_particles(self, acceleration):
        """Update particle positions based on accelerometer tilt and physics
        (Similar to PixelDust's iterate() method)"""
        # Accelerometer X and Y directly represent tilt direction
        # These values are the force direction for sand movement
        force_x = acceleration['x']  # Tilt left/right
        force_y = acceleration['y']  # Tilt forward/back
        
        # Gravity strength from Z component (when tilted, gravity is weaker in Z)
        # Use this to adjust base gravity
        gravity_strength = max(0.1, acceleration['z']) * self.gravity
        
        # Shake effect - add random forces when shaking
        shake_intensity = acceleration.get('shake', 0.0)
        shake_force_multiplier = min(shake_intensity * 0.5, 3.0)  # Cap shake force
        
        # Update particles in random order for more natural collisions
        particle_indices = list(range(len(self.particles)))
        random.shuffle(particle_indices)
        
        for i in particle_indices:
            particle = self.particles[i]
            x, y, vx, vy = particle
            
            # Apply tilt forces directly (like PixelDust)
            # Balanced forces for natural movement
            vx += force_x * 0.7
            vy += force_y * 0.7
            
            # Apply shake forces - make particles dance!
            if shake_intensity > 0:
                # Add random forces in all directions when shaking
                vx += random.uniform(-shake_force_multiplier, shake_force_multiplier)
                vy += random.uniform(-shake_force_multiplier, shake_force_multiplier)
            
            # Apply gravity (always downward, adjusted by tilt)
            # Reduce gravity when near bottom to allow repulsion to push particles up
            gravity_multiplier = 1.0
            if y > self.height - 0.5:  # Near bottom
                gravity_multiplier = 0.3  # Much less gravity to allow repulsion to work
            
            vy += gravity_strength * gravity_multiplier
            
            # Apply friction (very low for fluid movement)
            vx *= self.friction
            vy *= self.friction
            
            # Add repulsion force from nearby particles to prevent clustering
            # Stronger repulsion when near bottom to prevent clustering
            repulsion_range = 0.8
            repulsion_force = 0.12
            
            # Check if particle is near bottom (where clustering happens)
            near_bottom = y > self.height - 0.5
            
            for j, other_particle in enumerate(self.particles):
                if i == j:
                    continue
                
                ox, oy = other_particle[0], other_particle[1]
                dx = x - ox
                dy = y - oy
                distance = math.sqrt(dx**2 + dy**2)
                
                if 0 < distance < repulsion_range:
                    # Normalize and apply repulsion
                    if distance > 0:
                        dx /= distance
                        dy /= distance
                    
                    repulsion = repulsion_force * (1.0 - distance / repulsion_range)
                    
                    # MUCH stronger repulsion when near bottom to push particles up
                    if near_bottom:
                        if dy < 0:  # Other particle is above
                            # Strong upward repulsion to prevent bottom clustering
                            repulsion *= 3.0  # Much stronger upward push
                        else:
                            # Stronger horizontal repulsion too
                            repulsion *= 1.5
                    
                    vx += dx * repulsion
                    vy += dy * repulsion
            
            # Additional upward force when near bottom to counteract gravity
            if near_bottom:
                # Push particles upward to prevent clustering
                upward_force = 0.2
                vy -= upward_force  # Negative y is upward
            
            # Calculate new position
            new_x = x + vx
            new_y = y + vy
            
            # Check for particle collisions before moving
            # Only check collision if moving significantly to avoid over-checking
            collision_count = 0
            max_collisions = 3  # Limit collisions per frame to prevent clustering
            
            if abs(vx) > 0.01 or abs(vy) > 0.01:
                collision, other_pos = self.check_collision(new_x, new_y, exclude_index=i)
                
                if collision and collision_count < max_collisions:
                    # Resolve collision - particles push apart smoothly
                    new_x, new_y, vx, vy = self.resolve_collision(new_x, new_y, vx, vy, other_pos)
                    collision_count += 1
                    # Add small random offset only if still very close
                    if abs(new_x - other_pos[0]) < 0.2 and abs(new_y - other_pos[1]) < 0.2:
                        new_x += random.uniform(-0.2, 0.2)
                        new_y += random.uniform(-0.2, 0.2)
            
            # Update position
            x = new_x
            y = new_y
            
            # Simple boundary collision detection - keep particles visible
            # Allow full range [0, width) to access all pixels [0, width-1]
            if x < 0:
                x = 0.01
                vx *= -0.5  # Gentle bounce
            elif x >= self.width:
                x = self.width - 0.01  # Keep inside boundary
                vx *= -0.5
            
            if y < 0:
                y = 0.01
                vy *= -0.5
            elif y >= self.height:
                # CRITICAL: Prevent particles from clustering at bottom
                # Push them up slightly and allow upward movement
                y = self.height - 0.05  # Push up more to prevent clustering
                # If moving up, allow it; if moving down, reverse and reduce
                if vy > 0:  # Moving down
                    vy *= -0.7  # Reverse and reduce downward velocity
                else:  # Moving up
                    vy *= 0.9  # Allow upward movement
            
            # Ensure particles stay within bounds [0, width) for full pixel coverage
            # Use tighter bounds to prevent edge clustering
            x = max(0.01, min(self.width - 0.01, x))
            y = max(0.01, min(self.height - 0.05, y))  # Keep away from bottom edge
            
            # Update particle
            particle[0] = x
            particle[1] = y
            particle[2] = vx
            particle[3] = vy
    
    def render(self, debug=False, shake_active=False):
        """Render particles on the LED matrix with smooth sub-pixel rendering"""
        # Create a brightness map for the display
        display = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        
        particles_rendered = 0
        # Add particles to display with smooth sub-pixel rendering
        for particle in self.particles:
            x, y = particle[0], particle[1]
            
            # Clamp coordinates to valid range (safety check)
            x = max(0.0, min(self.width - 0.001, x))
            y = max(0.0, min(self.height - 0.001, y))
            
            # Check if particle is visible (allow full range including edges)
            if 0 <= x < self.width and 0 <= y < self.height:
                particles_rendered += 1
                
                # Get integer pixel coordinates - clamp to valid range [0, width-1] = [0, 7]
                # int() already floors for positive numbers, so this gives us pixel indices
                px = max(0, min(self.width - 1, int(x)))
                py = max(0, min(self.height - 1, int(y)))
                
                # Calculate sub-pixel offset for smooth interpolation
                fx, fy = x - px, y - py
                
                # Render particle with bilinear interpolation for smooth movement
                # This creates trails and smoother motion
                # Use full 3x3 range to ensure edge particles render properly
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        nx, ny = px + dx, py + dy
                        # Check bounds - this ensures we only render valid pixels [0-7]
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            # Calculate distance from particle center
                            dist_x = abs(dx - fx)
                            dist_y = abs(dy - fy)
                            distance = math.sqrt(dist_x**2 + dist_y**2)
                            
                            # Brightness falls off with distance (smooth interpolation)
                            if distance < 1.5:
                                brightness = max(0, 200 * (1.0 - distance / 1.5))
                                display[ny][nx] += brightness
        
        if debug:
            lit_pixels = sum(1 for row in display for val in row if val > 0)
            # Show which rows have lit pixels
            rows_lit = [y for y in range(self.height) if any(display[y][x] > 0 for x in range(self.width))]
            cols_lit = [x for x in range(self.width) if any(display[y][x] > 0 for y in range(self.height))]
            print(f"Rendering: {particles_rendered} particles visible, {lit_pixels} pixels lit")
            print(f"Rows with lit pixels: {rows_lit} (should be 0-7)")
            print(f"Cols with lit pixels: {cols_lit} (should be 0-7)")
            # Show pixel coordinate ranges
            if self.particles:
                px_coords = [max(0, min(self.width - 1, int(p[0]))) for p in self.particles]
                py_coords = [max(0, min(self.height - 1, int(p[1]))) for p in self.particles]
                print(f"Particle X range: {min(px_coords)} to {max(px_coords)} (pixels 0-7)")
                print(f"Particle Y range: {min(py_coords)} to {max(py_coords)} (pixels 0-7)")
                # Show actual coordinate ranges
                x_coords = [p[0] for p in self.particles]
                y_coords = [p[1] for p in self.particles]
                print(f"Actual X coord range: {min(x_coords):.2f} to {max(x_coords):.2f} (should be 0.0-7.999)")
                print(f"Actual Y coord range: {min(y_coords):.2f} to {max(y_coords):.2f} (should be 0.0-7.999)")
            
            # Visual ASCII representation of LED matrix
            print("\nLED Matrix Visualization (8x8):")
            print("   " + " ".join([str(i) for i in range(self.width)]))
            for y in range(self.height):
                row_str = f"{y} "
                for x in range(self.width):
                    brightness = display[y][x]
                    if brightness > 200:
                        row_str += "██"  # Very bright
                    elif brightness > 100:
                        row_str += "▓▓"  # Medium bright
                    elif brightness > 50:
                        row_str += "▒▒"  # Dim
                    elif brightness > 0:
                        row_str += "░░"  # Very dim
                    else:
                        row_str += "  "  # Off
                print(row_str)
            print()  # Empty line after matrix
        
        # Convert to Sense HAT pixel format and render
        # Sense HAT expects a flat list of RGB tuples (r, g, b) in row-major order
        pixels = []
        for y in range(self.height):
            for x in range(self.width):
                brightness = min(255.0, display[y][x])  # Cap brightness
                
                # More vibrant colors when shaking (dance mode!)
                if shake_active:
                    # Vibrant, colorful effect when shaking
                    # Mix colors based on position for rainbow effect
                    color_shift = (x + y) % 3
                    if color_shift == 0:
                        r = min(255, int(brightness * 1.0))
                        g = min(255, int(brightness * 0.5))
                        b = min(255, int(brightness * 0.2))
                    elif color_shift == 1:
                        r = min(255, int(brightness * 0.5))
                        g = min(255, int(brightness * 1.0))
                        b = min(255, int(brightness * 0.3))
                    else:
                        r = min(255, int(brightness * 0.8))
                        g = min(255, int(brightness * 0.4))
                        b = min(255, int(brightness * 1.0))
                else:
                    # Sand color: warm yellow/orange with smooth gradients
                    r = min(255, int(brightness * 0.9))
                    g = min(255, int(brightness * 0.7))
                    b = min(255, int(brightness * 0.3))
                
                pixels.append((r, g, b))  # Use tuple instead of list
        
        self.sense.set_pixels(pixels)
    
    def clear_screen(self):
        """Clear terminal screen and move cursor to top"""
        print("\033[2J\033[H", end="")  # Clear screen and move to top
    
    def run(self):
        """Main simulation loop"""
        print("Sand Simulation Started!")
        print("Tilt the Raspberry Pi to move the sand around")
        print("Shake it to make the LEDs dance! 🎉")
        print("Press Ctrl+C to exit")
        print("\n" * 2)  # Some space before debug output
        
        frame_count = 0
        debug_interval = 20  # Print debug info every 20 frames
        first_debug = True  # Track if this is the first debug output
        
        try:
            while True:
                # Get accelerometer data (tilt/gravity direction)
                debug = (frame_count % debug_interval == 0)
                acceleration = self.get_acceleration_vector(debug=debug)
                
                # Track shake state
                self.shake_active = acceleration.get('shake', 0.0) > 0
                
                if debug:
                    # Clear screen and redraw (except first time to show startup message)
                    if not first_debug:
                        self.clear_screen()
                    else:
                        first_debug = False
                    
                    print("Sand Simulation - Debug Output")
                    print("=" * 50)
                    print(f"Applied forces: x={acceleration['x']:.4f}, y={acceleration['y']:.4f}, z={acceleration['z']:.4f}")
                    if self.shake_active:
                        print(f"🎉 SHAKING! Intensity: {acceleration.get('shake', 0.0):.2f}")
                    print(f"Total particles: {len(self.particles)}")
                    
                    # Count particles in bounds
                    in_bounds = sum(1 for p in self.particles if 0 <= p[0] < self.width and 0 <= p[1] < self.height)
                    print(f"Particles in bounds: {in_bounds}/{len(self.particles)}")
                    
                    # Show particle distribution
                    if self.particles:
                        x_positions = [p[0] for p in self.particles]
                        y_positions = [p[1] for p in self.particles]
                        print(f"X range: {min(x_positions):.2f} to {max(x_positions):.2f}")
                        print(f"Y range: {min(y_positions):.2f} to {max(y_positions):.2f}")
                        print("Sample particles (x, y, vx, vy):")
                        for i, p in enumerate(self.particles[:5]):
                            in_bounds = "✓" if 0 <= p[0] < self.width and 0 <= p[1] < self.height else "✗"
                            print(f"  Particle {i} {in_bounds}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.4f}, {p[3]:.4f})")
                    print("-" * 50)
                
                # Update physics (like PixelDust's iterate())
                self.update_particles(acceleration)
                
                # Render with shake effect
                self.render(debug=debug, shake_active=self.shake_active)
                
                frame_count += 1
                
                # Control frame rate - faster for smoother, more fluid movement
                time.sleep(0.03)  # ~33 FPS for smoother animation
                
        except KeyboardInterrupt:
            print("\nExiting...")
            self.sense.clear()

def main():
    sim = SandSimulation()
    sim.run()

if __name__ == "__main__":
    main()



