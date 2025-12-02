"""
Diagnostic script to check what modules are available on Calliope mini MicroPython
Run this first to see what imports work
"""

print("Checking available modules...")
print()

# Check machine module
try:
    import machine
    print("✓ machine module available")
    print(f"  Contents: {dir(machine)}")
    try:
        from machine import Pin
        print("  ✓ Pin class available")
    except ImportError as e:
        print(f"  ✗ Pin class not available: {e}")
except ImportError as e:
    print(f"✗ machine module not available: {e}")

print()

# Check calliopemini module
try:
    import calliopemini
    print("✓ calliopemini module available")
    print(f"  Contents: {dir(calliopemini)}")
except ImportError as e:
    print(f"✗ calliopemini module not available: {e}")

print()

# Check neopixel module
try:
    import neopixel
    print("✓ neopixel module available")
    print(f"  Contents: {dir(neopixel)}")
except ImportError as e:
    print(f"✗ neopixel module not available: {e}")

print()

# Check for pin objects
print("Checking for pin objects...")
try:
    from calliopemini import *
    print("✓ calliopemini.* imported")
    # Check for common pin names
    pin_names = ['pin0', 'pin8', 'pin9', 'pin13', 'pin14', 'pin_RGB']
    for name in pin_names:
        try:
            pin = eval(name)
            print(f"  ✓ {name} available: {pin}")
        except:
            print(f"  ✗ {name} not available")
except Exception as e:
    print(f"✗ Could not import calliopemini.*: {e}")

print()
print("Check complete!")

