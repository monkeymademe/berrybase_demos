#!/usr/bin/env python3
"""
Convert GIF to optimized RGB565 frames for RP2040 LCD display.

Usage:
    python3 convert_gif.py input.gif output_folder num_frames

Examples:
    python3 convert_gif.py hamster.gif frames_opt 21
    python3 convert_gif.py hamster2.gif frames2_opt21 21
"""

from PIL import Image, ImageSequence
import os
import sys

def rgb_to_rgb565(img):
    """Convert PIL RGB image to RGB565 bytes for RP2040 LCD.
    Packed as: Red(5 bits) | Green(6 bits) | Blue(5 bits)
    Written as big-endian (MSB first) to match display format.
    """
    img = img.convert("RGB").resize((240, 240))
    buf = bytearray()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b = img.getpixel((x, y))
            r5 = (r >> 3) & 0x1F
            g6 = (g >> 2) & 0x3F
            b5 = (b >> 3) & 0x1F
            val = (r5 << 11) | (g6 << 5) | b5
            buf.append((val >> 8) & 0xFF)
            buf.append(val & 0xFF)
    return buf


def convert_gif(input_gif, output_folder, num_frames=21):
    """Convert GIF to optimized RGB565 frames.
    
    Args:
        input_gif: Path to input GIF file
        output_folder: Output directory for frames
        num_frames: Number of frames to extract (evenly sampled)
    """
    if not os.path.exists(input_gif):
        print(f"Error: Input file '{input_gif}' not found!")
        return False
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Open and collect all frames
    im = Image.open(input_gif)
    all_frames = []
    for frame in ImageSequence.Iterator(im):
        all_frames.append(frame.copy())
    
    N = len(all_frames)
    if N == 0:
        print(f"Error: No frames found in '{input_gif}'!")
        return False
    
    print(f"Total frames in {input_gif}: {N}")
    
    # Evenly sample frames across the animation
    if N >= num_frames:
        indexes = [round(i * (N - 1) / (num_frames - 1)) for i in range(num_frames)]
    else:
        indexes = list(range(N))
        print(f"Warning: GIF has only {N} frames, using all of them")
    
    print(f"Exporting {len(indexes)} frames to {output_folder}/")
    
    # Convert and save frames
    for i, idx in enumerate(indexes):
        frame = all_frames[idx].convert('RGB').resize((240, 240))
        data = rgb_to_rgb565(frame)
        out_path = os.path.join(output_folder, f'frame{i:03}.rgb565')
        with open(out_path, 'wb') as f:
            f.write(data)
        print(f"  Saved frame {i:03} (from source frame {idx})")
    
    print(f"\nDone! Created {len(indexes)} frames in {output_folder}/")
    return True


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nExample:")
        print("  python3 convert_gif.py hamster.gif frames_opt 21")
        sys.exit(1)
    
    input_gif = sys.argv[1]
    output_folder = sys.argv[2]
    num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 21
    
    success = convert_gif(input_gif, output_folder, num_frames)
    sys.exit(0 if success else 1)

