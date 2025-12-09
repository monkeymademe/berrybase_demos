#!/usr/bin/env python3
"""
Web application for streaming camera feed with Hailo AI object detection.
Uses Flask for the web UI and picamera2 for camera streaming.
"""

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
from picamera2.devices import Hailo
import cv2
import numpy as np
import io
import threading
import time
import subprocess
import os
import signal
import sys
import json
from PIL import Image

app = Flask(__name__)

# Camera and streaming setup
picam2 = None
hailo = None
camera_lock = threading.Lock()
streaming_output = None

def cleanup_camera():
    """Clean up camera and Hailo resources"""
    global picam2, hailo
    with camera_lock:
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
                picam2 = None
                print("Camera closed.")
            except:
                pass
        if hailo is not None:
            try:
                hailo.close()
                hailo = None
                print("Hailo closed.")
            except:
                pass

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal, cleaning up...")
    cleanup_camera()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Detection results are now computed directly in the frame capture thread

# COCO class names for YOLOv8
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def stop_camera_services():
    """Stop pipewire and wireplumber services that may be using the camera"""
    print("Stopping pipewire and wireplumber services...")
    try:
        # Stop systemd user services
        subprocess.run(['systemctl', '--user', 'stop', 
                       'pipewire.service', 'pipewire.socket', 
                       'pipewire-pulse.socket', 'wireplumber.service'],
                      capture_output=True, timeout=5, check=False)
        
        # Also kill any remaining processes (more aggressive)
        subprocess.run(['pkill', '-9', '-f', 'pipewire'], 
                      capture_output=True, timeout=2, check=False)
        subprocess.run(['pkill', '-9', '-f', 'wireplumber'], 
                      capture_output=True, timeout=2, check=False)
        
        # Wait longer and check if devices are free
        for i in range(5):
            time.sleep(0.5)
            try:
                # Check if any process is using media devices
                result = subprocess.run(['fuser', '/dev/media*'], 
                                      capture_output=True, timeout=1, check=False)
                if result.returncode != 0:  # fuser returns non-zero if no processes found
                    print("✓ Camera services stopped and devices are free")
                    return True
            except:
                pass
        
        # Final check with lsof
        try:
            result = subprocess.run(['lsof', '/dev/media*'], 
                                  capture_output=True, timeout=1, check=False)
            if result.returncode != 0 or len(result.stdout) == 0:
                print("✓ Camera services stopped")
                return True
            else:
                print("⚠ Warning: Some processes may still be using camera devices")
                print("  You may need to manually stop pipewire/wireplumber")
                return False
        except:
            print("✓ Camera services stopped (could not verify)")
            return True
            
    except Exception as e:
        print(f"Warning: Could not stop camera services: {e}")
        return False


def extract_detections(hailo_output, w, h, class_names, threshold=0.4):
    """
    Extract detections from the HailoRT-postprocess output.
    Based on picamera2 example: https://github.com/raspberrypi/picamera2/blob/main/examples/hailo/detect.py
    
    Args:
        hailo_output: Output from hailo.run() - list of detections per class
        w: Image width for coordinate scaling
        h: Image height for coordinate scaling
        class_names: List of class names
        threshold: Confidence threshold (default 0.4 to match JSON config)
    
    Returns:
        List of detections in format: [x1, y1, x2, y2, class_id, confidence]
    """
    results = []
    if hailo_output is None:
        return results
    
    # Hailo output is a list where each element is detections for that class
    for class_id, detections in enumerate(hailo_output):
        if class_id >= len(class_names):
            continue
        for detection in detections:
            if len(detection) < 5:
                continue
            score = detection[4]
            if score >= threshold:
                # Hailo format: [y0, x0, y1, x1, score]
                y0, x0, y1, x1 = detection[:4]
                # Convert normalized coordinates to pixel coordinates
                x1_px = int(x0 * w)
                y1_px = int(y0 * h)
                x2_px = int(x1 * w)
                y2_px = int(y1 * h)
                # Return in format: [x1, y1, x2, y2, class_id, confidence]
                results.append([x1_px, y1_px, x2_px, y2_px, class_id, score])
    return results


def draw_detections(image, detections, class_names=COCO_CLASSES):
    """
    Draw bounding boxes and labels for detected objects.
    Handles normalized (0-1) and pixel coordinates.
    """
    if detections is None or len(detections) == 0:
        return image
    
    h, w = image.shape[:2]
    drawn_count = 0
    
    for detection in detections:
        try:
            # Parse detection format: could be [x1, y1, x2, y2, class_id, confidence] or similar
            if isinstance(detection, (list, tuple, np.ndarray)):
                det_array = np.array(detection)
                
                if len(det_array) < 4:
                    continue
                
                x1, y1, x2, y2 = det_array[0], det_array[1], det_array[2], det_array[3]
                
                # Get class ID and confidence if available
                class_id = int(det_array[4]) if len(det_array) > 4 else 0
                confidence = float(det_array[5]) if len(det_array) > 5 else 1.0
                
                # Handle normalized coordinates (0-1)
                if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                else:
                    # Pixel coordinates - ensure they're integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Validate bounding box
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                if x2 > x1 and y2 > y1:
                    # Get class name
                    if 0 <= class_id < len(class_names):
                        class_name = class_names[class_id]
                    else:
                        class_name = f"Class {class_id}"
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green in BGR
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with class name and confidence as percentage
                    confidence_pct = int(confidence * 100)
                    label = f"{class_name} {confidence_pct}%"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = max(y1, label_size[1] + 10)
                    
                    # Draw label background
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(image, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    drawn_count += 1
        except Exception as e:
            print(f"Error drawing detection: {e}")
            continue
    
    return image


def start_camera():
    """Initialize and start the camera with object detection using Hailo directly"""
    global picam2, hailo, streaming_output
    
    # Stop pipewire/wireplumber before accessing camera
    stop_camera_services()
    
    # Check for any existing camera processes or app instances FIRST
    try:
        # Check for existing app.py processes (excluding this one)
        result = subprocess.run(['pgrep', '-f', 'python.*app.py'], 
                              capture_output=True, timeout=1, check=False)
        if result.returncode == 0:
            pids = result.stdout.decode().strip().split('\n')
            current_pid = str(os.getpid())
            for pid in pids:
                if pid and pid != current_pid:
                    print(f"⚠ Warning: Found existing app.py process (PID {pid}), killing it...")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                    except:
                        pass
        
        # Check for rpicam processes
        result = subprocess.run(['pgrep', '-f', 'rpicam'], 
                              capture_output=True, timeout=1, check=False)
        if result.returncode == 0:
            print("⚠ Warning: Found existing rpicam processes, killing them...")
            subprocess.run(['pkill', '-9', '-f', 'rpicam'], 
                          capture_output=True, timeout=2, check=False)
            time.sleep(1)
    except:
        pass
    
    # Clean up any existing camera instance (BEFORE initializing Hailo)
    # This only cleans up camera, not Hailo (hailo is None at this point)
    with camera_lock:
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
                picam2 = None
                print("Camera closed.")
            except:
                pass
    time.sleep(0.5)
    
    # Initialize Hailo with YOLOv8 model (AFTER cleanup)
    # Use H8 model (not H8L) for better performance on Hailo8 device
    hailo_model_path = '/usr/share/hailo-models/yolov8s_h8.hef'
    if not os.path.exists(hailo_model_path):
        # Fallback to H8L model if H8 not available
        hailo_model_path = '/usr/share/hailo-models/yolov8s_h8l.hef'
        if not os.path.exists(hailo_model_path):
            print(f"⚠ ERROR: Hailo model not found")
            return False
        print("⚠ Warning: Using H8L model (compiled for Hailo8L, may have lower performance)")
    else:
        print("✓ Using H8 model (optimized for Hailo8 device)")
    
    print("Initializing Hailo...")
    try:
        hailo = Hailo(hailo_model_path)
        model_h, model_w, _ = hailo.get_input_shape()
        print(f"✓ Hailo initialized. Model input shape: {model_w}x{model_h}")
    except Exception as e:
        print(f"⚠ ERROR: Failed to initialize Hailo: {e}")
        return False
    
    # Retry camera initialization up to 3 times
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with camera_lock:
                if attempt > 0:
                    print(f"Retrying camera initialization (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    stop_camera_services()  # Stop services again before retry
                
                picam2 = Picamera2()
                break  # Success, exit retry loop
        except RuntimeError as e:
            if "busy" in str(e).lower() or "Pipeline handler in use" in str(e):
                if attempt < max_retries - 1:
                    print(f"Camera busy, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"Error: Camera still busy after {max_retries} attempts")
                    print("Please ensure no other applications are using the camera")
                    return False
            else:
                raise  # Re-raise if it's a different error
    else:
        print("Error: Failed to initialize camera after retries")
        return False
    
    try:
        # Configure camera - lores must match model input size (640x640)
        # Main stream can be higher resolution for better display quality
        # Use 1280x720 (HD) or 1920x1080 (Full HD) for main stream
        # Lores stays at 640x640 for Hailo inference
        
        # Try higher resolution first, fall back if not supported
        video_w, video_h = 1280, 720  # HD resolution for display
        
        main = {'size': (video_w, video_h), 'format': 'RGB888'}
        lores = {'size': (model_w, model_h), 'format': 'RGB888'}  # Must match model input (640x640)
        controls = {'FrameRate': 30}
        
        try:
            config = picam2.create_video_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)
            print(f"✓ Camera configured: main={video_w}x{video_h} (HD), lores={model_w}x{model_h}")
        except Exception as e:
            # Fall back to lower resolution if HD not supported
            print(f"⚠ HD resolution not supported, trying 960x720: {e}")
            video_w, video_h = 960, 720
            main = {'size': (video_w, video_h), 'format': 'RGB888'}
            config = picam2.create_video_configuration(main, lores=lores, controls=controls)
            picam2.configure(config)
            print(f"✓ Camera configured: main={video_w}x{video_h}, lores={model_w}x{model_h}")
        
        picam2.start()
        print("✓ Camera started")
        
        # Start frame capture thread
        def frame_capture_thread():
                """Thread to capture frames and encode to MJPEG"""
                global streaming_output
                
                class StreamingOutput(io.BufferedIOBase):
                    def __init__(self):
                        self.frame = None
                        self.condition = threading.Condition()
                    
                    def write(self, buf):
                        with self.condition:
                            self.frame = buf
                            self.condition.notify_all()
                    
                    def read(self):
                        with self.condition:
                            self.condition.wait()
                            return self.frame
                
                streaming_output = StreamingOutput()
                frame_count = 0
                print("✓ StreamingOutput initialized in thread")
                
                # Write a test frame immediately so the stream starts
                test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.putText(test_frame, "Starting...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                test_pil = Image.fromarray(test_frame_rgb)
                test_buffer = io.BytesIO()
                test_pil.save(test_buffer, format='JPEG', quality=85)
                streaming_output.write(test_buffer.getvalue())
                print("✓ Test frame written to stream")
                
                while picam2 is not None:
                    try:
                        # Capture main stream frame for display first
                        request = picam2.capture_request()
                        frame = request.make_array("main")
                        
                        detections = []
                        
                        # Try Hailo inference if available
                        if hailo is not None:
                            try:
                                # Capture frame from lores stream (model input size) for inference
                                frame_lores = picam2.capture_array("lores")
                                # Run Hailo inference on the lores frame (this might be slow)
                                hailo_output = hailo.run(frame_lores)
                                # Extract detections from Hailo output
                                # Scale detections from lores (640x640) to main stream resolution
                                detections = extract_detections(hailo_output, video_w, video_h, COCO_CLASSES, threshold=0.4)
                            except Exception as hailo_error:
                                if frame_count < 5:
                                    print(f"⚠ Hailo inference error (frame {frame_count}): {hailo_error}")
                                detections = []
                        else:
                            # Hailo not available, continue without detections
                            if frame_count < 5:
                                print(f"⚠ Hailo not available, streaming without detection")
                        
                        # Fix colors: picamera2 RGB888 is actually BGR, so swap R and B channels
                        frame_swapped = frame.copy()
                        frame_swapped[:, :, [0, 2]] = frame_swapped[:, :, [2, 0]]
                        
                        # Convert to BGR for OpenCV drawing
                        frame_bgr = cv2.cvtColor(frame_swapped, cv2.COLOR_RGB2BGR)
                        
                        # Debug output (first 10 frames)
                        if frame_count < 10:
                            print(f"Frame {frame_count}: Processing frame, shape={frame.shape}")
                            if detections:
                                print(f"  Found {len(detections)} detections")
                            else:
                                print(f"  No detections (threshold=0.4)")
                        
                        # Draw test rectangle (first 10 frames) to verify drawing works
                        if frame_count < 10:
                            cv2.rectangle(frame_bgr, (10, 10), (200, 50), (0, 255, 255), 2)
                            cv2.putText(frame_bgr, f"Frame {frame_count}", (15, 35),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Draw detections on the frame (ALWAYS try to draw if we have detections)
                        if detections is not None and len(detections) > 0:
                            frame_bgr = draw_detections(frame_bgr, detections)
                            if frame_count < 10:
                                print(f"  Drawing {len(detections)} detection(s)")
                        
                        # Convert back to RGB for PIL
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        
                        # Encode to JPEG
                        try:
                            pil_image = Image.fromarray(frame_rgb)
                            jpeg_buffer = io.BytesIO()
                            pil_image.save(jpeg_buffer, format='JPEG', quality=85)
                            jpeg_bytes = jpeg_buffer.getvalue()
                            
                            # Write to streaming output
                            streaming_output.write(jpeg_bytes)
                            
                            if frame_count == 0:
                                print(f"✓ First frame written successfully, {len(jpeg_bytes)} bytes")
                        except Exception as encode_error:
                            print(f"⚠ Frame encoding error: {encode_error}")
                            import traceback
                            traceback.print_exc()
                            request.release()
                            continue
                        
                        # Always write a frame, even if no detections
                        if frame_count == 0:
                            print(f"✓ Writing first frame to stream, size={len(jpeg_bytes)} bytes")
                        
                        # Release request after we're done with it
                        request.release()
                        frame_count += 1
                        
                        # Print detection count every 30 frames
                        if frame_count % 30 == 0 and detections is not None:
                            detected_classes = []
                            for det in detections:
                                if isinstance(det, (list, tuple, np.ndarray)) and len(det) > 4:
                                    class_id = int(det[4])
                                    if 0 <= class_id < len(COCO_CLASSES):
                                        detected_classes.append(COCO_CLASSES[class_id])
                            
                            if detected_classes:
                                unique_classes = list(set(detected_classes))
                                print(f"Live detections: {len(detections)} objects detected [{', '.join(unique_classes)}]")
                            else:
                                print(f"Live detections: {len(detections)} objects detected")
                        
                    except Exception as e:
                        print(f"Error in frame capture: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(0.1)
        
        # Start the frame capture thread
        thread = threading.Thread(target=frame_capture_thread, daemon=True)
        thread.start()
        print("✓ Frame capture thread started")
        
        # Give thread time to initialize streaming_output
        time.sleep(0.5)
        
        print("✓ Camera started with object detection")
        return True
            
    except Exception as e:
        print(f"Error starting camera: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/')
def index():
    """Main page with video stream"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Object Detection Camera Stream</title>
    <style>
        :root {
            --bg-gradient-start: #3e91be;
            --bg-gradient-end: #cd3c65;
            --bg-color: #f8f8f8;
            --card-bg: white;
            --text-primary: #333;
            --text-secondary: #666;
            --text-muted: #999;
            --text-white: white;
            --border-color: #f0f0f0;
        }

        [data-theme="dark"] {
            --bg-gradient-start: #6b6a6a;
            --bg-gradient-end: #cd3c65;
            --bg-color: #0f1419;
            --card-bg: #1a1f2e;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
            --text-muted: #808080;
            --text-white: #e0e0e0;
            --border-color: #2a2f3e;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 100%);
            min-height: 100vh;
            padding: 20px;
            color: var(--text-primary);
            transition: background 0.3s ease;
        }

        html[data-theme="dark"] body {
            background: linear-gradient(135deg, var(--bg-gradient-start) 0%, var(--bg-gradient-end) 90%, var(--bg-gradient-end) 100%);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            color: var(--text-white);
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 600;
        }

        .video-container {
            text-align: center;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: background 0.3s ease;
        }

        img {
            max-width: 100%;
            height: auto;
            border: 2px solid var(--border-color);
            border-radius: 10px;
            transition: border-color 0.3s ease;
        }

        .info {
            margin-top: 20px;
            text-align: center;
            color: var(--text-white);
            opacity: 0.9;
            font-size: 0.9em;
        }

        .theme-toggle {
            position: fixed;
            top: calc(20px + 1.25em);
            right: 20px;
            background: transparent;
            border: none;
            padding: 0;
            width: auto;
            height: auto;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: opacity 0.3s ease;
            z-index: 1000;
            transform: translateY(-50%);
        }

        .theme-toggle:hover {
            opacity: 0.7;
        }

        .theme-toggle svg {
            width: 28px;
            height: 28px;
            fill: white;
        }
    </style>
</head>
<body>
    <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle dark mode">
        <svg id="themeIcon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 16 16">
            <path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>
        </svg>
    </button>

    <div class="container">
        <h1>🤖 AI Object Detection Camera Stream</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Video Stream">
        </div>
        <div class="info">
            <p>Detected objects are shown with green bounding boxes and labels.</p>
        </div>
    </div>

    <script>
        // Dark mode toggle
        function toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update toggle icon - show icon for the NEW theme
            updateThemeIcon(newTheme);
        }

        // Update theme icon based on current theme
        function updateThemeIcon(theme) {
            const svg = document.getElementById('themeIcon');
            if (!svg) return;
            
            if (theme === 'dark') {
                // Show sun icon when dark mode is ON (clicking will switch to light)
                svg.innerHTML = '<path d="M8 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8M8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0m0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13m8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5M3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8m10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0m-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0m9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707M4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708"/>';
            } else {
                // Show moon-fill icon when light mode is ON (clicking will switch to dark)
                svg.innerHTML = '<path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>';
            }
        }

        // Initialize theme from localStorage
        function initTheme() {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeIcon(savedTheme);
        }

        // Initialize on page load
        window.addEventListener('DOMContentLoaded', () => {
            initTheme();
        });
    </script>
</body>
</html>
    ''')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print("✓ Video feed route accessed")
    def generate():
        print("✓ Video feed generator started")
        frame_count = 0
        while True:
            if streaming_output is not None:
                try:
                    frame = streaming_output.read()
                    if frame:
                        frame_count += 1
                        if frame_count <= 3:
                            print(f"✓ Streaming frame {frame_count}, size={len(frame)} bytes")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        if frame_count == 0:
                            print("⚠ No frame data in read()")
                        time.sleep(0.01)
                except Exception as e:
                    print(f"⚠ Error in video feed generator: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
            else:
                if frame_count == 0:
                    print("⚠ streaming_output is None, waiting...")
                time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("=" * 60)
    print("Starting AI Object Detection Web Application")
    print("=" * 60)
    
    if start_camera():
        print("\n✓ Camera initialized successfully")
        print("✓ Starting Flask web server...")
        print("\nAccess the web UI at: http://<raspberry-pi-ip>:5000")
        print("=" * 60)
        
        try:
            # Disable reloader to prevent camera conflicts on code changes
            # The reloader spawns a parent process that keeps the camera open
            app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cleanup_camera()
    else:
        print("Failed to start camera. Exiting.")

