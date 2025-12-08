#!/usr/bin/env python3
"""
Merged AI Vision Application
- Object Detection mode: Uses Hailo YOLOv8 for object detection
- Face Recognition mode: Detects and recognizes faces
- Mode switcher: Switch between modes via /mode route
"""

from flask import Flask, Response, render_template_string, request, jsonify, redirect, url_for
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
import pickle
import base64
import datetime
from PIL import Image

app = Flask(__name__)

# Camera and streaming setup
picam2 = None
hailo = None
camera_lock = threading.Lock()
streaming_output = None
last_frame = None  # Keep last frame to send when camera is restarting
last_frame_lock = threading.Lock()

# Mode management
current_mode = 'object'  # 'object' or 'face'
mode_lock = threading.Lock()

# Face recognition storage
known_faces_file = 'known_faces.pkl'
known_faces = {}  # {face_id: {'name': str, 'embedding': np.array, 'image': np.array}}
face_id_counter = 0
match_debug_count = 0  # Debug counter for face matching
MAX_KNOWN_FACES = 5  # Maximum number of known faces to keep (FIFO - oldest removed when limit reached)

# Face detection/recognition models
face_detector = None
face_recognizer = None

# Capture request
capture_requested = False
capture_name = ""
capture_request_frame = 0  # Track when capture was requested (frame count)
detected_faces_for_selection = []  # Store detected faces with images for selection overlay
waiting_for_selection = False  # Flag to prevent auto-capture when waiting for user selection
detected_faces_lock = threading.Lock()

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

def cleanup_camera():
    """Clean up camera and Hailo resources"""
    global picam2, hailo, streaming_output
    with camera_lock:
        # Close streaming_output first to unblock any waiting reads
        if streaming_output is not None:
            try:
                if hasattr(streaming_output, 'mark_closed'):
                    streaming_output.mark_closed()
                streaming_output = None
                print("Streaming output closed.")
            except:
                pass
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

def load_known_faces():
    """Load known faces from disk"""
    global known_faces, face_id_counter
    if os.path.exists(known_faces_file):
        try:
            with open(known_faces_file, 'rb') as f:
                data = pickle.load(f)
                known_faces = data.get('faces', {})
                face_id_counter = data.get('counter', 0)
            print(f"✓ Loaded {len(known_faces)} known faces")
        except Exception as e:
            print(f"⚠ Error loading known faces: {e}")
            known_faces = {}
            face_id_counter = 0

def enforce_face_limit():
    """Enforce MAX_KNOWN_FACES limit by removing oldest face (lowest ID)"""
    global known_faces
    if len(known_faces) >= MAX_KNOWN_FACES:
        # Find the oldest face (lowest ID number)
        oldest_id = min(known_faces.keys())
        oldest_name = known_faces[oldest_id]['name']
        del known_faces[oldest_id]
        print(f"⚠ Removed oldest face '{oldest_name}' (ID: {oldest_id}) to maintain limit of {MAX_KNOWN_FACES} faces")

def save_known_faces():
    """Save known faces to disk"""
    try:
        with open(known_faces_file, 'wb') as f:
            pickle.dump({'faces': known_faces, 'counter': face_id_counter}, f)
        print(f"✓ Saved {len(known_faces)} known faces")
    except Exception as e:
        print(f"⚠ Error saving known faces: {e}")

def init_face_recognition():
    """Initialize face recognition models"""
    global face_detector, face_recognizer
    
    # Initialize OpenCV face detector (Haar Cascade)
    possible_paths = [
        '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml',
    ]
    
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            possible_paths.insert(0, cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    except:
        pass
    
    cascade_path = None
    for path in possible_paths:
        if os.path.exists(path):
            cascade_path = path
            break
    
    if not cascade_path:
        print("⚠ Error: Could not find Haar cascade file")
        return False
    
    try:
        face_detector = cv2.CascadeClassifier(cascade_path)
        if face_detector.empty():
            print("⚠ Warning: Could not load face detector cascade")
            return False
        print(f"✓ Face detector initialized (using {cascade_path})")
    except Exception as e:
        print(f"⚠ Error initializing face detector: {e}")
        return False
    
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✓ Face recognizer initialized")
    except:
        print("⚠ Using basic face matching (no advanced recognizer)")
        face_recognizer = None
    
    return True

def detect_faces(image):
    """Detect faces in image using OpenCV - optimized for speed"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def extract_face_embedding(face_roi):
    """Extract a face embedding for matching using histogram and spatial features"""
    # Resize to standard size for consistency (slightly larger for more detail)
    face_resized = cv2.resize(face_roi, (128, 128))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray_eq = cv2.equalizeHist(gray)
    
    # Method 1: Global histogram features (more bins for better discrimination)
    hist = cv2.calcHist([gray_eq], [0], None, [256], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-6)  # Normalize
    
    # Method 2: Spatial histogram features - divide face into 9 regions (3x3 grid)
    # More regions = better discrimination between different faces
    h, w = gray_eq.shape
    h_third = h // 3
    w_third = w // 3
    
    # Create 9 regions for better spatial discrimination
    regions = [
        gray_eq[0:h_third, 0:w_third],                    # Top-left
        gray_eq[0:h_third, w_third:2*w_third],          # Top-center
        gray_eq[0:h_third, 2*w_third:w],                # Top-right
        gray_eq[h_third:2*h_third, 0:w_third],          # Middle-left
        gray_eq[h_third:2*h_third, w_third:2*w_third],  # Middle-center (nose/mouth)
        gray_eq[h_third:2*h_third, 2*w_third:w],        # Middle-right
        gray_eq[2*h_third:h, 0:w_third],                # Bottom-left
        gray_eq[2*h_third:h, w_third:2*w_third],        # Bottom-center
        gray_eq[2*h_third:h, 2*w_third:w],              # Bottom-right
    ]
    
    # Calculate histograms for each region (more bins for better discrimination)
    region_hists = []
    for region in regions:
        region_hist = cv2.calcHist([region], [0], None, [64], [0, 256])
        region_hist = region_hist.flatten() / (region_hist.sum() + 1e-6)
        region_hists.append(region_hist)
    
    # Combine all features: global + 9 spatial regions
    embedding = np.concatenate([hist] + region_hists)
    
    # Don't normalize - normalization makes distances larger and harder to match
    # Just return the normalized histograms (each histogram is already normalized)
    
    return embedding.astype(np.float32)

def match_face(face_embedding, threshold=0.3):
    """Match a face embedding to known faces"""
    global match_debug_count
    
    if len(known_faces) == 0:
        return None, 0.0
    
    current_dim = len(face_embedding) if hasattr(face_embedding, '__len__') else face_embedding.shape[0]
    
    best_match = None
    best_distance = float('inf')
    all_distances = []
    skipped_old = 0
    
    for face_id, face_data in known_faces.items():
        known_embedding = face_data['embedding']
        known_dim = len(known_embedding) if hasattr(known_embedding, '__len__') else known_embedding.shape[0]
        
        if known_dim != current_dim:
            skipped_old += 1
            if match_debug_count < 3:
                print(f"  ⚠ Skipping face {face_id} ({face_data['name']}) - old embedding format ({known_dim}D vs {current_dim}D)")
            continue
        
        distance = np.linalg.norm(face_embedding - known_embedding)
        all_distances.append((face_id, distance, face_data['name']))
        if distance < best_distance:
            best_distance = distance
            best_match = face_id
    
    if skipped_old > 0 and match_debug_count < 3:
        print(f"  ⚠ Note: {skipped_old} face(s) skipped due to old embedding format. Please recapture them.")
        match_debug_count += 1
    
    if len(all_distances) == 0:
        return None, 0.0
    
    # Debug: print distances (only first few frames to avoid spam)
    if match_debug_count < 10:
        print(f"  Face matching distances: {[(name, f'{d:.3f}') for _, d, name in sorted(all_distances, key=lambda x: x[1])[:3]]}")
        match_debug_count += 1
    
    # Lower distance = better match
    # With histogram-based embeddings (global + 9 spatial histograms, no L2 normalization):
    # - Same person: distances typically 0.3-0.6
    # - Different people: distances typically 0.6-1.2
    # Need to be stricter to prevent misidentification
    max_distance = 0.65  # Stricter threshold - good matches are < 0.6
    
    # CRITICAL: If multiple faces, check if the best match is clearly better than second best
    # This prevents misidentification when two people have similar distances
    if len(all_distances) > 1:
        sorted_distances = sorted(all_distances, key=lambda x: x[1])
        best_dist = sorted_distances[0][1]
        second_best_dist = sorted_distances[1][1]
        distance_gap = second_best_dist - best_dist
        
        # Require at least 0.20 gap between best and second best (very strict)
        # This ensures the match is clearly better and prevents misidentification
        # Example: user=0.45, wife=0.50 -> gap=0.05 -> REJECT (too ambiguous)
        # Example: user=0.40, wife=0.65 -> gap=0.25 -> ACCEPT (clear match)
        if distance_gap < 0.20:
            if match_debug_count <= 20:
                print(f"  ✗ Ambiguous match (best: {best_dist:.3f} ({sorted_distances[0][2]}), 2nd: {second_best_dist:.3f} ({sorted_distances[1][2]}), gap: {distance_gap:.3f} < 0.20)")
            return None, 0.0
    
    if best_distance <= max_distance:
        # Convert distance to similarity (0-1), where 0 distance = 1.0 similarity
        similarity = 1.0 - (best_distance / max_distance)
        similarity = max(0.0, min(1.0, similarity))  # Clamp to 0-1
        
        # Require minimum 10% similarity to match
        if similarity < 0.10:
            if match_debug_count <= 10:
                print(f"  ✗ Match rejected - similarity too low: {similarity:.1%} < 10% (distance: {best_distance:.3f})")
            return None, 0.0
        
        if match_debug_count <= 10:
            print(f"  ✓ Matched: {known_faces[best_match]['name']} (distance: {best_distance:.3f}, similarity: {similarity:.1%})")
        return known_faces[best_match]['name'], similarity
    
    if match_debug_count <= 10:
        print(f"  ✗ No match (best distance: {best_distance:.3f} > threshold: {max_distance})")
    return None, 0.0

def stop_camera_services():
    """Stop pipewire and wireplumber services that may be using the camera"""
    print("Stopping pipewire and wireplumber services...")
    try:
        subprocess.run(['systemctl', '--user', 'stop', 
                       'pipewire.service', 'pipewire.socket', 
                       'pipewire-pulse.socket', 'wireplumber.service'],
                      capture_output=True, timeout=5, check=False)
        subprocess.run(['pkill', '-9', '-f', 'pipewire'], 
                      capture_output=True, timeout=2, check=False)
        subprocess.run(['pkill', '-9', '-f', 'wireplumber'], 
                      capture_output=True, timeout=2, check=False)
        for i in range(5):
            time.sleep(0.5)
            try:
                result = subprocess.run(['fuser', '/dev/media*'], 
                                      capture_output=True, timeout=1, check=False)
                if result.returncode != 0:
                    break
            except:
                pass
    except Exception as e:
        print(f"⚠ Error stopping camera services: {e}")

def extract_detections(hailo_output, w, h, class_names, threshold=0.4):
    """Extract detections from Hailo output"""
    results = []
    if hailo_output is None:
        return results
    
    for class_id, detections in enumerate(hailo_output):
        if class_id >= len(class_names):
            continue
        for detection in detections:
            if len(detection) < 5:
                continue
            score = detection[4]
            if score >= threshold:
                y0, x0, y1, x1 = detection[:4]
                x1_px = int(x0 * w)
                y1_px = int(y0 * h)
                x2_px = int(x1 * w)
                y2_px = int(y1 * h)
                results.append([x1_px, y1_px, x2_px, y2_px, class_id, score])
    return results

def draw_detections(image, detections, class_names=COCO_CLASSES):
    """Draw bounding boxes and labels for detected objects"""
    if detections is None or len(detections) == 0:
        return image
    
    h, w = image.shape[:2]
    
    for detection in detections:
        try:
            det_array = np.array(detection)
            if len(det_array) < 4:
                continue
            
            x1, y1, x2, y2 = det_array[0], det_array[1], det_array[2], det_array[3]
            class_id = int(det_array[4]) if len(det_array) > 4 else 0
            confidence = float(det_array[5]) if len(det_array) > 5 else 1.0
            
            if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                x1 = int(x1 * w)
                y1 = int(y1 * h)
                x2 = int(x2 * w)
                y2 = int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            if x2 > x1 and y2 > y1:
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                confidence_pct = int(confidence * 100)
                label = f"{class_name} {confidence_pct}%"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        except Exception as e:
            continue
    
    return image

def start_camera(mode):
    """Initialize and start the camera with the specified mode"""
    global picam2, hailo, streaming_output, current_mode
    
    with mode_lock:
        current_mode = mode
    
    # Stop pipewire/wireplumber before accessing camera
    stop_camera_services()
    
    # Check for existing processes
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*app'], 
                              capture_output=True, timeout=1, check=False)
        if result.returncode == 0:
            pids = result.stdout.decode().strip().split('\n')
            current_pid = str(os.getpid())
            for pid in pids:
                if pid and pid != current_pid:
                    print(f"⚠ Warning: Found existing app process (PID {pid}), killing it...")
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        time.sleep(1)
                    except:
                        pass
    except:
        pass
    
    # Clean up existing camera instance
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
    time.sleep(0.5)
    
    # Initialize Hailo for object detection mode
    if mode == 'object':
        hailo_model_path = '/usr/share/hailo-models/yolov8s_h8.hef'
        if not os.path.exists(hailo_model_path):
            hailo_model_path = '/usr/share/hailo-models/yolov8s_h8l.hef'
            if not os.path.exists(hailo_model_path):
                print(f"⚠ ERROR: Hailo model not found")
                return False
        print("Initializing Hailo...")
        try:
            hailo = Hailo(hailo_model_path)
            model_h, model_w, _ = hailo.get_input_shape()
            print(f"✓ Hailo initialized. Model input shape: {model_w}x{model_h}")
        except Exception as e:
            print(f"⚠ ERROR: Failed to initialize Hailo: {e}")
            return False
    
    # Initialize face recognition for face mode
    if mode == 'face':
        if not init_face_recognition():
            print("⚠ ERROR: Failed to initialize face recognition")
            return False
    
    # Retry camera initialization
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with camera_lock:
                if attempt > 0:
                    print(f"Retrying camera initialization (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                    stop_camera_services()
                
                picam2 = Picamera2()
                break
        except RuntimeError as e:
            if "busy" in str(e).lower() or "Pipeline handler in use" in str(e):
                if attempt < max_retries - 1:
                    print(f"Camera busy, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"Error: Camera still busy after {max_retries} attempts")
                    return False
            else:
                raise
    else:
        print("Error: Failed to initialize camera after retries")
        return False
    
    try:
        # Configure camera based on mode
        # Try Full HD (1920x1080) first, then fall back to HD (1280x720), then 960x720
        video_w, video_h = 1920, 1080
        
        if mode == 'object':
            # Object detection needs lores stream for Hailo
            main = {'size': (video_w, video_h), 'format': 'RGB888'}
            lores = {'size': (model_w, model_h), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            try:
                config = picam2.create_video_configuration(main, lores=lores, controls=controls)
                picam2.configure(config)
                print(f"✓ Camera configured: main={video_w}x{video_h} (Full HD), lores={model_w}x{model_h}")
            except Exception as e:
                print(f"⚠ Full HD resolution not supported, trying HD (1280x720): {e}")
                video_w, video_h = 1280, 720
                main = {'size': (video_w, video_h), 'format': 'RGB888'}
                try:
                    config = picam2.create_video_configuration(main, lores=lores, controls=controls)
                    picam2.configure(config)
                    print(f"✓ Camera configured: main={video_w}x{video_h} (HD), lores={model_w}x{model_h}")
                except Exception as e2:
                    print(f"⚠ HD resolution not supported, trying 960x720: {e2}")
                    video_w, video_h = 960, 720
                    main = {'size': (video_w, video_h), 'format': 'RGB888'}
                    config = picam2.create_video_configuration(main, lores=lores, controls=controls)
                    picam2.configure(config)
                    print(f"✓ Camera configured: main={video_w}x{video_h}, lores={model_w}x{model_h}")
        else:
            # Face recognition only needs main stream
            main = {'size': (video_w, video_h), 'format': 'RGB888'}
            controls = {'FrameRate': 30}
            try:
                config = picam2.create_video_configuration(main, controls=controls)
                picam2.configure(config)
                print(f"✓ Camera configured: {video_w}x{video_h} (Full HD)")
            except Exception as e:
                print(f"⚠ Full HD resolution not supported, trying HD (1280x720): {e}")
                video_w, video_h = 1280, 720
                main = {'size': (video_w, video_h), 'format': 'RGB888'}
                try:
                    config = picam2.create_video_configuration(main, controls=controls)
                    picam2.configure(config)
                    print(f"✓ Camera configured: {video_w}x{video_h} (HD)")
                except Exception as e2:
                    print(f"⚠ HD resolution not supported, trying 960x720: {e2}")
                    video_w, video_h = 960, 720
                    main = {'size': (video_w, video_h), 'format': 'RGB888'}
                    config = picam2.create_video_configuration(main, controls=controls)
                    picam2.configure(config)
                    print(f"✓ Camera configured: {video_w}x{video_h}")
        
        picam2.start()
        print("✓ Camera started")
        
        # Start frame capture thread
        def frame_capture_thread():
            global streaming_output, capture_requested, capture_name, face_id_counter, current_mode, detected_faces_for_selection, capture_request_frame, waiting_for_selection
            
            class StreamingOutput(io.BufferedIOBase):
                def __init__(self):
                    self.frame = None
                    self.condition = threading.Condition()
                    self._is_closed = False  # Use _is_closed instead of closed (which is read-only)
                
                def write(self, buf):
                    with self.condition:
                        if self._is_closed:
                            return
                        self.frame = buf
                        self.condition.notify_all()
                
                def read(self, timeout=0.1):
                    """Read a frame with timeout to avoid blocking forever"""
                    with self.condition:
                        if self._is_closed:
                            return None
                        if self.frame is not None:
                            return self.frame
                        # Wait with timeout
                        self.condition.wait(timeout=timeout)
                        if self._is_closed:
                            return None
                        return self.frame
                
                def mark_closed(self):
                    """Mark as closed to signal shutdown"""
                    with self.condition:
                        self._is_closed = True
                        self.condition.notify_all()
            
            streaming_output = StreamingOutput()
            frame_count = 0
            last_recognition_frame = -1
            recognition_skip_frames = 2
            face_names = {}
            # Track faces by position to maintain consistent labeling
            face_tracker = {}  # {(center_x, center_y, size): (name, similarity, frame_count)}
            
            # Write test frame
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
                    with mode_lock:
                        active_mode = current_mode
                    
                    # Capture frame
                    request = picam2.capture_request()
                    frame = request.make_array("main")
                    
                    # Fix colors
                    frame_swapped = frame.copy()
                    frame_swapped[:, :, [0, 2]] = frame_swapped[:, :, [2, 0]]
                    frame_bgr = cv2.cvtColor(frame_swapped, cv2.COLOR_RGB2BGR)
                    
                    if active_mode == 'object':
                        # Object detection mode
                        detections = []
                        if hailo is not None:
                            try:
                                frame_lores = picam2.capture_array("lores")
                                hailo_output = hailo.run(frame_lores)
                                detections = extract_detections(hailo_output, video_w, video_h, COCO_CLASSES, threshold=0.4)
                            except Exception as e:
                                if frame_count < 5:
                                    print(f"⚠ Hailo inference error: {e}")
                                detections = []
                        
                        if detections:
                            frame_bgr = draw_detections(frame_bgr, detections)
                    
                    elif active_mode == 'face':
                        # Face recognition mode
                        faces = detect_faces(frame_bgr)
                        
                        # Limit to top 5 largest faces (by area) for performance and clarity
                        MAX_FACES = 5
                        original_face_count = len(faces)
                        if original_face_count > MAX_FACES:
                            # Sort by face area (width * height), largest first
                            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:MAX_FACES]
                            if frame_count % 30 == 0:  # Print occasionally to avoid spam
                                print(f"⚠ Limited face detection to {MAX_FACES} largest faces (out of {original_face_count} detected)")
                        
                        # Handle capture request - keep checking every frame until timeout
                        if capture_requested:
                            # Track when capture was first requested
                            if capture_request_frame == 0:
                                capture_request_frame = frame_count
                            
                            frames_since_request = frame_count - capture_request_frame
                            max_check_frames = 90  # Check for ~3 seconds (at ~30fps) - but don't timeout if waiting for selection
                            
                            if len(faces) > 0:
                                # Check which faces are unknown (not recognized)
                                unknown_faces = []
                                print(f"🔍 Capture requested (frame {frame_count}): checking {len(faces)} detected faces...")
                                for idx, (x, y, w, h) in enumerate(faces):
                                    face_roi = frame_bgr[y:y+h, x:x+w]
                                    face_embedding = extract_face_embedding(face_roi)
                                    name, similarity = match_face(face_embedding)
                                    
                                    name_str = name if name else 'None'
                                    print(f"  Face {idx}: name={name_str}, similarity={similarity:.1%}")
                                    
                                    # Only include unknown faces (not recognized or low similarity)
                                    # A face is unknown if: no name returned OR similarity < 30%
                                    is_unknown = (name is None) or (similarity < 0.30)
                                    
                                    if is_unknown:
                                        print(f"  ✓ Face {idx} is UNKNOWN - adding to selection list")
                                        # Store the face ROI for later capture
                                        face_roi_copy = face_roi.copy()
                                        # Encode face image to base64 for display
                                        face_resized = cv2.resize(face_roi_copy, (150, 150))
                                        success, buffer = cv2.imencode('.jpg', face_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                        if success:
                                            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                                            # Also encode the full-size ROI for storage
                                            success_full, buffer_full = cv2.imencode('.jpg', face_roi_copy, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                            img_full_base64 = base64.b64encode(buffer_full.tobytes()).decode('utf-8') if success_full else img_base64
                                            unknown_faces.append({
                                                'index': idx,
                                                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                                                'image': img_base64,  # Thumbnail for display
                                                'image_full': img_full_base64,  # Full size for storage
                                                'embedding': face_embedding.tolist(),  # Convert numpy array to list for JSON
                                                'face_roi_data': None  # Will be set when capturing
                                            })
                                    else:
                                        print(f"  ✗ Face {idx} is KNOWN ({name} at {similarity:.1%}) - skipping")
                                
                                print(f"📊 Found {len(unknown_faces)} unknown faces out of {len(faces)} total")
                                
                                # Only update detected_faces_for_selection if we're not already waiting for user selection
                                # This prevents overwriting the faces while the user is selecting
                                if len(unknown_faces) > 0 and not waiting_for_selection:
                                    with detected_faces_lock:
                                        detected_faces_for_selection = unknown_faces
                                        # Store frame dimensions for validation
                                        for face_data in unknown_faces:
                                            face_data['frame_h'] = frame_bgr.shape[0]
                                            face_data['frame_w'] = frame_bgr.shape[1]
                                
                                # If only one unknown face, capture it directly (but only if we're not waiting for selection)
                                if len(unknown_faces) == 1 and not waiting_for_selection:
                                    face_data = unknown_faces[0]
                                    # Get the face ROI from current frame
                                    face_roi = frame_bgr[face_data['y']:face_data['y']+face_data['h'], 
                                                       face_data['x']:face_data['x']+face_data['w']].copy()
                                    face_embedding = np.array(face_data['embedding'], dtype=np.float32)
                                    
                                    # Store face
                                    enforce_face_limit()  # Remove oldest if at limit
                                    face_id_counter += 1
                                    name_to_save = capture_name if capture_name else f"Person {face_id_counter}"
                                    known_faces[face_id_counter] = {
                                        'name': name_to_save,
                                        'embedding': face_embedding,
                                        'image': face_roi.copy()
                                    }
                                    save_known_faces()
                                    print(f"✓ Captured face #{face_id_counter}: {name_to_save}")
                                    
                                    capture_requested = False
                                    capture_name = ""
                                    capture_request_frame = 0
                                    waiting_for_selection = False
                                    with detected_faces_lock:
                                        detected_faces_for_selection = []
                                # If multiple unknown faces, keep capture_requested = True for user selection
                                elif len(unknown_faces) > 1:
                                    print(f"✓ Found {len(unknown_faces)} unknown faces - waiting for user selection")
                                    waiting_for_selection = True  # Set flag to prevent auto-capture
                                    # Keep capture_requested = True, user will select which face to capture
                                # If no unknown faces in this frame, keep checking (don't clear yet)
                                else:
                                    if frames_since_request % 10 == 0:  # Print every 10 frames
                                        print(f"⚠ Frame {frame_count}: All {len(faces)} detected faces are already known, continuing to check...")
                            elif len(faces) == 0:
                                # No face detected when capture was requested
                                if frames_since_request % 30 == 0:  # Print every 30 frames to avoid spam
                                    print(f"⚠ Frame {frame_count}: Capture requested but no face detected, continuing to check...")
                            
                            # Clear capture request if we've been checking for too long
                            # BUT: Don't timeout if we're waiting for user selection - let them take their time
                            if frames_since_request >= max_check_frames and not waiting_for_selection:
                                print(f"⚠ Timeout: Cleared capture request after {max_check_frames} frames")
                                capture_requested = False
                                capture_name = ""
                                capture_request_frame = 0
                                waiting_for_selection = False
                                with detected_faces_lock:
                                    detected_faces_for_selection = []
                        
                        # Recognize faces
                        should_recognize = (frame_count - last_recognition_frame) > recognition_skip_frames
                        
                        # Update recognition results only when needed
                        if should_recognize:
                            # Match each detected face to a tracked face by position FIRST
                            # This ensures consistent labeling even if detection order changes
                            matched_indices = set()
                            updated_tracker = {}
                            
                            for idx, (x, y, w, h) in enumerate(faces):
                                face_roi = frame_bgr[y:y+h, x:x+w]
                                face_embedding = extract_face_embedding(face_roi)
                                
                                # Calculate face center and size for tracking
                                center_x = x + w // 2
                                center_y = y + h // 2
                                face_size = w * h
                                
                                # Find closest tracked face (within reasonable distance)
                                matched_tracker_key = None
                                min_distance = float('inf')
                                for (tx, ty, ts), (tname, tsim, tframe) in face_tracker.items():
                                    # Calculate distance between face centers
                                    dist = np.sqrt((center_x - tx)**2 + (center_y - ty)**2)
                                    # Also consider size similarity
                                    size_diff = abs(face_size - ts) / max(face_size, ts)
                                    # Combined distance metric (stricter matching)
                                    combined_dist = dist + size_diff * 200
                                    
                                    # Match if within 100 pixels and similar size (stricter threshold)
                                    if combined_dist < 100 and combined_dist < min_distance:
                                        min_distance = combined_dist
                                        matched_tracker_key = (tx, ty, ts)
                                
                                # Match to known faces
                                name, similarity = match_face(face_embedding)
                                
                                # Update tracker: use matched position or create new entry
                                if matched_tracker_key:
                                    # Update existing tracked face - ALWAYS use the tracked name (persistent labeling)
                                    old_name, old_sim, old_frame = face_tracker[matched_tracker_key]
                                    
                                    # Only update the name if:
                                    # 1. New recognition is confident (similarity > 50%)
                                    # 2. New recognition matches the old name (same person)
                                    # 3. Otherwise, keep the old name (prevent label swapping)
                                    if name and similarity > 0.50 and name == old_name:
                                        # Update with new similarity but keep same name
                                        updated_tracker[(center_x, center_y, face_size)] = (old_name, similarity, frame_count)
                                        face_names[idx] = (old_name, similarity)
                                    else:
                                        # Keep old name and similarity (prevent label swapping)
                                        updated_tracker[(center_x, center_y, face_size)] = (old_name, old_sim, frame_count)
                                        face_names[idx] = (old_name, old_sim)
                                    matched_indices.add(idx)
                                else:
                                    # New face - add to tracker only if we have a confident match
                                    if name and similarity > 0.30:  # Require at least 30% similarity for new faces
                                        updated_tracker[(center_x, center_y, face_size)] = (name, similarity, frame_count)
                                        face_names[idx] = (name, similarity)
                                        matched_indices.add(idx)
                                    else:
                                        face_names[idx] = (None, 0.0)
                            
                            # Update tracker with new positions
                            face_tracker = updated_tracker
                            
                            # Clean up old tracked faces (not seen for 30 frames)
                            face_tracker = {
                                k: v for k, v in face_tracker.items()
                                if (frame_count - v[2]) < 30
                            }
                            
                            last_recognition_frame = frame_count
                        else:
                            # When not recognizing, use tracked faces for consistent labeling
                            face_names = {}
                            for idx, (x, y, w, h) in enumerate(faces):
                                center_x = x + w // 2
                                center_y = y + h // 2
                                face_size = w * h
                                
                                # Find closest tracked face
                                matched_tracker_key = None
                                min_distance = float('inf')
                                for (tx, ty, ts), (tname, tsim, tframe) in face_tracker.items():
                                    dist = np.sqrt((center_x - tx)**2 + (center_y - ty)**2)
                                    size_diff = abs(face_size - ts) / max(face_size, ts)
                                    combined_dist = dist + size_diff * 200
                                    
                                    if combined_dist < 100 and combined_dist < min_distance:
                                        min_distance = combined_dist
                                        matched_tracker_key = (tx, ty, ts)
                                
                                if matched_tracker_key:
                                    tname, tsim, tframe = face_tracker[matched_tracker_key]
                                    face_names[idx] = (tname, tsim)
                                else:
                                    face_names[idx] = (None, 0.0)
                        
                        # Draw faces
                        for idx, (x, y, w, h) in enumerate(faces):
                            if idx in face_names:
                                name, similarity = face_names[idx]
                            else:
                                name, similarity = None, 0.0
                            
                            color = (0, 255, 0) if name else (0, 0, 255)
                            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
                            
                            if name:
                                label = f"{name} {int(similarity*100)}%"
                            else:
                                label = "Unknown"
                            
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame_bgr, (x, y - label_size[1] - 10),
                                        (x + label_size[0], y), color, -1)
                            cv2.putText(frame_bgr, label, (x, y - 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Convert back to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Encode to JPEG
                    pil_image = Image.fromarray(frame_rgb)
                    jpeg_buffer = io.BytesIO()
                    pil_image.save(jpeg_buffer, format='JPEG', quality=85)
                    jpeg_bytes = jpeg_buffer.getvalue()
                    
                    # Write to streaming output
                    streaming_output.write(jpeg_bytes)
                    
                    request.release()
                    frame_count += 1
                    
                except Exception as e:
                    print(f"Error in frame capture: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
        
        # Start the frame capture thread
        thread = threading.Thread(target=frame_capture_thread, daemon=True)
        thread.start()
        print("✓ Frame capture thread started")
        
        # Wait longer for camera to initialize and start producing frames
        # This ensures streaming_output is ready before returning
        time.sleep(2.0)  # Increased from 0.5 to 2.0 seconds
        
        # Verify streaming_output is ready
        if streaming_output is None:
            print("⚠ Warning: streaming_output not initialized after delay")
            return False
        
        print(f"✓ Camera started with {mode} mode")
        return True
            
    except Exception as e:
        print(f"Error starting camera: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/mode', methods=['GET', 'POST'])
def mode_switcher():
    """Mode switcher page with Object and Face buttons"""
    global current_mode
    
    if request.method == 'POST':
        new_mode = request.json.get('mode', 'object')
        if new_mode not in ['object', 'face']:
            return jsonify({'success': False, 'message': 'Invalid mode'})
        
        with mode_lock:
            if current_mode == new_mode:
                return jsonify({'success': True, 'message': f'Mode {new_mode} is already active', 'mode': current_mode})
            
            print(f"Switching mode from {current_mode} to {new_mode}")
            current_mode = new_mode
        
        # Stop current camera and restart with new mode
        cleanup_camera()
        time.sleep(1)  # Give time for cleanup
        
        if start_camera(new_mode):
            # Give extra time for camera to fully initialize before redirecting
            time.sleep(1)
            return jsonify({'success': True, 'message': f'Switched to {new_mode} mode', 'mode': new_mode})
        else:
            return jsonify({'success': False, 'message': 'Failed to start camera with new mode'})
    
    # GET request - show mode switcher page
    with mode_lock:
        active_mode = current_mode
    
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mode Switcher</title>
    <style>
        :root {
            --bg-gradient-start: #3e91be;
            --bg-gradient-end: #cd3c65;
            --card-bg: white;
            --text-primary: #333;
            --text-white: white;
        }
        [data-theme="dark"] {
            --bg-gradient-start: #6b6a6a;
            --bg-gradient-end: #cd3c65;
            --card-bg: #1a1f2e;
            --text-primary: #e0e0e0;
            --text-white: #e0e0e0;
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
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            max-width: 500px;
            width: 100%;
        }
        h1 {
            color: var(--text-primary);
            margin-bottom: 30px;
            font-size: 2em;
        }
        .mode-buttons {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .mode-button {
            padding: 20px 40px;
            font-size: 1.2em;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.2s;
            color: white;
        }
        .mode-button:hover {
            transform: translateY(-2px);
        }
        .mode-button.active {
            opacity: 0.7;
            cursor: not-allowed;
        }
        .mode-button.object {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .mode-button.face {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
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
        <h1>🎯 Select Mode</h1>
        <div class="mode-buttons">
            <button class="mode-button object" id="objectBtn" onclick="switchMode('object')">
                Object Detection
            </button>
            <button class="mode-button face" id="faceBtn" onclick="switchMode('face')">
                Face Recognition
            </button>
        </div>
        <div id="status"></div>
    </div>
    <script>
        const activeMode = '{{ active_mode }}';
        
        // Highlight active mode
        if (activeMode === 'object') {
            document.getElementById('objectBtn').classList.add('active');
            document.getElementById('objectBtn').textContent = 'Object Detection (Active)';
        } else {
            document.getElementById('faceBtn').classList.add('active');
            document.getElementById('faceBtn').textContent = 'Face Recognition (Active)';
        }
        
        function switchMode(mode) {
            if (mode === activeMode) {
                showStatus('This mode is already active', 'error');
                return;
            }
            
                    showStatus('Switching mode...', 'success');
            
            fetch('/mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mode: mode})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(data.message, 'success');
                    // Wait longer (3 seconds) to ensure camera is fully initialized
                    setTimeout(() => {
                        window.location.href = '/';
                    }, 3000);
                } else {
                    showStatus(data.message || 'Error switching mode', 'error');
                }
            })
            .catch(error => {
                showStatus('Error: ' + error.message, 'error');
            });
        }
        
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            setTimeout(() => {
                status.textContent = '';
                status.className = 'status';
            }, 3000);
        }
        
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
    ''', active_mode=active_mode)

@app.route('/')
def index():
    """Main page - shows different template based on current mode"""
    global current_mode
    
    with mode_lock:
        active_mode = current_mode
    
    if active_mode == 'object':
        # Object detection template
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
            --text-white: white;
            --border-color: #f0f0f0;
        }
        [data-theme="dark"] {
            --bg-gradient-start: #6b6a6a;
            --bg-gradient-end: #cd3c65;
            --bg-color: #0f1419;
            --card-bg: #1a1f2e;
            --text-primary: #e0e0e0;
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
            max-width: 100%;
            margin: 0 auto;
            padding: 0 20px;
        }
        .header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            gap: 20px;
        }
        .mode-link {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            backdrop-filter: blur(10px);
            white-space: nowrap;
        }
        .mode-link:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        h1 {
            flex: 1;
            text-align: center;
            color: var(--text-white);
            font-size: 2.5em;
            margin: 0;
        }
        .theme-toggle {
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
            white-space: nowrap;
        }
        .theme-toggle:hover {
            opacity: 0.7;
        }
        .theme-toggle svg {
            width: 28px;
            height: 28px;
            fill: white;
        }
        .feed-container {
            display: flex;
            gap: 20px;
            width: 100%;
        }
        .feed-description {
            width: 20%;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            color: var(--text-primary);
            font-size: 1.1em;
            line-height: 1.5;
        }
        .video-container {
            flex: 1;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        img {
            width: 100%;
            height: auto;
            display: block;
            border: 2px solid var(--border-color);
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-row">
            <a href="/mode" class="mode-link">🎯 Switch Mode</a>
            <h1>🤖 AI Object Detection Camera Stream</h1>
            <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle dark mode">
                <svg id="themeIcon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>
                </svg>
            </button>
        </div>
        <div class="feed-container">
            <div class="feed-description">
                <p>Detected objects are shown with green bounding boxes and labels.</p>
            </div>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
        </div>
    </div>
    <script>
        // Check mode periodically and reload if it changes
        // This ensures pages reload when mode is switched from another window/tab
        let currentMode = 'object';
        let checkCount = 0;
        
        // Initial check on page load - fetch actual mode from server
        fetch('/get_mode')
            .then(response => response.json())
            .then(data => {
                currentMode = data.mode || 'object';
                console.log('Initial mode:', currentMode);
            })
            .catch(err => {
                console.error('Error getting initial mode:', err);
                // Default to 'object' if fetch fails
                currentMode = 'object';
            });
        
        // Check every second for mode changes
        setInterval(() => {
            fetch('/get_mode')
                .then(response => response.json())
                .then(data => {
                    checkCount++;
                    if (data.mode && data.mode !== currentMode) {
                        console.log(`Mode changed from '${currentMode}' to '${data.mode}', reloading page...`);
                        currentMode = data.mode; // Update before reload
                        window.location.reload();
                    }
                })
                .catch(err => {
                    // Only log errors occasionally to avoid spam
                    if (checkCount % 10 === 0) {
                        console.error('Error checking mode:', err);
                    }
                });
        }, 1000); // Check every second
        
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
    else:
        # Face recognition template
        return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition Camera</title>
    <style>
        :root {
            --bg-gradient-start: #3e91be;
            --bg-gradient-end: #cd3c65;
            --bg-color: #f8f8f8;
            --card-bg: white;
            --text-primary: #333;
            --border-color: #f0f0f0;
        }
        [data-theme="dark"] {
            --bg-gradient-start: #6b6a6a;
            --bg-gradient-end: #cd3c65;
            --bg-color: #0f1419;
            --card-bg: #1a1f2e;
            --text-primary: #e0e0e0;
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
            max-width: 100%;
            margin: 0 auto;
            padding: 0 20px;
        }
        .header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            gap: 20px;
        }
        .mode-link {
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.2);
            color: white;
            text-decoration: none;
            border-radius: 5px;
            backdrop-filter: blur(10px);
            white-space: nowrap;
        }
        .mode-link:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        h1 {
            flex: 1;
            text-align: center;
            color: white;
            font-size: 2.5em;
            margin: 0;
        }
        .theme-toggle {
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
            white-space: nowrap;
        }
        .theme-toggle:hover {
            opacity: 0.7;
        }
        .theme-toggle svg {
            width: 28px;
            height: 28px;
            fill: white;
        }
        .feed-container {
            display: flex;
            flex-wrap: nowrap;
            gap: 20px;
            width: 100%;
            margin-bottom: 20px;
        }
        .feed-description {
            width: 20%;
            min-width: 250px;
            max-width: 20%;
            display: flex;
            flex-direction: column;
            gap: 20px;
            flex-shrink: 0;
        }
        .video-container {
            flex: 1;
            background: var(--card-bg);
            padding: 20px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        img {
            width: 100%;
            height: auto;
            display: block;
            border-radius: 10px;
        }
        .controls {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 20px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 15px;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            font-size: 16px;
            background: var(--card-bg);
            color: var(--text-primary);
        }
        button {
            width: 100%;
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:active {
            transform: translateY(0);
        }
        .known-faces {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 20px;
            flex: 1;
            overflow-y: auto;
        }
        .known-faces h2 {
            margin-bottom: 15px;
            color: var(--text-primary);
            font-size: 1.2em;
        }
        .faces-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .face-item {
            text-align: center;
            padding: 10px;
            background: var(--bg-color);
            border-radius: 8px;
            border: 2px solid var(--border-color);
        }
        .face-item img {
            width: 100%;
            height: auto;
            max-height: 100px;
            object-fit: cover;
            border-radius: 5px;
            margin-bottom: 8px;
        }
        .face-item .name {
            font-weight: bold;
            color: var(--text-primary);
        }
        .status {
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        /* Face selection overlay */
        .face-selection-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        .face-selection-overlay.active {
            display: flex;
        }
        .face-selection-modal {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 30px;
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
        }
        .face-selection-modal h2 {
            margin-top: 0;
            margin-bottom: 20px;
            color: var(--text-primary);
        }
        .face-selection-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .face-selection-item {
            cursor: pointer;
            border: 3px solid transparent;
            border-radius: 10px;
            padding: 10px;
            transition: all 0.3s ease;
            text-align: center;
        }
        .face-selection-item:hover {
            border-color: var(--accent-color);
            background: var(--bg-secondary);
        }
        .face-selection-item.selected {
            border-color: var(--accent-color);
            background: var(--bg-secondary);
        }
        .face-selection-item img {
            width: 100%;
            height: auto;
            border-radius: 8px;
            display: block;
        }
        .face-selection-actions {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 20px;
        }
        .face-selection-actions button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            transition: all 0.3s ease;
            width: auto;
        }
        .face-selection-actions button:hover:not(:disabled) {
            transform: translateY(-2px);
            opacity: 0.9;
        }
        .btn-cancel {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-cancel:hover:not(:disabled) {
            opacity: 0.9;
        }
        .btn-confirm {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-confirm:hover:not(:disabled) {
            opacity: 0.9;
        }
        .btn-confirm:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-row">
            <a href="/mode" class="mode-link">🎯 Switch Mode</a>
            <h1>Face Recognition Camera</h1>
            <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle dark mode">
                <svg id="themeIcon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>
                </svg>
            </button>
        </div>
        <div class="feed-container">
            <div class="feed-description">
                <div class="controls">
                    <div class="control-group">
                        <input type="text" id="personName" placeholder="Enter person's name..." />
                        <button onclick="captureFace()">Capture Face</button>
                    </div>
                    <div id="status"></div>
                </div>
                <div class="known-faces">
                    <h2>Known Faces</h2>
                    <div id="facesList" class="faces-list">
                        <p>No faces captured yet. Use the capture button above to add faces.</p>
                    </div>
                </div>
            </div>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
        </div>
    
    <!-- Face Selection Overlay -->
    <div id="faceSelectionOverlay" class="face-selection-overlay">
        <div class="face-selection-modal">
            <h2>Select Face to Capture</h2>
            <p id="selectionMessage" style="color: var(--text-secondary); margin-bottom: 20px;"></p>
            <div id="faceSelectionGrid" class="face-selection-grid"></div>
            <div class="face-selection-actions">
                <button class="btn-confirm" id="confirmSelectionBtn" onclick="confirmFaceSelection()" disabled>Accept</button>
                <button class="btn-cancel" onclick="cancelFaceSelection()">Cancel</button>
            </div>
        </div>
    </div>
    </div>
    <script>
        // Check mode periodically and reload if it changes
        // This ensures pages reload when mode is switched from another window/tab
        let currentMode = 'face';
        let checkCount = 0;
        
        // Initial check on page load
        fetch('/get_mode')
            .then(response => response.json())
            .then(data => {
                currentMode = data.mode;
                console.log('Initial mode:', currentMode);
            })
            .catch(err => console.error('Error getting initial mode:', err));
        
        // Check every second for mode changes
        setInterval(() => {
            fetch('/get_mode')
                .then(response => response.json())
                .then(data => {
                    checkCount++;
                    if (data.mode && data.mode !== currentMode) {
                        console.log(`Mode changed from '${currentMode}' to '${data.mode}', reloading page...`);
                        currentMode = data.mode; // Update before reload
                        window.location.reload();
                    }
                    // Debug: log every 10 checks (every 10 seconds)
                    if (checkCount % 10 === 0) {
                        console.log(`Mode check #${checkCount}: current='${currentMode}', server='${data.mode}'`);
                    }
                })
                .catch(err => {
                    // Only log errors occasionally to avoid spam
                    if (checkCount % 10 === 0) {
                        console.error('Error checking mode:', err);
                    }
                });
        }, 1000); // Check every second
        
        let selectedFaceIndex = null;
        let currentCaptureName = '';
        let availableFaces = [];
        
        function captureFace() {
            const name = document.getElementById('personName').value.trim();
            if (!name) {
                showStatus('Please enter a name first', 'error');
                return;
            }
            
            currentCaptureName = name;
            selectedFaceIndex = null;

            fetch('/capture_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name})
            })
            .then(response => response.json())
            .then(data => {
                console.log('Capture response:', data); // Debug log
                if (data.needs_selection && data.faces) {
                    // Show selection overlay
                    console.log(`Showing selection overlay with ${data.faces.length} faces`);
                    availableFaces = data.faces;
                    showFaceSelection(data.faces, name);
                } else if (data.success) {
                    showStatus(data.message, 'success');
                    document.getElementById('personName').value = '';
                    refreshFaces();
                } else {
                    console.error('Capture failed:', data.message);
                    showStatus(data.message, 'error');
                }
            })
            .catch(error => {
                showStatus(`Error: ${error.message}`, 'error');
            });
        }
        
        function showFaceSelection(faces, name) {
            const overlay = document.getElementById('faceSelectionOverlay');
            const grid = document.getElementById('faceSelectionGrid');
            const message = document.getElementById('selectionMessage');
            const confirmBtn = document.getElementById('confirmSelectionBtn');
            
            message.textContent = `Found ${faces.length} unknown faces. Select the face you want to capture as "${name}".`;
            grid.innerHTML = '';
            selectedFaceIndex = null;
            confirmBtn.disabled = true;
            
            // Store the faces array for later reference
            availableFaces = faces;
            
            faces.forEach((face, displayIndex) => {
                const item = document.createElement('div');
                item.className = 'face-selection-item';
                // Use the display index (array position) as the face_index for backend
                // The backend uses this as an array index into detected_faces_for_selection
                item.onclick = () => selectFace(displayIndex, item);
                
                const img = document.createElement('img');
                img.src = 'data:image/jpeg;base64,' + face.image;
                img.alt = `Face ${displayIndex + 1}`;
                
                const label = document.createElement('div');
                label.textContent = `Face ${displayIndex + 1}`;
                label.style.marginTop = '8px';
                label.style.color = 'var(--text-primary)';
                
                item.appendChild(img);
                item.appendChild(label);
                grid.appendChild(item);
            });
            
            overlay.classList.add('active');
        }
        
        function selectFace(faceIndex, element) {
            // Remove selected class from all items
            document.querySelectorAll('.face-selection-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            // Add selected class to clicked item
            element.classList.add('selected');
            selectedFaceIndex = faceIndex; // This is the array index (0, 1, 2...)
            console.log(`Selected face at array index: ${faceIndex}`);
            document.getElementById('confirmSelectionBtn').disabled = false;
        }
        
        function confirmFaceSelection() {
            if (selectedFaceIndex === null || !currentCaptureName) {
                return;
            }
            
            fetch('/capture_specific_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    name: currentCaptureName,
                    face_index: selectedFaceIndex
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(data.message, 'success');
                    document.getElementById('personName').value = '';
                    cancelFaceSelection();
                    refreshFaces();
                } else {
                    showStatus(data.message, 'error');
                }
            })
            .catch(error => {
                showStatus(`Error: ${error.message}`, 'error');
            });
        }
        
        function cancelFaceSelection() {
            const overlay = document.getElementById('faceSelectionOverlay');
            overlay.classList.remove('active');
            selectedFaceIndex = null;
            currentCaptureName = '';
            availableFaces = [];
            
            // Also cancel the capture request on server
            fetch('/capture_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: '', cancel: true})
            }).catch(() => {});
        }
        function refreshFaces() {
            fetch('/known_faces')
            .then(response => response.json())
            .then(data => {
                const list = document.getElementById('facesList');
                if (data.faces && Object.keys(data.faces).length > 0) {
                    list.innerHTML = '';
                    Object.entries(data.faces).forEach(([id, face]) => {
                        const item = document.createElement('div');
                        item.className = 'face-item';
                        // Image is already base64 encoded from server
                        const imgData = face.image;
                        item.innerHTML = `
                            <img src="data:image/jpeg;base64,${imgData}" alt="${face.name}" 
                                 onerror="console.error('Image load error for ${face.name}'); this.style.display='none';" />
                            <div class="name">${face.name}</div>
                            <button onclick="deleteFace(${id})" style="margin-top: 5px; padding: 5px 10px; font-size: 12px;">Delete</button>
                        `;
                        list.appendChild(item);
                    });
                } else {
                    list.innerHTML = '<p>No faces captured yet. Use the capture button above to add faces.</p>';
                }
            })
            .catch(error => {
                console.error('Error refreshing faces:', error);
                showStatus('Error loading faces', 'error');
            });
        }

        function deleteFace(faceId) {
            fetch('/delete_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({face_id: parseInt(faceId)})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus('Face deleted', 'success');
                    refreshFaces();
                } else {
                    showStatus(`Error: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Error: ${error.message}`, 'error');
            });
        }
        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            setTimeout(() => {
                status.textContent = '';
                status.className = 'status';
            }, 3000);
        }

        window.addEventListener('DOMContentLoaded', () => {
            refreshFaces();
            setInterval(refreshFaces, 5000); // Refresh every 5 seconds
        });
        
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

        // Initialize theme on page load
        initTheme();
    </script>
</body>
</html>
        ''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route - keeps feed alive with placeholder frames when camera is restarting"""
    def generate():
        global last_frame
        
        def create_placeholder_frame(message="Camera initializing..."):
            """Create a visible placeholder frame with a message"""
            # Use a larger size to match typical video feed (1280x720)
            h, w = 720, 1280
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Fill with a dark blue background (more visible than black)
            placeholder[:, :] = (40, 40, 80)  # Dark blue in BGR
            
            # Add a border
            cv2.rectangle(placeholder, (10, 10), (w-10, h-10), (100, 150, 255), 5)
            
            # Add main message
            text = message
            font_scale = 2.0
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = (w - text_width) // 2
            text_y = (h + text_height) // 2 - 50
            
            # Draw text without shadow
            cv2.putText(placeholder, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            timestamp_text = f"Time: {timestamp}"
            cv2.putText(placeholder, timestamp_text, (20, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Add animated dots (simple version - will update each frame)
            dot_count = (int(time.time() * 2) % 4)  # Changes every 0.5 seconds
            dots = "." * dot_count
            cv2.putText(placeholder, dots, (text_x + text_width + 10, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
            
            placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(placeholder_rgb)
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=85)
            return buffer.getvalue()
        
        # Create initial placeholder
        placeholder_frame = create_placeholder_frame("Camera initializing...")
        
        consecutive_failures = 0
        max_failures = 3  # After 3 consecutive failures, use placeholder (faster fallback)
        last_success_time = time.time()
        
        while True:
            frame_to_send = None
            current_time = time.time()
            
            # Check if streaming_output exists and is ready
            if streaming_output is None:
                # Camera is restarting - immediately use placeholder
                message = "Camera restarting... Please wait"
                frame_to_send = create_placeholder_frame(message)
                consecutive_failures = max_failures + 1  # Force placeholder mode
            else:
                # Try to get a frame from the camera (with timeout to avoid blocking)
                try:
                    # Use read() with short timeout - won't block forever
                    frame = streaming_output.read(timeout=0.05)  # 50ms timeout
                    if frame and len(frame) > 0:
                        # Success - update last_frame and reset failure counter
                        with last_frame_lock:
                            last_frame = frame
                        consecutive_failures = 0
                        last_success_time = current_time
                        frame_to_send = frame
                    else:
                        consecutive_failures += 1
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures == 1:  # Only log first failure
                        print(f"⚠ Error reading from streaming_output: {e}")
                except AttributeError:
                    # streaming_output doesn't have read() method or was replaced
                    consecutive_failures += 1
                    if consecutive_failures == 1:
                        print("⚠ streaming_output missing read() method")
            
            # If we don't have a frame, use fallback
            if frame_to_send is None:
                # Check how long since last success
                time_since_success = current_time - last_success_time
                
                if consecutive_failures >= max_failures or time_since_success > 1.0:
                    # Camera seems to be down - use placeholder with updating message
                    if streaming_output is None:
                        message = "Camera restarting... Please wait"
                    elif time_since_success > 2.0:
                        message = "Camera not responding... Waiting"
                    else:
                        message = "Waiting for camera frames..."
                    frame_to_send = create_placeholder_frame(message)
                else:
                    # Try to use last known frame (only if recent)
                    with last_frame_lock:
                        if last_frame is not None and time_since_success < 2.0:
                            frame_to_send = last_frame
                        else:
                            # No recent frame - use placeholder
                            frame_to_send = create_placeholder_frame("Camera initializing...")
            
            # Always send a frame to keep the stream alive
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            except Exception as e:
                print(f"⚠ Error yielding frame: {e}")
                # Even on error, try to send a placeholder
                try:
                    error_frame = create_placeholder_frame("Stream error - reconnecting...")
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
                except:
                    pass
                time.sleep(0.1)
                continue
            
            # Small delay to maintain reasonable frame rate
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """Capture and store a face - returns detected faces if multiple unknown"""
    global capture_requested, capture_name, detected_faces_for_selection, waiting_for_selection, capture_request_frame
    data = request.json
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    
    capture_name = name
    capture_requested = True
    capture_request_frame = 0  # Will be set by frame thread when it starts processing
    waiting_for_selection = False  # Reset flag when new capture is requested
    print(f"📸 Capture requested for '{name}'")
    
    # Wait a bit for the frame capture thread to process
    # Try multiple times to catch faces (they might not be detected immediately)
    faces_for_selection = []
    for attempt in range(6):  # Try 6 times (up to 3 seconds)
        time.sleep(0.5)
        with detected_faces_lock:
            faces_for_selection = detected_faces_for_selection.copy()
        if len(faces_for_selection) > 0 or not capture_requested:
            # Either we found faces or the request was processed (single face captured)
            break
    
    print(f"📊 After waiting: found {len(faces_for_selection)} unknown faces, capture_requested={capture_requested}")
    
    if len(faces_for_selection) > 1:
        # Multiple unknown faces detected - return them for user selection
        # Remove frame dimensions from response (not needed in frontend)
        faces_response = []
        for face in faces_for_selection:
            faces_response.append({
                'index': face['index'],
                'image': face['image'],  # Thumbnail
                'x': face['x'], 'y': face['y'], 'w': face['w'], 'h': face['h']
            })
        return jsonify({
            'success': False,
            'needs_selection': True,
            'faces': faces_response,
            'message': f'Found {len(faces_for_selection)} unknown faces. Please select which one to capture.'
        })
    elif len(faces_for_selection) == 1:
        # Single face was captured automatically
        capture_requested = False
        capture_name = ""
        with detected_faces_lock:
            detected_faces_for_selection = []
        return jsonify({'success': True, 'message': f'Face captured for {name}'})
    elif capture_requested:
        # Still requested means no unknown faces detected
        capture_requested = False
        capture_name = ""
        return jsonify({'success': False, 'message': 'No unknown faces detected. Please ensure an unknown face is visible.'})
    else:
        # Already processed
        return jsonify({'success': True, 'message': f'Face captured for {name}'})

@app.route('/capture_specific_face', methods=['POST'])
def capture_specific_face():
    """Capture a specific face by index from the detected faces"""
    global capture_requested, capture_name, face_id_counter, detected_faces_for_selection, waiting_for_selection
    data = request.json
    name = data.get('name', '').strip()
    face_index = data.get('face_index')
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    
    if face_index is None:
        return jsonify({'success': False, 'message': 'Face index is required'})
    
    # Get the face data from detected_faces_for_selection
    # face_index is the array position (0, 1, 2...) in the order faces were sent to frontend
    with detected_faces_lock:
        if len(detected_faces_for_selection) == 0:
            return jsonify({'success': False, 'message': 'No faces available for selection. The faces may have been cleared. Please try capturing again.'})
        
        if face_index >= len(detected_faces_for_selection):
            return jsonify({'success': False, 'message': f'Invalid face index: {face_index} (max: {len(detected_faces_for_selection)-1})'})
        
        face_data = detected_faces_for_selection[face_index]
        print(f"📸 Capturing face at array index {face_index} for '{name}' (total faces available: {len(detected_faces_for_selection)})")
        # Convert embedding back to numpy array
        face_embedding = np.array(face_data['embedding'], dtype=np.float32)
        
        # Decode the full-size image from base64
        try:
            img_bytes = base64.b64decode(face_data['image_full'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            face_roi = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if face_roi is None:
                return jsonify({'success': False, 'message': 'Failed to decode face image'})
            
            # Store face
            enforce_face_limit()  # Remove oldest if at limit
            face_id_counter += 1
            known_faces[face_id_counter] = {
                'name': name,
                'embedding': face_embedding,
                'image': face_roi.copy()
            }
            save_known_faces()
            print(f"✓ Captured face #{face_id_counter}: {name} (selected from {len(detected_faces_for_selection)} faces)")
            
            capture_requested = False
            capture_name = ""
            waiting_for_selection = False  # Clear the flag
            detected_faces_for_selection = []
            
            return jsonify({'success': True, 'message': f'Face captured for {name}'})
        except Exception as e:
            print(f"⚠ Error capturing specific face: {e}")
            return jsonify({'success': False, 'message': f'Error capturing face: {str(e)}'})

@app.route('/known_faces', methods=['GET'])
def get_known_faces():
    """Get list of known faces"""
    faces_data = {}
    for face_id, face_info in known_faces.items():
        try:
            face_image = face_info['image']
            if face_image is None or face_image.size == 0:
                continue
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                success, buffer = cv2.imencode('.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                continue
            if not success or buffer is None:
                continue
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            faces_data[str(face_id)] = {
                'name': face_info['name'],
                'image': img_base64
            }
        except Exception as e:
            continue
    
    return jsonify({'faces': faces_data})

@app.route('/delete_face', methods=['POST'])
def delete_face():
    """Delete a known face"""
    global known_faces
    data = request.json
    face_id = data.get('face_id')
    
    if face_id in known_faces:
        del known_faces[face_id]
        save_known_faces()
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Face not found'})

@app.route('/get_mode', methods=['GET'])
def get_mode():
    """Get current mode"""
    global current_mode
    with mode_lock:
        return jsonify({'mode': current_mode})

if __name__ == '__main__':
    print("=" * 60)
    print("AI Vision Application - Merged Object Detection & Face Recognition")
    print("=" * 60)
    
    # Load known faces
    load_known_faces()
    
    # Start camera with default mode (object)
    if start_camera('object'):
        print("=" * 60)
        print("Application started successfully!")
        print("Access the web interface at: http://localhost:5000")
        print("Switch modes at: http://localhost:5000/mode")
        print("=" * 60)
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        print("Failed to start camera. Exiting.")
        sys.exit(1)

