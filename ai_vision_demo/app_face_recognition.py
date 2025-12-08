#!/usr/bin/env python3
"""
Face Recognition Web Application
- Detects faces in video feed
- Allows capturing and naming faces
- Recognizes and displays names on detected faces
"""

from flask import Flask, Response, render_template_string, request, jsonify
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
from PIL import Image

app = Flask(__name__)

# Camera and streaming setup
picam2 = None
hailo = None
camera_lock = threading.Lock()
streaming_output = None

# Face recognition storage
known_faces_file = 'known_faces.pkl'
known_faces = {}  # {face_id: {'name': str, 'embedding': np.array, 'image': np.array}}
face_id_counter = 0
match_debug_count = 0  # Debug counter for face matching

# Face detection/recognition models
face_detector = None
face_recognizer = None

# Capture request
capture_requested = False
capture_name = ""

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
    # Try multiple possible paths
    possible_paths = [
        '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml',
    ]
    
    # Try to get from cv2.data if available
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
        print("  Tried paths:", possible_paths)
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
    
    # Initialize face recognizer (LBPH - Local Binary Patterns Histograms)
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✓ Face recognizer initialized")
    except:
        # Fallback: use basic face matching
        print("⚠ Using basic face matching (no advanced recognizer)")
        face_recognizer = None
    
    return True

def detect_faces(image):
    """Detect faces in image using OpenCV - optimized for speed"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Optimized parameters for faster detection
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,  # Larger steps = faster (was 1.1)
        minNeighbors=4,  # Fewer neighbors = faster (was 5)
        minSize=(40, 40),  # Larger min size = faster (was 30x30)
        flags=cv2.CASCADE_SCALE_IMAGE  # Faster flag
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
    
    # Get current embedding dimension
    current_dim = len(face_embedding) if hasattr(face_embedding, '__len__') else face_embedding.shape[0]
    
    best_match = None
    best_distance = float('inf')
    all_distances = []
    skipped_old = 0
    
    for face_id, face_data in known_faces.items():
        known_embedding = face_data['embedding']
        known_dim = len(known_embedding) if hasattr(known_embedding, '__len__') else known_embedding.shape[0]
        
        # Skip faces with mismatched embedding dimensions (old format)
        if known_dim != current_dim:
            skipped_old += 1
            if match_debug_count < 3:
                print(f"  ⚠ Skipping face {face_id} ({face_data['name']}) - old embedding format ({known_dim}D vs {current_dim}D)")
            continue
        
        # Calculate Euclidean distance between embeddings
        distance = np.linalg.norm(face_embedding - known_embedding)
        all_distances.append((face_id, distance, face_data['name']))
        if distance < best_distance:
            best_distance = distance
            best_match = face_id
    
    # Warn if we skipped old faces
    if skipped_old > 0 and match_debug_count < 3:
        print(f"  ⚠ Note: {skipped_old} face(s) skipped due to old embedding format. Please recapture them.")
        match_debug_count += 1
    
    if len(all_distances) == 0:
        # No compatible faces found
        return None, 0.0
    
    # Debug: print distances (only first few frames to avoid spam)
    if match_debug_count < 10:
        print(f"  Face matching distances: {[(name, f'{d:.3f}') for _, d, name in sorted(all_distances, key=lambda x: x[1])[:3]]}")
        match_debug_count += 1
    
    # Lower distance = better match
    # With histogram-based embeddings (global + 4 spatial histograms, no L2 normalization):
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
        
        # Require minimum 10% similarity to match (reduced from 20%)
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
        # Stop systemd user services
        subprocess.run(['systemctl', '--user', 'stop', 
                       'pipewire.service', 'pipewire.socket', 
                       'pipewire-pulse.socket', 'wireplumber.service'],
                      capture_output=True, timeout=5, check=False)
        
        # Also kill any remaining processes
        subprocess.run(['pkill', '-9', '-f', 'pipewire'], 
                      capture_output=True, timeout=2, check=False)
        subprocess.run(['pkill', '-9', '-f', 'wireplumber'], 
                      capture_output=True, timeout=2, check=False)
        
        # Wait and check if devices are free
        for i in range(5):
            time.sleep(0.5)
            try:
                result = subprocess.run(['fuser', '/dev/media*'], 
                                      capture_output=True, timeout=1, check=False)
                if result.returncode != 0:
                    print("✓ Camera services stopped and devices are free")
                    return True
            except:
                pass
        
        print("✓ Camera services stopped")
        return True
    except Exception as e:
        print(f"Warning: Could not stop camera services: {e}")
        return False

def start_camera():
    """Initialize and start the camera"""
    global picam2, streaming_output
    
    # Stop pipewire/wireplumber before accessing camera
    stop_camera_services()
    
    # Check for any existing camera processes or app instances FIRST
    try:
        result = subprocess.run(['pgrep', '-f', 'python.*app_face'], 
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
    
    # Clean up any existing camera instance
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
    
    # Retry camera initialization up to 3 times
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
        # Configure camera - Try Full HD (1920x1080) first, then fall back to HD (1280x720), then 960x720
        video_w, video_h = 1920, 1080
        
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
            global streaming_output, capture_requested, capture_name, face_id_counter
            
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
            last_recognition_frame = -1
            recognition_skip_frames = 2  # Only recognize every N frames (0 = every frame, 2 = every 3rd frame)
            face_names = {}  # Store recognition results to reuse across frames
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
                    # Capture frame
                    request = picam2.capture_request()
                    frame = request.make_array("main")
                    
                    # Fix colors (in-place swap is faster)
                    frame_swapped = frame.copy()
                    frame_swapped[:, :, [0, 2]] = frame_swapped[:, :, [2, 0]]
                    frame_bgr = cv2.cvtColor(frame_swapped, cv2.COLOR_RGB2BGR)
                    
                    # Detect faces
                    faces = detect_faces(frame_bgr)
                    
                    # Handle capture request
                    if capture_requested and len(faces) > 0:
                        # Use the largest face
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest_face
                        face_roi = frame_bgr[y:y+h, x:x+w].copy()  # Make a copy to ensure it's saved
                        
                        # Extract embedding
                        face_embedding = extract_face_embedding(face_roi)
                        
                        # Store face
                        face_id_counter += 1
                        name_to_save = capture_name if capture_name else f"Person {face_id_counter}"
                        known_faces[face_id_counter] = {
                            'name': name_to_save,
                            'embedding': face_embedding,
                            'image': face_roi.copy()  # Store a copy of the image
                        }
                        save_known_faces()
                        print(f"✓ Captured face #{face_id_counter}: {name_to_save} (image shape: {face_roi.shape})")
                        
                        capture_requested = False
                        capture_name = ""
                    elif capture_requested and len(faces) == 0:
                        # No face detected when capture was requested
                        if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                            print("⚠ Capture requested but no face detected")
                    
                    # Recognize faces (skip recognition on some frames for performance)
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
                                
                                # Match if within 80 pixels and similar size (stricter threshold)
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
                    
                    # Draw faces with recognition results
                    for idx, (x, y, w, h) in enumerate(faces):
                        # Get recognition result (or default to unknown)
                        if idx in face_names:
                            name, similarity = face_names[idx]
                        else:
                            name, similarity = None, 0.0
                        
                        # Draw bounding box
                        color = (0, 255, 0) if name else (0, 0, 255)
                        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
                        
                        # Draw label
                        if name:
                            label = f"{name} {int(similarity*100)}%"
                        else:
                            label = "Unknown"
                        
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        label_y = max(y, label_size[1] + 10)
                        
                        # Draw label background
                        cv2.rectangle(frame_bgr, (x, y - label_size[1] - 10),
                                    (x + label_size[0], y), color, -1)
                        
                        # Draw label text
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
        time.sleep(0.5)
        
        print("✓ Camera started with face recognition")
        return True
            
    except Exception as e:
        print(f"Error starting camera: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main page with video stream and face capture controls"""
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
            --text-secondary: #666;
            --border-color: #f0f0f0;
        }

        [data-theme="dark"] {
            --bg-gradient-start: #6b6a6a;
            --bg-gradient-end: #cd3c65;
            --bg-color: #0f1419;
            --card-bg: #1a1f2e;
            --text-primary: #e0e0e0;
            --text-secondary: #b0b0b0;
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
        h1 {
            flex: 1;
            text-align: center;
            color: white;
            font-size: 2.5em;
            margin: 0;
        }
        .feed-container {
            display: flex;
            gap: 20px;
            width: 100%;
            margin-bottom: 20px;
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
            border-radius: 10px;
        }

        .controls {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .control-group {
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }

        input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 12px;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            font-size: 16px;
            background: var(--card-bg);
            color: var(--text-primary);
        }

        button {
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
        }

        .known-faces h2 {
            margin-bottom: 15px;
            color: var(--text-primary);
        }

        .faces-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
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
            height: 120px;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header-row">
            <div></div>
            <h1>Face Recognition Camera</h1>
            <button class="theme-toggle" id="themeToggle" onclick="toggleTheme()" aria-label="Toggle dark mode">
                <svg id="themeIcon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>
                </svg>
            </button>
        </div>
        <div class="feed-container">
            <div class="feed-description">
                <p>Detect, capture, and recognize faces in real-time.</p>
            </div>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <input type="text" id="personName" placeholder="Enter person's name..." />
                <button onclick="captureFace()">Capture Face</button>
                <button onclick="refreshFaces()">Refresh List</button>
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

    <script>
        function captureFace() {
            const name = document.getElementById('personName').value.trim();
            if (!name) {
                showStatus('Please enter a name first', 'error');
                return;
            }

            fetch('/capture_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name: name})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showStatus(`Face captured: ${name}`, 'success');
                    document.getElementById('personName').value = '';
                    setTimeout(refreshFaces, 500);
                } else {
                    showStatus(`Error: ${data.message}`, 'error');
                }
            })
            .catch(error => {
                showStatus(`Error: ${error.message}`, 'error');
            });
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
            if (!confirm('Delete this face?')) return;
            
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

        function toggleTheme() {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcon(newTheme);
        }

        function updateThemeIcon(theme) {
            const svg = document.getElementById('themeIcon');
            if (theme === 'dark') {
                svg.innerHTML = '<path d="M8 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8M8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0m0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13m8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5M3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8m10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0m-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0m9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707M4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708"/>';
            } else {
                svg.innerHTML = '<path d="M6 .278a.77.77 0 0 1 .08.858 7.2 7.2 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277q.792-.001 1.533-.16a.79.79 0 0 1 .81.316.73.73 0 0 1-.031.893A8.35 8.35 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.75.75 0 0 1 6 .278"/>';
            }
        }

        function initTheme() {
            const savedTheme = localStorage.getItem('theme') || 'light';
            document.documentElement.setAttribute('data-theme', savedTheme);
            updateThemeIcon(savedTheme);
        }

        window.addEventListener('DOMContentLoaded', () => {
            initTheme();
            refreshFaces();
            setInterval(refreshFaces, 5000); // Refresh every 5 seconds
        });
    </script>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            if streaming_output is not None:
                try:
                    frame = streaming_output.read()
                    if frame:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        time.sleep(0.01)
                except Exception as e:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face', methods=['POST'])
def capture_face():
    """Capture and store a face"""
    global capture_requested, capture_name
    data = request.json
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    
    capture_name = name
    capture_requested = True
    
    # Wait a bit for the frame capture thread to process
    time.sleep(1)
    
    if capture_requested:  # Still requested means it wasn't processed
        capture_requested = False
        return jsonify({'success': False, 'message': 'No face detected. Please ensure a face is visible.'})
    
    return jsonify({'success': True, 'message': f'Face captured for {name}'})

@app.route('/known_faces', methods=['GET'])
def get_known_faces():
    """Get list of known faces"""
    faces_data = {}
    for face_id, face_info in known_faces.items():
        try:
            # Get the face image
            face_image = face_info['image']
            
            # Ensure it's a valid image array
            if face_image is None or face_image.size == 0:
                print(f"⚠ Warning: Face {face_id} has invalid image")
                continue
            
            # Ensure image is in correct format (BGR for OpenCV)
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Image is BGR, encode directly
                success, buffer = cv2.imencode('.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                print(f"⚠ Warning: Face {face_id} image has unexpected shape: {face_image.shape}")
                continue
            
            if not success or buffer is None:
                print(f"⚠ Warning: Failed to encode face {face_id} image")
                continue
            
            img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            faces_data[str(face_id)] = {
                'name': face_info['name'],
                'image': img_base64
            }
        except Exception as e:
            print(f"⚠ Error processing face {face_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✓ Returning {len(faces_data)} faces to web UI (total known: {len(known_faces)})")
    return jsonify({'faces': faces_data})

@app.route('/delete_face', methods=['POST'])
def delete_face():
    """Delete a known face"""
    data = request.json
    face_id = data.get('face_id')
    
    if face_id in known_faces:
        del known_faces[face_id]
        save_known_faces()
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Face not found'})

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Face Recognition Web Application")
    print("=" * 60)
    
    # Initialize face recognition
    if not init_face_recognition():
        print("⚠ Warning: Face recognition initialization had issues, continuing anyway...")
    
    # Load known faces
    load_known_faces()
    
    if start_camera():
        print("\n✓ Camera initialized successfully")
        print("✓ Starting Flask web server...")
        print("\nAccess the web UI at: http://<raspberry-pi-ip>:5000")
        print("=" * 60)
        
        try:
            app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cleanup_camera()
    else:
        print("Failed to start camera. Exiting.")