#!/usr/bin/env python3
"""
Face Recognition Web Application with Hailo AI
- Uses Hailo SCRFD model for face detection (hardware accelerated)
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
video_w, video_h = 1920, 1080  # Will be set during camera initialization
model_w, model_h = 640, 640  # SCRFD model input size (will be updated from model)

# Global storage for detections from post_callback
hailo_detections = []
hailo_detections_lock = threading.Lock()

# Face recognition storage
known_faces_file = 'known_faces.pkl'
known_faces = {}  # {face_id: {'name': str, 'embedding': np.array, 'image': np.array}}
face_id_counter = 0
match_debug_count = 0  # Debug counter for face matching
MAX_KNOWN_FACES = 5  # Maximum number of known faces to keep (FIFO - oldest removed when limit reached)

# Face detection/recognition models
face_detector = None  # Not used with Hailo, kept for compatibility
face_recognizer = None  # Not used, kept for compatibility

# Capture request
capture_requested = False
capture_name = ""
capture_request_frame = 0  # Track when capture was requested (frame count)
detected_faces_for_selection = []  # Store detected faces with images for selection overlay
waiting_for_selection = False  # Flag to prevent auto-capture when waiting for user selection
detected_faces_lock = threading.Lock()

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

def init_hailo_face_detection():
    """Initialize Hailo for face detection - tries YOLOv5 personface first (list output), then SCRFD"""
    global hailo, model_w, model_h
    
    # Clean up any existing Hailo instance first
    if hailo is not None:
        try:
            hailo.close()
            hailo = None
            print("Cleaned up existing Hailo instance")
            time.sleep(0.5)  # Give device time to release
        except:
            pass
    
    # Try YOLOv5 personface first (outputs in list format like object detection)
    # Then fall back to SCRFD (outputs raw feature maps)
    hailo_model_paths = [
        '/usr/share/hailo-models/yolov5s_personface_h8.hef',
        '/usr/share/hailo-models/yolov5s_personface_h8l.hef',
        '/usr/share/hailo-models/scrfd_2.5g_h8.hef',
        '/usr/share/hailo-models/scrfd_10g_h8.hef',
        '/usr/share/hailo-models/scrfd_2.5g_h8l.hef',
        '/usr/share/hailo-models/scrfd_10g_h8l.hef',
    ]
    
    hailo_model_path = None
    model_type = None
    for path in hailo_model_paths:
        if os.path.exists(path):
            hailo_model_path = path
            if 'yolov5' in path.lower():
                model_type = 'yolov5'
            elif 'scrfd' in path.lower():
                model_type = 'scrfd'
            break
    
    if not hailo_model_path:
        print("⚠ ERROR: Hailo face detection model not found")
        print("  Tried paths:", hailo_model_paths)
        print("  Please ensure a face detection model is installed in /usr/share/hailo-models/")
        return False
    
    print(f"Initializing Hailo with face detection model: {hailo_model_path} (type: {model_type})")
    if 'h8l' in hailo_model_path.lower():
        print("⚠ Note: Using H8L model on Hailo8 device - may have lower performance. H8 model preferred.")
    
    # Retry initialization with delays if device is busy
    max_retries = 3
    retry_delay = 1.0
    for attempt in range(max_retries):
        try:
            hailo = Hailo(hailo_model_path)
            model_h, model_w, _ = hailo.get_input_shape()
            print(f"✓ Hailo initialized. Model input shape: {model_w}x{model_h}, type: {model_type}")
            # Store model type for later use
            hailo._model_type = model_type
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠ Hailo initialization attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"⚠ ERROR: Failed to initialize Hailo after {max_retries} attempts: {e}")
                import traceback
                traceback.print_exc()
                return False
    return False

def decode_scrfd_detections(boxes_tensor, scores_tensor, scale, stride, frame_w, frame_h, threshold=0.3):
    """
    Decode SCRFD detections from feature maps
    boxes_tensor: shape (H, W, 4 or 8) - box predictions
    scores_tensor: shape (H, W, 2) - score predictions (background, face)
    scale: scale factor for this feature map
    stride: stride of this feature map
    """
    faces = []
    if boxes_tensor is None or scores_tensor is None:
        return faces
    
    try:
        h, w = scores_tensor.shape[:2]
        
        # Get face scores (second channel if 2D, or direct if 1D)
        if len(scores_tensor.shape) == 3 and scores_tensor.shape[2] >= 2:
            face_scores = scores_tensor[:, :, 1]
        elif len(scores_tensor.shape) == 2:
            face_scores = scores_tensor
        else:
            return faces
        
        # Debug: check score range (only once)
        if not hasattr(decode_scrfd_detections, '_score_debug'):
            print(f"🔍 Score range: min={face_scores.min():.4f}, max={face_scores.max():.4f}, mean={face_scores.mean():.4f}")
            print(f"🔍 Scores above threshold {threshold}: {np.sum(face_scores >= threshold)}")
            decode_scrfd_detections._score_debug = True
        
        # Lower threshold if no detections found
        threshold_used = threshold
        if np.sum(face_scores >= threshold) == 0:
            # Try much lower threshold
            threshold_used = max(0.05, threshold * 0.1)
            if not hasattr(decode_scrfd_detections, '_threshold_warned'):
                print(f"⚠ No faces at threshold {threshold}, trying {threshold_used}")
                decode_scrfd_detections._threshold_warned = True
        
        # Find locations where face score exceeds threshold
        valid_mask = face_scores >= threshold_used
        
        if not np.any(valid_mask):
            return faces
        
        # Get valid indices
        y_indices, x_indices = np.where(valid_mask)
        
        # Limit to top N detections per scale to avoid too many
        if len(y_indices) > 100:
            # Get top scores
            top_indices = np.argsort(face_scores[y_indices, x_indices])[-100:][::-1]
            y_indices = y_indices[top_indices]
            x_indices = x_indices[top_indices]
        
        for y_idx, x_idx in zip(y_indices, x_indices):
            score = float(face_scores[y_idx, x_idx])
            if score < threshold_used:
                continue
            
            # Get box prediction
            if len(boxes_tensor.shape) == 3 and boxes_tensor.shape[2] >= 4:
                box_pred = boxes_tensor[y_idx, x_idx, :4]
            else:
                continue
            
            # Decode box: SCRFD uses center-based format with offsets
            # Box format: [dx, dy, dw, dh] or [x1, y1, x2, y2]
            cx = (x_idx + 0.5) * stride
            cy = (y_idx + 0.5) * stride
            
            # Try interpreting as offsets first
            dx, dy, dw, dh = float(box_pred[0]), float(box_pred[1]), float(box_pred[2]), float(box_pred[3])
            
            # If values are very large, they might be absolute coordinates
            if abs(dx) > 100 or abs(dy) > 100:
                # Treat as absolute coordinates (normalized to model size)
                x1 = int(dx * scale)
                y1 = int(dy * scale)
                x2 = int(dw * scale)
                y2 = int(dh * scale)
            else:
                # Treat as offsets
                x1 = int((cx + dx - dw/2) * scale)
                y1 = int((cy + dy - dh/2) * scale)
                x2 = int((cx + dx + dw/2) * scale)
                y2 = int((cy + dy + dh/2) * scale)
            
            # Ensure x2 > x1 and y2 > y1
            if x2 <= x1:
                x2 = x1 + 1
            if y2 <= y1:
                y2 = y1 + 1
            
            # Clamp to image bounds
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))
            
            w_box = x2 - x1
            h_box = y2 - y1
            
            if w_box > 10 and h_box > 10:  # Minimum face size
                faces.append((x1, y1, w_box, h_box))
    except Exception as e:
        if not hasattr(decode_scrfd_detections, '_error_printed'):
            print(f"⚠ Error decoding SCRFD detections: {e}")
            import traceback
            traceback.print_exc()
            decode_scrfd_detections._error_printed = True
    
    return faces

def extract_face_detections(hailo_output, frame_w, frame_h, threshold=0.3):
    """
    Extract face detections from Hailo output
    Tries list format first (like object detection), then falls back to SCRFD dictionary format
    Returns list of faces in format: [(x, y, w, h), ...] matching OpenCV format
    """
    faces = []
    if hailo_output is None:
        return faces
    
    # Debug: check output structure (only once)
    if not hasattr(extract_face_detections, '_debug_printed'):
        print(f"🔍 Debug: Hailo output type: {type(hailo_output)}")
        if isinstance(hailo_output, dict):
            print(f"🔍 Debug: Dictionary keys: {list(hailo_output.keys())}")
            for key, value in hailo_output.items():
                if isinstance(value, np.ndarray):
                    print(f"🔍 Debug: Key '{key}': shape {value.shape}, dtype {value.dtype}")
        elif isinstance(hailo_output, (list, tuple)):
            print(f"🔍 Debug: List length: {len(hailo_output)}")
            if len(hailo_output) > 0:
                print(f"🔍 Debug: First element type: {type(hailo_output[0])}, length: {len(hailo_output[0]) if hasattr(hailo_output[0], '__len__') else 'N/A'}")
        extract_face_detections._debug_printed = True
    
    # Try list format first (like object detection - post-processed detections)
    if isinstance(hailo_output, (list, tuple)):
        # Format: list where each element is detections for that class
        # For YOLOv5 personface: class 0 = person, class 1 = face (label_offset: 1)
        # For other models: class 0 might be face
        if not hasattr(extract_face_detections, '_list_format_checked'):
            print(f"🔍 Processing list format: length={len(hailo_output)}")
            extract_face_detections._list_format_checked = True
        
        # Determine which class is face based on model type
        # YOLOv5 personface has label_offset: 1, so class 1 is face
        # Other models might have class 0 as face
        face_class_ids = []
        if hasattr(hailo, '_model_type') and hailo._model_type == 'yolov5':
            face_class_ids = [1]  # YOLOv5 personface: class 1 is face
        else:
            face_class_ids = [0]  # Default: class 0 is face
        
        for class_id, detections in enumerate(hailo_output):
            if not hasattr(detections, '__len__'):
                continue
            
            # Only process face class(es)
            if class_id not in face_class_ids:
                continue
                
            if not hasattr(extract_face_detections, '_list_debug_printed'):
                print(f"🔍 Class {class_id} (face): {len(detections)} detections")
            
            for detection in detections:
                if not hasattr(detection, '__len__') or len(detection) < 5:
                    continue
                score = detection[4]
                
                # Debug first detection
                if not hasattr(extract_face_detections, '_list_debug_printed'):
                    print(f"🔍 First detection: {detection}, score={score}, threshold={threshold}")
                    extract_face_detections._list_debug_printed = True
                
                if score >= threshold:
                    # Hailo format: [y0, x0, y1, x1, score] (normalized)
                    y0, x0, y1, x1 = detection[:4]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x1_px = int(x0 * frame_w)
                    y1_px = int(y0 * frame_h)
                    x2_px = int(x1 * frame_w)
                    y2_px = int(y1 * frame_h)
                    
                    # Convert to (x, y, w, h) format for compatibility
                    x = max(0, x1_px)
                    y = max(0, y1_px)
                    w = max(1, x2_px - x1_px)
                    h = max(1, y2_px - y1_px)
                    
                    # Ensure within frame bounds
                    x = min(x, frame_w - 1)
                    y = min(y, frame_h - 1)
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)
                    
                    if w > 0 and h > 0:
                        faces.append((x, y, w, h))
        
        if faces:
            if not hasattr(extract_face_detections, '_list_format_printed'):
                print(f"✓ Using list format (post-processed detections): found {len(faces)} faces")
                extract_face_detections._list_format_printed = True
            return faces
        elif not hasattr(extract_face_detections, '_list_format_printed'):
            print(f"⚠ List format processed but no faces found (threshold={threshold}, checked classes: {face_class_ids})")
            extract_face_detections._list_format_printed = True
    
    # Handle dictionary output (SCRFD model format)
    if isinstance(hailo_output, dict):
        # SCRFD outputs feature maps at multiple scales
        # Pattern: conv43/conv42 (80x8 boxes, 80x2 scores), conv50/conv49 (40x8, 40x2), conv56/conv55 (20x8, 20x2)
        # These need to be decoded using anchor-based decoding
        
        # Group tensors by spatial dimensions to match boxes with scores
        tensor_groups = {}
        for key, value in hailo_output.items():
            if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                spatial_key = (value.shape[0], value.shape[1])
                if spatial_key not in tensor_groups:
                    tensor_groups[spatial_key] = {'boxes': [], 'scores': []}
                
                # Identify boxes (8 channels) or scores (2 channels)
                if value.shape[-1] == 8:
                    tensor_groups[spatial_key]['boxes'].append((key, value))
                elif value.shape[-1] == 2:
                    tensor_groups[spatial_key]['scores'].append((key, value))
        
        # Decode detections from each scale
        model_size = 640  # SCRFD input size
        scale_factor = min(frame_w, frame_h) / model_size
        
        # Typical SCRFD scales and strides
        # Scale 0: 80x80 feature map, stride 8
        # Scale 1: 40x40 feature map, stride 16  
        # Scale 2: 20x20 feature map, stride 32
        
        for spatial_key, tensors in tensor_groups.items():
            h, w = spatial_key
            boxes_list = tensors['boxes']
            scores_list = tensors['scores']
            
            if boxes_list and scores_list:
                # Use first matching pair
                boxes_key, boxes_tensor = boxes_list[0]
                scores_key, scores_tensor = scores_list[0]
                
                # Estimate stride from feature map size
                # For 640x640 input: 80x80 -> stride 8, 40x40 -> stride 16, 20x20 -> stride 32
                if h == 80:
                    stride = 8
                elif h == 40:
                    stride = 16
                elif h == 20:
                    stride = 32
                else:
                    # Estimate stride
                    stride = model_size // h
                
                # Decode detections from this scale
                scale_faces = decode_scrfd_detections(
                    boxes_tensor, scores_tensor,
                    scale_factor, stride,
                    frame_w, frame_h, threshold
                )
                
                faces.extend(scale_faces)
                
                if not hasattr(extract_face_detections, '_scale_printed'):
                    print(f"🔍 Decoding scale {h}x{w}: boxes={boxes_key} {boxes_tensor.shape}, scores={scores_key} {scores_tensor.shape}, stride={stride}")
                    if scale_faces:
                        print(f"   Found {len(scale_faces)} faces at this scale")
        
        if not hasattr(extract_face_detections, '_scale_printed'):
            extract_face_detections._scale_printed = True
    
    return faces

def detect_faces_from_request(request, frame_w, frame_h):
    """
    Detect faces using Hailo AI from a camera request
    Uses the same approach as object detection: hailo.run() on lores frame
    Returns faces in format: [(x, y, w, h), ...] matching OpenCV format
    """
    global hailo
    
    if hailo is None:
        return []
    
    try:
        # Use the same approach as object detection
        # Get lores frame for Hailo inference
        frame_lores = request.make_array("lores")
        
        # Run Hailo inference - same as object detection
        hailo_output = hailo.run(frame_lores)
        
        # Debug output (first 10 frames)
        if not hasattr(detect_faces_from_request, '_frame_count'):
            detect_faces_from_request._frame_count = 0
        detect_faces_from_request._frame_count += 1
        
        # Extract face detections (handles both list and dict formats)
        # Try with very low threshold first to see if we get any detections
        faces = extract_face_detections(hailo_output, frame_w, frame_h, threshold=0.1)
        
        # Apply Non-Maximum Suppression to remove overlapping/duplicate detections
        # This prevents the same face from being detected multiple times (e.g., close-up and wide angle)
        if len(faces) > 1:
            faces_before = len(faces)
            faces = non_max_suppression_faces(faces, overlap_threshold=0.4)
            if detect_faces_from_request._frame_count <= 5 and len(faces) < faces_before:
                print(f"   NMS: Removed {faces_before - len(faces)} duplicate face(s)")
        
        if detect_faces_from_request._frame_count <= 10:
            print(f"🔍 Frame {detect_faces_from_request._frame_count}:")
            print(f"   Hailo output type: {type(hailo_output)}")
            if isinstance(hailo_output, list):
                print(f"   List length: {len(hailo_output)}")
                if len(hailo_output) > 0:
                    print(f"   First element type: {type(hailo_output[0])}, length: {len(hailo_output[0]) if hasattr(hailo_output[0], '__len__') else 'N/A'}")
                    if len(hailo_output[0]) > 0:
                        print(f"   First detection: {hailo_output[0][0] if hasattr(hailo_output[0], '__getitem__') else 'N/A'}")
            elif isinstance(hailo_output, dict):
                print(f"   Dictionary keys: {list(hailo_output.keys())[:5]}")  # First 5 keys
            print(f"   Detected {len(faces)} faces (threshold=0.1)")
        
        return faces
    except Exception as e:
        if not hasattr(detect_faces_from_request, '_error_count'):
            detect_faces_from_request._error_count = 0
        detect_faces_from_request._error_count += 1
        
        if detect_faces_from_request._error_count < 10:
            print(f"⚠ Hailo face detection error: {e}")
            import traceback
            traceback.print_exc()
        return []

def non_max_suppression_faces(faces, overlap_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping face detections
    faces: list of (x, y, w, h) tuples
    overlap_threshold: IoU threshold for considering faces as duplicates (0.5 = 50% overlap)
    Returns: filtered list of faces
    """
    if len(faces) <= 1:
        return faces
    
    # Convert to (x1, y1, x2, y2) format for easier IoU calculation
    boxes = []
    for x, y, w, h in faces:
        boxes.append((x, y, x + w, y + h))
    
    # Calculate areas
    areas = [w * h for x, y, w, h in faces]
    
    # Sort by area (largest first) - keep larger detections
    indices = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i in indices:
        if i in suppressed:
            continue
        
        keep.append(i)
        x1_i, y1_i, x2_i, y2_i = boxes[i]
        
        # Suppress overlapping boxes
        for j in indices:
            if j == i or j in suppressed:
                continue
            
            x1_j, y1_j, x2_j, y2_j = boxes[j]
            
            # Calculate intersection
            x1_inter = max(x1_i, x1_j)
            y1_inter = max(y1_i, y1_j)
            x2_inter = min(x2_i, x2_j)
            y2_inter = min(y2_i, y2_j)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                continue  # No overlap
            
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # Calculate IoU (Intersection over Union)
            area_i = areas[i]
            area_j = areas[j]
            union_area = area_i + area_j - inter_area
            
            if union_area > 0:
                iou = inter_area / union_area
                
                # If overlap is too high, suppress the smaller box
                if iou > overlap_threshold:
                    suppressed.add(j)
    
    # Return filtered faces
    return [faces[i] for i in keep]

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
    # With histogram-based embeddings (global + 9 spatial histograms, no L2 normalization):
    # - Same person: distances typically 0.3-0.6
    # - Different people: distances typically 0.6-1.2
    max_distance = 0.70  # Slightly more lenient to catch more matches
    
    # CRITICAL: If multiple faces, check if the best match is clearly better than second best
    # This prevents misidentification when two people have similar distances
    if len(all_distances) > 1:
        sorted_distances = sorted(all_distances, key=lambda x: x[1])
        best_dist = sorted_distances[0][1]
        second_best_dist = sorted_distances[1][1]
        distance_gap = second_best_dist - best_dist
        
        # Require at least 0.12 gap between best and second best (more lenient)
        # Reduced from 0.20 to 0.12 to allow more matches while still preventing misidentification
        # Example: user=0.45, wife=0.57 -> gap=0.12 -> ACCEPT (clear enough)
        # Example: user=0.45, wife=0.50 -> gap=0.05 -> REJECT (too ambiguous)
        if distance_gap < 0.12:
            if match_debug_count <= 20:
                print(f"  ✗ Ambiguous match (best: {best_dist:.3f} ({sorted_distances[0][2]}), 2nd: {second_best_dist:.3f} ({sorted_distances[1][2]}), gap: {distance_gap:.3f} < 0.12)")
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
    """Initialize and start the camera with Hailo face detection"""
    global picam2, hailo, streaming_output, video_w, video_h, model_w, model_h
    
    # Stop pipewire/wireplumber before accessing camera
    stop_camera_services()
    
    # Check for any existing camera processes or app instances FIRST
    try:
        # Check for existing app processes (including all app variants)
        patterns = ['python.*app_face', 'python.*app_object', 'python.*app_merged', 'python.*app.py']
        current_pid = str(os.getpid())
        for pattern in patterns:
            result = subprocess.run(['pgrep', '-f', pattern], 
                                  capture_output=True, timeout=1, check=False)
            if result.returncode == 0:
                pids = result.stdout.decode().strip().split('\n')
                for pid in pids:
                    if pid and pid != current_pid:
                        print(f"⚠ Warning: Found existing app process (PID {pid}), killing it...")
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            time.sleep(0.5)
                            # If still running, force kill
                            try:
                                os.kill(int(pid), 0)  # Check if still exists
                                time.sleep(0.5)
                                os.kill(int(pid), signal.SIGKILL)
                                print(f"  Force killed process {pid}")
                            except ProcessLookupError:
                                pass  # Process already dead
                        except:
                            pass
        # Give processes time to fully terminate and release resources
        time.sleep(1.5)
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
        if hailo is not None:
            try:
                hailo.close()
                hailo = None
                print("Hailo closed.")
            except:
                pass
    time.sleep(0.5)
    
    # Initialize Hailo for face detection
    if not init_hailo_face_detection():
        print("⚠ ERROR: Failed to initialize Hailo face detection")
        return False
    
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
        
        # Face detection needs lores stream for Hailo inference
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
        
        # Set up post-processing callback if available
        try:
            picam2.set_post_callback(post_callback)
            print("✓ Post-processing callback set")
        except:
            pass  # post_callback might not be available in all picamera2 versions
        
        picam2.start()
        print("✓ Camera started")
        
        # Start frame capture thread
        def frame_capture_thread():
            global streaming_output, capture_requested, capture_name, face_id_counter, detected_faces_for_selection, capture_request_frame, waiting_for_selection, video_w, video_h
            
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
                    
                    # Detect faces using Hailo (use the same request to get lores stream)
                    faces = detect_faces_from_request(request, video_w, video_h)
                    
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
                        max_check_frames = 90  # Check for ~3 seconds (at ~30fps)
                        
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
        
        print("✓ Camera started with Hailo face detection")
        return True
            
    except Exception as e:
        print(f"Error starting camera: {e}")
        import traceback
        traceback.print_exc()
        return False

# Copy the rest of the file from app_face_recognition.py (routes, HTML template, etc.)
# I'll read the template section and copy it
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

    <script>
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
    """Capture and store a face - returns detected faces if multiple unknown"""
    global capture_requested, capture_name, detected_faces_for_selection, waiting_for_selection
    data = request.json
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({'success': False, 'message': 'Name is required'})
    
    capture_name = name
    capture_requested = True
    capture_request_frame = 0  # Will be set by frame thread
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
    print("Starting Face Recognition Web Application with Hailo AI")
    print("=" * 60)
    
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