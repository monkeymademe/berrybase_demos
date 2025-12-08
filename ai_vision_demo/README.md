# AI Object Detection Web Application

A Flask web application that streams live camera feed with Hailo AI object detection overlays.

## Features

- **Live MJPEG streaming** from Raspberry Pi Camera V3
- **Real-time object detection** using Hailo YOLOv8 model
- **Visual overlays** showing detected objects with bounding boxes and labels
- **Web-based UI** accessible from any device on your network

## Requirements

- Raspberry Pi with Camera V3
- Hailo AI+ HAT installed
- Python 3.9+
- Required packages:
  - Flask
  - picamera2
  - OpenCV (cv2)
  - NumPy
  - Pillow (PIL)

## Installation

1. Install required Python packages:
```bash
pip3 install flask opencv-python numpy pillow
```

2. Ensure Hailo models are installed:
```bash
# Models should be in /usr/share/hailo-models/
ls /usr/share/hailo-models/yolov8s_h8*.hef
```

3. Ensure Hailo post-processing config exists:
```bash
ls /usr/share/rpi-camera-assets/hailo_yolov8_inference.json
```

## Usage

### Run Manually

1. **Optional:** If you get "device busy" errors, stop pipewire/wireplumber first:
```bash
./stop_camera_services.sh
```

2. Run the application:
```bash
python3 app.py
```

3. Open a web browser and navigate to:
```
http://<raspberry-pi-ip>:5000
```

Replace `<raspberry-pi-ip>` with your Raspberry Pi's IP address (e.g., `192.168.1.100`).

### Optional: Install as System Service

Install the application as a systemd service that will start automatically at boot:

```bash
sudo ./setup_service.sh
```

The script will:
- Create a systemd service file
- Enable the service to start at boot
- Optionally start the service immediately

After installation, you can manage the service with:
```bash
# Check status
sudo systemctl status ai-vision-demo.service

# View logs
sudo journalctl -u ai-vision-demo.service -f

# Start/Stop/Restart
sudo systemctl start ai-vision-demo.service
sudo systemctl stop ai-vision-demo.service
sudo systemctl restart ai-vision-demo.service
```

## How It Works

1. **Camera Setup**: Uses `picamera2` to capture frames at 640x480 resolution
2. **Hailo Inference**: Configures `post_process_file` to run YOLOv8 object detection
3. **Detection Drawing**: The `object_detect_draw_cv` stage draws bounding boxes on the lores stream
4. **Web Streaming**: Frames are encoded to JPEG and streamed via MJPEG to the web browser

## Detected Objects

The application can detect 80 different object classes from the COCO dataset, including:
- People, vehicles (cars, buses, trucks, motorcycles, bicycles)
- Animals (dogs, cats, birds, horses, etc.)
- Furniture (chairs, couches, beds, tables)
- Electronics (laptops, phones, TVs, keyboards)
- Food items (pizza, donuts, bananas, apples, etc.)
- And many more!

## Troubleshooting

- **No detections appearing**: Ensure objects are well-lit and clearly visible
- **Stream not loading**: Check that port 5000 is not blocked by firewall
- **Camera errors**: Ensure camera is properly connected and not in use by another process

## Notes

- The application uses the lores stream (640x480) which is processed by Hailo
- Detection results are drawn directly on the frame by the post-processing pipeline
- The stream updates in real-time as objects are detected

