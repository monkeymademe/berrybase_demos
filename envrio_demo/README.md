# Enviro Indoor Sensor Dashboard

A web-based dashboard for displaying real-time and historical sensor data from the Pimoroni Enviro Indoor board, displayed on a Raspberry Pi 4. The Enviro board sends data over WiFi directly to the Pi 4's web service.

## Features

- **Real-time Sensor Readings**: Display current temperature, humidity, pressure, light, and gas readings with animated meter bars
- **Historical Data Visualization**: Interactive graphs showing sensor data over time (6h, 24h, 48h, 7d)
- **Sleep Mode Indicator**: Visual watermark when the system is inactive (no recent data)
- **Beautiful UI**: Modern, responsive web interface with smooth animations
- **WiFi Communication**: No USB cables needed - Enviro board sends data wirelessly

## Hardware Requirements

- Raspberry Pi 4 (connected to your local WiFi network)
- Pimoroni Enviro Indoor board (with Raspberry Pi Pico W)
- Both devices must be on the same WiFi network

## Software Requirements

### Raspberry Pi 4
- Python 3.7+
- pip
- Flask web framework

### Enviro Indoor Board
- Comes preloaded with firmware (no additional setup needed!)
- Just needs to be provisioned via the web interface

## Installation

### 1. Install Python Dependencies on Pi 4

```bash
cd /home/pi/berrybase_demos/envrio_demo
pip3 install -r requirements.txt
```

### 2. Find Your Pi 4's IP Address

You'll need this IP address when provisioning the Enviro board:

```bash
hostname -I
```

Or check your router's admin panel for the Pi's IP address.

### 3. Start the Web Application

```bash
python3 web_app.py
```

The web service will start on port 5000. Keep this running - you'll need it to receive data from the Enviro board.

**Note**: The data endpoint will be: `http://<your-pi-ip>:5000/api/data`

### 4. Provision Your Enviro Indoor Board

The Enviro board comes preloaded with firmware. Follow these steps to configure it:

1. **Power up the board** - Plug in USB power or insert batteries
2. **Press the POKE button** - This wakes the board and starts provisioning mode
3. **Look for the WiFi network** - You should see "**Enviro Indoor Setup**" in your WiFi networks
4. **Connect to the setup network** - Use your phone, tablet, or computer
5. **Follow the provisioning wizard**:
   - **Choose a nickname** (e.g., "living-room", "bedroom-sensor")
   - **Select your WiFi network** and enter the password
   - **Set reading frequency** (recommended: every 15 minutes)
   - **Set upload frequency** (recommended: every 5 readings)
   - **Choose upload destination**: Select "**Custom HTTP endpoint**"
   - **Enter the endpoint URL**: `http://<your-pi-ip>:5000/api/data`
     - Example: `http://192.168.1.100:5000/api/data`
     - Replace `<your-pi-ip>` with your actual Pi 4 IP address

6. **Complete provisioning** - The board will save settings and start taking readings

For detailed provisioning instructions, see the [official Enviro documentation](https://github.com/pimoroni/enviro/blob/main/documentation/getting-started.md).

### 5. Access the Dashboard

Once the Enviro board starts sending data, open your web browser and navigate to:

- Local: `http://localhost:5000`
- Network: `http://<your-pi-ip>:5000`

## Running as a Service (Optional)

To run the web app automatically on boot, create a systemd service file:

```bash
sudo nano /etc/systemd/system/enviro-web.service
```

Add:
```ini
[Unit]
Description=Enviro Indoor Web Dashboard
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/berrybase_demos/envrio_demo
ExecStart=/usr/bin/python3 /home/pi/berrybase_demos/envrio_demo/web_app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable enviro-web.service
sudo systemctl start enviro-web.service
```

Check status:
```bash
sudo systemctl status enviro-web.service
```

## File Structure

```
envrio_demo/
├── web_app.py               # Flask web application (receives data & serves UI)
├── templates/
│   └── index.html          # Web UI dashboard
├── sensor_data.db          # SQLite database (created automatically)
├── requirements.txt        # Python dependencies
├── start_services.sh       # Convenience script (now just starts web app)
└── README.md               # This file
```

## API Endpoints

- `GET /` - Main dashboard page
- `POST /api/data` - Receive sensor data from Enviro board (used by Enviro)
- `GET /api/current` - Get latest sensor readings (used by dashboard)
- `GET /api/historical?hours=24` - Get historical data (default 24 hours)
- `GET /api/stats` - Get statistics for last 24 hours

## How It Works

1. **Enviro Board**: Takes sensor readings at configured intervals (default: every 15 minutes)
2. **Data Collection**: Stores readings locally, then uploads in batches (default: every 5 readings)
3. **WiFi Upload**: Sends data via HTTP POST to your Pi 4's `/api/data` endpoint
4. **Database Storage**: Pi 4 receives and stores data in SQLite database
5. **Web Dashboard**: Displays current readings and historical graphs

## Troubleshooting

### Enviro board not connecting to WiFi
- Check that you entered the correct WiFi password during provisioning
- Ensure the board is within range of your router
- The WARNING LED (red) will blink if there's a connection problem
- You can re-provision by pressing RESET, then POKE

### No data appearing in dashboard
- Verify the web app is running: `ps aux | grep web_app.py`
- Check that the Enviro board is powered and connected to WiFi
- Verify the endpoint URL in Enviro's config matches your Pi's IP
- Check web app logs for incoming POST requests
- Press POKE on the Enviro board to force an immediate reading/upload

### Web app won't start
- Check if port 5000 is available: `sudo lsof -i :5000`
- Verify Flask is installed: `pip3 list | grep Flask`
- Check for errors: `python3 web_app.py` (run in foreground to see errors)

### Finding your Pi's IP address
- Run: `hostname -I` or `ip addr show`
- Check your router's admin panel for connected devices
- Use a static IP or DHCP reservation to keep the IP consistent

### Enviro board endpoint configuration
- The endpoint URL format is: `http://<ip-address>:5000/api/data`
- Do NOT include a trailing slash
- Use `http://` not `https://` (unless you've set up SSL)
- Make sure port 5000 matches your web app port

## Notes

- The system detects "sleeping" state when no data is received for 10+ minutes
- Data gaps in graphs are normal when the Pi is turned off at night
- The database file (`sensor_data.db`) will grow over time - consider periodic cleanup if needed
- The Enviro board uses deep sleep between readings to conserve battery power
- Readings are uploaded in batches for power efficiency

## References

- [Enviro Getting Started Guide](https://github.com/pimoroni/enviro/blob/main/documentation/getting-started.md)
- [Enviro Custom HTTP Endpoint Documentation](https://github.com/pimoroni/enviro/blob/main/documentation/destinations/custom-http-endpoint.md)
- [Pimoroni Enviro GitHub Repository](https://github.com/pimoroni/enviro)

## License

This project is provided as-is for educational and personal use.
