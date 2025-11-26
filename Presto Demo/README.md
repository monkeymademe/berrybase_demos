# Presto Flight Card Display

A client application for the Pimoroni Presto display that connects to the flight tracker server and displays flight card information in a style similar to the web interface.

## Overview

This demo connects to the flight tracker server's Server-Sent Events (SSE) endpoint and displays flight information on a Presto e-ink display. The flight cards match the design aesthetic of the web interface (`index.html`) but are adapted for the smaller e-ink display.

## Features

- **Real-time flight updates**: Connects to the flight tracker server via SSE
- **Flight card display**: Shows callsign, route (origin → destination), altitude, speed, track, and more
- **Automatic selection**: Prefers flights with complete route information
- **E-ink optimized**: Designed for the Presto display's capabilities

## Requirements

- Raspberry Pi with Presto display connected
- Flight tracker server running (from `flightaware_demo`)
- Python 3.7+ (CPython on Raspberry Pi, not MicroPython)

**Important**: This script is designed to run on a Raspberry Pi that has the Presto display connected to it. It uses CPython (standard Python), not MicroPython. The Presto device itself runs MicroPython, but you control it from the Raspberry Pi using Python scripts.

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

   **Note**: The `presto` and `picovector` packages may need to be installed from Pimoroni's repository or via their installer. Check the [Pimoroni Presto documentation](https://learn.pimoroni.com/article/getting-started-with-presto) for the latest installation instructions.

2. Configure WiFi credentials:

   Create a `wifi_config.py` file (copy from `wifi_config.example.py`) with your WiFi credentials:
   
   ```python
   WIFI_SSID = "YourWiFiNetworkName"
   WIFI_PASSWORD = "YourWiFiPassword"
   ```
   
   Or create a `wifi_config.json` file:
   
   ```json
   {
     "WIFI_SSID": "YourWiFiNetworkName",
     "WIFI_PASSWORD": "YourWiFiPassword"
   }
   ```
   
   Alternatively, you can pass WiFi credentials via command-line arguments.

3. Make sure the flight tracker server is running (from the `flightaware_demo` directory):

```bash
cd ../flightaware_demo
python3 flight_tracker_server.py
```

## Usage

### Basic Usage

Connect to the default server (localhost:5050):

```bash
python3 presto_flight_client.py
```

### Connect to Remote Server

If the flight tracker server is running on a different machine:

```bash
python3 presto_flight_client.py --server-url http://192.168.1.100:5050
```

### Specify WiFi Credentials

You can also pass WiFi credentials via command-line (useful for testing):

```bash
python3 presto_flight_client.py --wifi-ssid "MyNetwork" --wifi-password "MyPassword"
```

### Simulation Mode

If you don't have a Presto display connected, the script will run in simulation mode and print flight information to the console.

## Display Layout

The flight card is displayed in three sections:

1. **Header** (top ~70px):
   - Callsign (large, bold)
   - ICAO code (smaller, monospace)

2. **Route Section** (middle ~100px):
   - Origin airport code (large)
   - Origin country name
   - Arrow/plane icon (center)
   - Destination airport code (large)
   - Destination country name

3. **Footer** (bottom ~70px):
   - Altitude, Speed, Track (first row)
   - Vertical Rate, Distance, Squawk (second row)

## Configuration

The client connects to the flight tracker server's SSE endpoint at `/events`. The server URL can be configured via command-line arguments.

## Troubleshooting

### Display Not Found

If you see "Presto library not available", make sure:
- The Presto display is properly connected to your Raspberry Pi
- The `presto` Python package is installed
- You're running on a Raspberry Pi (not on the Presto device itself)
- You're using CPython (python3), not MicroPython

### MicroPython Import Errors

If you see errors like "no module named 'abc'" or "no module named 'typing'", you're likely running the script in MicroPython. This script is designed to run on a Raspberry Pi using standard Python (CPython), not on the Presto device itself using MicroPython.

The Presto device runs MicroPython, but you write Python scripts on your Raspberry Pi that communicate with and control the Presto display.

### Connection Errors

If you see connection errors:
- Verify the flight tracker server is running
- Check the server URL and port (default: `http://localhost:5050`)
- Ensure the server is accessible from your machine

### No Flights Displayed

- Check that the flight tracker server is receiving data from dump1090
- Verify aircraft are being detected (check server logs)
- The client prefers flights with complete route information

## Integration with Flight Tracker

This client is designed to work with the flight tracker server from `flightaware_demo`. The server:
- Fetches flight data from dump1090
- Enriches it with route information, airline logos, etc.
- Broadcasts updates via Server-Sent Events (SSE)

The client subscribes to these updates and displays them on the Presto display.

## Future Enhancements

- Weather display integration (as mentioned in the project description)
- Multiple flight cards with scrolling
- Customizable display layouts
- Configuration file support

