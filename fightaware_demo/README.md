# Flight Tracker

A real-time flight tracking system that collects ADS-B data from dump1090 and displays it via terminal and web interface.

## Features

- **Data Collection**: Fetches aircraft data from dump1090
- **Route Enrichment**: Looks up origin/destination using adsb.lol API
- **Web Interface**: Modern, responsive web UI for flight visualization
- **Real-time Updates**: Live updates via WebSocket
- **Unified Server**: Single program handles everything

## Architecture

```
flight_server.py (unified server)
    ├── HTTP server (serves web interface)
    ├── WebSocket server (real-time updates)
    └── Flight data collection (from dump1090)
            ↓
    web/index.html (web client)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json`:
```json
{
    "dump1090_url": "http://your-dump1090-url/data/aircraft.json",
    "receiver_lat": 52.40585,
    "receiver_lon": 13.55214,
    "http_host": "0.0.0.0",
    "http_port": 8080,
    "websocket_host": "0.0.0.0",
    "websocket_port": 8765
}
```

## Usage

### Unified Server (Recommended)

Run the unified server that handles everything:
```bash
python3 flight_server.py
```

This starts:
- HTTP server on port 8080 (serves web interface)
- WebSocket server on port 8765 (real-time updates)
- Flight data collection (fetches from dump1090 every 5 seconds)

Then open your browser to:
```
http://localhost:8080/index.html
```

### Terminal Only Mode

If you just want terminal output without the web interface:
```bash
python3 flight_tracker.py
```

## Web Interface

The web interface provides:
- Real-time flight updates via WebSocket
- Beautiful card-based layout
- Connection status indicator
- Flight statistics
- Route information (origin → destination)
- Aircraft details (altitude, speed, track, position)

## Files

- `flight_server.py` - Unified server (HTTP + WebSocket + data collection)
- `flight_tracker.py` - Terminal-only flight tracker
- `flight_info.py` - Route lookup utilities (adsb.lol API)
- `web/index.html` - Web interface client
- `config.json` - Configuration file
- `dump1090_feed.py` - Original simple feed (backup/reference)

## Testing

Test flight lookup manually:
```bash
python3 flight_info.py <ICAO> <CALLSIGN> [LAT] [LON]
```

Example:
```bash
python3 flight_info.py 3c55c7 EWG1AN
```

## Troubleshooting

- **WebSocket not connecting**: Check firewall settings and ensure port 8765 is open
- **No flights displayed**: Verify dump1090 URL is correct and accessible
- **Routes not found**: Ensure aircraft have callsigns and are in adsb.lol database
