"""
Flask web application for displaying Enviro Indoor sensor data.
Provides API endpoints and serves the web UI.
"""

from flask import Flask, render_template, jsonify, request
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

app = Flask(__name__)
DB_PATH = "sensor_data.db"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter to suppress GET request logs but keep POST requests
class HTTPMethodFilter(logging.Filter):
    def filter(self, record):
        # Suppress GET requests, but keep POST, PUT, DELETE, and errors
        message = record.getMessage()
        if '"GET /' in message:
            return False  # Suppress all GET requests
        return True  # Allow everything else (POST, errors, etc.)

# Apply filter to werkzeug logger to reduce GET request spam
werkzeug_logger = logging.getLogger('werkzeug')
werkzeug_logger.addFilter(HTTPMethodFilter())

def get_db_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_latest_reading():
    """Get the most recent sensor reading."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM sensor_readings
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_historical_data(hours=24):
    """Get historical sensor data for the specified number of hours."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Calculate cutoff time
    cutoff_time = datetime.now().timestamp() - (hours * 3600)
    
    cursor.execute("""
        SELECT timestamp, temperature, humidity, pressure, light, gas, aqi, color_temperature
        FROM sensor_readings
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """, (cutoff_time,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def init_database():
    """Initialize the SQLite database with sensor data table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            light REAL,
            gas REAL,
            aqi REAL,
            color_temperature REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add new columns if they don't exist (for existing databases)
    try:
        cursor.execute("ALTER TABLE sensor_readings ADD COLUMN aqi REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE sensor_readings ADD COLUMN color_temperature REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Create index on timestamp for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON sensor_readings(timestamp)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def store_sensor_data(data):
    """Store sensor data in the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sensor_readings 
            (timestamp, temperature, humidity, pressure, light, gas, aqi, color_temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("timestamp"),
            data.get("temperature"),
            data.get("humidity"),
            data.get("pressure"),
            data.get("light"),
            data.get("gas"),
            data.get("aqi"),
            data.get("color_temperature")
        ))
        
        conn.commit()
        conn.close()
        logger.debug(f"Stored sensor data: {data}")
        return True
    except Exception as e:
        logger.error(f"Error storing sensor data: {e}")
        return False

def is_system_sleeping():
    """Check if system appears to be sleeping (no recent data)."""
    latest = get_latest_reading()
    if not latest:
        return True
    
    # If last reading is more than 10 minutes old, consider it sleeping
    time_diff = datetime.now().timestamp() - latest['timestamp']
    return time_diff > 600  # 10 minutes

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')

@app.route('/api/data', methods=['POST'])
def api_receive_data():
    """
    Receive sensor data from Enviro Indoor board via HTTP POST.
    The Enviro board sends data in batches as JSON.
    Expected format: {"readings": [{"timestamp": ..., "temperature": ..., ...}, ...]}
    """
    try:
        # Get raw JSON data from request for debugging (single line)
        raw_data = request.get_data(as_text=True)
        # Remove newlines and extra spaces for single-line output
        raw_data_single_line = ' '.join(raw_data.split())
        logger.info(f"RAW JSON DATA RECEIVED: {raw_data_single_line}")
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            logger.warning("Received empty or invalid JSON data")
            return jsonify({"success": False, "error": "Invalid JSON"}), 400
        
        # Log the received data structure for debugging (first time only, then use debug level)
        logger.info(f"Received data type: {type(data).__name__}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        
        # Handle different possible formats from Enviro
        readings = []
        
        # Format 1: Direct array of readings
        if isinstance(data, list):
            readings = data
        # Format 2: Object with "readings" key
        elif isinstance(data, dict) and "readings" in data:
            readings_data = data["readings"]
            # If readings is a dict (not a list), merge it with top-level metadata
            if isinstance(readings_data, dict):
                # Merge readings dict with top-level fields (like timestamp, nickname, etc.)
                merged_reading = dict(readings_data)
                # Add timestamp from top level if not in readings
                if "timestamp" not in merged_reading and "timestamp" in data:
                    merged_reading["timestamp"] = data["timestamp"]
                # Add other metadata if needed
                if "nickname" in data:
                    merged_reading["nickname"] = data["nickname"]
                readings = [merged_reading]
            else:
                readings = readings_data
        # Format 3: Single reading object (flat dictionary with sensor values)
        elif isinstance(data, dict):
            # Check if it looks like a sensor reading (has at least one sensor field)
            sensor_fields = ["temperature", "temp", "humidity", "hum", "pressure", "press", 
                           "light", "lux", "gas", "gas_resistance", "luminance", 
                           "color_temperature", "aqi"]
            if any(field in data for field in sensor_fields):
                readings = [data]
            else:
                # If it's a dict but doesn't have sensor fields, log it for debugging
                logger.warning(f"Dictionary received but no recognized sensor fields. Keys: {list(data.keys())}")
                logger.debug(f"Full data: {data}")
                return jsonify({"success": False, "error": "Unexpected data format"}), 400
        else:
            logger.warning(f"Unexpected data format (type: {type(data).__name__}): {data}")
            return jsonify({"success": False, "error": "Unexpected data format"}), 400
        
        # Ensure readings is always a list
        if not isinstance(readings, list):
            if isinstance(readings, dict):
                readings = [readings]
            else:
                logger.error(f"Readings is not a list or dict: {type(readings).__name__}, value: {readings}")
                return jsonify({"success": False, "error": "Invalid readings format"}), 400
        
        # Check if readings is a list of strings (field names) - this indicates a data format issue
        if readings and isinstance(readings[0], str):
            logger.warning(f"Readings appears to be a list of field names instead of readings: {readings}")
            logger.warning(f"Original data structure: {data}")
            # If the original data was a dict and has sensor fields, use it as a single reading
            # (ignore the "readings" key if it contains field names)
            if isinstance(data, dict):
                sensor_fields = ["temperature", "temp", "humidity", "hum", "pressure", "press", 
                               "light", "lux", "gas", "gas_resistance", "luminance", 
                               "color_temperature", "aqi"]
                # Check if data (excluding "readings" key) has sensor fields
                data_without_readings = {k: v for k, v in data.items() if k != "readings"}
                if any(field in data_without_readings for field in sensor_fields):
                    readings = [data_without_readings]
                    logger.info(f"Reconstructed reading from data (excluding 'readings' key)")
                elif any(field in data for field in sensor_fields):
                    readings = [data]
                    logger.info(f"Using original data as reading despite 'readings' key issue")
                else:
                    logger.error(f"Cannot reconstruct reading from field names. Original data: {data}")
                    return jsonify({"success": False, "error": "Invalid data format: readings contains field names, not values"}), 400
            else:
                logger.error(f"Cannot reconstruct reading from field names. Original data type: {type(data).__name__}")
                return jsonify({"success": False, "error": "Invalid data format: readings contains field names, not values"}), 400
        
        logger.debug(f"Processed readings list length: {len(readings)}, first item type: {type(readings[0]).__name__ if readings else 'N/A'}")
        
        # Store each reading
        stored_count = 0
        for reading in readings:
            # Skip if reading is not a dictionary
            if not isinstance(reading, dict):
                logger.warning(f"Skipping invalid reading (not a dict): {reading}, type: {type(reading).__name__}")
                continue
            
            # Create a copy to avoid modifying the original
            reading = dict(reading)
            
            # Ensure timestamp is present (use current time if not)
            if "timestamp" not in reading:
                reading["timestamp"] = datetime.now().timestamp()
            
            # Convert timestamp if it's a string
            if isinstance(reading["timestamp"], str):
                try:
                    dt = datetime.fromisoformat(reading["timestamp"].replace('Z', '+00:00'))
                    reading["timestamp"] = dt.timestamp()
                except:
                    reading["timestamp"] = datetime.now().timestamp()
            
            # Map Enviro field names to our database fields
            sensor_data = {
                "timestamp": reading.get("timestamp", datetime.now().timestamp()),
                "temperature": reading.get("temperature") or reading.get("temp"),
                "humidity": reading.get("humidity") or reading.get("hum"),
                "pressure": reading.get("pressure") or reading.get("press"),
                "light": reading.get("light") or reading.get("lux") or reading.get("luminance"),
                "gas": reading.get("gas") or reading.get("gas_resistance"),
                "aqi": reading.get("aqi"),
                "color_temperature": reading.get("color_temperature")
            }
            
            if store_sensor_data(sensor_data):
                stored_count += 1
        
        logger.info(f"Received and stored {stored_count} reading(s) from Enviro board")
        
        return jsonify({
            "success": True,
            "stored": stored_count,
            "total": len(readings)
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing incoming data: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/current')
def api_current():
    """API endpoint for current sensor readings."""
    reading = get_latest_reading()
    is_sleeping = is_system_sleeping()
    
    if reading:
        return jsonify({
            "success": True,
            "data": {
                "timestamp": reading['timestamp'],
                "temperature": reading['temperature'],
                "humidity": reading['humidity'],
                "pressure": reading['pressure'],
                "light": reading['light'],
                "gas": reading['gas'],
                "aqi": reading.get('aqi'),
                "color_temperature": reading.get('color_temperature')
            },
            "sleeping": is_sleeping
        })
    else:
        return jsonify({
            "success": False,
            "data": None,
            "sleeping": True
        })

@app.route('/api/historical')
def api_historical():
    """API endpoint for historical sensor data."""
    hours = int(request.args.get('hours', 24))
    data = get_historical_data(hours)
    
    return jsonify({
        "success": True,
        "data": data,
        "hours": hours
    })

@app.route('/api/stats')
def api_stats():
    """API endpoint for sensor statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get min/max/avg for last 24 hours
    cutoff_time = datetime.now().timestamp() - (24 * 3600)
    
    cursor.execute("""
        SELECT 
            MIN(temperature) as min_temp,
            MAX(temperature) as max_temp,
            AVG(temperature) as avg_temp,
            MIN(humidity) as min_humidity,
            MAX(humidity) as max_humidity,
            AVG(humidity) as avg_humidity,
            MIN(pressure) as min_pressure,
            MAX(pressure) as max_pressure,
            AVG(pressure) as avg_pressure,
            MIN(light) as min_light,
            MAX(light) as max_light,
            AVG(light) as avg_light,
            MIN(gas) as min_gas,
            MAX(gas) as max_gas,
            AVG(gas) as avg_gas
        FROM sensor_readings
        WHERE timestamp >= ?
    """, (cutoff_time,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return jsonify({
            "success": True,
            "stats": dict(row)
        })
    else:
        return jsonify({
            "success": False,
            "stats": None
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    Path('templates').mkdir(exist_ok=True)
    
    # Initialize database on startup
    init_database()
    
    logger.info("Starting Enviro Indoor web dashboard...")
    logger.info("Data endpoint: http://<pi-ip>:5000/api/data")
    logger.info("Dashboard: http://<pi-ip>:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

