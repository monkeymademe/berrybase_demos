#!/usr/bin/env python3
"""
Script to simulate 24 hours of sensor data with multiple gaps:
- 8-hour gap
- 1-hour gap  
- 40-minute gap
This helps visualize how the dashboard handles missing data periods.
"""

import sqlite3
from datetime import datetime, timedelta
import random
import math

DB_PATH = "sensor_data.db"

def generate_sensor_reading(timestamp, base_temp=23.0, base_humidity=45.0, base_pressure=1006.0):
    """Generate realistic sensor readings with some variation."""
    # Add some realistic variation based on time of day
    hour = datetime.fromtimestamp(timestamp).hour
    
    # Temperature varies throughout the day (cooler at night, warmer during day)
    temp_variation = 2 * math.sin((hour - 6) * math.pi / 12)  # Peak around 2pm
    temperature = base_temp + temp_variation + random.uniform(-0.5, 0.5)
    
    # Humidity inversely related to temperature (roughly)
    humidity = base_humidity - (temp_variation * 2) + random.uniform(-2, 2)
    humidity = max(30, min(60, humidity))  # Clamp between 30-60%
    
    # Pressure has small random variations
    pressure = base_pressure + random.uniform(-2, 2)
    
    # Light varies dramatically (0 at night, higher during day)
    if 6 <= hour <= 20:
        luminance = random.uniform(200, 800)
    else:
        luminance = random.uniform(0, 50)
    
    # Gas resistance varies slightly
    gas_resistance = random.uniform(28000, 35000)
    
    # AQI varies
    aqi = random.uniform(8, 18)
    
    # Color temperature varies (warmer at night, cooler during day)
    if 6 <= hour <= 20:
        color_temperature = random.uniform(4000, 5500)  # Daylight
    else:
        color_temperature = random.uniform(2500, 3000)  # Warm light
    
    return {
        "timestamp": timestamp,
        "temperature": round(temperature, 2),
        "humidity": round(humidity, 2),
        "pressure": round(pressure, 2),
        "light": round(luminance, 0),
        "gas": round(gas_resistance, 0),
        "aqi": round(aqi, 1),
        "color_temperature": round(color_temperature, 0)
    }

def simulate_data():
    """Generate 24 hours of data with multiple gaps (8 hours, 1 hour, and 40 minutes)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Start from 24 hours ago
    now = datetime.now()
    start_time = now - timedelta(hours=24)
    
    # Define gap periods:
    # Gap 1: 8 hours, starting 8 hours after start
    gap1_start = start_time + timedelta(hours=8)
    gap1_end = gap1_start + timedelta(hours=8)
    
    # Gap 2: 1 hour, starting 2 hours after start
    gap2_start = start_time + timedelta(hours=2)
    gap2_end = gap2_start + timedelta(hours=1)
    
    # Gap 3: 40 minutes, starting 20 hours after start
    gap3_start = start_time + timedelta(hours=20)
    gap3_end = gap3_start + timedelta(minutes=40)
    
    print("Generating simulated sensor data...")
    print(f"Start time: {start_time}")
    print(f"Gap 1 (8 hours): {gap1_start} to {gap1_end}")
    print(f"Gap 2 (1 hour): {gap2_start} to {gap2_end}")
    print(f"Gap 3 (40 minutes): {gap3_start} to {gap3_end}")
    print(f"End time: {now}")
    print()
    
    # First, delete any existing data in this time range to avoid conflicts
    print("Clearing existing data in time range...")
    cursor.execute("""
        DELETE FROM sensor_readings 
        WHERE timestamp >= ? AND timestamp <= ?
    """, (start_time.timestamp(), now.timestamp()))
    deleted_count = cursor.rowcount
    print(f"✓ Deleted {deleted_count} existing readings in time range")
    print()
    
    # Generate readings every 15 minutes (4 per hour)
    current_time = start_time
    reading_count = 0
    skipped_count = 0
    
    while current_time <= now:
        timestamp = current_time.timestamp()
        
        # Skip all gap periods - make sure we're not in any gap
        if (gap1_start <= current_time < gap1_end) or \
           (gap2_start <= current_time < gap2_end) or \
           (gap3_start <= current_time < gap3_end):
            skipped_count += 1
            current_time += timedelta(minutes=15)
            continue
        
        reading = generate_sensor_reading(timestamp)
        
        cursor.execute("""
            INSERT INTO sensor_readings 
            (timestamp, temperature, humidity, pressure, light, gas, aqi, color_temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            reading["timestamp"],
            reading["temperature"],
            reading["humidity"],
            reading["pressure"],
            reading["light"],
            reading["gas"],
            reading["aqi"],
            reading["color_temperature"]
        ))
        
        reading_count += 1
        current_time += timedelta(minutes=15)
    
    conn.commit()
    
    # Verify all gaps exist by checking for data in each gap period
    cursor.execute("""
        SELECT COUNT(*) FROM sensor_readings 
        WHERE timestamp >= ? AND timestamp < ?
    """, (gap1_start.timestamp(), gap1_end.timestamp()))
    gap1_data_count = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM sensor_readings 
        WHERE timestamp >= ? AND timestamp < ?
    """, (gap2_start.timestamp(), gap2_end.timestamp()))
    gap2_data_count = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM sensor_readings 
        WHERE timestamp >= ? AND timestamp < ?
    """, (gap3_start.timestamp(), gap3_end.timestamp()))
    gap3_data_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"✓ Generated {reading_count} sensor readings")
    print(f"✓ Skipped {skipped_count} readings during gap periods")
    print()
    print("Gap verification:")
    print(f"  Gap 1 (8h): {gap1_start.strftime('%H:%M')} to {gap1_end.strftime('%H:%M')} - {gap1_data_count} data points (should be 0)")
    print(f"  Gap 2 (1h): {gap2_start.strftime('%H:%M')} to {gap2_end.strftime('%H:%M')} - {gap2_data_count} data points (should be 0)")
    print(f"  Gap 3 (40m): {gap3_start.strftime('%H:%M')} to {gap3_end.strftime('%H:%M')} - {gap3_data_count} data points (should be 0)")
    print()
    
    if gap1_data_count == 0 and gap2_data_count == 0 and gap3_data_count == 0:
        print("✓ All gaps verified: No data in any gap period!")
    else:
        print(f"⚠ Warning: Found data points in gap periods!")
    
    print()
    print("Data simulation complete! Refresh your dashboard to see the results.")

if __name__ == "__main__":
    simulate_data()

