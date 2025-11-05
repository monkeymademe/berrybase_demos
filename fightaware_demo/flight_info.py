#!/usr/bin/env python3
"""
Flight information lookup utility
Uses adsb.lol API to get origin and destination
"""

import requests
import sys

def get_current_position_adsblol(icao):
    """Get current aircraft position from adsb.lol by ICAO"""
    url = f"https://api.adsb.lol/v2/hex/{icao}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            ac_list = data.get('ac', [])
            if ac_list and len(ac_list) > 0:
                ac = ac_list[0]
                lat = ac.get('lat')
                lon = ac.get('lon')
                if lat and lon:
                    return {'lat': lat, 'lon': lon}
    except:
        pass
    return None

def get_flight_route_adsblol(callsign, icao=None, lat=None, lon=None):
    """Get flight route from adsb.lol routeset API
    
    Args:
        callsign: Flight callsign (required)
        icao: ICAO24 hex code (optional, used to get position if lat/lon not provided)
        lat: Latitude (optional, will be fetched if not provided)
        lon: Longitude (optional, will be fetched if not provided)
    
    Returns:
        dict with 'origin', 'destination', 'source' keys, or None if not found
    """
    if not callsign:
        return None
    
    # Get position if not provided
    if (lat is None or lon is None) and icao:
        pos = get_current_position_adsblol(icao)
        if pos:
            lat = pos.get('lat')
            lon = pos.get('lon')
    
    # We need lat/lon for the API
    if lat is None or lon is None:
        return None
    
    url = "https://api.adsb.lol/api/0/routeset"
    payload = {
        "planes": [{
            "callsign": callsign.strip(),
            "lat": lat,
            "lng": lon
        }]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            routes = response.json()
            if routes and len(routes) > 0:
                route = routes[0]
                airport_codes = route.get('airport_codes', '')
                airports = route.get('_airports', [])
                
                if airports and len(airports) >= 2:
                    # Prefer IATA codes (more user-friendly) but fallback to ICAO
                    origin = airports[0].get('iata') or airports[0].get('icao')
                    destination = airports[1].get('iata') or airports[1].get('icao')
                    
                    if origin and destination:
                        return {
                            'origin': origin,
                            'destination': destination,
                            'source': 'adsb.lol'
                        }
    except Exception as e:
        # Return error info for debugging
        return {
            'origin': None,
            'destination': None,
            'source': 'adsb.lol',
            'error': str(e)
        }
    
    return None

def get_flight_route(icao, callsign=None, lat=None, lon=None):
    """
    Get flight origin and destination from adsb.lol routeset API
    
    Uses only adsb.lol (free, no rate limits, requires callsign + position)
    
    Args:
        icao: ICAO24 hex code (required, used to get position if not provided)
        callsign: Flight callsign (required)
        lat: Latitude (optional, will be fetched from adsb.lol if not provided)
        lon: Longitude (optional, will be fetched from adsb.lol if not provided)
    
    Returns:
        dict with 'origin', 'destination', 'source' keys, or None if not found
    """
    # Only use adsb.lol - requires callsign
    if not callsign:
        return None
    
    return get_flight_route_adsblol(callsign, icao, lat, lon)

if __name__ == '__main__':
    # Allow manual lookup from command line
    if len(sys.argv) < 3:
        print("Usage: python3 flight_info.py <ICAO> <CALLSIGN> [LAT] [LON]")
        print()
        print("Examples:")
        print("  python3 flight_info.py 3c55c7 EWG1AN")
        print("  python3 flight_info.py 3c55c7 EWG1AN 52.4 13.5")
        print()
        sys.exit(1)
    
    icao = sys.argv[1]
    callsign = sys.argv[2]
    
    # Parse optional lat/lon
    lat = None
    lon = None
    if len(sys.argv) > 3:
        try:
            lat = float(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid latitude '{sys.argv[3]}', ignoring")
    if len(sys.argv) > 4:
        try:
            lon = float(sys.argv[4])
        except ValueError:
            print(f"Warning: Invalid longitude '{sys.argv[4]}', ignoring")
    
    print(f"Looking up flight: {callsign} (ICAO: {icao})")
    if lat and lon:
        print(f"Using position: {lat}, {lon}")
    print()
    
    result = get_flight_route(icao, callsign, lat, lon)
    
    if result:
        if result.get('error'):
            print(f"❌ Error: {result.get('error')}")
        else:
            print("✅ Flight info found:")
            print(f"  Origin: {result.get('origin', 'Unknown')}")
            print(f"  Destination: {result.get('destination', 'Unknown')}")
            print(f"  Source: {result.get('source', 'Unknown')}")
    else:
        print("❌ Flight info not found")
        print("   (This could mean the flight is not in adsb.lol database,")
        print("    or position data is needed but unavailable)")
