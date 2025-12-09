#!/bin/bash
# Startup script to run the Enviro Indoor web dashboard

# Change to script directory
cd "$(dirname "$0")"

# Check if web app is already running
if pgrep -f "web_app.py" > /dev/null; then
    echo "Web app is already running"
    echo "Dashboard: http://$(hostname -I | awk '{print $1}'):5000"
else
    echo "Starting Enviro Indoor web dashboard..."
    python3 web_app.py &
    echo "Web app started (PID: $!)"
    echo ""
    echo "Dashboard available at: http://localhost:5000"
    echo "Or from network: http://$(hostname -I | awk '{print $1}'):5000"
    echo ""
    echo "Data endpoint for Enviro board: http://$(hostname -I | awk '{print $1}'):5000/api/data"
fi

echo ""
echo "To stop the service, use: pkill -f web_app.py"


