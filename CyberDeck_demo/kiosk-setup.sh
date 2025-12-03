#!/bin/bash
# Raspberry Pi Kiosk Mode Setup Script for Debian Trixie
# This script sets up automatic kiosk mode to display index.html in fullscreen

set -e

echo "=========================================="
echo "Raspberry Pi Kiosk Mode Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HTML_FILE="$SCRIPT_DIR/index.html"

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ] || ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "Warning: This doesn't appear to be a Raspberry Pi. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Install Chromium if not already installed
echo "Step 1: Installing Chromium..."
if ! command -v chromium &> /dev/null && ! command -v chromium-browser &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y chromium chromium-sandbox
else
    echo "Chromium is already installed."
fi

# Determine Chromium command name
if command -v chromium &> /dev/null; then
    CHROMIUM_CMD="chromium"
elif command -v chromium-browser &> /dev/null; then
    CHROMIUM_CMD="chromium-browser"
else
    echo "Error: Could not find Chromium installation"
    exit 1
fi

echo "Using Chromium command: $CHROMIUM_CMD"
echo ""

# Get absolute path to HTML file
if [ ! -f "$HTML_FILE" ]; then
    echo "Error: Could not find index.html at $HTML_FILE"
    echo "Please make sure you're running this script from the CyberDeck_demo directory"
    exit 1
fi

HTML_PATH=$(realpath "$HTML_FILE")
echo "HTML file path: $HTML_PATH"
echo ""

# Create systemd service file
echo "Step 2: Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/cyberdeck-kiosk.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=CyberDeck Kiosk Mode
After=graphical.target

[Service]
Type=simple
User=$USER
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/$USER/.Xauthority
ExecStartPre=/bin/sleep 3
ExecStart=$CHROMIUM_CMD --noerrdialogs --disable-infobars --kiosk --app=file://$HTML_PATH
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
EOF

echo "Service file created at $SERVICE_FILE"
echo ""

# Enable autologin (optional - uncomment if needed)
echo "Step 3: Setting up autologin..."
echo "Do you want to enable autologin for user '$USER'? (y/n)"
read -r autologin_response
if [ "$autologin_response" = "y" ]; then
    # For Debian Trixie with GDM3
    if [ -f /etc/gdm3/daemon.conf ]; then
        echo "Configuring GDM3 autologin..."
        sudo sed -i "s/#  AutomaticLoginEnable = true/  AutomaticLoginEnable = true/" /etc/gdm3/daemon.conf
        sudo sed -i "s/#  AutomaticLogin = user1/  AutomaticLogin = $USER/" /etc/gdm3/daemon.conf
    fi
    
    # For LightDM (alternative display manager)
    if [ -f /etc/lightdm/lightdm.conf ]; then
        echo "Configuring LightDM autologin..."
        sudo sed -i "s/#autologin-user=/autologin-user=$USER/" /etc/lightdm/lightdm.conf
        sudo sed -i "s/#autologin-user-timeout=0/autologin-user-timeout=0/" /etc/lightdm/lightdm.conf
    fi
    
    echo "Autologin configured."
else
    echo "Skipping autologin setup. You'll need to log in manually."
fi
echo ""

# Disable screen blanking
echo "Step 4: Disabling screen blanking..."
if [ -f /etc/X11/xorg.conf ]; then
    echo "xorg.conf already exists. You may need to manually configure screen blanking."
else
    sudo tee /etc/X11/xorg.conf > /dev/null <<EOF
Section "ServerLayout"
    Identifier "ServerLayout0"
EndSection

Section "ServerFlags"
    Option "blank time" "0"
    Option "standby time" "0"
    Option "suspend time" "0"
    Option "off time" "0"
EndSection
EOF
    echo "Screen blanking disabled."
fi
echo ""

# Disable screen saver for user
echo "Step 5: Disabling screensaver..."
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/xscreensaver-disable.service <<EOF
[Unit]
Description=Disable XScreenSaver
After=graphical-session.target

[Service]
Type=oneshot
ExecStart=/usr/bin/xset s off
ExecStart=/usr/bin/xset -dpms
ExecStart=/usr/bin/xset s noblank

[Install]
WantedBy=default.target
EOF

systemctl --user enable xscreensaver-disable.service
echo "Screensaver disabled."
echo ""

# Enable and start the service
echo "Step 6: Enabling kiosk service..."
sudo systemctl daemon-reload
sudo systemctl enable cyberdeck-kiosk.service
echo "Service enabled. It will start automatically on boot."
echo ""

# Ask if user wants to start it now
echo "Do you want to start the kiosk mode now? (y/n)"
read -r start_now
if [ "$start_now" = "y" ]; then
    echo "Starting kiosk service..."
    sudo systemctl start cyberdeck-kiosk.service
    echo "Kiosk mode started!"
    echo ""
    echo "To stop it, run: sudo systemctl stop cyberdeck-kiosk.service"
    echo "To disable it, run: sudo systemctl disable cyberdeck-kiosk.service"
else
    echo "Kiosk mode will start automatically on next boot."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  Start kiosk:   sudo systemctl start cyberdeck-kiosk.service"
echo "  Stop kiosk:    sudo systemctl stop cyberdeck-kiosk.service"
echo "  Status:        sudo systemctl status cyberdeck-kiosk.service"
echo "  View logs:     journalctl -u cyberdeck-kiosk.service -f"
echo "  Disable:       sudo systemctl disable cyberdeck-kiosk.service"
echo ""
echo "To exit kiosk mode, press Alt+F4 or Ctrl+Alt+F1 to switch to terminal"
echo ""

