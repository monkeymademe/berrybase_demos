# Raspberry Pi Kiosk Mode Setup Guide

This guide will help you set up your Raspberry Pi running Debian Trixie to automatically display the CyberDeck demo in fullscreen kiosk mode.

## Quick Setup (Easiest Method)

1. **Copy the files to your Raspberry Pi:**
   ```bash
   scp -r CyberDeck_demo pi@raspberrypi.local:~/
   ```

2. **SSH into your Raspberry Pi:**
   ```bash
   ssh pi@raspberrypi.local
   ```

3. **Navigate to the directory and run the setup script:**
   ```bash
   cd ~/CyberDeck_demo
   chmod +x kiosk-setup.sh
   ./kiosk-setup.sh
   ```

4. **Follow the prompts** - the script will:
   - Install Chromium if needed
   - Create a systemd service for kiosk mode
   - Optionally set up autologin
   - Disable screen blanking
   - Enable the service to start on boot

5. **Reboot your Pi:**
   ```bash
   sudo reboot
   ```

After reboot, the page should automatically open in fullscreen kiosk mode!

## Manual Setup (Alternative Method)

If you prefer to set things up manually:

### 1. Install Chromium
```bash
sudo apt-get update
sudo apt-get install -y chromium chromium-sandbox
```

### 2. Create the systemd service
Create `/etc/systemd/system/cyberdeck-kiosk.service`:

```ini
[Unit]
Description=CyberDeck Kiosk Mode
After=graphical.target

[Service]
Type=simple
User=pi
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority
ExecStartPre=/bin/sleep 3
ExecStart=chromium --noerrdialogs --disable-infobars --kiosk --app=file:///home/pi/CyberDeck_demo/index.html
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
```

**Important:** Replace `pi` with your username and update the file path if needed.

### 3. Enable and start the service
```bash
sudo systemctl daemon-reload
sudo systemctl enable cyberdeck-kiosk.service
sudo systemctl start cyberdeck-kiosk.service
```

### 4. Set up autologin (optional)

**For GDM3 (default on Debian Trixie):**
Edit `/etc/gdm3/daemon.conf`:
```ini
[daemon]
AutomaticLoginEnable = true
AutomaticLogin = pi
```

**For LightDM:**
Edit `/etc/lightdm/lightdm.conf`:
```ini
[Seat:*]
autologin-user=pi
autologin-user-timeout=0
```

### 5. Disable screen blanking

Create `/etc/X11/xorg.conf`:
```ini
Section "ServerLayout"
    Identifier "ServerLayout0"
EndSection

Section "ServerFlags"
    Option "blank time" "0"
    Option "standby time" "0"
    Option "suspend time" "0"
    Option "off time" "0"
EndSection
```

Also disable screensaver:
```bash
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
```

## Useful Commands

- **Start kiosk mode:** `sudo systemctl start cyberdeck-kiosk.service`
- **Stop kiosk mode:** `sudo systemctl stop cyberdeck-kiosk.service`
- **Check status:** `sudo systemctl status cyberdeck-kiosk.service`
- **View logs:** `journalctl -u cyberdeck-kiosk.service -f`
- **Disable on boot:** `sudo systemctl disable cyberdeck-kiosk.service`

## Exiting Kiosk Mode

- Press `Alt+F4` to close Chromium
- Press `Ctrl+Alt+F1` to switch to a terminal (F7 to return to GUI)
- SSH into the Pi and stop the service: `sudo systemctl stop cyberdeck-kiosk.service`

## Troubleshooting

### Chromium doesn't start
- Check logs: `journalctl -u cyberdeck-kiosk.service -n 50`
- Make sure you're logged into the graphical session
- Verify the HTML file path is correct
- Try running Chromium manually: `chromium --kiosk --app=file:///home/pi/CyberDeck_demo/index.html`

### Screen goes blank
- Check that screen blanking is disabled
- Verify xset commands are running: `xset q`
- Make sure the xscreensaver-disable service is enabled

### Service doesn't start on boot
- Check if autologin is configured
- Verify the service is enabled: `sudo systemctl is-enabled cyberdeck-kiosk.service`
- Check systemd logs: `journalctl -b`

### Wrong display resolution
- Configure display settings in `/boot/firmware/config.txt` (for Raspberry Pi OS)
- Or use `xrandr` to set resolution after login

## Notes

- The service waits 3 seconds after boot to ensure the display is ready
- Chromium will automatically restart if it crashes
- The page will use the system hostname (or "CYBERDECK" as fallback) for the node name
- If accessing via network, the hostname will be detected from the URL

