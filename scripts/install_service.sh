#!/bin/bash
# Install the scanner as a systemd service
# This keeps it running 24/7, auto-restarts on failure, starts on boot

set -e

PROJECT_DIR="/home/ec2-user/sports_trading"
SERVICE_NAME="sports-scanner"

# Create the systemd service file
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=Sports Trading Scanner Daemon
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=${PROJECT_DIR}
Environment="PATH=${PROJECT_DIR}/.venv/bin:/usr/bin"
ExecStart=${PROJECT_DIR}/.venv/bin/python -m src.trading.scanner --daemon
Restart=always
RestartSec=60
StandardOutput=append:${PROJECT_DIR}/logs/scanner.log
StandardError=append:${PROJECT_DIR}/logs/scanner.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
mkdir -p ${PROJECT_DIR}/logs

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

echo "=========================================="
echo "Scanner service installed!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  sudo systemctl status ${SERVICE_NAME}    # Check status"
echo "  sudo systemctl stop ${SERVICE_NAME}      # Stop scanner"
echo "  sudo systemctl start ${SERVICE_NAME}     # Start scanner"
echo "  sudo systemctl restart ${SERVICE_NAME}   # Restart scanner"
echo "  tail -f ${PROJECT_DIR}/logs/scanner.log  # View logs"
echo ""
