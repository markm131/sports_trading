#!/bin/bash
# EC2 Setup Script for Sports Trading Bot
# Run this after launching a fresh Amazon Linux 2023 or Ubuntu instance

set -e

echo "=========================================="
echo "Sports Trading Bot - EC2 Setup"
echo "=========================================="

# Update system
echo "Updating system..."
sudo yum update -y 2>/dev/null || sudo apt update -y

# Install Python 3.11
echo "Installing Python 3.11..."
sudo yum install python3.11 python3.11-pip git -y 2>/dev/null || sudo apt install python3.11 python3.11-venv git -y

# Install AWS CLI (for S3 backups)
echo "Installing AWS CLI..."
sudo yum install awscli -y 2>/dev/null || sudo apt install awscli -y

# Create project directory
echo "Setting up project directory..."
mkdir -p ~/sports_trading
cd ~/sports_trading

# Clone or upload your code here
echo ""
echo "=========================================="
echo "NEXT STEPS (manual):"
echo "=========================================="
echo ""
echo "1. Upload your project files:"
echo "   scp -r -i your-key.pem C:\\Projects\\Code_Hub\\Python\\sports_trading\\* ec2-user@YOUR-IP:~/sports_trading/"
echo ""
echo "2. Upload Betfair certificates:"
echo "   scp -i your-key.pem C:\\Projects\\Code_Hub\\Python\\sports_trading\\certs\\* ec2-user@YOUR-IP:~/sports_trading/certs/"
echo ""
echo "3. Create virtual environment:"
echo "   cd ~/sports_trading"
echo "   python3.11 -m venv .venv"
echo "   source .venv/bin/activate"
echo "   pip install -r requirements.txt"
echo ""
echo "4. Create .env file with your Betfair credentials:"
echo "   nano .env"
echo "   # Add: BETFAIR_USERNAME=xxx"
echo "   # Add: BETFAIR_PASSWORD=xxx"
echo "   # Add: BETFAIR_APP_KEY=xxx"
echo ""
echo "5. Run the service installer:"
echo "   sudo bash scripts/install_service.sh"
echo ""
