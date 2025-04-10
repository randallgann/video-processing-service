#!/bin/bash
# Setup script for YouTube Video Transcription Service on Ubuntu VM

set -e  # Exit on error

echo "Setting up YouTube Video Transcription Service..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git

# Create virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r replicate-requirements.txt

# Configure environment
echo "Configuring environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Setup Google Cloud credentials
echo "Setting up Google Cloud authentication..."
echo "NOTE: You need to copy your Google Cloud service account key file to this directory"
echo "      and set the GOOGLE_APPLICATION_CREDENTIALS variable in your .env file"

# Create a directory for credentials if it doesn't exist
mkdir -p credentials

echo "To copy your service account key file from your local machine to this server, run:"
echo "scp /path/to/your-key.json username@server-ip:$(pwd)/credentials/"

# Set up service
echo "Setting up systemd service..."
sudo cp video-transcription.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable video-transcription.service

echo
echo "Installation complete!"
echo
echo "Next steps:"
echo "1. Edit the .env file with your configuration"
echo "2. Start the service with: sudo systemctl start video-transcription.service"
echo "3. Check status with: sudo systemctl status video-transcription.service"
echo "4. View logs with: sudo journalctl -u video-transcription.service -f"
echo
echo "To test the service without installing as a system service, run:"
echo "source venv/bin/activate"
echo "python cloud-vm-processor.py --test --video-url https://www.youtube.com/watch?v=your_video_id"