# Ubuntu VM Setup for Whisper Transcription with Replicate API

This guide details how to set up a standard Ubuntu VM to run the YouTube transcription service using Replicate's Whisper API instead of local GPU processing.

## 1. VM Requirements

* **OS**: Ubuntu 20.04 LTS or newer
* **CPU**: 4-8 cores
* **RAM**: 16GB recommended (8GB minimum)
* **Storage**: 100GB+ SSD
* **Network**: Good internet connection for video downloads and API calls

No GPU is required as transcription is handled via API calls.

## 2. Initial System Setup

Update the system and install required system dependencies:

```bash
# Update packages
sudo apt update
sudo apt upgrade -y

# Install required dependencies
sudo apt install -y python3 python3-pip python3-venv ffmpeg git
```

## 3. Project Setup

Clone and set up the repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-video-transcriptor.git
cd youtube-video-transcriptor

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r replicate-requirements.txt
```

## 4. Configuration

Configure environment variables:

```bash
# Create a configuration file
touch .env

# Add the following to the .env file
echo "REPLICATE_API_TOKEN=your_replicate_api_token" >> .env
echo "PROJECT_ID=your_gcp_project_id" >> .env
echo "SUBSCRIPTION_ID=your_pubsub_subscription_id" >> .env
echo "BUCKET_NAME=your_gcs_bucket_name" >> .env
echo "PROGRESS_TOPIC_ID=your_progress_topic_id" >> .env
```

Create a script to load the environment variables:

```bash
# Create load-env.sh
cat > load-env.sh << 'EOF'
#!/bin/bash
export $(grep -v '^#' .env | xargs)
EOF

# Make it executable
chmod +x load-env.sh
```

## 5. GCP Credentials Setup

If you're using Google Cloud Storage for result uploads, set up the credentials:

```bash
# Create credentials directory
mkdir -p credentials

# Copy your service account credentials 
# You'll need to upload these files to the VM
cp /path/to/rag-widget-pubsub-publisher-key.json credentials/
cp /path/to/rag-widget-pubsub-subscriber-key.json credentials/

# Create symbolic links in the project root
ln -s credentials/rag-widget-pubsub-publisher-key.json .
ln -s credentials/rag-widget-pubsub-subscriber-key.json .
```

## 6. Testing the Setup

Test the installation with a short video:

```bash
# Load environment variables
source load-env.sh

# Test the installation with a short video
python3 direct-transcribe-replicate.py "https://www.youtube.com/watch?v=example_video_id"
```

## 7. Setting Up as a Service

Create a systemd service to run the transcription service continuously:

```bash
# Create the service file
sudo tee /etc/systemd/system/video-transcriber.service > /dev/null << 'EOF'
[Unit]
Description=YouTube Video Transcription Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/youtube-video-transcriptor
ExecStart=/home/ubuntu/youtube-video-transcriptor/venv/bin/python k8s-processor.py
EnvironmentFile=/home/ubuntu/youtube-video-transcriptor/.env
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl enable video-transcriber
sudo systemctl start video-transcriber
```

Check the status:

```bash
sudo systemctl status video-transcriber
```

## 8. Server Monitoring

Set up basic monitoring:

```bash
# Install monitoring tools
sudo apt install -y htop iotop

# View logs
sudo journalctl -u video-transcriber -f
```

## 9. Firewall Configuration

If you're using the mini_audio_server to serve audio files for the Replicate API, open the required port:

```bash
# Open port 8000 for the audio server
sudo ufw allow 8000/tcp
sudo ufw enable
```

## 10. Optional: Automatic Updates

Set up automatic updates:

```bash
# Install and configure automatic updates
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

## Troubleshooting

Common issues and solutions:

1. **API token not working**: Verify your Replicate API token at https://replicate.com/account/api-tokens

2. **Audio server not accessible**: Check that port 8000 is open and that the VM has a public IP

3. **Transcription failures**: Check the logs with `sudo journalctl -u video-transcriber -f` for details

4. **Missing dependencies**: Run `pip install -r replicate-requirements.txt` to ensure all dependencies are installed