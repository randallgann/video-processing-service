# Deployment Guide for YouTube Video Transcription Service

This guide provides step-by-step instructions for deploying the YouTube Video Transcription Service on a cloud VM or any server with Docker support.

## Requirements

- Ubuntu 20.04 or newer server
- Docker and Docker Compose
- Google Cloud project with proper configuration
- Replicate API token

## Step 1: Set Up the Server

### Update and Install Dependencies

```bash
# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add current user to docker group
sudo usermod -aG docker $USER
# You'll need to log out and back in for this to take effect
```

## Step 2: Clone the Repository

```bash
# Create application directory
mkdir -p ~/youtube-transcription
cd ~/youtube-transcription

# Clone the repository
git clone https://github.com/your-repo/youtube-video-transcriptor.git .
```

## Step 3: Configure Google Cloud Authentication

### Option 1: Using Service Account Keys

```bash
# Create directories for credentials
mkdir -p credentials

# Copy your service account keys to the credentials directory
# You need to upload these files to your server
cp /path/to/your-service-account-key.json credentials/
```

### Option 2: Using Google Cloud Workload Identity (for GCP VMs)

If running on a Google Cloud VM with appropriate service account attached:

```bash
# Install gcloud CLI
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install -y google-cloud-cli

# Login and configure application default credentials
gcloud auth login
gcloud auth application-default login
```

## Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your configuration
nano .env
```

Make sure to set all required environment variables:

```
# Required settings
PROJECT_ID=your-gcp-project-id
SUBSCRIPTION_ID=your-pubsub-subscription-id
BUCKET_NAME=your-gcs-bucket
REPLICATE_API_TOKEN=your-replicate-api-token

# Optional settings
PROGRESS_TOPIC_ID=your-progress-topic-id
PUBLIC_IP=your-server-public-ip
OPENAI_WHISPER=openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e
FAST_WHISPER=vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c
DEFAULT_MODEL_TYPE=openai
```

## Step 5: Run the Service

### Build and Start the Docker Container

```bash
# Create data directory for persistence
mkdir -p data

# Build and start the service
docker-compose up -d
```

### Verify the Service is Running

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

## Step 6: Test the Service

You can test the service by running it in test mode:

```bash
# Test with a specific YouTube video
docker-compose run --rm transcriber --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

## Step 7: Configure Auto-Start (Optional)

To ensure the service starts automatically on system boot:

```bash
# Create a systemd service file
sudo nano /etc/systemd/system/youtube-transcription.service
```

Add the following content:

```
[Unit]
Description=YouTube Video Transcription Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
Group=docker
WorkingDirectory=/home/ubuntu/youtube-transcription
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable youtube-transcription.service
sudo systemctl start youtube-transcription.service
```

## Step 8: Set Up Monitoring (Optional)

The service can publish progress updates to a PubSub topic. You can use this to build a monitoring dashboard or set up alerting.

```bash
# Run the progress monitor (development only)
docker-compose --profile dev up progress-monitor
```

## Troubleshooting

### Check Container Logs

```bash
docker-compose logs -f transcriber
```

### Check System Resource Usage

```bash
docker stats
```

### Test API Access

```bash
# Test Replicate API access
docker-compose run --rm transcriber python debug-replicate.py --help
```

### Common Issues and Solutions

- **PubSub Connection Errors**: Verify your service account credentials have the proper permissions
- **Missing Credentials**: Ensure your GCP credentials are properly mounted
- **Replicate API Errors**: Check your API token is valid and not expired
- **Space Issues**: Ensure you have enough disk space for processing large videos

## Maintenance

### Updating the Service

```bash
# Pull latest changes
git pull

# Rebuild and restart containers
docker-compose down
docker-compose up -d --build
```

### Backup Configuration

```bash
# Backup env file and credentials
cp .env .env.backup
tar -czf credentials-backup.tar.gz credentials/
```