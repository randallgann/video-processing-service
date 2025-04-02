# Setting Up YouTube Transcription Service on Ubuntu 22.04 with RTX 4090 GPUs

This guide walks through setting up the YouTube transcription service on a dedicated Ubuntu 22.04 server with RTX 4090 GPUs.

## 1. Initial Server Setup

### Update System
```bash
sudo apt update
sudo apt upgrade -y
```

### Set Up Basic Security
```bash
# Configure firewall
sudo apt install -y ufw
sudo ufw allow ssh
sudo ufw allow 22/tcp
sudo ufw enable
```

## 2. Install NVIDIA Drivers and CUDA

### Install NVIDIA Drivers
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NVIDIA drivers, CUDA, and related packages
sudo apt-get install -y cuda-drivers cuda nvidia-cuda-toolkit
```

### Verify GPU Installation
```bash
# Verify NVIDIA drivers are loaded
nvidia-smi
```

You should see output showing your RTX 4090 GPUs with driver version and CUDA version.

## 3. Install Docker

```bash
# Install prerequisites
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Add your user to the docker group (to run docker without sudo)
sudo usermod -aG docker $USER
```

Log out and log back in for the group changes to take effect, or run:
```bash
newgrp docker
```

## 4. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA Container Toolkit repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify installation
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## 5. Set Up the Transcription Service

### Create Project Directory
```bash
mkdir -p ~/transcription-service
cd ~/transcription-service
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/youtube-video-transcriptor.git
cd youtube-video-transcriptor
```

### Install Python Dependencies Directly
```bash
# Install Python and pip if not already installed
sudo apt install -y python3 python3-pip python3-venv ffmpeg git

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch>=2.0.0 torchaudio>=2.0.0 numpy>=1.20.0 openai-whisper>=1.0.0 pydub>=0.25.1 yt-dlp>=2023.3.4 requests>=2.28.0 ffmpeg-python>=0.2.0 google-cloud-pubsub>=2.13.0 google-cloud-storage>=2.5.0 google-auth>=2.15.0
```

### Create Direct Transcription Script
Create a file named `direct-transcribe.py`:

```python
import sys
import subprocess
import os
import json
import time
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using Whisper")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: medium)")
    parser.add_argument("--output", default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--gpus", default="2", help="Number of GPUs to use (default: 2)")
    args = parser.parse_args()
    
    video_url = args.url
    model_name = args.model
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the video
    print(f"Downloading: {video_url}")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    download_cmd = [
        "python3", "yt-dlp-aduio-processor-v1.py",
        "--url", video_url,
        "--output", temp_dir
    ]
    result = subprocess.run(download_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading video: {result.stderr}")
        return
    
    # Find the downloaded audio file
    audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
    if not audio_files:
        print("No audio file found after download")
        return
    
    audio_path = os.path.join(temp_dir, audio_files[0])
    desc_path = os.path.join(temp_dir, os.path.splitext(audio_files[0])[0] + '.txt')
    
    if not os.path.exists(desc_path):
        # Create a basic description file if none exists
        with open(desc_path, 'w') as f:
            f.write(f"Upload Date: {time.strftime('%m-%d-%Y')}\n")
            f.write(f"Episode_number: 1\n")
            f.write(f"Title: {os.path.basename(audio_path)}\n\n")
    
    # Transcribe the video
    print(f"Transcribing: {audio_path} with model {model_name}")
    transcribe_cmd = [
        "python3", "transcribe-whisper-gpu.py",
        "--audio", audio_path,
        "--desc", desc_path,
        "--model", model_name,
        "--gpus", args.gpus
    ]
    subprocess.run(transcribe_cmd)
    
    # Move results to output directory
    output_file = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.json"
    if os.path.exists(output_file):
        # Print summary of transcription
        with open(output_file, 'r') as f:
            data = json.load(f)
            print(f"Transcription complete! Generated {len(data)} text segments.")
            print(f"Output file: {os.path.abspath(output_file)}")
    else:
        print("Transcription may have failed. Check for errors above.")

if __name__ == "__main__":
    main()
```

Make the script executable:
```bash
chmod +x direct-transcribe.py
```

### Create Configuration for GCP (if using GCS/PubSub)

```bash
# Install gcloud CLI
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-464.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-464.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Authenticate
gcloud auth login
gcloud auth application-default login

# Create service account key file if needed
# Replace with your actual project and service account details
gcloud iam service-accounts keys create key.json \
  --iam-account=transcription-processor@rag-widget.iam.gserviceaccount.com

# Set environment variable for GCP authentication
echo 'export GOOGLE_APPLICATION_CREDENTIALS="'$(pwd)'/key.json"' >> ~/.bashrc
source ~/.bashrc
```

## 6. Test the Transcription Service

### Run a Test Transcription
```bash
# Activate virtual environment if not already activated
source venv/bin/activate

# Run a test transcription
python direct-transcribe.py https://www.youtube.com/watch?v=YOUR_TEST_VIDEO_ID --model medium
```

### Create a Batch Processing Script
Create a file named `batch-process.sh`:

```bash
#!/bin/bash

# Check if file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <url_file>"
    exit 1
fi

URL_FILE=$1

if [ ! -f "$URL_FILE" ]; then
    echo "Error: File $URL_FILE does not exist."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Process each URL in the file
echo "Starting batch processing..."
while IFS= read -r url || [[ -n "$url" ]]; do
    # Skip empty lines and comments
    if [[ -z "$url" || "$url" == \#* ]]; then
        continue
    fi
    
    echo "----------------------------------------"
    echo "Processing: $url"
    echo "Start time: $(date)"
    
    python direct-transcribe.py "$url" --model medium
    
    echo "Finished at: $(date)"
    echo "----------------------------------------"
    echo ""
done < "$URL_FILE"

echo "All videos processed!"
```

Make it executable:
```bash
chmod +x batch-process.sh
```

## 7. Set Up Automatic Startup (Optional)

### Create a Systemd Service for PubSub Processing
If you want to run the service continuously to process Pub/Sub messages:

```bash
# Create service file
sudo tee /etc/systemd/system/transcription.service > /dev/null << EOL
[Unit]
Description=YouTube Transcription Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/transcription-service/video-processing-service
ExecStart=/root/transcription-service/video-processing-service/venv/bin/python3 k8s-processor.py
Environment="GOOGLE_APPLICATION_CREDENTIALS=/root/transcription-service/video-processing-service/key.json"
Environment="PROJECT_ID=rag-widget"
Environment="SUBSCRIPTION_ID=video-transcription-processor"
Environment="BUCKET_NAME=rag-widget-transcription-outputs"
Environment="GPU_COUNT=2"
Environment="MODEL_NAME=medium"
Environment="MAX_MESSAGES=1"
Restart=always

[Install]
WantedBy=multi-user.target
```

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable transcription.service
sudo systemctl start transcription.service

# Check status
sudo systemctl status transcription.service
```

## 8. Monitoring and Maintenance

### Monitor GPU Usage
```bash
# Install monitoring tools
sudo apt-get install -y htop

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# For more detailed monitoring, consider installing nvtop
sudo apt-get install -y nvtop
nvtop
```

### Log Files and Debugging
```bash
# View service logs if using systemd
sudo journalctl -u transcription.service -f

# Create a dedicated log directory
mkdir -p ~/transcription-service/logs

# Modify the service to log output
sudo systemctl edit transcription.service
```

Add these lines:
```
[Service]
StandardOutput=append:/home/YOUR_USERNAME/transcription-service/logs/service.log
StandardError=append:/home/YOUR_USERNAME/transcription-service/logs/error.log
```

Then reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart transcription.service
```

## 9. Performance Optimization

### Fine-tune GPU Memory Allocation
Add to your direct-transcribe.py script to limit memory growth:

```python
# Add at the beginning of the script
import os
os.environ['TF_MEMORY_ALLOCATION'] = '0.8'  # Use 80% of GPU memory
```

### Use Disk Caching for Large Files
Create a temporary storage area with good I/O performance:

```bash
# Create a RAM disk for temporary files (adjust size as needed)
sudo mkdir -p /mnt/ramdisk
sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk

# Use it in your scripts
temp_dir = "/mnt/ramdisk/temp"
```

## 10. Additional Tips

### Create a Convenient Alias
```bash
echo 'alias transcribe="cd ~/transcription-service/youtube-video-transcriptor && source venv/bin/activate && python direct-transcribe.py"' >> ~/.bashrc
source ~/.bashrc

# Now you can simply run:
transcribe https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

### Set Up Automated Cleanup
```bash
# Create a cleanup script
cat > cleanup.sh << 'EOF'
#!/bin/bash
# Remove temporary files older than 7 days
find ~/transcription-service/youtube-video-transcriptor/outputs/temp -type f -mtime +7 -delete
# Remove empty directories
find ~/transcription-service/youtube-video-transcriptor/outputs/temp -type d -empty -delete
EOF

chmod +x cleanup.sh

# Add to crontab to run daily
(crontab -l 2>/dev/null; echo "0 0 * * * $(pwd)/cleanup.sh") | crontab -
```

---

This setup gives you a robust, high-performance transcription service on your Ubuntu 22.04 server with RTX 4090 GPUs. The service can be used in three ways:

1. **Direct usage**: Transcribe videos one at a time with `direct-transcribe.py`
2. **Batch processing**: Process multiple videos with `batch-process.sh`
3. **Continuous processing**: Listen for messages on Google Cloud Pub/Sub using the systemd service

All processing takes advantage of your RTX 4090 GPUs for maximum performance.