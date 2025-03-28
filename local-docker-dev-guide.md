# Local Docker Development Guide for Video Transcription Pipeline

This guide contains commands and instructions for local development and testing of the video transcription Docker container before deploying to GKE.

## Prerequisites

- Docker installed on your development machine
- NVIDIA Container Toolkit installed (for GPU support)
- Google Cloud SDK installed (for GCP authentication)

## Environment Setup

### 1. Install NVIDIA Container Toolkit (if not already installed)

```bash
# For Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Authenticate with Google Cloud

```bash
# Login to GCP
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create application default credentials
gcloud auth application-default login
```

## Building the Docker Image

```bash
# Navigate to your project directory
cd /path/to/youtube-video-transcriptor

# Build the Docker image
docker build -t video-transcriber:latest .
```

## Running the Container Locally

### 1. Test Mode (Process a Single Video)

```bash
# Run with GPU support in test mode
docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=dummy-value \
  -e BUCKET_NAME=your-bucket-name \
  -e GPU_COUNT=1 \
  -e MODEL_NAME=medium \
  video-transcriber:latest \
  --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

### 2. Interactive Shell Access

```bash
# Get a shell inside the container
docker run --rm -it --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=dummy-value \
  -e BUCKET_NAME=your-bucket-name \
  video-transcriber:latest \
  /bin/bash
```

### 3. Run with Local Credentials

```bash
# Mount your GCP credentials
docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=your-subscription-id \
  -e BUCKET_NAME=your-bucket-name \
  -e GPU_COUNT=1 \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/credentials.json \
  -v $HOME/.config/gcloud/application_default_credentials.json:/tmp/keys/credentials.json:ro \
  video-transcriber:latest
```

### 4. Run with Local Storage Volume

```bash
# Mount a local directory to store results
docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=your-subscription-id \
  -e BUCKET_NAME=your-bucket-name \
  -e GPU_COUNT=1 \
  -v $(pwd)/outputs:/outputs \
  video-transcriber:latest
```

## Debugging and Troubleshooting

### 1. Check Logs

```bash
# View logs of a running container
docker logs -f CONTAINER_ID

# Run with increased log verbosity
docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=your-subscription-id \
  -e BUCKET_NAME=your-bucket-name \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=DEBUG \
  video-transcriber:latest
```

### 2. Verify GPU Availability

```bash
# Check if container can see GPUs
docker run --rm --gpus all nvidia/cuda:12.3.1-base-ubuntu22.04 nvidia-smi

# Check CUDA version in your container
docker run --rm --gpus all video-transcriber:latest python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), ', Device count:', torch.cuda.device_count())"
```

### 3. Test Whisper Directly

```bash
# Run whisper model directly
docker run --rm --gpus all video-transcriber:latest python3 -c "import whisper; model = whisper.load_model('tiny'); print('Model loaded successfully')"
```

## Pushing to Google Artifact Registry

```bash
# Tag your image for Artifact Registry
docker tag video-transcriber:latest REGION-docker.pkg.dev/PROJECT_ID/video-transcriber/transcriber:latest

# Configure Docker to use GCloud credentials
gcloud auth configure-docker REGION-docker.pkg.dev

# Push the image
docker push REGION-docker.pkg.dev/PROJECT_ID/video-transcriber/transcriber:latest
```

## Performance Tuning

### 1. Resource Allocation

```bash
# Limit CPU and memory (for testing resource constraints)
docker run --rm --gpus all \
  --cpus 4 \
  --memory 8g \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=your-subscription-id \
  -e BUCKET_NAME=your-bucket-name \
  -e GPU_COUNT=1 \
  video-transcriber:latest
```

### 2. Testing Different Models

```bash
# Test with smaller model for faster processing
docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=dummy-value \
  -e BUCKET_NAME=your-bucket-name \
  -e MODEL_NAME=small \
  video-transcriber:latest \
  --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

### 3. Benchmark Processing Time

```bash
# Run with time measurement
time docker run --rm --gpus all \
  -e PROJECT_ID=your-project-id \
  -e SUBSCRIPTION_ID=dummy-value \
  -e BUCKET_NAME=your-bucket-name \
  video-transcriber:latest \
  --test --video-url https://www.youtube.com/watch?v=YOUR_VIDEO_ID
```

## Common Issues and Solutions

### GPU Not Detected

1. Verify NVIDIA drivers are installed:
   ```bash
   nvidia-smi
   ```

2. Check NVIDIA Container Toolkit:
   ```bash
   sudo docker info | grep -i runtime
   ```
   You should see "nvidia" listed as a runtime.

3. Try running with `--gpus device=0` instead of `--gpus all`.

### Authentication Issues

1. Re-authenticate:
   ```bash
   gcloud auth login && gcloud auth application-default login
   ```

2. Check service account permissions:
   ```bash
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
     --role="roles/pubsub.subscriber"
   
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
     --role="roles/storage.objectCreator"
   ```

### Container Exits Immediately

1. Run with interactive shell:
   ```bash
   docker run -it --entrypoint /bin/bash video-transcriber:latest
   ```

2. Check for missing environment variables:
   ```bash
   python3 -c "import os; print('PROJECT_ID:', os.environ.get('PROJECT_ID', 'MISSING!'))"
   ```