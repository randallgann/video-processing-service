# GKE GPU Implementation for Video Transcription Pipeline

This document outlines a cost-optimized implementation of the YouTube video transcription pipeline using Google Kubernetes Engine (GKE) with GPU nodes.

## Architecture Overview

```
GCP Pub/Sub → GKE GPU Cluster → Cloud Storage
       ↑              ↓
       └───── Cloud Logging ────┘
```

## Cost Optimization Strategy

1. **GKE Autopilot with Spot VMs** - Save 60-90% on compute costs
2. **GPU Time-Sharing** - Process multiple videos on a single GPU where possible
3. **Ephemeral GPU Nodes** - Scale to zero when no processing required
4. **Efficient GPU Selection** - Use L4 GPUs (best price/performance for transcription)
5. **Pub/Sub Message Batching** - Process multiple messages in batches when possible

## Setup Process

### 1. Containerize the Transcription Pipeline

**1.1 Create Dockerfile**

```Dockerfile
FROM nvidia/cuda:12.3.1-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY transcribe-whisper-gpu.py /app/
COPY yt-dlp-aduio-processor-v1.py /app/

WORKDIR /app

# Install additional dependencies
RUN pip3 install --no-cache-dir google-cloud-pubsub google-cloud-storage

# Copy processor script
COPY k8s-processor.py /app/

# Run the processor script
CMD ["python3", "k8s-processor.py"]
```

**1.2 Create requirements.txt**

```
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.20.0
whisper>=1.0.0
google-cloud-pubsub>=2.13.0
google-cloud-storage>=2.5.0
google-auth>=2.15.0
pydub>=0.25.1
yt_dlp>=2023.3.4
```

**1.3 Create k8s-processor.py**

```python
import os
import time
import json
import subprocess
import tempfile
from google.cloud import pubsub_v1
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
PROJECT_ID = os.environ.get('PROJECT_ID')
SUBSCRIPTION_ID = os.environ.get('SUBSCRIPTION_ID')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MAX_MESSAGES = int(os.environ.get('MAX_MESSAGES', '1'))
GPU_COUNT = int(os.environ.get('GPU_COUNT', '1'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'medium')

def download_video(video_url):
    """Download video using yt-dlp and return the audio path"""
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Downloading video: {video_url} to {temp_dir}")
    
    # Run the yt-dlp processor script
    command = [
        "python3", "yt-dlp-aduio-processor-v1.py", 
        "--url", video_url,
        "--output", temp_dir
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error downloading video: {result.stderr}")
        raise Exception(f"Failed to download video: {result.stderr}")
    
    # Find the downloaded audio file
    audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
    if not audio_files:
        raise Exception("No audio file found after download")
    
    audio_path = os.path.join(temp_dir, audio_files[0])
    desc_path = os.path.join(temp_dir, os.path.splitext(audio_files[0])[0] + '.txt')
    
    return audio_path, desc_path, temp_dir

def transcribe_video(audio_path, desc_path, message_id):
    """Transcribe the video using whisper-gpu script"""
    output_dir = tempfile.mkdtemp()
    logger.info(f"Transcribing audio: {audio_path} with model {MODEL_NAME}")
    
    command = [
        "python3", "transcribe-whisper-gpu.py",
        "--audio", audio_path,
        "--desc", desc_path,
        "--model", MODEL_NAME,
        "--gpus", str(GPU_COUNT),
        "--output", output_dir
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Error transcribing video: {result.stderr}")
        raise Exception(f"Failed to transcribe video: {result.stderr}")
    
    return output_dir

def upload_results(output_dir, message_id):
    """Upload transcription results to GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    results_uploaded = 0
    logger.info(f"Uploading results from {output_dir} to gs://{BUCKET_NAME}/{message_id}/")
    
    for root, _, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, output_dir)
            blob_path = f"{message_id}/{relative_path}"
            
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            results_uploaded += 1
    
    logger.info(f"Uploaded {results_uploaded} files to GCS")
    return results_uploaded

def process_message(message):
    """Process a single Pub/Sub message"""
    try:
        message_id = message.message_id
        logger.info(f"Processing message {message_id}")
        
        # Parse message data
        data = json.loads(message.data.decode('utf-8'))
        video_url = data.get('video_url')
        
        if not video_url:
            logger.error("Message missing video_url")
            message.ack()
            return
        
        # Process the video
        audio_path, desc_path, temp_dir = download_video(video_url)
        output_dir = transcribe_video(audio_path, desc_path, message_id)
        upload_results(output_dir, message_id)
        
        # Cleanup
        for dir_path in [temp_dir, output_dir]:
            try:
                subprocess.run(["rm", "-rf", dir_path])
            except Exception as e:
                logger.warning(f"Error cleaning up directory {dir_path}: {e}")
        
        # Acknowledge the message
        message.ack()
        logger.info(f"Successfully processed message {message_id}")
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        # Don't acknowledge to allow redelivery

def main():
    """Main processing loop"""
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
    
    # Configure flow control
    flow_control = pubsub_v1.types.FlowControl(max_messages=MAX_MESSAGES)
    
    def callback(message):
        process_message(message)
    
    # Subscribe to the subscription
    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback, flow_control=flow_control
    )
    
    logger.info(f"Listening for messages on {subscription_path}")
    logger.info(f"Using GPU count: {GPU_COUNT}")
    logger.info(f"Using model: {MODEL_NAME}")
    
    try:
        # Keep the main thread from exiting
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        logger.info("Subscription canceled")
    except Exception as e:
        logger.error(f"Streaming pull error: {e}")
        streaming_pull_future.cancel()
        raise

if __name__ == "__main__":
    main()
```

### 2. Build and Push Docker Image

#### Steps in GCP Console:

1. **Navigate to Artifact Registry**
   - Go to GCP Console → Artifact Registry → Repositories
   - Click "Create Repository"
   - Name: `video-transcriber`
   - Format: Docker
   - Location type: Region (select your preferred region)
   - Click "Create"

2. **Build and Push Using Cloud Build**
   - Go to Cloud Build → Triggers
   - Click "Create Trigger"
   - Name: `build-transcriber-image`
   - Event: Manual invocation
   - Source: Connect to your repository (GitHub/BitBucket/etc.)
   - Configuration: Dockerfile
   - Dockerfile directory: `/` (or your directory)
   - Dockerfile name: `Dockerfile`
   - Image name: `[REGION]-docker.pkg.dev/[PROJECT_ID]/video-transcriber/transcriber:latest`
   - Click "Create"
   - Click "Run" to build the image

### 3. Create GKE Autopilot Cluster

1. **Navigate to GKE**
   - Go to GCP Console → Kubernetes Engine → Clusters
   - Click "Create"
   - Select "Autopilot" mode
   - Click "Configure"

2. **Configure Cluster Basics**
   - Name: `transcription-cluster`
   - Region: Choose your preferred region (where L4 GPUs are available)
   - Release channel: Standard
   - Click "Networking"

3. **Configure Networking**
   - Network: default (or your VPC)
   - Click "Security"

4. **Configure Security**
   - Leave defaults
   - Click "Create"

5. **Wait for Cluster Creation**
   - This will take 5-10 minutes

### 4. Configure GPU Resources

1. **Enable GPUs for Autopilot**
   - Go to your cluster → Features
   - Find "GPU resources" in the list
   - Click "Enable" (if not already enabled)
   - Select "NVIDIA L4" (best price/performance)

### 5. Create Pub/Sub Topic and Subscription

1. **Navigate to Pub/Sub**
   - Go to GCP Console → Pub/Sub → Topics
   - Click "Create Topic"
   - Topic ID: `video-transcription-requests`
   - Click "Create"

2. **Create Subscription**
   - In the topic details page, click "Create Subscription"
   - Subscription ID: `video-transcription-processor`
   - Delivery type: Pull
   - Acknowledgement deadline: 600 seconds (10 minutes)
   - Message retention: 7 days
   - Click "Create"

### 6. Create Storage Bucket

1. **Navigate to Cloud Storage**
   - Go to GCP Console → Cloud Storage → Buckets
   - Click "Create"
   - Name: `[PROJECT_ID]-transcription-outputs`
   - Location type: Region (same as your GKE cluster)
   - Storage class: Standard
   - Access control: Uniform
   - Click "Create"

### 7. Create Kubernetes Deployment

1. **Navigate to GKE Workloads**
   - Go to GCP Console → Kubernetes Engine → Workloads
   - Click "Deploy"

2. **Configure Deployment**
   - Container image: `[REGION]-docker.pkg.dev/[PROJECT_ID]/video-transcriber/transcriber:latest`
   - Click "Continue"

3. **Configure Container**
   - Name: `transcriber`
   - Click "Add environment variable" (add the following):
     - Name: `PROJECT_ID`, Value: `[YOUR_PROJECT_ID]`
     - Name: `SUBSCRIPTION_ID`, Value: `video-transcription-processor`
     - Name: `BUCKET_NAME`, Value: `[PROJECT_ID]-transcription-outputs`
     - Name: `GPU_COUNT`, Value: `1`
     - Name: `MODEL_NAME`, Value: `medium`
     - Name: `MAX_MESSAGES`, Value: `1`
   - Click "Container resources" 
   - Request GPU: Type - NVIDIA L4, Count - 1
      - G2 VMs: these VMs have NVIDIA L4 GPUs automatically attached.
   - Click "Continue" 

4. **Configure Workload Details**
   - Name: `video-transcriber`
   - Namespace: default
   - Labels: Add `app=video-transcriber`
   - Click "Deployment options"

5. **Configure Deployment Options**
   - Replicas: 0 (start with 0 and scale with HPA)
   - Click "Auto scaling" tab
   - Enable autoscaling
   - Minimum replicas: 0
   - Maximum replicas: 5 (adjust based on your needs)
   - Click "Create"

### 8. Create Horizontal Pod Autoscaler (HPA)

1. **Navigate to GKE Clusters**
   - Go to GCP Console → Kubernetes Engine → Clusters
   - Click on your cluster
   - Click "Connect" > "Run in Cloud Shell"

2. **Create HPA YAML**
   - In Cloud Shell, create a file named `transcriber-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: video-transcriber-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: video-transcriber
  minReplicas: 0
  maxReplicas: 5
  metrics:
  - type: External
    external:
      metric:
        name: pubsub.googleapis.com|subscription|num_undelivered_messages
        selector:
          matchLabels:
            resource.labels.subscription_id: video-transcription-processor
      target:
        type: AverageValue
        averageValue: 1
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
```

3. **Apply HPA YAML**
   - Run: `kubectl apply -f transcriber-hpa.yaml`

### 9. Create Service Account and IAM Permissions

1. **Navigate to IAM & Admin**
   - Go to GCP Console → IAM & Admin → Service accounts
   - Click "Create Service Account"
   - Name: `transcription-processor`
   - Click "Create and Continue"
   - Add the following roles:
     - Pub/Sub Subscriber
     - Storage Object Creator
   - Click "Done"

2. **Create and Download Key**
   - Find the service account in the list
   - Click on the three dots → "Manage Keys"
   - Click "Add Key" → "Create new key"
   - Select JSON format
   - Click "Create"
   - Save the downloaded key

3. **Create Kubernetes Secret**
   - Go to Kubernetes Engine → Clusters
   - Click on your cluster
   - Click "Connect" > "Run in Cloud Shell"
   - Run: `kubectl create secret generic gcp-credentials --from-file=key.json=/path/to/downloaded/key.json`

4. **Update the Deployment to Use the Secret**
   - Go to Kubernetes Engine → Workloads
   - Find your `video-transcriber` deployment
   - Click "Edit"
   - In the "Environment variables" section, add:
     - Name: `GOOGLE_APPLICATION_CREDENTIALS`, Value: `/var/secrets/google/key.json`
   - Add a volume mount:
     - Volume: Create new volume named `gcp-creds`
     - Type: Secret
     - Secret name: `gcp-credentials`
     - Mount path: `/var/secrets/google`
   - Click "Update"

## Testing the Pipeline

1. **Send a Test Message to Pub/Sub**
   - Go to Pub/Sub → Topics
   - Select your `video-transcription-requests` topic
   - Click "Publish message"
   - Message body:
     ```json
     {
       "video_url": "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID"
     }
     ```
   - Click "Publish"

2. **Monitor Pod Creation and Scaling**
   - Go to Kubernetes Engine → Workloads
   - Watch as the HPA creates pods to process the message

3. **Check Results in Storage**
   - Go to Cloud Storage → Buckets
   - Navigate to your `[PROJECT_ID]-transcription-outputs` bucket
   - Verify the transcription results are uploaded

## Cost Optimization Tips

1. **Use Preemptible/Spot VMs**
   - In your GKE configuration, enable Spot VMs for significant savings
   - Spot VMs cost ~70% less than regular VMs

2. **Scale to Zero**
   - The HPA is configured to scale to zero when there are no messages
   - This means you only pay for resources when processing videos

3. **Batch Processing**
   - Adjust `MAX_MESSAGES` to process multiple videos when there's a backlog
   - This improves GPU utilization

4. **Monitor and Adjust**
   - Regularly check your GPU utilization metrics
   - If consistently high, consider using A100 GPUs for faster processing
   - If consistently low, adjust to use smaller GPUs like T4

5. **GPU Sharing**
   - Consider enabling time-slicing to share a GPU between multiple pods
   - This works well when processing shorter videos

## Troubleshooting

1. **Pod Failures**
   - Check pod logs: Kubernetes Engine → Workloads → video-transcriber → Logs
   - Common issues include memory limits, GPU availability, or permission problems

2. **Pub/Sub Issues**
   - Check subscription backlog: Pub/Sub → Subscriptions → video-transcription-processor
   - Verify messages are being acknowledged after successful processing

3. **Storage Issues**
   - Check IAM permissions if uploads fail
   - Verify bucket location matches your GKE cluster's region for best performance

4. **GPU Availability**
   - If pods are stuck in "Pending" state, check GPU quota in your region
   - Consider requesting additional quota or trying a different region