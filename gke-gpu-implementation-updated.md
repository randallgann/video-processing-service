# GKE GPU Implementation for Video Transcription Pipeline (Updated)

This document outlines a cost-optimized implementation of the YouTube video transcription pipeline using Google Kubernetes Engine (GKE) Autopilot with GPU support.

## Architecture Overview

```
GCP Pub/Sub → GKE GPU Cluster (Autopilot with L4 GPUs) → Cloud Storage
       ↑              ↓
       └───── Cloud Logging ────┘
```

## Setup Process Summary

1. **Create a GKE Autopilot cluster in a region with L4 GPU support**
   - Cluster name: `transcriber-service-autopilot`
   - Region: `us-central1` (supports NVIDIA L4 GPUs)
   - Project: `rag-widget`

2. **Build and push the Docker image to Artifact Registry**
   - Repository: `video-transcriber` (in us-south1 region)
   - Image: `us-south1-docker.pkg.dev/rag-widget/video-transcriber/transcriber:latest`

3. **Create Pub/Sub resources**
   - Topic: `video-transcription-requests`
   - Subscription: `video-transcription-processor`

4. **Create Cloud Storage bucket**
   - Bucket name: `rag-widget-transcription-outputs`

5. **Create Service Account with required permissions**
   - Name: `transcription-processor`
   - Roles: Pub/Sub Subscriber, Storage Object Creator

6. **Deploy the transcription service with GPU support**
   - Apply the `transcriber-deployment.yaml` configuration

## GPU Configuration in Autopilot

With GKE Autopilot version 1.31.6-gke.1020000, we can request GPUs directly in our workload manifests. Autopilot will automatically:

1. Schedule the Pod on a node with the requested GPU type and count
2. Install the appropriate NVIDIA GPU drivers
3. Configure the necessary CUDA environment
4. Apply appropriate taints to ensure proper scheduling

## Implementation Details

### Deployment Configuration

The deployment is configured to:

1. **Use L4 GPUs with nodeSelector** 
   ```yaml
   nodeSelector:
     cloud.google.com/gke-accelerator: "nvidia-l4"
   ```

2. **Request appropriate resources for GPU workloads**
   ```yaml
   resources:
     requests:
       cpu: "4"
       memory: "16Gi"
       nvidia.com/gpu: 1
     limits:
       cpu: "8"
       memory: "24Gi"
       nvidia.com/gpu: 1
   ```

3. **Enable horizontal autoscaling based on Pub/Sub message count**
   - Scale up when there are undelivered messages
   - Scale down to zero when idle for cost optimization

### Cost Optimization Strategy

1. **GKE Autopilot with L4 GPUs** - Provides the best price/performance ratio for transcription workloads
2. **Scale to Zero** - Avoid paying for idle resources when there are no messages to process
3. **Pub/Sub Message Batching** - Process multiple messages in batches when possible by adjusting the `MAX_MESSAGES` environment variable

## Deployment and Usage Instructions

For detailed deployment commands, see [gke-commands.md](gke-commands.md).

To deploy the transcription service:

1. Create the necessary GCP resources (Pub/Sub, GCS bucket)
2. Build and push the Docker image
3. Apply the Kubernetes configuration with `kubectl apply -f transcriber-deployment.yaml`

To submit a video for transcription:

```bash
gcloud pubsub topics publish video-transcription-requests --message="{\"video_url\": \"https://www.youtube.com/watch?v=YOUR_VIDEO_ID\"}"
```

## Monitoring and Management

Monitor the service using:

1. **GKE Dashboard** - View pod status, logs, and events
2. **Cloud Monitoring** - Track GPU utilization and application performance
3. **Cloud Logging** - View application logs for debugging

## Limitations and Considerations

1. **GPU Availability** - NVIDIA L4 GPUs may have limited availability in some regions
2. **Cold Start Time** - When scaling from zero, expect a delay while GKE provisions a GPU node
3. **Resource Usage** - Whisper transcription is memory intensive; adjust memory requests if needed
4. **Quota Limits** - Ensure you have sufficient GPU quota in your project

## Troubleshooting

For common issues and their solutions, refer to the troubleshooting section in [gke-commands.md](gke-commands.md).

---

*Note: This implementation has been optimized for the transcriber-service-autopilot cluster in us-central1 with NVIDIA L4 GPUs.*