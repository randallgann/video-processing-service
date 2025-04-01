# GKE GPU Transcription Service - Deployment Commands

## Prerequisites

1. GCP project: `rag-widget`
2. GKE Autopilot cluster: `transcriber-service-autopilot` in `us-central1` region
3. NVIDIA L4 GPUs available in the cluster region
4. Docker image built and pushed to Artifact Registry

## Resource Creation Commands

### 1. Set up Pub/Sub Resources

```bash
# Create Pub/Sub topic for transcription requests
gcloud pubsub topics create video-processing-requests

# Create Pub/Sub subscription for processing
gcloud pubsub subscriptions create video-transcription-processor \
  --topic=video-processing-requests \
  --ack-deadline=600 \
  --message-retention-duration=7d
```

### 2. Create GCS Bucket for Outputs

```bash
# Create a Cloud Storage bucket for transcription outputs
gcloud storage buckets create gs://rag-widget-transcription-outputs \
  --location=us-central1 \
  --uniform-bucket-level-access
```

### 3. Build and Push Docker Image

```bash
# Configure Docker for GCP Authentication
gcloud auth configure-docker us-south1-docker.pkg.dev

# Create Artifact Registry Repository (if not already created)
gcloud artifacts repositories create video-transcriber \
  --repository-format=docker \
  --location=us-south1 \
  --description="Repository for video transcription service"

# Build the Docker image
docker build -t us-central1-docker.pkg.dev/rag-widget/video-transcriber/transcriber:latest .

# Push the Docker image
docker push us-central1-docker.pkg.dev/rag-widget/video-transcriber/transcriber:latest
```

### 4. Create IAM Service Account with Required Permissions

```bash
# Create Service Account
gcloud iam service-accounts create transcription-processor --display-name="Transcription Processor Service Account"

# Add Pub/Sub Subscriber Role
gcloud projects add-iam-policy-binding rag-widget --member="serviceAccount:transcription-processor@rag-widget.iam.gserviceaccount.com" --role="roles/pubsub.subscriber"

# Add Storage Object Creator Role
gcloud projects add-iam-policy-binding rag-widget --member="serviceAccount:transcription-processor@rag-widget.iam.gserviceaccount.com" --role="roles/storage.objectCreator"

# Create and download the service account key
gcloud iam service-accounts keys create key.json --iam-account=transcription-processor@rag-widget.iam.gserviceaccount.com
```

### 5. Connect to the GKE Cluster

```bash
gcloud container clusters get-credentials transcriber-service-autopilot --region=us-central1
gcloud container clusters get-credentials autopilot-cluster-1 --region=us-central1
```

### 6. Apply the Kubernetes Configuration

```bash
# Create a Kubernetes secret for the service account key
kubectl create secret generic gcp-credentials --from-file=key.json=./key.json

# Apply the deployment configuration
kubectl apply -f transcriber-deployment.yaml
```

### 7. Monitor the Deployment

```bash
# Check the status of the pods
kubectl get pods -l app=video-transcriber

# Check the HPA
kubectl get hpa video-transcriber-hpa

# View pod logs
kubectl logs -l app=video-transcriber

# Describe the pod to check GPU allocation
kubectl describe pods -l app=video-transcriber
```

## Testing the Pipeline

### Send a Test Message to Pub/Sub

```bash
gcloud pubsub topics publish video-transcription-requests --message="{\"video_url\": \"https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID\"}"
```

### Watch Pod Scaling

```bash
# Watch as pods scale up to process messages
kubectl get pods -l app=video-transcriber -w
```

### Check Results in Storage

```bash
# List files in the output bucket
gcloud storage ls gs://rag-widget-transcription-outputs/
```

## Troubleshooting

### Check Pod Status and Events

```bash
kubectl describe pods -l app=video-transcriber
```

### View Container Logs

```bash
kubectl logs -l app=video-transcriber
```

### Check GPU Node Availability

```bash
kubectl get nodes --selector=cloud.google.com/gke-accelerator=nvidia-l4
```

### Check GPU Driver Installation

```bash
kubectl logs --selector=k8s-app=nvidia-gpu-device-plugin \
  --container="nvidia-gpu-device-plugin" \
  --tail=-1 \
  --namespace=kube-system | grep Driver
```

```bash
  kubectl delete deployment video-transcriber
  kubectl apply -f video-transcriber-test-pod.yaml
```