#!/bin/bash

echo "🚀 Building optimized YouTube Video Transcriptor image..."

# Use Minikube's Docker environment
eval $(minikube docker-env)

# Build with optimized Dockerfile
echo "📦 Building optimized image (should be much smaller)..."
docker build -f Dockerfile.optimized -t youtube-video-transcriptor:optimized .

# Show image sizes for comparison
echo "📊 Image size comparison:"
docker images | grep youtube-video-transcriptor

echo "✅ Optimized build complete!"
echo ""
echo "💡 To use the optimized image, update your Kubernetes deployment to use:"
echo "   image: youtube-video-transcriptor:optimized"