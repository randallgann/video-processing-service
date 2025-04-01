# Cross-Region Deployment: Artifact Registry in us-south1 and GKE in us-central1

It is completely acceptable to have your container image stored in a different region (us-south1) than your GKE cluster (us-central1). Here's what you should know:

## Benefits and Considerations

1. **Network Latency**:
   - When your GKE cluster pulls the container image, there will be a small cross-region network latency
   - This only affects the initial pod startup time when the image is being pulled
   - After the image is cached on the node, there is no additional latency

2. **Network Egress Costs**:
   - There may be small egress costs for cross-region data transfer
   - This is only for the initial image pull and is typically minimal

3. **Pod Startup Time**:
   - The first pull of the image may take slightly longer due to cross-region transfer
   - Subsequent pulls are faster if the image is already cached on the node
   - To minimize impact, consider using node pools with a higher disk cache size

## Mitigating Cross-Region Impact

1. **Image Caching**:
   - GKE automatically caches images on nodes
   - Once cached, no additional pulls are needed unless the image tag changes

2. **Image Size Optimization**:
   - Keep your Docker images as small as possible
   - Use multi-stage builds to reduce final image size
   - Remove unnecessary packages and files

3. **Rolling Updates**:
   - When updating your deployment, use rolling updates to minimize downtime
   - This spreads the image pull latency across pods

## When to Consider Moving the Image

Moving your Artifact Registry to the same region as your GKE cluster is beneficial if:

1. **High-frequency deployments**: You deploy new images multiple times per day
2. **Very large images**: Your container images are extremely large (many GB)
3. **Critical startup performance**: Your application needs minimal startup time

For most use cases, including your transcription service, the cross-region setup is perfectly acceptable and the impact on performance is minimal.

## Conclusion

For your YouTube transcription service, having the image in us-south1 while running the GKE cluster in us-central1 will work well. The overhead is minimal, especially considering:

1. The service scales based on Pub/Sub messages, not frequent deployments
2. The GPU startup time will likely dominate the overall pod startup time
3. The image pull happens infrequently since the service is designed to scale to zero

If you later decide to optimize further, you can create a new repository in us-central1 and push your images there.