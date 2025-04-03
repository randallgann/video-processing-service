# Setting Up Public IP Access for Replicate API

To make your audio files accessible to Replicate's API, you need to configure your VM and network to allow external access to your audio server.

## 1. Determine Your Public IP

First, identify your VM's public IP address:

```bash
# Get your public IP address
PUBLIC_IP=$(curl -s https://api.ipify.org)
echo "Your public IP address is: $PUBLIC_IP"
```

Add this to your environment variables:

```bash
# Add to your .env file
echo "PUBLIC_IP=$PUBLIC_IP" >> .env

# Or export directly for immediate use
export PUBLIC_IP=$PUBLIC_IP
```

## 2. Configure Firewall Rules

You need to open port 8000 (or your chosen port) on your VM and any upstream firewalls:

### VM Firewall (UFW)

```bash
# Open port 8000 on the VM
sudo ufw allow 8000/tcp
sudo ufw status
```

### Cloud Provider Firewall Rules

Depending on your cloud provider, you'll need to configure additional firewall rules:

#### Google Cloud Platform (GCP)

```bash
# Create a firewall rule to allow inbound traffic on port 8000
gcloud compute firewall-rules create allow-audio-server \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8000 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=audio-server

# Tag your VM with the 'audio-server' tag
gcloud compute instances add-tags YOUR_VM_NAME --tags=audio-server --zone=YOUR_ZONE
```

#### AWS

```bash
# If using AWS EC2, add a security group rule
aws ec2 authorize-security-group-ingress \
    --group-id YOUR_SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0
```

#### Azure

```bash
# If using Azure, add a network security rule
az network nsg rule create \
    --name AllowAudioServer \
    --nsg-name YOUR_NSG_NAME \
    --priority 1000 \
    --resource-group YOUR_RESOURCE_GROUP \
    --access Allow \
    --protocol Tcp \
    --direction Inbound \
    --source-address-prefix '*' \
    --source-port-range '*' \
    --destination-address-prefix '*' \
    --destination-port-range 8000
```

## 3. Test Connectivity

To verify your setup, start the audio server and test accessibility:

```bash
# Start the audio server explicitly (normally handled by the script)
python3 mini_audio_server.py --directory /path/to/audio/files

# Test with curl from another machine or your local computer
curl http://$PUBLIC_IP:8000/
```

You should see a directory listing or a response from the server.

## 4. Security Considerations

Opening your server to the internet introduces security concerns:

1. **Temporary Access**: The mini_audio_server is designed to shut down after a period (default: 1 hour)
2. **Limited Content**: Only serve directories containing audio files, not sensitive data
3. **Firewall Restriction**: Consider limiting access to only Replicate's IP ranges if available
4. **HTTPS**: For production use, consider setting up HTTPS with a reverse proxy

## 5. Troubleshooting

If Replicate still can't access your files, check:

1. **Firewalls**: Ensure all firewall levels (VM, cloud provider, network) allow port 8000
2. **IP Address**: Confirm you're using the correct public IP in the PUBLIC_IP environment variable
3. **Port Binding**: Make sure the server is binding to 0.0.0.0 (all interfaces) not just localhost
4. **NAT Issues**: If behind NAT, you may need port forwarding configuration
5. **Server Logs**: Check mini_audio_server logs for connection attempts

## 6. Alternative Solutions

If direct access remains problematic, consider these alternatives:

1. **Cloud Storage**: Upload audio to GCS/S3 and generate signed URLs
2. **Tunneling Services**: Use ngrok or cloudflared for temporary public access
3. **CDN**: For production use, consider a CDN for serving audio files