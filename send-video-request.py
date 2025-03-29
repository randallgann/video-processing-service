#!/usr/bin/env python3
"""
Send a video processing request to the video-processing-requests Pub/Sub topic.
"""

import os
import json
import argparse
from google.cloud import pubsub_v1

def publish_video_request(project_id, topic_id, video_url):
    """Publish a video processing request to the specified topic."""
    try:
        # Initialize publisher client
        publisher = pubsub_v1.PublisherClient.from_service_account_json(
            "rag-widget-pubsub-publisher-key.json")
        
        # Construct topic path
        topic_path = publisher.topic_path(project_id, topic_id)
        
        # Create message data
        message = {
            "video_url": video_url
        }
        message_data = json.dumps(message).encode('utf-8')
        
        # Publish message
        future = publisher.publish(topic_path, data=message_data)
        message_id = future.result()
        
        print(f"Published message with ID: {message_id}")
        print(f"Video URL: {video_url}")
        print(f"Topic: {topic_path}")
        
        return message_id
    
    except Exception as e:
        print(f"Error publishing message: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Send a video processing request')
    parser.add_argument('--project', default=os.environ.get('PROJECT_ID', ''),
                       help='Google Cloud project ID')
    parser.add_argument('--topic', default='video-processing-requests',
                       help='Pub/Sub topic for video requests')
    parser.add_argument('--video', required=True,
                       help='YouTube video URL to process')
    
    args = parser.parse_args()
    
    if not args.project:
        print("Error: PROJECT_ID is not set. Use --project or set the PROJECT_ID environment variable.")
        return
    
    message_id = publish_video_request(args.project, args.topic, args.video)
    
    if message_id:
        print("\nRequest sent successfully!")
        print("To monitor progress, run:")
        print(f"python monitor-progress.py --project {args.project}")

if __name__ == "__main__":
    main()