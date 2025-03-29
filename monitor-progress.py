#!/usr/bin/env python3
"""
Monitor progress updates from the video-processing-progress Pub/Sub topic.
This script creates a temporary subscription and listens for progress updates.
"""

import os
import json
import time
import argparse
from datetime import datetime
from google.cloud import pubsub_v1

def create_subscription(project_id, topic_id, subscription_id):
    """Create a subscription to the specified topic."""
    subscriber = pubsub_v1.SubscriberClient.from_service_account_json(
        "rag-widget-pubsub-subscriber-key.json")
    
    topic_path = subscriber.topic_path(project_id, topic_id)
    subscription_path = subscriber.subscription_path(project_id, subscription_id)
    
    try:
        # Check if subscription exists
        subscriber.get_subscription(subscription=subscription_path)
        print(f"Subscription {subscription_id} already exists")
    except Exception:
        # Create new subscription
        subscriber.create_subscription(
            request={"name": subscription_path, "topic": topic_path}
        )
        print(f"Created subscription: {subscription_path}")
    
    return subscription_path

def callback(message):
    """Process received progress updates."""
    try:
        data = json.loads(message.data.decode('utf-8'))
        
        # Get timestamp and format it
        timestamp = data.get('timestamp', datetime.now().isoformat())
        if 'T' in timestamp:
            timestamp = timestamp.split('T')[1].split('.')[0]  # Get just the time part
        
        # Format progress bar
        progress = data.get('progress_percent', 0)
        bar_length = 20
        completed = int(bar_length * progress / 100)
        progress_bar = '█' * completed + '░' * (bar_length - completed)
        
        # Get status info
        status = data.get('status', 'unknown')
        stage = data.get('current_stage', 'unknown')
        stage_progress = data.get('stage_progress_percent', 0)
        
        # Get time info
        proc_time = data.get('processing_time_seconds', 0)
        remaining = data.get('estimated_time_remaining_seconds', 0)
        
        # Clear line and print progress
        print('\r', end='')
        print(f"[{timestamp}] [{status.upper()}] [{stage}] ", end='')
        print(f"[{progress_bar}] {progress:.1f}% ", end='')
        
        if status != "completed":
            print(f"(Stage: {stage_progress:.1f}%) ", end='')
            print(f"Time: {proc_time}s Remaining: {remaining}s ", end='')
        else:
            print(f"Completed in {proc_time}s", end='')
            
        # Acknowledge the message
        message.ack()
        
        # If we're done, add a newline
        if status == "completed" or status == "failed":
            print("")
            
            # If error, print the error message
            if status == "failed" and 'error' in data and data['error']:
                print(f"Error: {data['error']}")
    
    except Exception as e:
        print(f"\nError processing message: {e}")
        message.ack()  # Acknowledge to avoid redelivery

def main():
    parser = argparse.ArgumentParser(description='Monitor video processing progress')
    parser.add_argument('--project', default=os.environ.get('PROJECT_ID', ''),
                       help='Google Cloud project ID')
    parser.add_argument('--topic', default=os.environ.get('PROGRESS_TOPIC_ID', 'video-processing-progress'),
                       help='Pub/Sub topic for progress updates')
    parser.add_argument('--subscription', default='progress-monitor-temp',
                       help='Temporary subscription name to create')
    
    args = parser.parse_args()
    
    if not args.project:
        print("Error: PROJECT_ID is not set. Use --project or set the PROJECT_ID environment variable.")
        return
        
    try:
        # Create or reuse subscription
        subscription_path = create_subscription(args.project, args.topic, args.subscription)
        
        # Initialize subscriber client
        subscriber = pubsub_v1.SubscriberClient.from_service_account_json(
            "rag-widget-pubsub-subscriber-key.json")
        
        # Set up streaming pull
        print(f"Listening for progress updates on {args.topic}...")
        streaming_pull_future = subscriber.subscribe(
            subscription_path, callback=callback
        )
        
        # Keep the main thread from exiting
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            print("\nMonitoring stopped.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()