#!/usr/bin/env python3
"""
Download YouTube videos to local directory
"""

import argparse
import os
import sys
import yt_dlp
from datetime import datetime

def download_video(url, output_dir='downloaded-videos'):
    """
    Download a YouTube video to the specified directory
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the video (default: downloaded-videos)
    
    Returns:
        Path to the downloaded video file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',  # Prefer MP4 format
        'outtmpl': os.path.join(output_dir, '%(title)s-%(id)s.%(ext)s'),
        'restrictfilenames': True,  # Avoid special characters in filenames
        'quiet': False,
        'no_warnings': False,
        'merge_output_format': 'mp4',  # Ensure output is MP4
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading video from: {url}")
            info = ydl.extract_info(url, download=True)
            
            # Get the actual filename
            filename = ydl.prepare_filename(info)
            # Handle merged format extension
            if not filename.endswith('.mp4'):
                filename = filename.rsplit('.', 1)[0] + '.mp4'
            
            print(f"\nVideo downloaded successfully!")
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 0) // 60} minutes {info.get('duration', 0) % 60} seconds")
            print(f"File saved to: {filename}")
            
            # Also save metadata
            metadata_file = filename.rsplit('.', 1)[0] + '_metadata.txt'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {info.get('title', 'Unknown')}\n")
                f.write(f"URL: {url}\n")
                f.write(f"Video ID: {info.get('id', 'Unknown')}\n")
                f.write(f"Duration: {info.get('duration', 0)} seconds\n")
                f.write(f"Upload Date: {info.get('upload_date', 'Unknown')}\n")
                f.write(f"Uploader: {info.get('uploader', 'Unknown')}\n")
                f.write(f"View Count: {info.get('view_count', 'Unknown')}\n")
                f.write(f"Downloaded: {datetime.now().isoformat()}\n")
                f.write(f"\nDescription:\n{info.get('description', 'No description available')}\n")
            
            print(f"Metadata saved to: {metadata_file}")
            
            return filename
            
    except Exception as e:
        print(f"Error downloading video: {str(e)}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description='Download YouTube videos')
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('-o', '--output-dir', default='downloaded-videos',
                        help='Output directory for videos (default: downloaded-videos)')
    parser.add_argument('--list-formats', action='store_true',
                        help='List available formats for the video')
    
    args = parser.parse_args()
    
    if args.list_formats:
        # List available formats
        ydl_opts = {'listformats': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(args.url, download=False)
    else:
        # Download the video
        result = download_video(args.url, args.output_dir)
        if result:
            print(f"\nDownload complete: {result}")
        else:
            sys.exit(1)

if __name__ == '__main__':
    main()