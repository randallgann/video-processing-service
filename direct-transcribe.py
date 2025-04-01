import sys
import subprocess
import os
import json
import time
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using Whisper")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model to use (default: medium)")
    parser.add_argument("--output", default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--gpus", default="2", help="Number of GPUs to use (default: 2)")
    args = parser.parse_args()
    
    video_url = args.url
    model_name = args.model
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the video
    print(f"Downloading: {video_url}")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    download_cmd = [
        "python3", "yt-dlp-aduio-processor-v1.py",
        "--url", video_url,
        "--output", temp_dir
    ]
    result = subprocess.run(download_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading video: {result.stderr}")
        return
    
    # Find the downloaded audio file
    audio_files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
    if not audio_files:
        print("No audio file found after download")
        return
    
    audio_path = os.path.join(temp_dir, audio_files[0])
    desc_path = os.path.join(temp_dir, os.path.splitext(audio_files[0])[0] + '.txt')
    
    if not os.path.exists(desc_path):
        # Create a basic description file if none exists
        with open(desc_path, 'w') as f:
            f.write(f"Upload Date: {time.strftime('%m-%d-%Y')}\n")
            f.write(f"Episode_number: 1\n")
            f.write(f"Title: {os.path.basename(audio_path)}\n\n")
    
    # Transcribe the video
    print(f"Transcribing: {audio_path} with model {model_name}")
    transcribe_cmd = [
        "python3", "transcribe-whisper-gpu.py",
        "--audio", audio_path,
        "--desc", desc_path,
        "--model", model_name,
        "--gpus", args.gpus
    ]
    subprocess.run(transcribe_cmd)
    
    # Move results to output directory
    output_file = os.path.splitext(os.path.basename(audio_path))[0] + "_transcript.json"
    if os.path.exists(output_file):
        # Print summary of transcription
        with open(output_file, 'r') as f:
            data = json.load(f)
            print(f"Transcription complete! Generated {len(data)} text segments.")
            print(f"Output file: {os.path.abspath(output_file)}")
    else:
        print("Transcription may have failed. Check for errors above.")

if __name__ == "__main__":
    main()