import sys
import subprocess
import os
import json
import time
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos using Replicate Whisper API")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--model-version", default="8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e", 
                      help="Replicate model version to use")
    parser.add_argument("--output", default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--max-concurrent", default="5", help="Maximum concurrent API calls (default: 5)")
    parser.add_argument("--use-wav", action="store_true", help="Use WAV format for audio chunks (better API compatibility)")
    args = parser.parse_args()
    
    # Ensure REPLICATE_API_TOKEN is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: REPLICATE_API_TOKEN environment variable not set")
        print("Please set it with: export REPLICATE_API_TOKEN=your_token_here")
        sys.exit(1)
    
    video_url = args.url
    model_version = args.model_version
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the video
    print(f"Downloading: {video_url}")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    download_cmd = [
        "python3", "yt-dlp-aduio-processor-v1.py",
        video_url,
        "--output", temp_dir
    ]
    # Run the command and show output in real-time
    print("Running download command:", " ".join(download_cmd))
    result = subprocess.run(download_cmd, text=True)
    if result.returncode != 0:
        print(f"Error downloading video. Return code: {result.returncode}")
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
    print(f"Transcribing: {audio_path} with Replicate model version {model_version}")
    transcribe_cmd = [
        "python3", "transcribe-whisper-replicate.py",
        "--audio", audio_path,
        "--desc", desc_path,
        "--model-version", model_version,
        "--max-concurrent", args.max_concurrent
    ]
    
    # Add WAV option if specified
    if args.use_wav:
        transcribe_cmd.append("--use-wav")
        print("Using WAV format for audio chunks (better compatibility with Replicate API)")
    
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