import yt_dlp
import os

def download_audio(url, download_path="./"):
    """
    Downloads YouTube audio using yt-dlp
    """
    
    # Ensure download_path is an absolute path
    download_path = os.path.abspath(download_path)
    print(f"Using download path: {download_path}")

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality
        'paths': {'home': download_path},  # Set the base download path
        'outtmpl': {'default': os.path.join(download_path, '%(title)s.%(ext)s')},
        'quiet': False,  # Show progress
        'no_warnings': False,
        'extract_flat': False,
        'restrictfilenames': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192'
        }]
    }
    
    try:
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting video information...")
            info = ydl.extract_info(url, download=True)
            title = info['title']
            # Get the base filename and ensure it's in the correct output directory
            base_filename = ydl.prepare_filename(info)
            # Make sure we're using the full path
            if not os.path.isabs(base_filename):
                base_filename = os.path.join(download_path, os.path.basename(base_filename))
            base_filename_no_ext, _ = os.path.splitext(base_filename)
            audio_path = base_filename_no_ext + ".mp3"
            print(f"Expected audio path: {audio_path}")
            
            # Verify file exists
            if os.path.exists(audio_path):
                print(f"Audio file exists at: {audio_path}")
            else:
                print(f"Audio file not found at: {audio_path}")
                # Try to find the file in the download directory
                for file in os.listdir(download_path):
                    if file.endswith('.mp3'):
                        audio_path = os.path.join(download_path, file)
                        print(f"Found audio file at: {audio_path}")
                        break
            description = info.get('description', '')
            upload_date = info.get('upload_date')
            print(f"Download completed: {audio_path}")
            return audio_path, title, info['title'], description, upload_date
    except Exception as e:
        print(f"Download error: {str(e)}")
        raise

# def extract_audio(video_path, title, cleaned_title, download_path="./"):
#     """Extract audio from video file"""
#     base_path = os.path.splitext(video_path)[0]
#     audio_path = os.path.join(download_path, f"{base_path}.mp3")
#     try:
#         video = VideoFileClip(video_path)
#         audio = video.audio
#         print(f"Extracting audio to: {audio_path}")
#         audio.write_audiofile(audio_path)
#         video.close()
#         audio.close()
#         return audio_path
#     except Exception as e:
#         print(f"Audio extraction error: {str(e)}")
#         raise


def download_and_process_audio(url, download_path="./"):
    """
    Downloads a YouTube video and transcribes it with speaker identification
    """
    try:
        # Ensure download_path exists
        download_path = os.path.abspath(download_path)
        os.makedirs(download_path, exist_ok=True)
        print(f"Downloading audio to: {download_path}")
        
        # Download video using yt-dlp
        # video_path, video_title, cleaned_title = download_video(url, download_path)
        # print(f"Video saved as: {video_path}")

        # Download best audio using yt-dlp
        audio_path, audio_title, cleaned_title, description, upload_date = download_audio(url, download_path)
        print(f"Audio saved as: {audio_path}")

        # Extract episode number from audio_title
        # We'll iterate from the start until we hit a non-digit character
        episode_number = ""
        for char in audio_title:
            if char.isdigit():
                episode_number += char
            else:
                break

        # Now find the first alphabetic character after the digits to define where the cleaned title should start
        # If none found, we just use whatever is left
        import re
        alpha_match = re.search('[A-Za-z]', audio_title)
        if alpha_match:
            start_idx = alpha_match.start()
            final_cleaned_title = audio_title[start_idx:].strip()
        else:
            # No alpha found, just use the remainder
            final_cleaned_title = audio_title.strip()

        # Convert upload_date if present
        formatted_date = None
        if upload_date:
            # upload_date format is typically YYYYMMDD
            year = upload_date[:4]
            month = upload_date[4:6]
            day = upload_date[6:8]
            formatted_date = f"{month}-{day}-{year}"

        # Save the description to a text file
        base_path = os.path.splitext(audio_path)[0]
        desc_file_path = f"{base_path}.txt"
        with open(desc_file_path, "w", encoding="utf-8") as desc_file:
            if formatted_date:
                desc_file.write(f"Upload Date: {formatted_date}\n")
            if episode_number:
                desc_file.write(f"Episode_number: {episode_number}\n")
            desc_file.write(f"Title: {final_cleaned_title}\n\n")
            desc_file.write(description)
        print(f"Description saved as: {desc_file_path}")
        
        # Extract audio
        # print("Extracting audio...")
        # audio_path = extract_audio(video_path, video_title, cleaned_title, download_path)
        # print(f"Audio saved as: {audio_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Download audio from YouTube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", default="./", help="Output directory (default: current directory)")
    args = parser.parse_args()
    
    download_and_process_audio(args.url, args.output)