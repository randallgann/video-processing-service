import yt_dlp
import os
import re

def download_audio(url, download_path="./"):
    """
    Downloads YouTube audio using yt-dlp
    """

    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality
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
            base_filename = ydl.prepare_filename(info)
            base_filename_no_ext, _ = os.path.splitext(base_filename)
            audio_path = base_filename_no_ext + ".mp3"
            description = info.get('description', '')
            upload_date = info.get('upload_date')
            print(f"Download completed: {audio_path}")
            return audio_path, title, info['title'], description, upload_date
    except Exception as e:
        print(f"Download error: {str(e)}")
        raise

def extract_episode_number(title):
    """
    Extract episode number from title, handling both formats:
    - Simple number format (e.g., "606:")
    - Season/Episode format (e.g., "S01E01:")
    Returns the episode number as a string (e.g., "606" or "101")
    """
    # Try to match S##E## format first
    season_episode_match = re.search(r'S(\d{2})E(\d{2})', title, re.IGNORECASE)
    if season_episode_match:
        season = int(season_episode_match.group(1))
        episode = int(season_episode_match.group(2))
        # Convert to overall episode number (S01E01 becomes 101)
        return str(season * 100 + episode)
    
    # If no S##E## format, try the original number format
    number_match = re.match(r'^\s*(\d+)', title)
    if number_match:
        return number_match.group(1)
    
    return ""

def download_and_process_audio(url, download_path="./"):
    """
    Downloads a YouTube video and transcribes it with speaker identification
    """
    try:
        # Download best audio using yt-dlp
        audio_path, audio_title, cleaned_title, description, upload_date = download_audio(url, download_path)
        print(f"Audio saved as: {audio_path}")

        # Extract episode number using the new function
        episode_number = extract_episode_number(audio_title)

        # Find the first alphabetic character after the episode number/colon to define where the cleaned title should start
        # Look for either the colon after the episode number or the first letter
        colon_match = re.search(':', audio_title)
        alpha_match = re.search('[A-Za-z]', audio_title)
        
        if colon_match:
            start_idx = colon_match.end()
            final_cleaned_title = audio_title[start_idx:].strip()
        elif alpha_match:
            start_idx = alpha_match.start()
            final_cleaned_title = audio_title[start_idx:].strip()
        else:
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
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=tiOPkHfJKbA"
    download_and_process_audio(video_url)