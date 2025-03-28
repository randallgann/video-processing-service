import requests

file_url = 'https://storage.googleapis.com/mm-audio-files/Under_The_Umbrella_ep1._-_The_.mp3'
response = requests.head(file_url)
print("Content-Type:", response.headers.get('Content-Type'))

response = requests.get(file_url, stream=True)
print("Content-Type:", response.headers.get('Content-Type'))
print("First few bytes:", response.raw.read(16).hex())  # MP3 files typically start with 'ID3' or 'FF FB'