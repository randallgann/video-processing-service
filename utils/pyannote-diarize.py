import requests

url = "https://api.pyannote.ai/v1/diarize"
API_KEY = "sk_ce7aca2cd799474eb30dbc3126bde508"
file_url = 'https://storage.googleapis.com/mm-audio-files/606_-_Hawk_Tuah_Crypto_Scam_1991_s_Best_Films_Precognition_Time_Loops_History_funny_btc.mp3'
webhook_url = 'https://webhook-service-879200966247.us-south1.run.app/your-webhook-url'
headers = {
   "Authorization": f"Bearer {API_KEY}"
}
data = {
    'webhook': webhook_url,
    'url': file_url
}
response = requests.post(url, headers=headers, json=data)

print(response.status_code)
# 200

print(response.json()) 
# {
#  "jobId": "bd7e97c9-0742-4a19-bd5a-9df519ce8c74",
#  "message": "Job added to queue",
#  "status": "pending"
# }
