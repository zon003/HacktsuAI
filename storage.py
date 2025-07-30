from google.cloud import storage
import json

def save_chat_history(bucket_name: str, user_id: str, history: list):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"chat_history/user_{user_id}.json")
    blob.upload_from_string(json.dumps(history, ensure_ascii=False), content_type='application/json')

def load_chat_history(bucket_name: str, user_id: str) -> list:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"chat_history/user_{user_id}.json")
    if blob.exists():
        return json.loads(blob.download_as_text())
    return []