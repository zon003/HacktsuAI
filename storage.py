# storage.py
import os
import json
from google.cloud import storage

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
CHAT_HISTORY_PREFIX = "chat_histories"

client = storage.Client()


def _blob_path(user_id: str) -> str:
    return f"{CHAT_HISTORY_PREFIX}/{user_id}.json"


def save_chat_history(user_id: str, history: list):
    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(user_id))
        blob.upload_from_string(json.dumps(
            history), content_type="application/json")
        print(f"[GCS] Chat history saved for user {user_id}")
    except Exception as e:
        print(f"[GCS ERROR] Failed to save history for user {user_id}: {e}")


def load_chat_history(user_id: str) -> list:
    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(user_id))
        if not blob.exists():
            print(f"[GCS] No history found for user {user_id}")
            return []
        content = blob.download_as_text()
        print(f"[GCS] Chat history loaded for user {user_id}")
        return json.loads(content)
    except Exception as e:
        print(f"[GCS ERROR] Failed to load history for user {user_id}: {e}")
        return []


def clear_chat_history(user_id: str):
    """履歴ファイル自体を削除（存在しなければ何もしない）"""
    try:
        bucket = client.get_bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(_blob_path(user_id))
        if blob.exists():
            blob.delete()
            print(f"[GCS] Chat history deleted for user {user_id}")
        else:
            print(f"[GCS] No history to delete for user {user_id}")
    except Exception as e:
        print(f"[GCS ERROR] Failed to clear history for user {user_id}: {e}")
