# auth.py
import os
import logging
from fastapi import HTTPException
from jose import jwt, JWTError
from dotenv import load_dotenv

# .env のロード（ローカル起動用）
load_dotenv()

logger = logging.getLogger(__name__)

SECRET_KEY        = os.getenv("MY_AI_JWT_SECRET_KEY")
ALGORITHM         = "HS256"
# JWT に入れている値をデフォルトに設定
EXPECTED_ISSUER   = os.getenv("EXPECTED_ISSUER", "https://hacktsu.doyou.love")
EXPECTED_AUDIENCE = os.getenv("EXPECTED_AUDIENCE", "my-ai-chat-app")

def decode_jwt_token(token: str):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=EXPECTED_AUDIENCE,
            issuer=EXPECTED_ISSUER,
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT Decode Failed: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid JWT: {e}")
