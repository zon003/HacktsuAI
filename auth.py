# auth.py
import os, logging
from fastapi import HTTPException
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SECRET_KEY        = os.getenv("MY_AI_JWT_SECRET_KEY")
ALGORITHM         = "HS256"

# カンマ区切りの ENV 値をリストに
raw_issuers       = os.getenv("EXPECTED_ISSUER", "").split(",")
EXPECTED_ISSUER   = [u.strip() for u in raw_issuers if u.strip()]

EXPECTED_AUDIENCE = os.getenv("EXPECTED_AUDIENCE", "my-ai-chat-app")

def decode_jwt_token(token: str):
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=EXPECTED_AUDIENCE,
            issuer=EXPECTED_ISSUER,    # list を渡せます
        )
        return payload
    except JWTError as e:
        logger.error(f"JWT Decode Failed: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid JWT: {e}")
