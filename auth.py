import os
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import logging
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

security = HTTPBearer()
SECRET_KEY = os.getenv("MY_AI_JWT_SECRET_KEY")
ALGORITHM = "HS256"
EXPECTED_ISSUER = os.getenv("EXPECTED_ISSUER", "https://hacktsu.doyou.love")  # ← デフォルトもOK
EXPECTED_AUDIENCE = "my-ai-chat-app"

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
        logger.error(f"JWT Decode Failed: {str(e)}")  # ログ出力を追加
        raise HTTPException(status_code=401, detail=f"Invalid JWT: {str(e)}")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_jwt_token(token)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="user_id not found in token")
    return user_id