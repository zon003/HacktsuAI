# auth.py
import os
import logging
from fastapi import HTTPException
from jose import jwt, JWTError
from dotenv import load_dotenv

# ローカル起動のときだけ .env をロード
load_dotenv()

logger = logging.getLogger(__name__)

SECRET_KEY        = os.getenv("MY_AI_JWT_SECRET_KEY")
ALGORITHM         = "HS256"

# カンマ区切りで複数の ISSUER を許可できるように
# 例: EXPECTED_ISSUER="https://hacktsu.doyou.love,http://localhost:3000"
_expected_iss = os.getenv("EXPECTED_ISSUER", "")
ALLOWED_ISSUERS   = [s.strip() for s in _expected_iss.split(",") if s.strip()]

EXPECTED_AUDIENCE = os.getenv("EXPECTED_AUDIENCE", "my-ai-chat-app")

def decode_jwt_token(token: str):
    try:
        # jwt.decode に渡すオプションを組み立てる
        decode_kwargs = {
            "key": SECRET_KEY,
            "algorithms": [ALGORITHM],
            "audience": EXPECTED_AUDIENCE,
        }
        # 発行者チェックを有効にしたい場合だけ issuer を追加
        if ALLOWED_ISSUERS:
            # jose は単一の issuer 文字列しか取れないので、
            # デコード後に自前でチェックします
            decode_kwargs.pop("issuer", None)
            payload = jwt.decode(token, **decode_kwargs)
            iss = payload.get("iss")
            if iss not in ALLOWED_ISSUERS:
                raise JWTError(f"Issuer {iss!r} not allowed")
        else:
            # ISSUER 指定がなければ issuer チェックをスキップ
            payload = jwt.decode(token, **decode_kwargs)

        return payload

    except JWTError as e:
        logger.error(f"JWT Decode Failed: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid JWT: {e}")
