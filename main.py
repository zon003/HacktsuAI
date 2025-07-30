# main.py (FastAPIアプリケーションのメインファイル)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from jwt import decode, PyJWTError
import uvicorn

# rag_pipeline.py から必要な関数をインポート
from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()  # .envファイルをロード

app = FastAPI()

# --- CORSミドルウェアの設定 ---
allowed_origins_str = os.getenv("WORDPRESS_FRONTEND_URLS")
origins = []
if allowed_origins_str:
    origins = [origin.strip() for origin in allowed_origins_str.split(',')]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- JWT秘密鍵の取得 ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY 環境変数が設定されていません。")

# --- RAGチェーンの初期化 ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME 環境変数が設定されていません。")

rag_chain_instance = None
vectorstore_instance = None

@app.on_event("startup")
async def startup_event():
    global rag_chain_instance, vectorstore_instance
    try:
        print("FastAPI: RAGチェーンの初期化を開始します...")
        vectorstore_instance = load_vectorstore(GCS_BUCKET_NAME)
        rag_chain_instance = build_rag_chain(vectorstore_instance)
        print("FastAPI: RAGチェーンの初期化が完了しました。")
    except Exception as e:
        print(f"FastAPI: RAGチェーンの初期化中にエラーが発生しました: {e}")
        raise

# --- ヘルスチェックエンドポイント ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI service is running."}

# --- AIチャットエンドポイント ---
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message")
        jwt_token = data.get("jwtToken")
        wp_user_id = data.get("userId")

        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required.")
        if not jwt_token:
            raise HTTPException(status_code=401, detail="JWT token is required.")

        try:
            payload = decode(jwt_token, JWT_SECRET_KEY, algorithms=["HS256"])
            if payload.get("user_id") != wp_user_id:
                raise HTTPException(status_code=403, detail="JWT user ID mismatch.")
        except PyJWTError as e:
            print(f"JWT Verification Failed: {e}")
            raise HTTPException(status_code=401, detail=f"Invalid JWT token: {e}")

        raw_chat_history = data.get("chatHistory", [])
        langchain_chat_history = []
        for msg in raw_chat_history:
            if msg["role"] == "user":
                langchain_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_chat_history.append(AIMessage(content=msg["content"]))

        if rag_chain_instance is None:
            raise HTTPException(status_code=500, detail="AI service not initialized.")

        print(f"Received message for user {wp_user_id}: {user_message}")
        ai_response = run_query(rag_chain_instance, user_message, langchain_chat_history)
        return {"response": ai_response}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {e}")

# --- ローカル起動用 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Runが指定するPORTを使う
    uvicorn.run("main:app", host="0.0.0.0", port=port)