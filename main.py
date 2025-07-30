# main.py (FastAPIアプリケーションのメインファイル)
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jwt import decode, PyJWTError
from langchain_core.messages import HumanMessage, AIMessage

from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query
from storage import load_chat_history, save_chat_history
from models import ChatRequest, ChatResponse, ChatMessage

# --- 環境変数の読み込み (.env) ---
load_dotenv()

# --- アプリケーション初期化 ---
app = FastAPI()

# --- CORS設定（WordPressからのリクエストを許可） ---
allowed_origins_str = os.getenv("WORDPRESS_FRONTEND_URLS")
origins = [origin.strip() for origin in allowed_origins_str.split(',')] if allowed_origins_str else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 秘密鍵とバケット名の検証 ---
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY が未設定です。")
if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME が未設定です。")

# --- グローバルRAGインスタンス定義 ---
vectorstore_instance = None
rag_chain_instance = None

@app.on_event("startup")
async def startup_event():
    global vectorstore_instance, rag_chain_instance
    print("✅ RAGチェーンの初期化を開始...")
    vectorstore_instance = load_vectorstore(GCS_BUCKET_NAME)
    rag_chain_instance = build_rag_chain(vectorstore_instance)
    print("✅ RAGチェーンの初期化が完了しました。")

# --- ヘルスチェック ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI service is running."}

# --- チャットエンドポイント ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    user_message = payload.message
    jwt_token = payload.jwtToken
    wp_user_id = payload.userId
    posted_history = payload.chatHistory or []

    # --- 入力検証 ---
    if not user_message:
        raise HTTPException(status_code=400, detail="message is required.")
    if not jwt_token:
        raise HTTPException(status_code=401, detail="jwtToken is required.")
    if not wp_user_id:
        raise HTTPException(status_code=400, detail="userId is required.")

    # --- JWT検証 ---
    try:
        decoded = decode(jwt_token, JWT_SECRET_KEY, algorithms=["HS256"])
        if decoded.get("user_id") != wp_user_id:
            raise HTTPException(status_code=403, detail="JWT user_id mismatch.")
    except PyJWTError as e:
        print(f"❌ JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid JWT token.")

    # --- 履歴の統合（過去＋今回） ---
    persisted_history = load_chat_history(wp_user_id)
    combined_history = persisted_history + [
        {"role": msg.role, "content": msg.content} for msg in posted_history
    ]

    langchain_chat_history = []
    for msg in combined_history:
        if msg["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_chat_history.append(AIMessage(content=msg["content"]))

    # --- RAG実行 ---
    if not rag_chain_instance:
        raise HTTPException(status_code=500, detail="AI engine not initialized.")

    print(f"🤖 user_id={wp_user_id} | message={user_message}")
    ai_response = run_query(rag_chain_instance, user_message, langchain_chat_history)

    # --- 履歴の保存 ---
    updated_history = combined_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_response},
    ]
    save_chat_history(wp_user_id, updated_history)

    return {"response": ai_response}

# --- ローカル起動用（開発環境） ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)