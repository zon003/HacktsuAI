# main.py
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query
from storage import load_chat_history, save_chat_history
from models import ChatRequest, ChatResponse
from auth import get_current_user  # 🔐 JWT検証はこちらに委譲

# --- 環境変数の読み込み (.env) ---
load_dotenv()

# --- アプリケーション初期化 ---
app = FastAPI()

# --- CORS設定 ---
allowed_origins_str = os.getenv("WORDPRESS_FRONTEND_URLS")
origins = [origin.strip() for origin in allowed_origins_str.split(',')] if allowed_origins_str else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 必須の環境変数確認 ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME が未設定です。")

# --- グローバル RAG インスタンス ---
vectorstore_instance = None
rag_chain_instance = None

@app.on_event("startup")
async def startup_event():
    global vectorstore_instance, rag_chain_instance
    print("✅ RAGチェーンの初期化を開始...")
    vectorstore_instance = load_vectorstore(GCS_BUCKET_NAME)
    rag_chain_instance = build_rag_chain(vectorstore_instance)
    print("✅ RAGチェーンの初期化が完了しました。")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI service is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, wp_user_id: str = Depends(get_current_user)):
    if payload.userId != wp_user_id:
        raise HTTPException(status_code=403, detail="JWT user_id mismatch.")

    user_message = payload.message
    posted_history = payload.chatHistory or []

    if not user_message:
        raise HTTPException(status_code=400, detail="message is required.")

    # 履歴統合
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

    # RAG 実行
    if not rag_chain_instance:
        raise HTTPException(status_code=500, detail="AI engine not initialized.")

    print(f"🤖 user_id={wp_user_id} | message={user_message}")
    ai_response = run_query(rag_chain_instance, user_message, langchain_chat_history)

    # 履歴保存
    updated_history = combined_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ai_response},
    ]
    save_chat_history(wp_user_id, updated_history)

    return {"response": ai_response}

@app.get("/history")
async def get_chat_history(user_id: str, wp_user_id: str = Depends(get_current_user)):
    if str(user_id) != str(wp_user_id):
        raise HTTPException(status_code=403, detail="JWT user_id mismatch.")

    try:
        history = load_chat_history(user_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load history: {e}")

# --- ローカル開発用 ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)