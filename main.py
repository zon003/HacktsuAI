# main.py
import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from auth import decode_jwt_token
from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query
from storage import load_chat_history, save_chat_history, clear_chat_history
from models import ChatRequest, ChatResponse

# 環境変数ロード
load_dotenv()

# FastAPI アプリケーション
app = FastAPI()

# CORS 設定（WordPress 側の URL をカンマ区切りで）
origins = [
    origin.strip()
    for origin in os.getenv("WORDPRESS_FRONTEND_URLS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],  # DELETE もOK
    allow_headers=["*"],
)

# 必須 envvar チェック
if not os.getenv("MY_AI_JWT_SECRET_KEY"):
    raise ValueError("MY_AI_JWT_SECRET_KEY が未設定です。")
if not os.getenv("GCS_BUCKET_NAME"):
    raise ValueError("GCS_BUCKET_NAME が未設定です。")

# RAG 初期化用グローバル
vectorstore_instance = None
rag_chain_instance = None


@app.on_event("startup")
async def startup_event():
    """Cloud Run 起動時に RAG を初期化"""
    global vectorstore_instance, rag_chain_instance
    bucket = os.getenv("GCS_BUCKET_NAME")
    print("✅ RAGチェーンの初期化を開始...")
    vectorstore_instance = load_vectorstore(bucket)
    rag_chain_instance = build_rag_chain(vectorstore_instance)
    print("✅ RAGチェーンの初期化が完了しました。")


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "FastAPI service is running."}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    # Authorization ヘッダー取得
    auth: str = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Authorization header missing or malformed")
    token = auth.split(" ", 1)[1]

    # JWT デコード
    decoded = decode_jwt_token(token)
    if str(decoded.get("user_id")) != str(payload.userId):
        raise HTTPException(403, "JWT user_id mismatch.")

    # 既存履歴 + 今回POSTされた履歴（あれば） を統合
    old = load_chat_history(payload.userId)
    posted = [
        {"role": m.role, "content": m.content}
        for m in (payload.chatHistory or [])
    ]
    combined = old + posted

    # LangChain メッセージに変換
    history_msgs = []
    for m in combined:
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))

    # RAG 実行
    if rag_chain_instance is None:
        raise HTTPException(500, "AI engine not initialized.")
    answer = run_query(rag_chain_instance, payload.message, history_msgs)

    # 新規分を保存
    new_hist = combined + [
        {"role": "user", "content": payload.message},
        {"role": "assistant", "content": answer},
    ]
    save_chat_history(payload.userId, new_hist)

    return ChatResponse(response=answer)


# 既存互換：POST でクリア（フロントがこれを呼んでいる場合向け）
@app.post("/history/clear")
async def clear_history_endpoint(user_id: str, request: Request):
    # Authorization ヘッダー取得
    auth: str = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Authorization header missing or malformed")
    token = auth.split(" ", 1)[1]

    # JWT デコード & ユーザ一致チェック
    decoded = decode_jwt_token(token)
    if str(decoded.get("user_id")) != str(user_id):
        raise HTTPException(403, "JWT user_id mismatch.")

    clear_chat_history(user_id)
    return {"ok": True}


# 新規：DELETE /history?user_id=xxx でもクリアできる（推奨）
@app.delete("/history")
async def delete_history_endpoint(user_id: str, request: Request):
    # Authorization ヘッダー取得
    auth: str = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Authorization header missing or malformed")
    token = auth.split(" ", 1)[1]

    # JWT デコード & ユーザ一致チェック
    decoded = decode_jwt_token(token)
    if str(decoded.get("user_id")) != str(user_id):
        raise HTTPException(403, "JWT user_id mismatch.")

    clear_chat_history(user_id)
    return {"ok": True}


@app.get("/history")
async def get_chat_history(user_id: str, request: Request):
    # Authorization ヘッダー取得
    auth: str = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Authorization header missing or malformed")
    token = auth.split(" ", 1)[1]

    # JWT デコード
    decoded = decode_jwt_token(token)
    if str(decoded.get("user_id")) != str(user_id):
        raise HTTPException(403, "JWT user_id mismatch.")

    # 履歴ロード
    return {"history": load_chat_history(user_id)}


# ローカル開発用エントリ
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
