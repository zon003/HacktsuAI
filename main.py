# main.py (FastAPIアプリケーションのメインファイル)
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# JWT関連のインポート
from Firebase.JWT import JWT, PyJWTError # Firebase\JWT ではなく Firebase.JWT をインポート
from Firebase.JWT import algorithms # アルゴリズムを指定するために必要

# rag_pipeline.py から必要な関数をインポート
# プロジェクトの構造に合わせてパスを調整してください。
# 例: FastAPIのmain.pyがHacktsuAI/app/にある場合、rag_pipeline.pyはHacktsuAI/rag/にあるので
# from ..rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query
# または、fastapi-app の Dockerfile 内でPYTHONPATHを設定している場合、直接インポートできることもあります
from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv() # .envファイルをロード

app = FastAPI()

# --- CORSミドルウェアの設定 ---
# まず、環境変数から文字列を取得
allowed_origins_str = os.getenv("WORDPRESS_FRONTEND_URLS")

# 文字列をカンマで分割し、空白を除去してリストに変換
# 環境変数が設定されていない場合は空のリストをデフォルトとする
origins = []
if allowed_origins_str:
    origins = [origin.strip() for origin in allowed_origins_str.split(',')]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # POST, GET, OPTIONS を許可
    allow_headers=["*"],  # Content-Type, Authorization を許可
)

# --- JWT秘密鍵の取得 ---
# wp-config.php と同じ値をCloud Runの環境変数に設定してください (MY_AI_JWT_SECRET_KEYと同じ値を推奨)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") # Cloud Runの環境変数から取得

if not JWT_SECRET_KEY:
    raise ValueError("JWT_SECRET_KEY 環境変数が設定されていません。")

# --- RAGチェーンの初期化 ---
# アプリケーション起動時に一度だけ実行
# 環境変数からGCSバケット名を取得
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

if not GCS_BUCKET_NAME:
    raise ValueError("GCS_BUCKET_NAME 環境変数が設定されていません。")

# グローバル変数としてRAGチェーンを保持
rag_chain_instance = None
vectorstore_instance = None

@app.on_event("startup")
async def startup_event():
    """FastAPIアプリケーション起動時に実行される処理"""
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
        # リクエストボディからJSONデータを取得
        data = await request.json()
        user_message = data.get("message")
        jwt_token = data.get("jwtToken")
        wp_user_id = data.get("userId") # WordPressから送られてくるユーザーID

        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required.")
        if not jwt_token:
            raise HTTPException(status_code=401, detail="JWT token is required.")

        # JWTの検証
        try:
            # PyJWTError は PyJWT モジュールからのエラー
            # algorithms は Firebase.JWT.algorithms からインポート
            payload = JWT.decode(jwt_token, JWT_SECRET_KEY, algorithms=["HS256"])

            # JWTのペイロードに含まれるuser_idとWordPressから送られてきたuser_idが一致するか確認
            if payload.get("user_id") != wp_user_id:
                 raise HTTPException(status_code=403, detail="JWT user ID mismatch.")

            # JWTの有効期限もdecode()が自動でチェックします

        except PyJWTError as e:
            # JWTが無効な場合の具体的なエラーログ
            print(f"JWT Verification Failed: {e}")
            raise HTTPException(status_code=401, detail=f"Invalid JWT token: {e}")

        # chat_historyの処理（JavaScriptから受け取った履歴をLangChainの形式に変換）
        # WordPressのJSから送られてくるchat_historyの形式による
        # 例: [{"role": "user", "content": "..."}]
        raw_chat_history = data.get("chatHistory", [])
        langchain_chat_history = []
        for msg in raw_chat_history:
            if msg["role"] == "user":
                langchain_chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_chat_history.append(AIMessage(content=msg["content"]))

        # RAGチェーンが初期化されていることを確認
        if rag_chain_instance is None:
            raise HTTPException(status_code=500, detail="AI service not initialized.")

        print(f"Received message for user {wp_user_id}: {user_message}")

        # rag_pipeline.py の run_query 関数を呼び出してAI応答を取得
        ai_response = run_query(rag_chain_instance, user_message, langchain_chat_history)

        return {"response": ai_response}

    except HTTPException as e:
        raise e # FastAPIのHTTPExceptionはそのまま再スロー

    except Exception as e:
        # その他の予期せぬエラー
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {e}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)