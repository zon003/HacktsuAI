import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage

# rag_pipeline.py から必要な関数をインポートします
# load_vectorstore と build_rag_chain をインポートして、このファイルでRAGチェーンを構築します
from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query

# --- 環境変数の設定 ---
# GCS_BUCKET_NAME は .env ファイルまたはStreamlit Secretsから取得される想定
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# OpenAI API Key が設定されているか確認 (Streamlit Secretsで設定を推奨)
if not os.getenv("OPENAI_API_KEY"):
    st.error("エラー: OPENAI_API_KEY 環境変数が設定されていません。Streamlit Secretsまたは.envファイルで設定してください。")
    st.stop()

# GCS_BUCKET_NAME が設定されているか確認
if not GCS_BUCKET_NAME:
    st.error("エラー: GCS_BUCKET_NAME 環境変数が設定されていません。Streamlit Secretsまたは.envファイルで設定してください。")
    st.stop()


st.title("あなたのメンターAI")

# --- Streamlit Session State の初期化 ---
# アプリケーションの状態をセッション間で保持するために使用します
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # 会話履歴を格納するリスト


# --- RAGチェーンの初期化（@st.cache_resource でキャッシュ） ---
# この関数は、ベクトルストアのロードやRAGチェーンの構築といった
# 時間のかかる処理を、アプリのセッション開始時に一度だけ実行し、結果をキャッシュします。
@st.cache_resource
def get_rag_chain(bucket_name):
    st.info("💡 ベクトルストアとRAGチェーンを初期化中です。初回は時間がかかります...")
    try:
        vectorstore = load_vectorstore(bucket_name)
        rag_chain = build_rag_chain(vectorstore)
        st.success("✨ 初期化完了！メンターAIと話し始めましょう。")
        return rag_chain
    except Exception as e:
        st.error(f"エラーが発生しました: RAGチェーンの初期化に失敗しました。{e}")
        st.stop() # エラー時はアプリの実行を停止します

# アプリケーションの起動時にチェーンをロード
if st.session_state.rag_chain is None:
    st.session_state.rag_chain = get_rag_chain(GCS_BUCKET_NAME)


# --- 過去のチャット履歴を表示 ---
# st.session_state.chat_history に保存されているメッセージを順に表示します
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- ユーザーからの入力エリア ---
user_query = st.chat_input("何について相談しますか？")

if user_query:
    # ユーザーのメッセージを履歴に追加し、画面に表示
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # AIの応答を取得し、画面に表示
    with st.chat_message("assistant"):
        with st.spinner("AIが回答を考えています..."):
            # run_query 関数に現在のチャット履歴を渡します
            # rag_chain は st.session_state から取得します
            ai_response = run_query(st.session_state.rag_chain, user_query, st.session_state.chat_history)
            st.markdown(ai_response)
        
        # AIの応答を履歴に追加
        st.session_state.chat_history.append(AIMessage(content=ai_response))

# --- サイドバーに履歴をクリアするボタン（デバッグやリセット用） ---
st.sidebar.title("設定・操作")
if st.sidebar.button("チャット履歴をクリア"):
    st.session_state.chat_history = [] # 履歴を空にする
    # RAGチェーンも再初期化する場合は、以下の行のコメントを外す
    # st