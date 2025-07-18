import streamlit as st
import sys
import os

# プロジェクトルートのパスをシステムパスに追加し、モジュールをインポート可能にする
# これにより、rag.rag_pipeline などとインポートできるようになる
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# rag/rag_pipeline.py から必要な関数をインポート
# load_vectorstore と build_rag_chain はRAGチェーンの初期化用
# run_query は実際の質問応答用
from rag.rag_pipeline import load_vectorstore, build_rag_chain, run_query

# LangChainのMessages型をStreamlitの表示用に変換するためのヘルパー関数
from langchain_core.messages import HumanMessage, AIMessage

# --- 設定 ---
FAISS_DB_PATH = os.path.join(project_root, "faiss_index") # faiss_index の絶対パス

# --- RAGシステムの初期化 ---
# @st.cache_resource を使うと、関数の結果がキャッシュされ、
# アプリ再起動時以外は再実行されないため、高速に起動できる
@st.cache_resource
def initialize_rag():
    st.write("RAGシステムを初期化中...") # 初期化中のメッセージを表示
    vectorstore = load_vectorstore(FAISS_DB_PATH)
    rag_chain = build_rag_chain(vectorstore)
    st.write("RAGシステム初期化完了！")
    return rag_chain

# RAGシステムを初期化
rag_chain = initialize_rag()

# --- Streamlit UI の構築 ---
st.title("HackTsuAI メンターAI")

# 会話履歴をセッション状態に保存
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 最初のメッセージとしてシステムからの挨拶を設定（任意）
    st.session_state.messages.append({"role": "assistant", "content": "何かお困りですか？どんな些細なことでもメンターとしてお答えします（医療行為は行いません）。"})

# 既存の会話履歴を表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け取るチャット入力ボックス
if prompt := st.chat_input("質問を入力してください..."):
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの回答を生成して表示
    with st.chat_message("assistant"):
        with st.spinner("AIが回答を生成中です..."):
            # run_query に渡すチャット履歴をLangChainのMessages形式に変換
            # Streamlitのセッション履歴は{"role": ..., "content": ...}形式
            langchain_chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_chat_history.append(AIMessage(content=msg["content"]))

            # RAGチェーンを実行して回答を取得
            response_content = run_query(rag_chain, prompt, langchain_chat_history)
            st.markdown(response_content)
        # AIの回答を履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response_content})