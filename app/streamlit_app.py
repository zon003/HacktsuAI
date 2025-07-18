import streamlit as st
import sys
import os

# プロジェクトルートのパスをシステムパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# rag/rag_pipeline.py から必要な関数をインポート
from rag.rag_pipeline import load_vectorstore, build_rag_chain # run_queryは通常不要、直接RAGチェーンを呼び出すため

# LangChainのMessages型をStreamlitの表示用に変換するためのヘルパー関数
from langchain_core.messages import HumanMessage, AIMessage

# --- 設定 ---
# FAISS_DB_PATH はもうローカルファイルパスではないので削除または変更
# GCSバケット名は共通の設定として、rag/rag_pipeline.py からインポートするか、ここで再度定義
# ここでは一旦、分かりやすいように再定義しますが、config.py などにまとめるのが理想的です。
GCS_BUCKET_NAME = "hacktsuai-rag-data-bucket-unique-id" # ★ あなたが作成したGCSバケット名に置き換える


# --- RAGシステムの初期化 ---
@st.cache_resource
def initialize_rag():
    st.write("RAGシステムを初期化中...")
    # load_vectorstore にGCSバケット名を渡す
    vectorstore = load_vectorstore(GCS_BUCKET_NAME) # ★ ここを変更
    rag_chain = build_rag_chain(vectorstore)
    st.write("RAGシステム初期化完了！")
    return rag_chain

rag_chain = initialize_rag()

# --- Streamlit UI の構築 (以下は変更なし) ---
st.title("HackTsuAI メンターAI")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "何かお困りですか？どんな些細なことでも、メンターとしてお答えします（医療行為は行いません）。"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("質問を入力してください..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AIが回答を生成中です..."):
            langchain_chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    langchain_chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_chat_history.append(AIMessage(content=msg["content"]))

            # run_queryではなく、直接rag_chainをinvokeする
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": langchain_chat_history
            })
            response_content = response["answer"]
            st.markdown(response_content)
        st.session_state.messages.append({"role": "assistant", "content": response_content})