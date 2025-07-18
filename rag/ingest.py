import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 環境変数をロード (あれば)
from dotenv import load_dotenv
load_dotenv()

# --- 設定 ---
# スクリプトがあるディレクトリの絶対パスを取得
script_dir = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリのパスを計算 (ragの親ディレクトリ)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

# data/yanbaru への絶対パスを構築
DATA_DIR = os.path.join(project_root, "data", "yanbaru")

# FAISS DBのパスも同様にプロジェクトルートからのパスにする
FAISS_DB_PATH = os.path.join(project_root, "faiss_index")

# --- 1. データ読み込み ---
def load_all_documents(data_dir):
    all_documents = []

    # .txt ファイルの読み込み
    text_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    all_documents.extend(text_loader.load())

    # .pdf ファイルの読み込み
    # PDFLoaderは各ページを個別のDocumentとして読み込む
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    all_documents.extend(pdf_loader.load())

    print(f"合計 {len(all_documents)} 個のドキュメントを読み込みました。")
    return all_documents

# --- 2. テキスト分割（チャンク化）---
def split_documents_into_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"ドキュメントを {len(chunks)} 個のチャンクに分割しました。")
    return chunks

# --- 3. 埋め込み生成とベクトルストアへの保存 ---
def create_and_save_vectorstore(chunks, db_path):
    print("埋め込みを生成し、ベクトルストアを構築します...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") # 埋め込みモデル
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(db_path) # ローカルに保存
    print(f"ベクトルストアを {db_path} に保存しました。")
    return vectorstore

if __name__ == "__main__":
    print("データ取り込みプロセスを開始します...")
    documents = load_all_documents(DATA_DIR)
    chunks = split_documents_into_chunks(documents)
    vectorstore = create_and_save_vectorstore(chunks, FAISS_DB_PATH)
    print("データ取り込みプロセスが完了しました！")