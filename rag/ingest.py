import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from google.cloud import storage # ★ GCSライブラリをインポート
import json # ★ サービスアカウントキーをJSONとしてロードするために必要

load_dotenv()

# --- 設定 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))

DATA_DIR = os.path.join(project_root, "data", "yanbaru") # 生データはローカルから読み込む（後でGCSから読む方法も説明）
LOCAL_FAISS_DIR = os.path.join(project_root, "faiss_index_temp") # ★ 一時的にローカルに保存するディレクトリ

# ★ GCS関連の設定
GCS_BUCKET_NAME = "hacktsuai-rag-data-bucket-unique-id" # ★ あなたが作成したGCSバケット名に置き換える
# ★ サービスアカウントキーファイルのパス
# Codespacesやローカルで実行する場合のパス。本番デプロイでは環境変数で渡すのが一般的。
# 通常、このキーファイルはGitに含めない。
GCP_SERVICE_ACCOUNT_KEY_PATH = os.path.join(project_root, "gcp_keys", "hacktsuai-rag-project-e65ee603943f.json") # ★ 適切なパスに置き換える

# 環境変数からサービスアカウントキーを読み込む設定 (推奨)
# Streamlit Community Cloud の Secrets に設定する場合など
# JSON文字列全体を環境変数に設定することが可能
GCP_SERVICE_ACCOUNT_KEY_JSON = os.getenv("GCP_SERVICE_ACCOUNT_KEY_JSON")


# --- ヘルパー関数: GCSへアップロード ---
def upload_to_gcs(bucket_name, source_directory, destination_blob_prefix):
    """ディレクトリの内容をGCSバケットにアップロードする"""
    print(f"GCSにファイルをアップロード中: gs://{bucket_name}/{destination_blob_prefix}/")
    storage_client = storage.Client.from_service_account_json(GCP_SERVICE_ACCOUNT_KEY_PATH) # または .from_service_account_info(json.loads(GCP_SERVICE_ACCOUNT_KEY_JSON))
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(source_directory):
        for file in files:
            local_file_path = os.path.join(root, file)
            # GCS上のパス (ディレクトリ構造を維持)
            relative_path = os.path.relpath(local_file_path, start=source_directory)
            gcs_blob_name = os.path.join(destination_blob_prefix, relative_path).replace("\\", "/") # Windowsパス対策

            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to {gcs_blob_name}")
    print("GCSへのアップロード完了。")

# --- 1. データ読み込み (変更なし) ---
def load_all_documents(data_dir):
    # ... (既存のコード) ...
    all_documents = []
    text_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    all_documents.extend(text_loader.load())
    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    all_documents.extend(pdf_loader.load())
    print(f"合計 {len(all_documents)} 個のドキュメントを読み込みました。")
    return all_documents

# --- 2. テキスト分割（チャンク化）(変更なし) ---
def split_documents_into_chunks(documents):
    # ... (既存のコード) ...
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"ドキュメントを {len(chunks)} 個のチャンクに分割しました。")
    return chunks

# --- 3. 埋め込み生成とベクトルストアへの保存（変更あり）---
def create_and_save_vectorstore(chunks, local_db_path, gcs_bucket_name):
    print("埋め込みを生成し、ベクトルストアを構築します...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # まずローカルの一時ディレクトリに保存
    os.makedirs(local_db_path, exist_ok=True)
    vectorstore.save_local(local_db_path)
    print(f"ベクトルストアを一時的にローカルの {local_db_path} に保存しました。")

    # 次にGCSにアップロード
    upload_to_gcs(gcs_bucket_name, local_db_path, "faiss_index") # GCS上のプレフィックス（フォルダ名）
    
    # 一時的なローカルディレクトリをクリーンアップ（任意）
    import shutil
    shutil.rmtree(local_db_path)
    print(f"一時ディレクトリ {local_db_path} を削除しました。")

    return vectorstore

if __name__ == "__main__":
    print("データ取り込みプロセスを開始します...")
    # 環境変数 GCP_SERVICE_ACCOUNT_KEY_JSON が設定されている場合、その情報で認証
    if GCP_SERVICE_ACCOUNT_KEY_JSON:
        print("GCP_SERVICE_ACCOUNT_KEY_JSON 環境変数を使用して認証します。")
        # GCSクライアントを初期化する際、from_service_account_infoを使用
        # upload_to_gcs関数内でClientの初期化を修正する必要がある
    elif not os.path.exists(GCP_SERVICE_ACCOUNT_KEY_PATH):
        print(f"エラー: サービスアカウントキーファイルが見つかりません: {GCP_SERVICE_ACCOUNT_KEY_PATH}")
        print("GCP_SERVICE_ACCOUNT_KEY_PATH を正しく設定するか、環境変数 GCP_SERVICE_ACCOUNT_KEY_JSON を設定してください。")
        exit(1)

    documents = load_all_documents(DATA_DIR)
    chunks = split_documents_into_chunks(documents)
    create_and_save_vectorstore(chunks, LOCAL_FAISS_DIR, GCS_BUCKET_NAME)
    print("データ取り込みプロセスが完了しました！")