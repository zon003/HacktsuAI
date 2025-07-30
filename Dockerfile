# Dockerfile
# Pythonの公式スリムイメージを使用（軽量で推奨）
FROM python:3.11-slim-buster

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# requirements.txt をコピーして依存関係をインストール
# Dockerのキャッシュを利用するため、まずrequirements.txtのみコピーしてpip install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコード全体をコピー
# main.py と rag ディレクトリ、その他の必要なファイルを /app にコピー
# あなたのFastAPIアプリケーションのルートディレクトリにあるすべての必要なファイルをコピーします
COPY . .

# FastAPIアプリケーションがリッスンするポートを設定 (Cloud Runの標準ポートは8080)
ENV PORT 8080

EXPOSE 8080

# アプリケーションの実行コマンド
# uvicorn はASGIサーバーで、FastAPIアプリケーションを起動するために必要です。
# main:app は、main.py ファイル内の `app` という名前のFastAPIインスタンスを指します。
CMD ["python", "main.py"]