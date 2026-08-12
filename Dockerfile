FROM python:3.11-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

# Build and install SQLite 3.45.0 to fix ChromaDB compatibility
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc wget \
    && cd /tmp \
    && wget https://www.sqlite.org/2024/sqlite-autoconf-3450000.tar.gz \
    && tar xzf sqlite-autoconf-3450000.tar.gz \
    && cd sqlite-autoconf-3450000 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && ldconfig \
    && cd /app \
    && rm -rf /tmp/sqlite-* \
    && apt-get remove -y build-essential gcc wget \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# nltk.download("stopwords") is commented out in metadata_inference.py, so a
# fresh container has no corpora - pre-download at build time instead.
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt punkt_tab stopwords

COPY . .

RUN mkdir -p /app/DB/auditCSVFiles /app/vector_store

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]