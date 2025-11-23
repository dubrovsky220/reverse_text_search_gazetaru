FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка модели
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
try:
    SentenceTransformer("intfloat/multilingual-e5-base")
    print("Model preloaded.")
except Exception as e:
    print("Model preload failed:", e)
EOF

# Копирование файлов
COPY src/ ./src
COPY index/ ./index
COPY embeddings/ ./embeddings
COPY data/ ./data

# Streamlit порт
EXPOSE 8501

# Запуск Streamlit UI
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
