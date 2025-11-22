import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from loguru import logger


@st.cache_resource
def load_model(model_name="intfloat/multilingual-e5-base", device="cpu"):
    """
    Загружает модель SentenceTransformer.
    :param model_name: Название модели.
    :param device: Устройство для загрузки модели.
    :return: Загруженная модель
    """
    logger.info(f"Загрузка модели SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    return model


@st.cache_resource
def load_index(index_path="index/faiss_index.bin"):
    """
    Загружает индекс Faiss.
    :param index_path: Путь к файлу .bin с индексом.
    :return: Загруженный индекс.
    """
    logger.info(f"Загрузка FAISS индекса: {index_path}")
    index = faiss.read_index(index_path)
    return index


@st.cache_resource
def load_metadata(metadata_path="embeddings/metadata.json"):
    """
    Загружает метаданные
    :param metadata_path: Путь к файлу .json с метаданными.
    :return: Загруженные метаданные.
    """
    logger.info(f"Загрузка метаданных: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata


def encode_query(query: str, model=None) -> np.ndarray:
    """
    Возвращает нормализованный эмбеддинг запроса (float32).
    :param query: Запрос.
    :param model: Модель SentenceTransformer.
    :return: Эмбеддинг запроса.
    """

    if model is None:
        model = load_model()

    logger.info(f"Создание эмбеддинга для запроса: {query}")

    embedding = model.encode([f"query: {query}"], normalize_embeddings=True)
    embedding = embedding.astype("float32")
    return embedding


def search(
        query: str,
        top_k: int = 10,
        index=None,
        metadata=None,
        model=None,
) -> list[dict]:
    """
    Выполняет поиск по FAISS индексу.

    :param query: Запрос.
    :param top_k: Количество лучших результатов для вывода.
    :param index: Индекс Faiss.
    :param metadata: Метаданные.
    :param model: Модель SentenceTransformer.
    :return: Возвращает список словарей:
    {
       "score": float,
       "id": ...,
       "summary": ...,
       "url": ...,
    }
    """

    index = load_index() if not index else index
    metadata = load_metadata() if not metadata else metadata
    model = load_model() if not model else model

    logger.info(f"Поиск по запросу: {query}")

    embedding = encode_query(query, model)

    # 2. Поиск в FAISS
    scores, indexes = index.search(embedding, top_k)
    scores = scores[0]
    indexes = indexes[0]

    results = []

    for score, idx in zip(scores, indexes):
        item = metadata[idx]

        result = {
            "score": float(score),
            "id": item.get("id"),
            "summary": item.get("summary"),
            "url": item.get("url"),
        }

        results.append(result)

    return results

if __name__ == "__main__":
    print(search("МЧС прилагает все усилия для тушения лесных пожаров в Западной Сибири"))
    print(search("Президент России Владимир Путин провёл встречу с министром МВД Колокольцевым"))
    print(search("Британские учёные обнаружили новый вид рыб в Тихом океане"))
    print(search("Астронавты НАСА провели выход в открытый космос на МКС"))
