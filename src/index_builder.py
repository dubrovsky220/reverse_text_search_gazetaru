import os
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from loguru import logger


def create_faiss_index(embeddings_path: str, index_dir: str):
    """
    Создает индекс faiss для эмбеддингов текстов.
    :param embeddings_path: Путь к файлу с эмбеддингами в формате .npy
    :param index_dir: Директория для сохранения файла индекса.
    """
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")

    logger.info("Загрузка эмбеддингов...")
    embeddings = np.load(embeddings_path).astype("float32")

    logger.info("Нормализация эмбеддингов...")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    logger.info("Создание индекса IndexFlatIP (dim={dim})...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("Сохранение индекса на диск...")
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_path)

    logger.info("Создание индекса завершено!")



def test_search(
        query: str,
        index_path: str,
        metadata_path: str,
        top_k=10,
        model_name="intfloat/multilingual-e5-base"
):
    """
    Функция для тестирования работы поиска.
    :param query: Текст запроса.
    :param index_path: Путь к сохраненному индексу Faiss.
    :param metadata_path: Путь к файлу с метаданными в формате json.
    :param top_k: Количество лучших совпадений для вывода.
    :param model_name: Название используемой модели.
    """
    logger.info(f"Тестовый запрос: {query}")

    logger.info("Загрузка индекса...")
    index = faiss.read_index(index_path)

    logger.info("Загрузка метаданных...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    logger.info(f"Загрузка модели: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info("Создание эмбеддинга...")
    embedding = model.encode([f"query: {query}"], normalize_embeddings=True)
    embedding = embedding.astype("float32")

    scores, idx = index.search(embedding, top_k)

    print("\nРезультаты поиска:")
    for rank, (score, i) in enumerate(zip(scores[0], idx[0]), start=1):

        item = metadata[i]

        real_id = item.get("id", None)
        url = item.get("url", "<нет title>")
        summary = item.get("summary", "<нет summary>")

        print(f"{rank}. id={real_id}, score={score:.4f}")
        print(f"   url: {url}")
        print(f"   summary: {summary}\n")


if __name__ == "__main__":
    EMBEDDINGS_PATH = "../embeddings/embeddings.npy"
    METADATA_PATH = "../embeddings/metadata.json"
    INDEX_DIR = "../index"
    INDEX_PATH = "../index/faiss_index.bin"
    MODEL_NAME = "intfloat/multilingual-e5-base"

    create_faiss_index(EMBEDDINGS_PATH, INDEX_DIR)

    test_search(
        "Президент России Владимир Путин провёл встречу с министром МВД Колокольцевым",
        top_k=10,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME
    )
