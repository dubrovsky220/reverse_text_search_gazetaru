import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger


def main(
    input_path,
    output_dir,
    model_name="intfloat/multilingual-e5-base-instruct",
    batch_size=64,
    device="cpu",
):
    """
    Создает эмбеддинги текстов и сохраняет их на диск.
    :param input_path: Путь к предобработнному датасету в формате parquet.
    :param output_dir: Путь к директории для сохранения эмбеддингов и метаданных.
    :param model_name: Модель, используемая для создания эмбеддингов.
    :param batch_size: Размер батча, подаваемого на вход модели.
    :param device: Устройство для загрузки модели.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Загрузка датасета из parquet...")
    df = pd.read_parquet(input_path)
    if not {"summary", "url"}.issubset(df.columns):
        raise ValueError("Датасет должен содержать следующие колонки: summary, url")

    logger.info(f".Количество строк в загруженном датасете: {len(df)}")
    # passage добавляется для улучшения качества работы модели multilingual-e5-base
    texts = ["passage: " + t for t in df["summary"].astype(str).tolist()]

    logger.info(f"Загрузка модели: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    logger.info("Создание эмбеддингов текстов...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    logger.info(f"Кодирование завершено! Сохранение на диск...")
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Эмбеддинги сохранены!")

    logger.info(f"Сохранение метаданных...")
    metadata = [
        {
            "id": i,
            "summary": row["summary"],
            "url": row["url"]
        }
        for i, row in df.iterrows()
    ]

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Метаданные сохранены!")
    logger.info("Создание эмбеддингов завершено!")


if __name__ == "__main__":
    INPUT_PATH = "../data/data_processed.parquet"
    OUTPUT_DIR = "../embeddings"
    MODEL_NAME = "intfloat/multilingual-e5-base"
    BATCH_SIZE = 64
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    main(INPUT_PATH, OUTPUT_DIR, MODEL_NAME, BATCH_SIZE, DEVICE)
