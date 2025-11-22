import re
import pandas as pd
from datasets import load_dataset
from pathlib import Path
from loguru import logger


def remove_repeating_spaces(s: str) -> str:
    """
    Удаляет повторяющиеся пробелы в строке.
    :param s: Исходная строка.
    :return: Обработанная строка.
    """
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main(data_path: str):
    """
    Загружает датасет, выполняет его очистку и сохраняет в формате parquet.
    :param data_path: Путь для сохранения обработанных данных.
    """
    data_path = Path(data_path)
    data_path.parent.mkdir(exist_ok=True)

    logger.info("Загрузка датасета...")
    dataset = load_dataset("IlyaGusev/gazeta", split="train")
    dataset = dataset.select_columns(["summary", "url"])

    logger.info("Фильтрация пустых и коротких записей...")
    dataset = dataset.filter(lambda x: x["summary"] is not None and x["summary"].strip() != "")
    dataset = dataset.filter(lambda x: len(x["summary"].split()) >= 5)

    logger.info("Удаление лишних пробелов...")
    dataset = dataset.map(lambda x: {"summary": remove_repeating_spaces(x["summary"])})

    logger.info("Преобразование в Pandas DataFrame...")
    df = dataset.to_pandas()

    logger.info(f"Размер итогового датасета: {len(df)}")
    logger.info(f"Пример данных: {df.head()}")

    logger.info(f"Сохранение в директорию {data_path}")
    df.to_parquet(data_path, index=False)

    logger.info("Загрузка датасета завершена!")


if __name__ == "__main__":
    DATA_PATH = "../data/data_processed.parquet"
    main(DATA_PATH)
