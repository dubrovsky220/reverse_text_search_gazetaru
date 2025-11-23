import os
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

MODEL = "x-ai/grok-4.1-fast"

def rerank_with_llm(query: str, candidates: list[dict]) -> list[dict]:
    """
    Переранжирование с помощью Grok 4.1 Fast через OpenRouter.
    :param query: Поисковый запрос.
    :param candidates: Кандидаты для переранжирования: [{ "summary": ..., "url": ..., "score": ..., "id": ... }, ...]
    :return: Переранжированный список словарей с кандидатами.
    """

    logger.info("Создание промпта для переранжирования...")

    items = [c["summary"] for c in candidates]

    numbered_items = "\n".join(f"{i+1}. {text}" for i, text in enumerate(items))

    prompt = f"""
        Отсортируй следующие короткие описания по релевантности поисковому запросу:
        
        Запрос: "{query}"
        
        Список текстов:
        {numbered_items}
        
        Требования:
        1. Верни только список номеров (индексов), разделенных пробелами, например: "3 1 2 4".
        2. Не добавляй объяснений, комментариев, текста.
    """

    try:
        logger.info("Отправка запроса к OpenRouter...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a ranking assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        llm_answer = response.choices[0].message.content.strip()
        logger.info(f"Ответ LLM: {llm_answer}")

        new_order = [int(x.strip()) - 1 for x in llm_answer.split()]

        if sorted(new_order) != list(range(len(candidates))):
            raise ValueError("LLM вернула неверные значения!")

        reranked = [candidates[i] for i in new_order]
        return reranked

    except Exception as e:
        logger.error(f"Ошибка при переранжировании через LLM : {e}")
        logger.warning("Использование оригинального ранжирования Faiss.")
        return candidates

if __name__ == "__main__":
    candidates = [
        {"summary": "Новость про экономику России", "url": "..."},
        {"summary": "Материал про спорт", "url": "..."},
        {"summary": "Статья об инвестициях", "url": "..."},
    ]

    print(rerank_with_llm("экономика", candidates))