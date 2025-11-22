import streamlit as st
import torch
from search_api import search, load_index, load_metadata, load_model
from loguru import logger


# Загрузочный экран
if "initialized" not in st.session_state:
    loading_box = st.empty()
    loading_box.title("Загрузка...")

    progress = st.progress(0)
    status = st.empty()

    status.write("1/3: Загрузка модели SentenceTransformer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state.model = load_model(device=device)
    progress.progress(33)

    status.write("2/3: Загрузка индекса Faiss...")
    st.session_state.index = load_index()
    progress.progress(66)

    status.write("3/3: Загрузка метаданных...")
    st.session_state.metadata = load_metadata()
    progress.progress(100)

    status.write("Готово!")

    loading_box.empty()
    status.empty()
    progress.empty()

    st.session_state.initialized = True


# Настройки страницы
st.set_page_config(
    page_title="reverse_text_search_gazetaru",
    page_icon="🔍",
    layout="wide"
)

st.title("reverse_text_search_gazetaru")

# Основная форма поиска
query = st.text_area(
    "Введите запрос:",
    value=st.session_state.get("query", ""),
    max_chars=200,
    height=120
)

top_k = st.number_input(
    "Количество результатов для вывода:",
    min_value=1, max_value=50, value=10
)

rerank_flag = st.checkbox(
    "Переранжировать результаты через LLM",
    value=False
)

search_btn = st.button("Найти")


# Основная логика при нажатии
if search_btn:
    if not query.strip():
        st.warning("Введите текст запроса!")
        st.stop()

    with st.spinner("Поиск..."):
        try:
            results = search(
                query,
                top_k=top_k,
                index=st.session_state.index,
                metadata=st.session_state.metadata,
                model=st.session_state.model
            )

            # Если включён реранк - заглушка
            if rerank_flag:
                st.info("Перенжирование включено")

        except Exception as e:
            st.error(f"Ошибка поиска: {e}")
            logger.exception(e)
            st.stop()


    # Отображение результатов
    st.subheader("Результаты поиска")

    if not results:
        st.info("Ничего не найдено!")
        st.stop()

    for i, item in enumerate(results, start=1):
        score = item["score"]
        summary = item.get("summary", "—")
        url = item.get("url", "")
        doc_id = item.get("id", "")

        st.markdown(f"### {i}. {summary}")
        st.markdown(f"**ID:** {doc_id} | **Score:** {score:.3f}")

        if url:
            st.markdown(f"[Открыть источник]({url})")

        st.write("---")
