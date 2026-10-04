# app.py — с пошаговым интерфейсом
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import joblib


# =============================================================
# НАСТРОЙКИ
# =============================================================
st.set_page_config(
    page_title="Определение сегмента B2B-компании",
    page_icon="🎯",
    layout="centered",
)

CLASS_ORDER = [
    "Новаторы", "Ранние последователи", "Раннее большинство",
    "Позднее большинство", "Консерваторы",
]

# =============================================================
# МОБИЛЬНАЯ АДАПТАЦИЯ
# =============================================================
st.markdown("""
<style>
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    [data-testid="column"] { width: 100% !important; }
    .stButton>button { width: 100% !important; padding: 0.8rem !important; }
}
</style>
""", unsafe_allow_html=True)


# =============================================================
# ЗАГРУЗКА МОДЕЛИ
# =============================================================
MODEL_PATH = "segment_model_final.pkl"

@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


def sanitize(name):
    name = re.sub(r'[\[\]\{\}":,\.\(\)/\\\-\–\—]', '', str(name))
    name = re.sub(r'\s+', '_', name.strip())
    return name or "feature"


def predict(bundle, company: dict) -> dict:
    model = bundle["model"]
    class_order = bundle["class_order"]
    feature_names = bundle["feature_names"]
    san_to_orig = bundle["san_to_orig"]

    row = {}
    for c in feature_names:
        orig = san_to_orig[c]
        row[c] = company.get(orig, 0)

    row_df = pd.DataFrame([row])
    for c in row_df.columns:
        conv = pd.to_numeric(row_df[c], errors="coerce")
        if conv.isna().all():
            row_df[c] = 0.0
        else:
            row_df[c] = conv.fillna(0).astype("float64")

    proba = model.predict_proba(row_df)[0]
    pred = class_order[int(proba.argmax())]
    sorted_p = np.sort(proba)[::-1]

    return {
        "segment": pred,
        "confidence": round(float(sorted_p[0]), 3),
        "is_borderline": bool((sorted_p[0] - sorted_p[1]) < 0.15),
        "probabilities": {
            class_order[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
    }


# =============================================================
# СПИСКИ ЗНАЧЕНИЙ
# =============================================================
OTRASLI = [
    "Информационные технологии и разработка ПО",
    "Строительство и девелопмент",
    "Производство и промышленность",
    "Торговля (дистрибьютеры)",
    "Консалтинг и прочие услуги для бизнеса",
    "Логистика и транспорт",
    "Реклама и маркетинг",
    "Финансы",
    "Банк",
    "Образование",
    "Туризм",
    "Гостиничная отрасль",
    "Телекоммуникация",
    "Нефтяной трейдинг",
    "Музыкальный бизнес",
    "Event-агентство",
    "негосударственный пенсионный фонд",
]

KANALY = [
    "Прямые офлайн-продажи (личные встречи, выезды к клиентам, звонки и т.д.)",
    "Прямые онлайн-продажи (сайт, интернет-магазин, соцсети)",
    "Продажи через дистрибьютеров, дилеров, агентов",
    "Продажи через B2B-маркетплейсы и электронные торговые площадки",
    "Многоканальная модель (нет явного доминирования одного канала)",
]

TYP_PRODUKTA = [
    "Индивидуальные решения (каждый проект или заказ адаптируется под требования клиента)",
    "Стандартная (типовая) продукция (каталог готовых решений)",
    "Смешанный тип (часть продуктов стандартная, часть – индивидуальная)",
]

SOTRUDNIKI = ["До 15", "16-100", "101-250", "Более 251"]

POKOLENIYA = [
    "преобладает Z поколение",
    "преобладают миллениалы",
    "преобладает X поколение",
    "преобладают бэби-бумеры",
]

INSTRUMENTS = [
    "Чат-боты",
    "Наружная реклама (баннеры, биллборды)",
    "Мессенджеры, скорость ответа клиентов",
    "Таргетированная реклама (реклама в социальных сетях)",
    "Теплые звонки (клиентам, с которым уже работали ранее)",
    "Стимулирование сбыта (акции, скидки, конкурсы)",
    "Длинные горизонтальные видео",
    "Пресс-конференции",
    "E-mail рассылки",
    "Сувенирная продукция с логотипом",
    "Спонсорство",
    "Контекстная реклама (Яндекс.Директ)",
    "SEO-продвижение",
    "Участие в отраслевых выставках",
    "Холодные звонки (потенциальным клиентам)",
    "Печатная реклама (листовки, каталоги, брошюры и др.)",
    "Ведение социальных сетей",
    "Ведение корпоративного блога",
    "Публикации в СМИ",
    "Размещение на агрегаторах",
    "Приложения дополненной реальности (AR)",
    "NFT",
]


# =============================================================
# СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# =============================================================
if "step" not in st.session_state:
    st.session_state.step = "start"
if "company" not in st.session_state:
    st.session_state.company = {}
if "result" not in st.session_state:
    st.session_state.result = None


def go_to(step):
    st.session_state.step = step
    st.rerun()


# =============================================================
# ЭКРАН 1: ПРИВЕТСТВИЕ
# =============================================================
if st.session_state.step == "start":
    st.title("🎯 Определение сегмента B2B-компании")
    st.markdown("""
    ### Что это?
    
    Приложение помогает определить, к какому сегменту по шкале инноваций 
    относится ваша компания:
    
- 🔵 Новаторы — тестирует и внедряет новые инструменты и технологии, даже если они пока не получили широкого распространения.
- 🟢 Ранние последователи — следят за трендами и начинают использовать новые решения раньше большинства коллег, опираясь на собственный анализ и пилотные проекты.
- 🟡 Раннее большинство — внедряют новые инструменты тогда, когда они уже доказали свою эффективность у заметной части рынка, и появляются первые проверенные кейсы.
- 🟠 Позднее большинство — предпочитают переходить на новые технологии только после того, как они стали устоявшимся стандартом нашей отрасли.
- 🔴 Консерваторы — стараются максимально долго использовать традиционные, проверенные временем методы. Новые инструменты внедряем вынужденно, когда без них уже невозможно работать.
    
    ### Что нужно?
    
    Ответить на **27 вопросов**:
    - 5 — о компании (отрасль, размер, тип продукта)
    - 22 — об используемых инструментах маркетинговых коммуникаций
    
    Это займёт **3–5 минут**.
    """)

    st.markdown("---")
    if st.button("🚀 Начать", type="primary", use_container_width=True):
        go_to("passport")


# =============================================================
# ЭКРАН 2: ПРОФИЛЬ КОМПАНИИ
# =============================================================
elif st.session_state.step == "passport":
    st.title("📋 Шаг 1 из 2: Профиль компании")
    st.progress(0.5)
    st.caption("Ответьте на 5 вопросов о вашей компании")

    company = st.session_state.company

    company["Отрасль"] = st.selectbox(
        "1. Отрасль",
        options=OTRASLI,
        index=OTRASLI.index(company["Отрасль"]) if company.get("Отрасль") in OTRASLI else 0,
    )

    company["канал продаж"] = st.selectbox(
        "2. Основной канал продаж",
        options=KANALY,
        index=KANALY.index(company["канал продаж"]) if company.get("канал продаж") in KANALY else 0,
    )

    company["тип продукта"] = st.selectbox(
        "3. Тип продукта",
        options=TYP_PRODUKTA,
        index=TYP_PRODUKTA.index(company["тип продукта"]) if company.get("тип продукта") in TYP_PRODUKTA else 0,
    )

    company["Сколько сотрудников в вашей компании?"] = st.selectbox(
        "4. Сколько сотрудников в компании?",
        options=SOTRUDNIKI,
        index=SOTRUDNIKI.index(company["Сколько сотрудников в вашей компании?"]) if company.get("Сколько сотрудников в вашей компании?") in SOTRUDNIKI else 0,
    )

    company["поколенческий состав"] = st.selectbox(
        "5. Поколенческий состав лиц, принимающих решения",
        options=POKOLENIYA,
        index=POKOLENIYA.index(company["поколенческий состав"]) if company.get("поколенческий состав") in POKOLENIYA else 0,
    )

    st.session_state.company = company

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Назад", use_container_width=True):
            go_to("start")
    with col2:
        if st.button("Далее →", type="primary", use_container_width=True):
            go_to("instruments")


# =============================================================
# ЭКРАН 3: ИНСТРУМЕНТЫ
# =============================================================
elif st.session_state.step == "instruments":
    st.title("🛠 Шаг 2 из 2: Используемые инструменты")
    st.progress(1.0)
    st.caption("Отметьте, насколько активно компания использует каждый инструмент")

    company = st.session_state.company

    with st.expander("ℹ️ Как отвечать", expanded=False):
        st.markdown("""
        - **0** — не используем
        - **1** — используем иногда / редко
        - **2** — используем активно / постоянно
        """)

    for instr in INSTRUMENTS:
        company[instr] = st.slider(
            instr,
            min_value=0,
            max_value=2,
            value=int(company.get(instr, 0)),
            step=1,
        )

    st.session_state.company = company

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Назад", use_container_width=True):
            go_to("passport")
    with col2:
        if st.button("🎯 Определить сегмент", type="primary", use_container_width=True):
            bundle = get_model()
            result = predict(bundle, company)
            st.session_state.result = result
            go_to("result")


# =============================================================
# ЭКРАН 4: РЕЗУЛЬТАТ
# =============================================================
elif st.session_state.step == "result":
    st.title("📊 Результат")
    result = st.session_state.result

    st.markdown(f"## Ваш сегмент: **{result['segment']}**")
    st.metric("Уверенность модели", f"{result['confidence']:.1%}")

    if result["is_borderline"]:
        st.warning(
            "⚠ Пограничный случай: модель колеблется между двумя сегментами. "
            "Рекомендуется экспертная оценка."
        )

    st.markdown("### Вероятности по всем сегментам")

    proba_df = pd.DataFrame({
        "Сегмент": list(result["probabilities"].keys()),
        "Вероятность": list(result["probabilities"].values()),
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if s == result["segment"] else "#95a5a6"
              for s in proba_df["Сегмент"]]
    bars = ax.barh(proba_df["Сегмент"], proba_df["Вероятность"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Вероятность")
    ax.invert_yaxis()
    for i, v in enumerate(proba_df["Вероятность"]):
        ax.text(v + 0.01, i, f"{v:.1%}", va="center")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Изменить ответы", use_container_width=True):
            go_to("instruments")
    with col2:
        if st.button("🔄 Пройти заново", type="primary", use_container_width=True):
            st.session_state.company = {}
            st.session_state.result = None
            go_to("start")

    with st.expander("ℹ️ Что означает мой сегмент?"):
        descriptions = {
            "Новаторы": "Тестируют и внедряют новые инструменты и технологии, даже если они пока не получили широкого распространения.",
            "Ранние последователи": "Следят за трендами и начинают использовать новые решения раньше большинства коллег, опираясь на собственный анализ и пилотные проекты.",
            "Раннее большинство": "Внедряют новые инструменты тогда, когда они уже доказали свою эффективность у заметной части рынка, и появляются первые проверенные кейсы.",
            "Позднее большинство": "Предпочитают переходить на новые технологии только после того, как они стали устоявшимся стандартом нашей отрасли.",
            "Консерваторы": "Стараются максимально долго использовать традиционные, проверенные временем методы. Новые инструменты внедряем вынужденно, когда без них уже невозможно работать.",
        }
        st.markdown(f"**{result['segment']}:** {descriptions[result['segment']]}")


# =============================================================
# ФУТЕР
# =============================================================
st.markdown("---")
st.caption(
    "Модель обучена на 136 B2B-компаниях. "
    "Результат носит рекомендательный характер."
)
