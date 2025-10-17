# # ===============================================================
# # 📉 Прогноз Відтоку Клієнтів — Streamlit App (оновлена версія)
# # ===============================================================
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import time
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_auc_score,
# )
# import warnings

# # ---------------------------------------------------------------
# # 1. Налаштування сторінки
# # ---------------------------------------------------------------
# st.set_page_config(page_title="Прогноз Відтоку Клієнтів", page_icon="📊", layout="wide")
# warnings.filterwarnings("ignore", message="Glyph.*missing from font")
# plt.rcParams["font.family"] = "DejaVu Sans"


# # ---------------------------------------------------------------
# # 2. Завантаження моделі та скейлера
# # ---------------------------------------------------------------
# @st.cache_resource
# def load_model():
#     model_path = "models/best_model_LightGBM.pkl"
#     scaler_path = "models/scaler.pkl"

#     model, scaler = None, None
#     if os.path.exists(model_path) and os.path.exists(scaler_path):
#         try:
#             model = joblib.load(model_path)
#             scaler = joblib.load(scaler_path)
#             st.success("✅ Модель та скейлер успішно завантажено.")
#         except Exception as e:
#             st.error(f"❌ Помилка при завантаженні: {e}")
#     else:
#         st.warning("⚠️ Модель або скейлер не знайдені. Увімкнено демо-режим.")
#     return model, scaler


# model, scaler = load_model()


# # =====================================================================
# # 🔧 Універсальна функція для узгодження назв ознак
# # =====================================================================
# def align_features(input_df, scaler):
#     expected = getattr(scaler, "feature_names_in_", None)
#     if expected is not None:
#         input_df = input_df.reindex(columns=expected, fill_value=0)
#     return input_df


# # ---------------------------------------------------------------
# # 3. Підготовка введених даних
# # ---------------------------------------------------------------
# def prepare_input_data(
#     is_tv_subscriber,
#     is_movie_package_subscriber,
#     subscription_age,
#     bill_avg,
#     reamining_contract,
#     service_failure_count,
#     download_avg,
#     upload_avg,
#     download_over_limit,
# ):
#     data = {
#         "is_tv_subscriber": 1 if is_tv_subscriber == "Так" else 0,
#         "is_movie_package_subscriber": 1 if is_movie_package_subscriber == "Так" else 0,
#         "subscription_age": subscription_age,
#         "bill_avg": bill_avg,
#         "reamining_contract": reamining_contract,
#         "service_failure_count": service_failure_count,
#         "download_avg": download_avg,
#         "upload_avg": upload_avg,
#         "download_over_limit": 1 if download_over_limit == "Так" else 0,
#     }
#     return pd.DataFrame([data])


# # =====================================================================
# # 🧍‍♂️ 4. Ручне прогнозування
# # =====================================================================
# def manual_input_page(model, scaler):
#     try:
#         logo = Image.open("app/assets/logo.png")
#         st.image(logo, width=150)
#     except FileNotFoundError:
#         st.markdown("## 📊 Прогнозування Відтоку Клієнтів")
#         st.caption("*(💡 Місце для логотипу: app/assets/logo.png)*")

#     st.markdown(
#         "### 👥 Аналітична система прогнозування поведінки клієнтів телеком-компанії"
#     )
#     st.divider()

#     if model is None or scaler is None:
#         st.info(
#             "⭐ **ДЕМО-РЕЖИМ:** Прогнози будуть випадковими, оскільки модель не завантажена."
#         )

#     st.header("🔧 Введіть дані клієнта:")
#     col1, col2 = st.columns(2)

#     with col1:
#         is_tv_subscriber = st.selectbox("Підписка на ТБ", ["Так", "Ні"])
#         is_movie_package_subscriber = st.selectbox("Пакет фільмів", ["Так", "Ні"])
#         subscription_age = st.number_input(
#             "Тривалість підписки (років)", 0.0, 20.0, 2.0, 0.1
#         )
#         bill_avg = st.number_input("Середній рахунок ($)", 0.0, 500.0, 25.0, 1.0)

#     with col2:
#         reamining_contract = st.number_input(
#             "Залишок контракту (років)", 0.0, 10.0, 1.0, 0.1
#         )
#         service_failure_count = st.number_input("Кількість збоїв сервісу", 0, 20, 0)
#         download_avg = st.number_input("Середнє завантаження (GB)", 0.0, 5000.0, 50.0)
#         upload_avg = st.number_input("Середнє відвантаження (GB)", 0.0, 500.0, 5.0)
#         download_over_limit = st.selectbox(
#             "Перевищення ліміту завантаження", ["Так", "Ні"]
#         )

#     st.divider()

#     if st.button("🔮 Прогнозувати"):
#         with st.spinner("Проводимо аналіз даних клієнта..."):
#             time.sleep(1.5)

#         input_df = prepare_input_data(
#             is_tv_subscriber,
#             is_movie_package_subscriber,
#             subscription_age,
#             bill_avg,
#             reamining_contract,
#             service_failure_count,
#             download_avg,
#             upload_avg,
#             download_over_limit,
#         )

#         if model is None or scaler is None:
#             st.warning("⚠️ Модель не завантажена — використовується ДЕМО прогноз.")
#             prediction = np.random.choice([0, 1], p=[0.7, 0.3])
#             probability = np.random.uniform(0.1, 0.9)
#         else:
#             try:
#                 input_df = align_features(input_df, scaler)
#                 X_scaled = scaler.transform(input_df)
#                 prediction = model.predict(X_scaled)[0]
#                 probability = model.predict_proba(X_scaled)[0][1]
#             except Exception as e:
#                 st.error(f"❌ Помилка при прогнозуванні: {e}")
#                 st.write(
#                     "Очікувані ознаки:", getattr(scaler, "feature_names_in_", None)
#                 )
#                 st.write("Отримані ознаки:", input_df.columns.tolist())
#                 return

#         st.subheader("📈 Результат прогнозу:")
#         if prediction == 1:
#             st.error(
#                 f"Клієнт **ймовірно піде** 😔 (ймовірність: **{probability:.2%}**)"
#             )
#         else:
#             st.success(
#                 f"Клієнт **ймовірно залишиться** 😊 (ймовірність: **{probability:.2%}**)"
#             )

#         fig, ax = plt.subplots(figsize=(4, 3))
#         ax.bar(
#             ["Залишиться", "Відтік"],
#             [1 - probability, probability],
#             color=["#28a745", "#dc3545"],
#         )
#         ax.set_ylim(0, 1)
#         ax.set_ylabel("Ймовірність")
#         ax.set_title("Ймовірність Відтоку / Утримання")
#         st.pyplot(fig)


# # =====================================================================
# # 📂 5. Пакетне прогнозування CSV
# # =====================================================================
# def batch_upload_page(model, scaler):
#     st.header("📤 Завантаження файлу для пакетного прогнозування")
#     st.info(
#         "Завантажте **CSV** файл із тими самими ознаками, що використовувались при навчанні моделі."
#     )
#     st.divider()

#     uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv")

#     if uploaded_file is not None:
#         try:
#             data = pd.read_csv(uploaded_file)
#             st.subheader("📄 Попередній перегляд даних:")
#             st.dataframe(data.head())

#             if st.button("🚀 Запустити Прогнозування"):
#                 with st.spinner("Обробка та прогнозування..."):
#                     time.sleep(1)

#                 if model is None or scaler is None:
#                     st.warning(
#                         "⚠️ Модель не завантажена — використовується ДЕМО прогноз."
#                     )
#                     data["Prediction"] = np.random.choice([0, 1], len(data))
#                     data["Probability"] = np.random.uniform(0.05, 0.95, len(data))
#                 else:
#                     try:
#                         data_prepared = align_features(data.copy(), scaler)
#                         X_scaled = scaler.transform(data_prepared)
#                         data["Prediction"] = model.predict(X_scaled)
#                         data["Probability"] = model.predict_proba(X_scaled)[:, 1]
#                     except Exception as e:
#                         st.error(f"❌ Помилка при прогнозуванні: {e}")
#                         st.write(
#                             "Очікувані:", getattr(scaler, "feature_names_in_", None)
#                         )
#                         st.write("Отримані:", data.columns.tolist())
#                         return

#                 data["Статус"] = data["Prediction"].map({1: "Відтік", 0: "Залишиться"})

#                 st.subheader("✅ Результати прогнозування:")
#                 st.dataframe(data)

#                 csv_out = data.to_csv(index=False).encode("utf-8")
#                 st.download_button(
#                     "📥 Завантажити результати (CSV)",
#                     csv_out,
#                     "churn_predictions.csv",
#                     "text/csv",
#                 )

#                 st.divider()
#                 st.markdown(f"**📊 Кількість клієнтів:** {len(data)}")
#                 st.markdown(f"**📉 Відтік:** {sum(data['Prediction'] == 1)}")
#                 st.markdown(f"**📈 Залишаються:** {sum(data['Prediction'] == 0)}")

#         except Exception as e:
#             st.error(f"❌ Помилка при читанні файлу: {e}")


# # =====================================================================
# # 📈 6. Аналітика моделі
# # =====================================================================
# def analytics_page():
#     st.header("📊 Аналітика Моделі (LightGBM)")
#     st.markdown("Оцінка моделі на різних наборах даних:")

#     combined = pd.DataFrame(
#         {
#             "dataset": ["Train", "Test", "Holdout"],
#             "accuracy": [0.972, 0.959, 0.954],
#             "precision": [0.981, 0.969, 0.962],
#             "recall": [0.958, 0.956, 0.947],
#             "f1": [0.969, 0.963, 0.955],
#             "roc_auc": [0.996, 0.994, 0.991],
#         }
#     )

#     st.dataframe(combined.style.format("{:.3f}"))

#     # Побудова графіка
#     melted = combined.melt(id_vars="dataset", var_name="metric", value_name="value")
#     plt.figure(figsize=(8, 5))
#     ax = sns.barplot(
#         data=melted, x="metric", y="value", hue="dataset", palette="coolwarm"
#     )
#     plt.title(
#         "📊 Порівняння якості моделі на Train/Test/Holdout",
#         fontsize=12,
#         fontweight="bold",
#     )
#     plt.ylabel("Значення метрики")
#     plt.ylim(0.9, 1.0)
#     plt.legend(title="Дані", loc="lower right")

#     for p in ax.patches:
#         ax.text(
#             p.get_x() + p.get_width() / 2.0,
#             p.get_height() + 0.003,
#             f"{p.get_height():.3f}",
#             ha="center",
#             va="bottom",
#             fontsize=9,
#             fontweight="bold",
#         )

#     st.pyplot(plt)

#     st.info(
#         """
#     ✅ **Висновок:**
#     Модель стабільна, точна і добре узагальнює дані.
#     LightGBM показує найкращі результати серед протестованих моделей.
#     """
#     )


# # =====================================================================
# # 🧭 7. Навігація
# # =====================================================================
# menu = st.sidebar.radio(
#     "🧭 Оберіть розділ", ["Ручний прогноз", "Пакетне прогнозування", "Аналітика"]
# )

# if menu == "Ручний прогноз":
#     manual_input_page(model, scaler)
# elif menu == "Пакетне прогнозування":
#     batch_upload_page(model, scaler)
# else:
#     analytics_page()

# st.sidebar.caption("© 2025 Аналітична команда прогнозування відтоку клієнтів 📡")


# ===============================================================
# 📉 Прогноз Відтоку Клієнтів — Streamlit App (ОСТАТОЧНА ВЕРСІЯ)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
from typing import Tuple, Any

# ---------------------------------------------------------------
# 1. Налаштування сторінки
# ---------------------------------------------------------------
st.set_page_config(page_title="Прогноз Відтоку Клієнтів", page_icon="📊", layout="wide")
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

# Налаштування стилю для графіків
plt.rcParams["font.family"] = "DejaVu Sans"
plt.style.use("ggplot")

# ---------------------------------------------------------------
# 🎨 Кастомні Стилі CSS
# ---------------------------------------------------------------
st.markdown(
    """
<style>
    /* Основна палітра кольорів */
    :root {
        --primary-dark: #324851;
        --primary-green: #86AC41;
        --secondary-teal: #34675C;
        --accent-light: #7DA3A1;
    }

    /* Загальні стилі */
    # .stApp {
    #     background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f2 100%);
    # }

    /* Заголовки */
    h1, h2, h3 {
        color: #324851 !important;
        font-weight: 700 !important;
    }

    /* Бокова панель */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #324851 0%, #34675C 100%);
    }

    [data-testid="stSidebar"] * {
        color: #324851 !important;
    }

    /* Кнопки */
    .stButton{
    display: flex !important;
        justify-content: center !important;
        margin: 2rem auto !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #86AC41 0%, #34675C 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 33% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #34675C 0%, #86AC41 100%);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }

    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(134, 172, 65, 0.3);
        outline: none;
    }

    /* Метрики */
    [data-testid="stMetricValue"] {
        color: #324851;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #34675C;
        font-weight: 600 !important;
    }

    /* Картки та контейнери */
    .element-container {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        #box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        #width: 500px !important;
    }

    /* Інфо блоки */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid #86AC41;
        background-color: rgba(134, 172, 65, 0.1);
    }

    /* Селектбокси та інпути */


    /* Таблиці */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Success/Error повідомлення */
    .stSuccess {
        background-color: rgba(134, 172, 65, 0.15);
        color: #34675C;
        border-radius: 8px;
        padding: 1rem;
    }

    .stError {
        background-color: rgba(220, 53, 69, 0.15);
        border-radius: 8px;
        padding: 1rem;
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #7DA3A1 0%, #34675C 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #34675C 0%, #7DA3A1 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Радіо кнопки в сайдбарі */
    [data-testid="stSidebar"] .stRadio > label {
        background-color: rgba(255, 255, 255, 0.1);
        width: fit-content;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
    }

    [data-testid="stSidebar"] .stRadio > label:hover {
        background-color: rgba(255, 255, 255, 0.2);
        width: fit-content;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Словник для перейменування ознак
FEATURE_NAMES_MAP = {
    "reamining_contract": "Залишок контракту",
    "subscription_age": "Тривалість підписки",
    "service_failure_count": "Кількість збоїв сервісу",
    "bill_avg": "Середній рахунок",
    "download_avg": "Середнє завантаження",
    "upload_avg": "Середнє скачування",
    "is_tv_subscriber": "Підписка на ТБ",
    "is_movie_package_subscriber": "Пакет фільмів",
    "download_over_limit": "Перевищення ліміту завантаження",
}


# ---------------------------------------------------------------
# 2. Завантаження моделі та скейлера
# ---------------------------------------------------------------
@st.cache_resource
def load_model() -> Tuple[Any, Any]:
    """Завантажує модель та скейлер з дисків. Повертає None, якщо файли не знайдено."""
    model_path = "models/best_model_LightGBM.pkl"
    scaler_path = "models/scaler.pkl"

    model, scaler = None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Модель та скейлер успішно завантажено.")
        except Exception as e:
            st.error(f"❌ Помилка при завантаженні файлів моделі/скейлера: {e}")
    else:
        st.error(
            f"""
        ❌ **КРИТИЧНА ПОМИЛКА:** Модель або скейлер не знайдені.
        Будь ласка, переконайтеся, що файли існують за шляхами:
        - Модель: `{model_path}`
        - Скейлер: `{scaler_path}`
        Прогнозування неможливе.
        """
        )

    return model, scaler


model, scaler = load_model()
MODEL_LOADED = model is not None and scaler is not None


# =====================================================================
# 🔧 Універсальні функції
# =====================================================================
def align_features(input_df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    """Вирівнює стовпці вхідного DataFrame відповідно до очікуваних моделлю."""
    expected = getattr(scaler, "feature_names_in_", None)
    if expected is not None:
        input_df = input_df.reindex(columns=expected, fill_value=0)
    return input_df


def prepare_input_data(
    is_tv_subscriber,
    is_movie_package_subscriber,
    subscription_age,
    bill_avg,
    reamining_contract,
    service_failure_count,
    download_avg,
    upload_avg,
    download_over_limit,
) -> pd.DataFrame:
    data = {
        "is_tv_subscriber": 1 if is_tv_subscriber == "Так" else 0,
        "is_movie_package_subscriber": 1 if is_movie_package_subscriber == "Так" else 0,
        "download_over_limit": 1 if download_over_limit == "Так" else 0,
        "subscription_age": subscription_age,
        "bill_avg": bill_avg,
        "reamining_contract": reamining_contract,
        "service_failure_count": service_failure_count,
        "download_avg": download_avg,
        "upload_avg": upload_avg,
    }
    return pd.DataFrame([data])


# ---------------------------------------------------------------
# 4. Ручне прогнозування (Вкладка 1)
# ---------------------------------------------------------------
def manual_input_page(model, scaler):
    st.markdown("## 📊 Прогноз Відтоку Клієнтів: Ручний Ввід")
    st.caption("Аналітична система прогнозування поведінки клієнтів телеком-компанії.")

    if not MODEL_LOADED:
        st.warning("⚠️ Функціонал прогнозування недоступний. Модель не завантажена.")
        return

    st.header("🔧 Введіть дані клієнта:")
    col1, col2 = st.columns(2)

    with col1:
        is_tv_subscriber = st.selectbox("Підписка на ТБ", ["Так", "Ні"])
        is_movie_package_subscriber = st.selectbox("Пакет фільмів", ["Так", "Ні"])
        subscription_age = st.number_input(
            "Тривалість підписки (років)", 0.0, 20.0, 2.0, 0.1
        )
        bill_avg = st.number_input("Середній рахунок ($)", 0.0, 500.0, 25.0, 1.0)

    with col2:
        reamining_contract = st.number_input(
            "Залишок контракту (років)", 0.0, 10.0, 1.0, 0.1
        )
        service_failure_count = st.number_input("Кількість збоїв сервісу", 0, 20, 0)
        download_avg = st.number_input("Середнє завантаження (GB)", 0.0, 5000.0, 50.0)
        upload_avg = st.number_input("Середнє відвантаження (GB)", 0.0, 500.0, 5.0)
        download_over_limit = st.selectbox(
            "Перевищення ліміту завантаження", ["Так", "Ні"]
        )

    if st.button("🔮 Прогнозувати", use_container_width=True):
        with st.spinner("Проводимо аналіз даних клієнта..."):
            time.sleep(1.0)

        input_df = prepare_input_data(
            is_tv_subscriber,
            is_movie_package_subscriber,
            subscription_age,
            bill_avg,
            reamining_contract,
            service_failure_count,
            download_avg,
            upload_avg,
            download_over_limit,
        )

        try:
            input_df = align_features(input_df, scaler)
            X_scaled = scaler.transform(input_df)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            st.error(f"❌ Помилка при прогнозуванні: {e}")
            return

        st.subheader("📈 Результат прогнозу:")
        if prediction == 1:
            st.error(
                f"Клієнт **ймовірно піде** 😔 (ймовірність: **{probability:.2%}**)"
            )
        else:
            st.success(
                f"Клієнт **ймовірно залишиться** 😊 (ймовірність: **{probability:.2%}**)"
            )

        fig, ax = plt.subplots(figsize=(6, 4))
        probabilities = [1 - probability, probability]
        labels = ["Залишиться", "Відтік"]
        colors = ["#86AC41", "#dc3545"]
        bars = ax.bar(
            labels, probabilities, color=colors, edgecolor="#324851", linewidth=2
        )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Ймовірність", fontsize=10, color="#324851", fontweight="bold")
        ax.set_title(
            "Ймовірність Відтоку / Утримання",
            fontsize=12,
            color="#324851",
            fontweight="bold",
            pad=20,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#7DA3A1")
        ax.spines["bottom"].set_color("#7DA3A1")
        ax.tick_params(colors="#324851")

        for i, (prob, bar) in enumerate(zip(probabilities, bars)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.03,
                f"{prob:.2%}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                color="#324851",
            )

        plt.tight_layout()
        st.pyplot(fig)


# ---------------------------------------------------------------
# 5. Пакетне прогнозування CSV (Вкладка 2)
# ---------------------------------------------------------------
def batch_upload_page(model, scaler):
    st.markdown("## 📤 Пакетне Прогнозування (Завантаження CSV)")
    st.info(
        "Завантажте **CSV** файл із даними клієнтів (без цільової змінної) для масового прогнозу."
    )

    if not MODEL_LOADED:
        st.warning("⚠️ Функціонал прогнозування недоступний. Модель не завантажена.")
        return

    uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv")

    if uploaded_file is None:
        st.markdown(
            """
        **✅ Вимоги до файлу:** Файл має містити такі стовпці:
        * **`is_tv_subscriber`**: 0 або 1
        * **`is_movie_package_subscriber`**: 0 або 1
        * **`subscription_age`**: Роки (0-15)
        * **`bill_avg`**: Сума (0-1000)
        * **`reamining_contract`**: Роки (0-10)
        * **`service_failure_count`**: Кількість (0-40)
        * **`download_avg`**: ГБ (0-8000)
        * **`upload_avg`**: ГБ (0-800)
        * **`download_over_limit`**: 0 або 1
        """
        )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("📄 Попередній перегляд даних:")
            st.dataframe(data.head())

            if st.button("🚀 Запустити Прогнозування", use_container_width=True):
                with st.spinner("Обробка та прогнозування..."):
                    time.sleep(1)

                try:
                    data_prepared = align_features(data.copy(), scaler)
                    required_features = set(getattr(scaler, "feature_names_in_", []))
                    if not required_features.issubset(set(data_prepared.columns)):
                        st.error(
                            f"Відсутні необхідні ознаки у завантаженому файлі: {required_features - set(data_prepared.columns)}"
                        )
                        return

                    X_scaled = scaler.transform(data_prepared)
                    data["Prediction"] = model.predict(X_scaled)
                    data["Probability"] = model.predict_proba(X_scaled)[:, 1]
                except Exception as e:
                    st.error(f"❌ Помилка при прогнозуванні: {e}")
                    return

                data["Статус"] = data["Prediction"].map({1: "Відтік", 0: "Залишиться"})
                data["Ймовірність"] = data["Probability"].apply(lambda x: f"{x:.2%}")

                st.subheader("✅ Результати прогнозування:")
                st.dataframe(
                    data.drop(columns=["Prediction", "Probability"], errors="ignore")
                )

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("📊 Кількість клієнтів", len(data))
                col_res2.metric("📉 Прогноз Відтоку", sum(data["Prediction"] == 1))
                col_res3.metric("📈 Прогноз Утримання", sum(data["Prediction"] == 0))

                csv_out = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Завантажити результати (CSV)",
                    csv_out,
                    "churn_predictions.csv",
                    "text/csv",
                )

        except Exception as e:
            st.error(f"❌ Помилка при читанні або обробці файлу: {e}")


# ---------------------------------------------------------------
# 6. Аналітика (Вкладка 3) - Розділена на підвкладки
# ---------------------------------------------------------------
def data_analysis_page():
    st.markdown("## 🔎 Огляд Даних та Ключових Факторів Відтоку")
    st.caption("Аналіз впливу ознак на ймовірність відтоку клієнтів.")

    st.subheader("💡 Рекомендації для Стратегій Утримання")
    st.warning(
        """
    **Рекомендації:**
    Сфокусуйтеся на клієнтах з високою важливістю відтоку: **пропонуйте продовження контракту** та **мінімізуйте збої сервісу** для їх утримання.
    """
    )

    st.subheader("🔑 Аналіз Факторів Відтоку (Feature Importance)")
    st.markdown(
        "Цей графік показує, які ознаки мають найбільший вплив на прогноз відтоку клієнтів (згідно з моделлю LightGBM)."
    )

    feature_importance_df = pd.DataFrame(
        {
            "Ознака": [
                "reamining_contract",
                "subscription_age",
                "service_failure_count",
                "bill_avg",
                "download_avg",
                "upload_avg",
                "is_tv_subscriber",
                "is_movie_package_subscriber",
                "download_over_limit",
            ],
            "Важливість": [0.35, 0.25, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01],
        }
    ).sort_values(by="Важливість", ascending=False)

    feature_importance_df["Ознака"] = feature_importance_df["Ознака"].replace(
        FEATURE_NAMES_MAP
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors_gradient = [
        "#86AC41",
        "#7DA3A1",
        "#34675C",
        "#324851",
        "#7DA3A1",
        "#86AC41",
        "#34675C",
        "#324851",
        "#7DA3A1",
    ]
    bars = ax.barh(
        feature_importance_df["Ознака"],
        feature_importance_df["Важливість"],
        color=colors_gradient[: len(feature_importance_df)],
        edgecolor="#324851",
        linewidth=1.5,
    )

    ax.set_xlabel(
        "Відносна Важливість", fontsize=12, color="#324851", fontweight="bold"
    )
    ax.set_ylabel("Ознака", fontsize=12, color="#324851", fontweight="bold")
    ax.set_title(
        "Ключові Фактори, що Впливають на Відтік",
        fontsize=14,
        color="#324851",
        fontweight="bold",
        pad=20,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7DA3A1")
    ax.spines["bottom"].set_color("#7DA3A1")
    ax.tick_params(colors="#324851", labelsize=10)
    ax.set_xlim(0, max(feature_importance_df["Важливість"]) * 1.15)

    for i, (val, bar) in enumerate(zip(feature_importance_df["Важливість"], bars)):
        width = bar.get_width()
        ax.text(
            width + 0.008,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f}",
            va="center",
            ha="left",
            fontweight="bold",
            fontsize=10,
            color="#324851",
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Детальний опис ознак")
    st.markdown(
        "Розуміння значень ознак, які використовує модель, є ключем до розробки ефективних стратегій."
    )

    st.markdown(
        f"""
    * **{FEATURE_NAMES_MAP['reamining_contract']}:** **Ключовий фактор.** Тривалість (у роках), що залишилася до завершення поточного контракту.
    * **{FEATURE_NAMES_MAP['subscription_age']}:** Вік клієнта в компанії (у роках). Нові клієнти схильні до більшого відтоку.
    * **{FEATURE_NAMES_MAP['service_failure_count']}:** Загальна кількість технічних проблем. Прямий показник незадоволеності.
    * **{FEATURE_NAMES_MAP['bill_avg']}:** Середній місячний рахунок клієнта ($).
    * **{FEATURE_NAMES_MAP['download_avg']}:** Середній обсяг даних, які клієнт завантажує за місяць (GB).
    * **{FEATURE_NAMES_MAP['upload_avg']}:** Середній обсяг даних, які клієнт відвантажує за місяць (GB).
    * **{FEATURE_NAMES_MAP['is_tv_subscriber']}:** Бінарна ознака (1/0).
    * **{FEATURE_NAMES_MAP['is_movie_package_subscriber']}:** Бінарна ознака (1/0).
    * **{FEATURE_NAMES_MAP['download_over_limit']}:** Бінарна ознака (1/0).
    """
    )


def model_performance_page():
    st.markdown("## 📈 Аналіз Роботи Моделі")
    st.caption("Оцінка якості прогнозуючої здатності моделі LightGBM.")

    st.subheader("1. Оцінка Якості Моделі (LightGBM)")
    st.markdown("Показники на різних наборах даних (Train/Test/Holdout):")

    combined = pd.DataFrame(
        {
            "Набір даних": ["Train", "Test", "Holdout"],
            "Accuracy": [0.972, 0.959, 0.954],
            "Precision": [0.981, 0.969, 0.962],
            "Recall": [0.958, 0.956, 0.947],
            "F1-Score": [0.969, 0.963, 0.955],
            "ROC AUC": [0.996, 0.994, 0.991],
        }
    )

    st.dataframe(
        combined.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
                "ROC AUC": "{:.3f}",
            }
        ),
        hide_index=True,
    )

    metric_options = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
    selected_metric = st.selectbox("Оберіть метрику для порівняння", metric_options)

    melted = combined.melt(
        id_vars="Набір даних", var_name="Метрика", value_name="Значення"
    )
    filtered_melted = melted[melted["Метрика"] == selected_metric]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors_bar = ["#86AC41", "#7DA3A1", "#34675C"]
    bars = ax.bar(
        filtered_melted["Набір даних"],
        filtered_melted["Значення"],
        color=colors_bar,
        edgecolor="#324851",
        linewidth=2,
    )

    ax.set_ylabel("Значення метрики", fontsize=12, color="#324851", fontweight="bold")
    ax.set_xlabel("Набір даних", fontsize=12, color="#324851", fontweight="bold")
    ax.set_title(
        f"📊 Порівняння **{selected_metric}** на Train/Test/Holdout",
        fontsize=14,
        color="#324851",
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(0.9, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7DA3A1")
    ax.spines["bottom"].set_color("#7DA3A1")
    ax.tick_params(colors="#324851", labelsize=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.003,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#324851",
        )

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        f"""
    ✅ **Висновок:** Модель демонструє високі показники по метриці **{selected_metric}** (близько 0.95-0.99)
    на всіх наборах даних, що свідчить про її **стабільність** і хорошу **узагальнюючу здатність**.
    """
    )


# =====================================================================
# 7. Основна Навігація
# =====================================================================
st.sidebar.title("🧭 Меню")

main_menu = st.sidebar.radio(
    "Оберіть розділ",
    ["Ручний ввід", "Пакетний ввід", "Аналітика та Звіти"],
    index=0,
)

if main_menu == "Ручний ввід":
    manual_input_page(model, scaler)
elif main_menu == "Пакетний ввід":
    batch_upload_page(model, scaler)
else:
    st.markdown("## 📈 Аналітичний Портал")
    st.caption("Виберіть підрозділ аналітики.")

    analytic_tab = st.selectbox(
        "Виберіть тип аналітики", ["Огляд Даних та Факторів Відтоку", "Оцінка Моделі"]
    )

    if analytic_tab == "Огляд Даних та Факторів Відтоку":
        data_analysis_page()
    elif analytic_tab == "Оцінка Моделі":
        model_performance_page()

st.sidebar.caption("© 2025 Аналітична команда прогнозування відтоку клієнтів 📡")
