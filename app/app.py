# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from utils.preprocessing import prepare_input_data
# from PIL import Image
# import matplotlib.pyplot as plt

# # ============ Налаштування сторінки ============
# st.set_page_config(
#     page_title="Прогноз Відтоку Клієнтів",
#     page_icon="📉",
#     layout="centered",
# )

# # ============ Логотип ============
# try:
#     logo = Image.open("app/assets/logo.png")
#     st.image(logo, width=150)
# except:
#     st.title("📊 Прогнозування Відтоку Клієнтів")

# st.markdown("### 👥 Аналітична система прогнозування поведінки клієнтів телеком-компанії")
# st.divider()

# # ============ Завантаження моделі ============
# @st.cache_resource
# def load_model():
#     model = joblib.load("models/best_model.pkl")
#     scaler = joblib.load("models/scaler.pkl")
#     return model, scaler

# try:
#     model, scaler = load_model()
# except:
#     st.warning("⚠️ Модель або scaler поки що не завантажені. Використовується демо-режим.")
#     model, scaler = None, None

# # ============ Ввід користувацьких даних ============
# st.header("🔧 Введіть дані клієнта:")

# col1, col2 = st.columns(2)

# with col1:
#     is_tv_subscriber = st.selectbox("Підписка на ТБ", ["Так", "Ні"])
#     is_movie_package_subscriber = st.selectbox("Пакет фільмів", ["Так", "Ні"])
#     subscription_age = st.number_input("Тривалість підписки (років)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
#     bill_avg = st.number_input("Середній рахунок ($)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)

# with col2:
#     reamining_contract = st.number_input("Залишок контракту (років)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
#     service_failure_count = st.number_input("Кількість збоїв сервісу", min_value=0, max_value=20, value=0)
#     download_avg = st.number_input("Середнє завантаження (GB)", min_value=0.0, max_value=5000.0, value=50.0)
#     upload_avg = st.number_input("Середнє відвантаження (GB)", min_value=0.0, max_value=500.0, value=5.0)
#     download_over_limit = st.selectbox("Перевищення ліміту завантаження", ["Так", "Ні"])

# st.divider()

# # ============ Підготовка даних ============
# input_data = prepare_input_data(
#     is_tv_subscriber,
#     is_movie_package_subscriber,
#     subscription_age,
#     bill_avg,
#     reamining_contract,
#     service_failure_count,
#     download_avg,
#     upload_avg,
#     download_over_limit
# )

# # ============ Прогноз ============
# if st.button("🔮 Прогнозувати"):
#     if model is None:
#         st.error("❌ Модель не завантажена. Додайте 'best_model.pkl' і 'scaler.pkl' у папку models/")
#     else:
#         X_scaled = scaler.transform(input_data)
#         prediction = model.predict(X_scaled)[0]
#         probability = model.predict_proba(X_scaled)[0][1]

#         st.subheader("📈 Результат прогнозу:")
#         if prediction == 1:
#             st.error(f"Клієнт **ймовірно піде** 😔 (ймовірність відтоку: {probability:.2%})")
#         else:
#             st.success(f"Клієнт **ймовірно залишиться** 😊 (ймовірність відтоку: {probability:.2%})")

#         # Додаткова візуалізація
#         fig, ax = plt.subplots(figsize=(4,3))
#         ax.bar(["Залишиться", "Відтік"], [1 - probability, probability], color=["green", "red"])
#         ax.set_ylim(0, 1)
#         ax.set_ylabel("Ймовірність")
#         st.pyplot(fig)

# st.caption("© 2025 Аналітична команда з прогнозування відтоку клієнтів 📡")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import os
import io
import time  # Додаємо для імітації роботи прогнозу


# ==============================================================================
# Функція для підготовки вхідних даних (для першої вкладки)
# ==============================================================================
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
):
    """
    Перетворює введені користувачем дані у формат DataFrame, готовий
    для масштабування та прогнозування.
    """
    data = {
        "is_tv_subscriber": 1 if is_tv_subscriber == "Так" else 0,
        "is_movie_package_subscriber": 1 if is_movie_package_subscriber == "Так" else 0,
        "subscription_age": subscription_age,
        "bill_avg": bill_avg,
        "reamining_contract": reamining_contract,
        "service_failure_count": service_failure_count,
        "download_avg": download_avg,
        "upload_avg": upload_avg,
        "download_over_limit": 1 if download_over_limit == "Так" else 0,
    }
    input_df = pd.DataFrame([data])
    return input_df


# ==============================================================================
# НАЛАШТУВАННЯ ТА ЗАВАНТАЖЕННЯ РЕСУРСІВ
# ==============================================================================

# Налаштування сторінки
st.set_page_config(
    page_title="Прогноз Відтоку Клієнтів",
    page_icon="📉",
    layout="wide",
)


# ============ ЗАГЛУШКА ДЛЯ ЗАВАНТАЖЕННЯ МОДЕЛІ ============
@st.cache_resource
def load_model():
    """
    Завантажує збережену модель та скейлер.
    !!! ЗАГЛУШКА: Для запуску без файлів модель та скейлер встановлюються в None.
    """
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    model, scaler = None, None

    # Перевіряємо наявність файлів
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Модель та скейлер успішно завантажені.")
        except Exception as e:
            st.error(f"❌ Помилка при завантаженні моделі/скейлера: {e}")
            st.info(
                "💡 **ПРИМІТКА:** Модель та скейлер не завантажені, додаток працює в ДЕМО-режимі."
            )
    else:
        st.warning(f"⚠️ Модель ({model_path}) або скейлер ({scaler_path}) не знайдені.")
        st.info(
            "💡 **ПРИМІТКА:** Додаток працює в **ДЕМО-режимі** з імітованими прогнозами."
        )

    return model, scaler


model, scaler = load_model()

# ==============================================================================
# ФУНКЦІЇ ДЛЯ ВКЛАДОК (РОЗДІЛІВ)
# ==============================================================================


def manual_input_page(model, scaler):
    """
    Реалізує сторінку ручного вводу даних клієнта та прогнозування.
    """
    # ============ ЗАГЛУШКА ДЛЯ ЛОГОТИПУ ============
    try:
        logo = Image.open("app/assets/logo.png")
        st.image(logo, width=150)
    except FileNotFoundError:
        # Якщо логотипу немає, відображаємо текст
        st.markdown(f"## 📊 Прогнозування Відтоку Клієнтів")
        st.caption("*(💡 Місце для логотипу: app/assets/logo.png)*")

    st.markdown(
        "### 👥 Аналітична система прогнозування поведінки клієнтів телеком-компанії"
    )
    st.divider()

    # Попередження, якщо модель не завантажена
    if model is None:
        st.info(
            "⭐ **ДЕМО-РЕЖИМ:** Прогнози будуть випадковими, оскільки модель не завантажена."
        )

    # ============ Ввід користувацьких даних ============
    st.header("🔧 Введіть дані клієнта:")

    # ... (інтерфейс вводу залишається без змін) ...
    col1, col2 = st.columns(2)

    with col1:
        is_tv_subscriber = st.selectbox("Підписка на ТБ", ["Так", "Ні"], key="tv_sub")
        is_movie_package_subscriber = st.selectbox(
            "Пакет фільмів", ["Так", "Ні"], key="movie_sub"
        )
        subscription_age = st.number_input(
            "Тривалість підписки (років)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            key="sub_age",
        )
        bill_avg = st.number_input(
            "Середній рахунок ($)",
            min_value=0.0,
            max_value=500.0,
            value=25.0,
            step=1.0,
            key="bill_avg",
        )

    with col2:
        reamining_contract = st.number_input(
            "Залишок контракту (років)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key="contract",
        )
        service_failure_count = st.number_input(
            "Кількість збоїв сервісу",
            min_value=0,
            max_value=20,
            value=0,
            key="failure_count",
        )
        download_avg = st.number_input(
            "Середнє завантаження (GB)",
            min_value=0.0,
            max_value=5000.0,
            value=50.0,
            key="download_avg",
        )
        upload_avg = st.number_input(
            "Середнє відвантаження (GB)",
            min_value=0.0,
            max_value=500.0,
            value=5.0,
            key="upload_avg",
        )
        download_over_limit = st.selectbox(
            "Перевищення ліміту завантаження", ["Так", "Ні"], key="over_limit"
        )

    st.divider()

    # ============ Прогноз ============
    if st.button("🔮 Прогнозувати"):
        # Імітація завантаження
        with st.spinner("Проводимо аналіз даних клієнта..."):
            time.sleep(1.5)  # Імітація часу обробки

        if model is None:
            # === ДЕМО-ЛОГІКА ПРОГНОЗУВАННЯ ===
            # Зробимо випадковий прогноз, але з ухилом до "залишиться"
            is_churn = np.random.choice([0, 1], p=[0.7, 0.3])
            probability = (
                np.random.uniform(0.1, 0.4)
                if is_churn == 0
                else np.random.uniform(0.6, 0.9)
            )
            prediction = is_churn
            # =================================

        else:
            # === РЕАЛЬНА ЛОГІКА ПРОГНОЗУВАННЯ ===
            try:
                input_data = prepare_input_data(
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
                X_scaled = scaler.transform(input_data)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1]
            except Exception as e:
                st.error(
                    f"❌ Помилка при прогнозуванні. Перевірте формат вхідних даних: {e}"
                )
                return
            # ====================================

        st.subheader("📈 Результат прогнозу:")
        if prediction == 1:
            st.error(
                f"Клієнт **ймовірно піде** 😔 (ймовірність відтоку: **{probability:.2%}**)"
            )
        else:
            st.success(
                f"Клієнт **ймовірно залишиться** 😊 (ймовірність відтоку: **{probability:.2%}**)"
            )

        # Додаткова візуалізація
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(
            ["Залишиться", "Відтік"],
            [1 - probability, probability],
            color=["#28a745", "#dc3545"],
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Ймовірність")
        ax.set_title("Ймовірності")
        st.pyplot(fig)


def file_upload_page(model, scaler):
    """
    Реалізує сторінку для завантаження CSV файлу з даними клієнтів.
    """
    st.header("📤 Завантаження Файлу для Пакетного Прогнозування")
    st.info("Будь ласка, завантажте **CSV** файл.")
    st.divider()

    uploaded_file = st.file_uploader("Оберіть файл CSV", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("Перегляд завантажених даних:")
            st.dataframe(data.head())

            if st.button("🚀 Зробити Пакетний Прогноз"):
                with st.spinner("Обробка та прогнозування даних у файлі..."):
                    time.sleep(2)  # Імітація часу обробки

                if model is None or scaler is None:
                    # === ДЕМО-ЛОГІКА ПАКЕТНОГО ПРОГНОЗУВАННЯ ===
                    st.warning("❌ Модель не завантажена. Виконується ДЕМО-прогноз.")
                    data["Prediction"] = np.random.randint(0, 2, size=len(data))
                    data["Churn_Probability"] = np.random.uniform(
                        0.05, 0.95, size=len(data)
                    )
                    # ==========================================
                else:
                    # === РЕАЛЬНА ЛОГІКА ПАКЕТНОГО ПРОГНОЗУВАННЯ ===
                    try:
                        data_to_predict = data.copy()
                        # !!! ВСТАВТЕ ФУНКЦІЮ ПЕРЕДОБРОБКИ ТУТ !!!
                        # data_processed = your_preprocessing_function(data_to_predict)

                        X_scaled = scaler.transform(data_to_predict)
                        predictions = model.predict(X_scaled)
                        probabilities = model.predict_proba(X_scaled)[:, 1]

                        data["Prediction"] = predictions
                        data["Churn_Probability"] = probabilities
                    except Exception as e:
                        st.error(
                            f"❌ Помилка при обробці даних або прогнозуванні. Перевірте відповідність стовпців: {e}"
                        )
                        return
                    # ============================================

                data["Churn_Status"] = data["Prediction"].apply(
                    lambda x: "Відтік" if x == 1 else "Залишиться"
                )

                st.subheader("✅ Результати Пакетного Прогнозування:")
                st.dataframe(data)

                # Надаємо можливість завантажити результат
                csv_output = data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Завантажити Результати (CSV)",
                    data=csv_output,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )

                # Коротка аналітика
                churn_counts = data["Churn_Status"].value_counts()
                st.markdown(f"**Всього прогнозованих клієнтів:** {len(data)}")
                st.markdown(
                    f"**Прогнозований Відтік:** {churn_counts.get('Відтік', 0)}"
                )
                st.markdown(
                    f"**Прогнозовано Залишаться:** {churn_counts.get('Залишиться', 0)}"
                )

        except Exception as e:
            st.error(f"❌ Помилка при читанні файлу: {e}")


def analytics_page():
    """
    Реалізує сторінку з візуалізацією аналітики роботи моделі.
    !!! ЗАГЛУШКА: Використовуються імітовані дані. !!!
    """
    st.header("📈 Аналітика та Оцінка Роботи Моделі")
    st.info(
        "Цей розділ демонструє **ІМІТОВАНІ** метрики. Будь ласка, замініть їх на реальні дані з вашого тестового набору."
    )
    st.divider()

    st.subheader("Ключові Метрики Оцінки")
    metrics_data = {
        "Метрика": [
            "Accuracy",
            "Precision (Відтік)",
            "Recall (Відтік)",
            "F1-Score (Відтік)",
            "AUC-ROC",
        ],
        "Значення": ["0.85", "0.78", "0.75", "0.76", "0.92"],
    }
    st.dataframe(pd.DataFrame(metrics_data).set_index("Метрика"))

    st.markdown(
        """
    ---
    **Пояснення:**
    - **Accuracy (Точність)**: Загальна частка правильних прогнозів.
    - **Precision (Точність)**: З усіх клієнтів, яких ми передбачили як 'Відтік', скільки справді пішло.
    - **Recall (Повнота)**: З усіх клієнтів, які справді пішли, скільки ми змогли визначити.
    - **AUC-ROC**: Міра роздільної здатності моделі. Значення понад 0.9 вважається відмінним.
    """
    )

    st.subheader("Візуалізації Роботи Моделі")

    # Приклад для імітації Матриці Плутанини
    st.text("Матриця Плутанини (на тестових даних):")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    # Імітація даних Матриці Плутанини
    cm = np.array([[850, 50], [100, 250]])
    ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(x=j, y=i, s=cm[i, j], va="center", ha="center", fontsize=18)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Залишиться (Прогноз)", "Відтік (Прогноз)"])
    ax_cm.set_yticklabels(["Залишиться (Факт)", "Відтік (Факт)"])
    ax_cm.set_title("Матриця Плутанини (Імітація)")
    st.pyplot(fig_cm)


# ==============================================================================
# ОСНОВНА СТРУКТУРА STREAMLIT
# ==============================================================================

# Бокова панель
st.sidebar.title("🛠 Навігація Проєктом")

# ============ ЗАГЛУШКА ДЛЯ ЛОГОТИПУ (БОКОВА ПАНЕЛЬ) ============
try:
    # Шлях до логотипу
    st.sidebar.image(
        "app/assets/logo.png", use_column_width=True, caption="Прогноз Відтоку"
    )
except:
    st.sidebar.markdown("### 📊 Прогноз Відтоку")
    st.sidebar.caption("*(💡 Місце для логотипу)*")
# ===============================================================

menu_options = {
    "Ручний Ввід": "single_predict",
    "Завантаження Файлу": "batch_predict",
    "Аналітика Моделі": "model_analytics",
}

# Меню вибору вкладки
selection = st.sidebar.radio("Оберіть розділ:", list(menu_options.keys()))

st.sidebar.divider()
st.sidebar.caption("Проєкт Python DS & ML Курсу")

# Відображення обраної вкладки
if selection == "Ручний Ввід":
    manual_input_page(model, scaler)
elif selection == "Завантаження Файлу":
    file_upload_page(model, scaler)
elif selection == "Аналітика Моделі":
    analytics_page()

st.caption("© 2025 Аналітична команда з прогнозування відтоку клієнтів 📡")
