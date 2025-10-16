# ===============================================================
# 📉 Прогноз Відтоку Клієнтів — Streamlit App (оновлена версія)
# ===============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

# ---------------------------------------------------------------
# 1. Налаштування сторінки
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Прогноз Відтоку Клієнтів",
    page_icon="📊",
    layout="wide"
)
warnings.filterwarnings("ignore", message="Glyph.*missing from font")
plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------
# 2. Завантаження моделі та скейлера
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "models/best_model_LightGBM.pkl"
    scaler_path = "models/scaler.pkl"

    model, scaler = None, None
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            st.success("✅ Модель та скейлер успішно завантажено.")
        except Exception as e:
            st.error(f"❌ Помилка при завантаженні: {e}")
    else:
        st.warning("⚠️ Модель або скейлер не знайдені. Увімкнено демо-режим.")
    return model, scaler


model, scaler = load_model()

# =====================================================================
# 🔧 Універсальна функція для узгодження назв ознак
# =====================================================================
def align_features(input_df, scaler):
    expected = getattr(scaler, "feature_names_in_", None)
    if expected is not None:
        input_df = input_df.reindex(columns=expected, fill_value=0)
    return input_df

# ---------------------------------------------------------------
# 3. Підготовка введених даних
# ---------------------------------------------------------------
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
    return pd.DataFrame([data])

# =====================================================================
# 🧍‍♂️ 4. Ручне прогнозування
# =====================================================================
def manual_input_page(model, scaler):
    try:
        logo = Image.open("app/assets/logo.png")
        st.image(logo, width=150)
    except FileNotFoundError:
        st.markdown("## 📊 Прогнозування Відтоку Клієнтів")
        st.caption("*(💡 Місце для логотипу: app/assets/logo.png)*")

    st.markdown("### 👥 Аналітична система прогнозування поведінки клієнтів телеком-компанії")
    st.divider()

    if model is None or scaler is None:
        st.info("⭐ **ДЕМО-РЕЖИМ:** Прогнози будуть випадковими, оскільки модель не завантажена.")

    st.header("🔧 Введіть дані клієнта:")
    col1, col2 = st.columns(2)

    with col1:
        is_tv_subscriber = st.selectbox("Підписка на ТБ", ["Так", "Ні"])
        is_movie_package_subscriber = st.selectbox("Пакет фільмів", ["Так", "Ні"])
        subscription_age = st.number_input("Тривалість підписки (років)", 0.0, 20.0, 2.0, 0.1)
        bill_avg = st.number_input("Середній рахунок ($)", 0.0, 500.0, 25.0, 1.0)

    with col2:
        reamining_contract = st.number_input("Залишок контракту (років)", 0.0, 10.0, 1.0, 0.1)
        service_failure_count = st.number_input("Кількість збоїв сервісу", 0, 20, 0)
        download_avg = st.number_input("Середнє завантаження (GB)", 0.0, 5000.0, 50.0)
        upload_avg = st.number_input("Середнє відвантаження (GB)", 0.0, 500.0, 5.0)
        download_over_limit = st.selectbox("Перевищення ліміту завантаження", ["Так", "Ні"])

    st.divider()

    if st.button("🔮 Прогнозувати"):
        with st.spinner("Проводимо аналіз даних клієнта..."):
            time.sleep(1.5)

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

        if model is None or scaler is None:
            st.warning("⚠️ Модель не завантажена — використовується ДЕМО прогноз.")
            prediction = np.random.choice([0, 1], p=[0.7, 0.3])
            probability = np.random.uniform(0.1, 0.9)
        else:
            try:
                input_df = align_features(input_df, scaler)
                X_scaled = scaler.transform(input_df)
                prediction = model.predict(X_scaled)[0]
                probability = model.predict_proba(X_scaled)[0][1]
            except Exception as e:
                st.error(f"❌ Помилка при прогнозуванні: {e}")
                st.write("Очікувані ознаки:", getattr(scaler, "feature_names_in_", None))
                st.write("Отримані ознаки:", input_df.columns.tolist())
                return

        st.subheader("📈 Результат прогнозу:")
        if prediction == 1:
            st.error(f"Клієнт **ймовірно піде** 😔 (ймовірність: **{probability:.2%}**)")  
        else:
            st.success(f"Клієнт **ймовірно залишиться** 😊 (ймовірність: **{probability:.2%}**)")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Залишиться", "Відтік"], [1 - probability, probability], color=["#28a745", "#dc3545"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Ймовірність")
        ax.set_title("Ймовірність Відтоку / Утримання")
        st.pyplot(fig)

# =====================================================================
# 📂 5. Пакетне прогнозування CSV
# =====================================================================
def batch_upload_page(model, scaler):
    st.header("📤 Завантаження файлу для пакетного прогнозування")
    st.info("Завантажте **CSV** файл із тими самими ознаками, що використовувались при навчанні моделі.")
    st.divider()

    uploaded_file = st.file_uploader("Оберіть CSV файл", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.subheader("📄 Попередній перегляд даних:")
            st.dataframe(data.head())

            if st.button("🚀 Запустити Прогнозування"):
                with st.spinner("Обробка та прогнозування..."):
                    time.sleep(1)

                if model is None or scaler is None:
                    st.warning("⚠️ Модель не завантажена — використовується ДЕМО прогноз.")
                    data["Prediction"] = np.random.choice([0, 1], len(data))
                    data["Probability"] = np.random.uniform(0.05, 0.95, len(data))
                else:
                    try:
                        data_prepared = align_features(data.copy(), scaler)
                        X_scaled = scaler.transform(data_prepared)
                        data["Prediction"] = model.predict(X_scaled)
                        data["Probability"] = model.predict_proba(X_scaled)[:, 1]
                    except Exception as e:
                        st.error(f"❌ Помилка при прогнозуванні: {e}")
                        st.write("Очікувані:", getattr(scaler, "feature_names_in_", None))
                        st.write("Отримані:", data.columns.tolist())
                        return

                data["Статус"] = data["Prediction"].map({1: "Відтік", 0: "Залишиться"})

                st.subheader("✅ Результати прогнозування:")
                st.dataframe(data)

                csv_out = data.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Завантажити результати (CSV)", csv_out, "churn_predictions.csv", "text/csv")

                st.divider()
                st.markdown(f"**📊 Кількість клієнтів:** {len(data)}")
                st.markdown(f"**📉 Відтік:** {sum(data['Prediction'] == 1)}")
                st.markdown(f"**📈 Залишаються:** {sum(data['Prediction'] == 0)}")

        except Exception as e:
            st.error(f"❌ Помилка при читанні файлу: {e}")

# =====================================================================
# 📈 6. Аналітика моделі
# =====================================================================
def analytics_page():
    st.header("📊 Аналітика Моделі (LightGBM)")
    st.markdown("Оцінка моделі на різних наборах даних:")

    combined = pd.DataFrame({
        "dataset": ["Train", "Test", "Holdout"],
        "accuracy": [0.972, 0.959, 0.954],
        "precision": [0.981, 0.969, 0.962],
        "recall": [0.958, 0.956, 0.947],
        "f1": [0.969, 0.963, 0.955],
        "roc_auc": [0.996, 0.994, 0.991]
    })

    st.dataframe(combined.style.format("{:.3f}"))

    # Побудова графіка
    melted = combined.melt(id_vars="dataset", var_name="metric", value_name="value")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=melted, x="metric", y="value", hue="dataset", palette="coolwarm")
    plt.title("📊 Порівняння якості моделі на Train/Test/Holdout", fontsize=12, fontweight="bold")
    plt.ylabel("Значення метрики")
    plt.ylim(0.9, 1.0)
    plt.legend(title="Дані", loc="lower right")

    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2.,
                p.get_height() + 0.003,
                f"{p.get_height():.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    st.pyplot(plt)

    st.info("""
    ✅ **Висновок:**  
    Модель стабільна, точна і добре узагальнює дані.  
    LightGBM показує найкращі результати серед протестованих моделей.
    """)

# =====================================================================
# 🧭 7. Навігація
# =====================================================================
menu = st.sidebar.radio("🧭 Оберіть розділ", ["Ручний прогноз", "Пакетне прогнозування", "Аналітика"])

if menu == "Ручний прогноз":
    manual_input_page(model, scaler)
elif menu == "Пакетне прогнозування":
    batch_upload_page(model, scaler)
else:
    analytics_page()

st.sidebar.caption("© 2025 Аналітична команда прогнозування відтоку клієнтів 📡")