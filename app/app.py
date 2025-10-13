import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.preprocessing import prepare_input_data
from PIL import Image
import matplotlib.pyplot as plt

# ============ Налаштування сторінки ============
st.set_page_config(
    page_title="Прогноз Відтоку Клієнтів",
    page_icon="📉",
    layout="centered",
)

# ============ Логотип ============
try:
    logo = Image.open("app/assets/logo.png")
    st.image(logo, width=150)
except:
    st.title("📊 Прогнозування Відтоку Клієнтів")

st.markdown("### 👥 Аналітична система прогнозування поведінки клієнтів телеком-компанії")
st.divider()

# ============ Завантаження моделі ============
@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except:
    st.warning("⚠️ Модель або scaler поки що не завантажені. Використовується демо-режим.")
    model, scaler = None, None

# ============ Ввід користувацьких даних ============
st.header("🔧 Введіть дані клієнта:")

col1, col2 = st.columns(2)

with col1:
    is_tv_subscriber = st.selectbox("Підписка на ТБ", ["Так", "Ні"])
    is_movie_package_subscriber = st.selectbox("Пакет фільмів", ["Так", "Ні"])
    subscription_age = st.number_input("Тривалість підписки (років)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    bill_avg = st.number_input("Середній рахунок ($)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)

with col2:
    reamining_contract = st.number_input("Залишок контракту (років)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    service_failure_count = st.number_input("Кількість збоїв сервісу", min_value=0, max_value=20, value=0)
    download_avg = st.number_input("Середнє завантаження (GB)", min_value=0.0, max_value=5000.0, value=50.0)
    upload_avg = st.number_input("Середнє відвантаження (GB)", min_value=0.0, max_value=500.0, value=5.0)
    download_over_limit = st.selectbox("Перевищення ліміту завантаження", ["Так", "Ні"])

st.divider()

# ============ Підготовка даних ============
input_data = prepare_input_data(
    is_tv_subscriber,
    is_movie_package_subscriber,
    subscription_age,
    bill_avg,
    reamining_contract,
    service_failure_count,
    download_avg,
    upload_avg,
    download_over_limit
)

# ============ Прогноз ============
if st.button("🔮 Прогнозувати"):
    if model is None:
        st.error("❌ Модель не завантажена. Додайте 'best_model.pkl' і 'scaler.pkl' у папку models/")
    else:
        X_scaled = scaler.transform(input_data)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        st.subheader("📈 Результат прогнозу:")
        if prediction == 1:
            st.error(f"Клієнт **ймовірно піде** 😔 (ймовірність відтоку: {probability:.2%})")
        else:
            st.success(f"Клієнт **ймовірно залишиться** 😊 (ймовірність відтоку: {probability:.2%})")

        # Додаткова візуалізація
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(["Залишиться", "Відтік"], [1 - probability, probability], color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Ймовірність")
        st.pyplot(fig)

st.caption("© 2025 Аналітична команда з прогнозування відтоку клієнтів 📡")