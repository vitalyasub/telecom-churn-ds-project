# ===============================================================
# Прогноз Відтоку Клієнтів — Streamlit App
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
from math import pi
from datetime import datetime

# ---------------------------------------------------------------
# 1. Налаштування сторінки
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Прогноз Відтоку Клієнтів",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.style.use("ggplot")

# ---------------------------------------------------------------
# 2. Кастомні Стилі CSS
# ---------------------------------------------------------------
st.markdown(
    """
<style>
    :root {
        --primary-dark: #324851;
        --primary-green: #86AC41;
        --secondary-teal: #34675C;
        --accent-light: #7DA3A1;
        --white: #ffffff;
        --error: #dc3545;
        --warning: #ffc107;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--primary-dark) !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #324851 0%, #34675C 100%);
    }

    [data-testid="stSidebar"] * {
        color: var(--white) !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }

    [data-testid="stSidebar"] .stRadio > div:hover {
        background-color: rgba(255, 255, 255, 0.2);
    }

    .stButton {
        display: flex !important;
        justify-content: center !important;
        margin: 2rem auto !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary-green) 0%, var(--secondary-teal) 100%);
        color: var(--white);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--secondary-teal) 0%, var(--primary-green) 100%);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--accent-light) 0%, var(--secondary-teal) 100%);
        color: var(--white);
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, var(--secondary-teal) 0%, var(--accent-light) 100%);
        transform: translateY(-2px);
    }

    [data-testid="stMetricValue"] {
        color: var(--primary-dark);
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--secondary-teal);
        font-weight: 600 !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-green) 0%, var(--secondary-teal) 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# 3. Константи
# ---------------------------------------------------------------
FEATURE_NAMES_MAP = {
    "reamining_contract": "Залишок контракту",
    "subscription_age": "Тривалість підписки",
    "service_failure_count": "Кількість збоїв сервісу",
    "bill_avg": "Середній рахунок",
    "download_avg": "Середнє завантаження",
    "upload_avg": "Середнє відвантаження",
    "is_tv_subscriber": "Підписка на ТБ",
    "is_movie_package_subscriber": "Пакет фільмів",
    "download_over_limit": "Перевищення ліміту завантаження",
}

FIELD_TOOLTIPS = {
    "subscription_age": "Тривалість активної підписки клієнта у роках",
    "bill_avg": "Середньомісячний рахунок клієнта в доларах США",
    "reamining_contract": "Кількість років до закінчення поточного контракту",
    "service_failure_count": "Загальна кількість технічних збоїв",
    "download_avg": "Середній місячний обсяг завантажених даних у ГБ",
    "upload_avg": "Середній місячний обсяг відправлених даних у ГБ",
}


# ---------------------------------------------------------------
# 4. Завантаження моделі
# ---------------------------------------------------------------
@st.cache_resource
def load_model() -> Tuple[Any, Any]:
    """
    Завантажує збережену модель машинного навчання (LightGBM) та скейлер (Scaler)
    з файлової системи.

    Ця функція кешується Streamlit'ом, щоб уникнути повторного завантаження.

    Returns:
        Tuple[Any, Any]: Кортеж, що містить завантажену модель та скейлер.
                         Повертає (None, None) у разі помилки або відсутності файлів.
    """
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
        st.error(f"❌ Файли моделі не знайдені: {model_path}, {scaler_path}")

    return model, scaler


model, scaler = load_model()
MODEL_LOADED = model is not None and scaler is not None


# ---------------------------------------------------------------
# 5. Допоміжні функції
# ---------------------------------------------------------------
def align_features(input_df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    """
    Вирівнює стовпці вхідного DataFrame відповідно до очікуваних ознак
    скейлера (або моделі) і заповнює відсутні стовпці нулями.

    Це критично важливо для коректного застосування моделі.

    Args:
        input_df (pd.DataFrame): Вхідний DataFrame з даними клієнта.
        scaler (Any): Об'єкт скейлера (наприклад, StandardScaler) з атрибутом
                      `feature_names_in_` або іншим способом отримання очікуваних ознак.

    Returns:
        pd.DataFrame: DataFrame з вирівняними стовпцями.
    """
    expected = getattr(scaler, "feature_names_in_", None)
    if expected is not None:
        input_df = input_df.reindex(columns=expected, fill_value=0)
    return input_df


def prepare_input_data(
    is_tv, is_movie, sub_age, bill, contract, failures, down, up, over_limit
):
    """
    Формує DataFrame з однієї стрічки на основі вхідних даних користувача.
    Виконує кодування бінарних ознак ("Так"/"Ні" в 1/0).

    Args:
        is_tv (str): Чи є підписка на ТБ ("Так" або "Ні").
        is_movie (str): Чи є пакет фільмів ("Так" або "Ні").
        sub_age (float): Тривалість підписки (років).
        bill (float): Середній рахунок ($).
        contract (float): Залишок контракту (років).
        failures (int): Кількість збоїв сервісу.
        down (float): Середнє завантаження (GB).
        up (float): Середнє відвантаження (GB).
        over_limit (str): Чи було перевищення ліміту завантаження ("Так" або "Ні").

    Returns:
        pd.DataFrame: DataFrame, готовий для стандартизації та прогнозування.
    """
    data = {
        "is_tv_subscriber": 1 if is_tv == "Так" else 0,
        "is_movie_package_subscriber": 1 if is_movie == "Так" else 0,
        "download_over_limit": 1 if over_limit == "Так" else 0,
        "subscription_age": sub_age,
        "bill_avg": bill,
        "reamining_contract": contract,
        "service_failure_count": failures,
        "download_avg": down,
        "upload_avg": up,
    }
    return pd.DataFrame([data])


def get_risk_category(probability: float) -> Tuple[str, str, str]:
    """
    Визначає категорію ризику відтоку клієнта на основі ймовірності.

    Args:
        probability (float): Ймовірність відтоку клієнта (значення від 0 до 1).

    Returns:
        Tuple[str, str, str]: Кортеж, що містить:
                                1. Мітку ризику (напр., "Високий ризик").
                                2. Тип стану Streamlit (напр., "error").
                                3. Колір (HEX, напр., "#dc3545").
    """
    if probability >= 0.7:
        return "🔴 Високий ризик", "error", "#dc3545"
    elif probability >= 0.4:
        return "🟡 Середній ризик", "warning", "#ffc107"
    else:
        return "🟢 Низький ризик", "success", "#86AC41"


def generate_recommendations(input_data: dict, probability: float) -> list:
    """
    Генерує персоналізовані рекомендації щодо утримання клієнта на основі
    його вхідних параметрів та прогнозованої ймовірності відтоку.

    Args:
        input_data (Dict[str, Any]): Словник з ключовими ознаками клієнта.
        probability (float): Прогнозована ймовірність відтоку.

    Returns:
        List[str]: Список рядків з рекомендаціями.
    """
    recommendations = []

    if input_data["reamining_contract"] < 0.5:
        recommendations.append(
            "⚠️ **Термінова дія:** Контракт закінчується. Запропонуйте продовження."
        )

    if input_data["service_failure_count"] > 3:
        recommendations.append(
            "🔧 **Технічна підтримка:** Висока кількість збоїв. Потрібна консультація."
        )

    if input_data["bill_avg"] > 100 and probability > 0.5:
        recommendations.append(
            "💰 **Фінансова пропозиція:** Клієнт з високим чеком під ризиком. Розгляньте знижку."
        )

    if input_data["download_over_limit"] == 1:
        recommendations.append(
            "📊 **Оптимізація тарифу:** Перевищення ліміту. Запропонуйте більший тариф."
        )

    if (
        not input_data["is_tv_subscriber"]
        and not input_data["is_movie_package_subscriber"]
    ):
        recommendations.append(
            "📺 **Додаткові послуги:** Запропонуйте пробний період ТВ або кінопакету."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "✅ Клієнт має стабільний профіль. Підтримуйте якість обслуговування."
        )

    return recommendations


# ---------------------------------------------------------------
# 6. СТОРІНКА 1: Ручне прогнозування
# ---------------------------------------------------------------
def manual_input_page(model, scaler):
    """
    Відображає сторінку для ручного введення даних клієнта, виконання
    прогнозу відтоку та відображення результатів і рекомендацій.

    Args:
        model (Any): Завантажена модель ML.
        scaler (Any): Завантажений скейлер для нормалізації даних.
    """
    st.markdown("## 🎯 Індивідуальний Прогноз Відтоку")
    st.markdown("---")

    if not MODEL_LOADED:
        st.warning("⚠️ Модель не завантажена.")
        return

    with st.expander("ℹ️ Інструкція"):
        st.markdown(
            """
        1. Заповніть поля з даними клієнта
        2. Натисніть "Прогнозувати"
        3. Отримайте аналіз та рекомендації
        """
        )

    st.subheader("🔧 Введіть дані клієнта")
    col1, col2 = st.columns(2)

    with col1:
        is_tv = st.selectbox("📺 Підписка на ТБ", ["Так", "Ні"])
        is_movie = st.selectbox("🎬 Пакет фільмів", ["Так", "Ні"])
        sub_age = st.number_input(
            "⏱️ Тривалість підписки (років)",
            0.0,
            20.0,
            2.0,
            0.1,
            help=FIELD_TOOLTIPS["subscription_age"],
        )
        bill = st.number_input(
            "💵 Середній рахунок ($)",
            0.0,
            500.0,
            25.0,
            1.0,
            help=FIELD_TOOLTIPS["bill_avg"],
        )

    with col2:
        contract = st.number_input(
            "📋 Залишок контракту (років)",
            0.0,
            10.0,
            1.0,
            0.1,
            help=FIELD_TOOLTIPS["reamining_contract"],
        )
        failures = st.number_input(
            "⚠️ Кількість збоїв сервісу",
            0,
            20,
            0,
            help=FIELD_TOOLTIPS["service_failure_count"],
        )
        down = st.number_input(
            "⬇️ Середнє завантаження (GB)",
            0.0,
            5000.0,
            50.0,
            help=FIELD_TOOLTIPS["download_avg"],
        )
        up = st.number_input(
            "⬆️ Середнє відвантаження (GB)",
            0.0,
            500.0,
            5.0,
            help=FIELD_TOOLTIPS["upload_avg"],
        )
        over_limit = st.selectbox("🚫 Перевищення ліміту", ["Так", "Ні"])

    if st.button("🔮 Прогнозувати"):
        progress_bar = st.progress(0)
        status = st.empty()

        status.text("⏳ Підготовка...")
        progress_bar.progress(25)
        time.sleep(0.3)

        input_df = prepare_input_data(
            is_tv, is_movie, sub_age, bill, contract, failures, down, up, over_limit
        )

        status.text("🔄 Прогнозування...")
        progress_bar.progress(50)
        time.sleep(0.3)

        try:
            input_df_aligned = align_features(input_df, scaler)
            X_scaled = scaler.transform(input_df_aligned)
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]

            progress_bar.progress(100)
            time.sleep(0.2)
            status.empty()
            progress_bar.empty()

        except Exception as e:
            st.error(f"❌ Помилка: {e}")
            return

        st.markdown("---")
        st.subheader("📈 Результат прогнозу")

        risk_label, risk_type, risk_color = get_risk_category(probability)

        col1, col2, col3 = st.columns(3)
        col1.metric("Статус", "Відтік ❌" if prediction == 1 else "Залишиться ✅")
        col2.metric("Ймовірність відтоку", f"{probability:.1%}")
        col3.markdown(f"### {risk_label}")

        fig, ax = plt.subplots(figsize=(10, 6))
        probs = [1 - probability, probability]
        labels = ["Залишиться", "Відтік"]
        colors = ["#86AC41", "#dc3545"]

        bars = ax.bar(
            labels, probs, color=colors, edgecolor="#324851", linewidth=2, width=0.6
        )
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Ймовірність", fontsize=13, fontweight="bold")
        ax.set_title(
            "Ймовірність Відтоку / Утримання", fontsize=15, fontweight="bold", pad=20
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        for prob, bar in zip(probs, bars):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{prob:.1%}",
                ha="center",
                fontweight="bold",
                fontsize=13,
            )

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("💡 Рекомендації")

        input_dict = {
            "is_tv_subscriber": 1 if is_tv == "Так" else 0,
            "is_movie_package_subscriber": 1 if is_movie == "Так" else 0,
            "subscription_age": sub_age,
            "bill_avg": bill,
            "reamining_contract": contract,
            "service_failure_count": failures,
            "download_avg": down,
            "upload_avg": up,
            "download_over_limit": 1 if over_limit == "Так" else 0,
        }

        for i, rec in enumerate(generate_recommendations(input_dict, probability), 1):
            st.info(f"**{i}.** {rec}")


# ---------------------------------------------------------------
# 7. СТОРІНКА 2: Пакетне прогнозування
# ---------------------------------------------------------------
def batch_upload_page(model, scaler):
    """
    Відображає сторінку для пакетного завантаження CSV-файлу,
    виконання масового прогнозу відтоку та відображення зведених результатів.

    Args:
        model (Any): Завантажена модель ML.
        scaler (Any): Завантажений скейлер для нормалізації даних.
    """
    st.markdown("## 📤 Пакетне Прогнозування")
    st.markdown("---")
    st.info("Завантажте CSV файл для масового прогнозу.")

    if not MODEL_LOADED:
        st.warning("⚠️ Модель не завантажена.")
        return

    with st.expander("📋 Завантажити шаблон"):
        template = pd.DataFrame(
            {
                "is_tv_subscriber": [1, 0, 1],
                "is_movie_package_subscriber": [0, 1, 1],
                "subscription_age": [2.5, 5.0, 1.2],
                "bill_avg": [45.0, 78.0, 32.0],
                "reamining_contract": [1.0, 0.5, 2.0],
                "service_failure_count": [2, 5, 1],
                "download_avg": [120.0, 450.0, 80.0],
                "upload_avg": [15.0, 35.0, 10.0],
                "download_over_limit": [0, 1, 0],
            }
        )
        st.download_button(
            "📥 Завантажити",
            template.to_csv(index=False).encode("utf-8"),
            "template.csv",
            "text/csv",
        )

    uploaded = st.file_uploader("Оберіть CSV", type="csv")

    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            st.subheader("📄 Попередній перегляд")
            st.dataframe(data.head(10), use_container_width=True)

            if st.button("🚀 Запустити"):
                progress = st.progress(0)
                status = st.empty()

                status.text("⏳ Обробка...")
                progress.progress(30)
                time.sleep(0.3)

                try:
                    data_prep = align_features(data.copy(), scaler)
                    status.text("🔄 Прогнозування...")
                    progress.progress(60)

                    X_scaled = scaler.transform(data_prep)
                    data["Prediction"] = model.predict(X_scaled)
                    data["Probability"] = model.predict_proba(X_scaled)[:, 1]

                    progress.progress(100)
                    time.sleep(0.2)
                    status.empty()
                    progress.empty()

                except Exception as e:
                    st.error(f"❌ Помилка: {e}")
                    return

                data["Статус"] = data["Prediction"].map({1: "Відтік", 0: "Залишиться"})
                data["Ймовірність"] = data["Probability"].apply(lambda x: f"{x:.2%}")
                data["Ризик"] = data["Probability"].apply(
                    lambda x: get_risk_category(x)[0]
                )

                st.subheader("✅ Результати")
                st.dataframe(
                    data.drop(columns=["Prediction", "Probability"], errors="ignore"),
                    use_container_width=True,
                )

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("📊 Всього", len(data))
                col2.metric("📉 Відтік", sum(data["Prediction"] == 1))
                col3.metric("📈 Утримання", sum(data["Prediction"] == 0))
                col4.metric("🔴 Високий ризик", sum(data["Probability"] >= 0.7))

                st.markdown("---")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(
                    data["Probability"],
                    bins=20,
                    color="#86AC41",
                    edgecolor="#324851",
                    alpha=0.7,
                )
                ax.axvline(
                    0.5, color="#dc3545", linestyle="--", linewidth=2, label="Поріг"
                )
                ax.set_xlabel("Ймовірність відтоку", fontsize=12, fontweight="bold")
                ax.set_ylabel("Кількість", fontsize=12, fontweight="bold")
                ax.set_title("Розподіл ймовірностей", fontsize=14, fontweight="bold")
                ax.legend()
                st.pyplot(fig)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "📥 Завантажити результати",
                    data.to_csv(index=False).encode("utf-8"),
                    f"predictions_{timestamp}.csv",
                    "text/csv",
                )

        except Exception as e:
            st.error(f"❌ Помилка читання: {e}")


# ---------------------------------------------------------------
# 8. СТОРІНКА 3: Аналітика
# ---------------------------------------------------------------
def data_analysis_page():
    """
    Відображає сторінку аналізу даних, зокрема, важливість ознак
    для моделі (Feature Importance).
    """
    st.markdown("## 🔎 Аналіз Факторів Відтоку")
    st.markdown("---")

    st.subheader("💡 Рекомендації")
    st.warning(
        "Сфокусуйтеся на клієнтах з високою важливістю відтоку: пропонуйте продовження контракту та мінімізуйте збої."
    )

    if hasattr(model, "feature_importances_"):
        try:
            fi = pd.Series(
                model.feature_importances_,
                index=getattr(
                    scaler, "feature_names_in_", range(len(model.feature_importances_))
                ),
            )
            fi = fi.sort_values(ascending=False)
            fi_df = fi.reset_index()
            fi_df.columns = ["Ознака", "Важливість"]
            fi_df["Ознака"] = fi_df["Ознака"].replace(FEATURE_NAMES_MAP)
        except:
            fi_df = pd.DataFrame(
                {
                    "Ознака": list(FEATURE_NAMES_MAP.values()),
                    "Важливість": np.linspace(0.35, 0.01, len(FEATURE_NAMES_MAP)),
                }
            )
    else:
        fi_df = pd.DataFrame(
            {
                "Ознака": list(FEATURE_NAMES_MAP.values()),
                "Важливість": np.linspace(0.35, 0.01, len(FEATURE_NAMES_MAP)),
            }
        )

    fi_df = fi_df[~fi_df["Ознака"].str.contains("_missing", case=False, na=False)]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(fi_df)))
    bars = ax.barh(
        fi_df["Ознака"],
        fi_df["Важливість"],
        color=colors,
        edgecolor="#324851",
        linewidth=1.5,
    )
    ax.set_xlabel("Важливість", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ознака", fontsize=12, fontweight="bold")
    ax.set_title("Ключові фактори відтоку", fontsize=14, fontweight="bold", pad=20)
    ax.invert_yaxis()

    for bar in bars:
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------
# 9. СТОРІНКА 4: Оцінка моделі
# ---------------------------------------------------------------
def model_performance_page():
    """
    Відображає сторінку з порівнянням метрик різних моделей ML,
    включаючи таблицю результатів, діаграми "точність vs швидкість",
    радарну діаграму та теплову карту кореляції метрик.
    """
    st.markdown("## 📈 Порівняння Моделей ML")
    st.markdown("---")

    # Завантаження даних
    results_path = "models/models_comparison_summary.csv"
    df = None

    if os.path.exists(results_path):
        try:
            df = pd.read_csv(results_path)
            st.success("✅ Дані завантажено з файлу")

            # Показуємо структуру даних для діагностики
            with st.expander("🔍 Структура даних"):
                st.write("**Стовпці:**", list(df.columns))
                st.dataframe(df.head(3))

        except Exception as e:
            st.error(f"❌ Помилка читання файлу: {e}")
            df = None

    # Якщо файл не завантажився, використовуємо демо-дані
    if df is None:
        st.warning("⚠️ Використовуються демонстраційні дані")
        df = pd.DataFrame(
            {
                "model": [
                    "LightGBM",
                    "RandomForest",
                    "CatBoost",
                    "GradientBoosting",
                    "MLP",
                    "SVC_linear",
                    "LogisticRegression",
                ],
                "train_time_s": [1.25, 16.11, 38.67, 14.46, 3348.95, 1201.85, 0.13],
                "accuracy": [0.947, 0.943, 0.944, 0.940, 0.937, 0.881, 0.878],
                "precision": [0.962, 0.960, 0.961, 0.960, 0.956, 0.876, 0.878],
                "recall": [0.942, 0.937, 0.937, 0.930, 0.930, 0.917, 0.907],
                "f1": [0.952, 0.948, 0.949, 0.945, 0.943, 0.896, 0.892],
                "roc_auc": [0.983, 0.982, 0.982, 0.974, 0.974, 0.936, 0.934],
            }
        )

    # Нормалізація назв стовпців (видаляємо пробіли, переводимо в lowercase)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    )

    # Мапінг можливих варіантів назв стовпців
    column_mapping = {
        "train_time_s": ["train_time_s", "traintime", "training_time", "time"],
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision", "prec"],
        "recall": ["recall", "rec"],
        "f1": ["f1", "f1_score", "f1score"],
        "roc_auc": ["roc_auc", "rocauc", "auc", "roc"],
    }

    # Функція для пошуку правильної назви стовпця
    def find_column(standard_name, alternatives):
        for alt in alternatives:
            if alt in df.columns:
                return alt
        return None

    # Знаходимо відповідні стовпці
    col_map = {}
    for standard, alternatives in column_mapping.items():
        found = find_column(standard, alternatives)
        if found:
            col_map[standard] = found
        else:
            st.warning(
                f"⚠️ Стовпець '{standard}' не знайдено. Графіки можуть бути неповними."
            )

    # Перевіряємо наявність необхідних стовпців
    required_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    missing_cols = [col for col in required_cols if col not in col_map]

    if missing_cols:
        st.error(f"❌ Відсутні обов'язкові стовпці: {missing_cols}")
        st.info(
            "💡 Очікувані назви стовпців: model, accuracy, precision, recall, f1, roc_auc, train_time_s (опціонально)"
        )
        return

    st.subheader("📊 Загальна таблиця результатів")
    st.dataframe(df, use_container_width=True)

    # ГРАФІК 1: Scatter plot - Точність vs Швидкість (якщо є train_time)
    if "train_time_s" in col_map:
        st.markdown("---")
        st.subheader("⚡ Баланс між Точністю та Швидкістю Навчання")

        fig1, ax1 = plt.subplots(figsize=(12, 7))

        time_col = col_map["train_time_s"]
        acc_col = col_map["accuracy"]
        f1_col = col_map["f1"]
        roc_col = col_map["roc_auc"]

        scatter = ax1.scatter(
            df[time_col],
            df[acc_col],
            s=df[f1_col] * 500,
            c=df[roc_col],
            cmap="RdYlGn",
            alpha=0.7,
            edgecolors="#324851",
            linewidth=2,
        )

        ax1.set_xscale("log")
        ax1.set_xlabel(
            "Час навчання (секунди, лог-шкала)", fontsize=13, fontweight="bold"
        )
        ax1.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
        ax1.set_title(
            "Порівняння моделей: Точність vs Швидкість навчання",
            fontsize=15,
            fontweight="bold",
            pad=20,
        )
        ax1.grid(True, alpha=0.3)

        # Додаємо назви моделей
        model_col = "model" if "model" in df.columns else df.columns[0]
        for i, model in enumerate(df[model_col]):
            ax1.annotate(
                model,
                (df[time_col].iloc[i], df[acc_col].iloc[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("ROC AUC", fontsize=11, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig1)

        st.info(
            "💡 **Розмір точок** відповідає F1-score. **Колір** показує ROC AUC (зелений = кращий)."
        )
    else:
        st.info(
            "ℹ️ Графік 'Точність vs Швидкість' недоступний (відсутні дані про час навчання)"
        )

    # ГРАФІК 2: Radar Chart - Багатовимірне порівняння топ-3 моделей
    st.markdown("---")
    st.subheader("🎯 Багатовимірне Порівняння Топ-3 Моделей")

    # Відбираємо топ-3 за F1
    f1_col = col_map["f1"]
    top3 = df.nlargest(3, f1_col)

    categories = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]

    fig2, ax2 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]

    colors_radar = ["#86AC41", "#34675C", "#7DA3A1"]

    model_col = "model" if "model" in df.columns else df.columns[0]

    for idx, (i, row) in enumerate(top3.iterrows()):
        values = [
            row[col_map["accuracy"]],
            row[col_map["precision"]],
            row[col_map["recall"]],
            row[col_map["f1"]],
            row[col_map["roc_auc"]],
        ]
        values += values[:1]

        ax2.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=row[model_col],
            color=colors_radar[idx],
        )
        ax2.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax2.set_ylim(0.85, 1.0)
    ax2.set_title(
        "Порівняння метрик найкращих моделей", fontsize=15, fontweight="bold", pad=30
    )
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(fig2)

    # ГРАФІК 3: Heatmap кореляції метрик
    st.markdown("---")
    st.subheader("🔥 Кореляція між Метриками")

    metrics_cols = [
        col_map[key] for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ]
    metrics_df = df[metrics_cols]

    # Перейменовуємо для кращого відображення
    metrics_df.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    corr = metrics_df.corr()

    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax3,
        vmin=-1,
        vmax=1,
    )
    ax3.set_title(
        "Матриця кореляції метрик якості", fontsize=15, fontweight="bold", pad=20
    )

    plt.tight_layout()
    st.pyplot(fig3)

    st.info(
        "💡 Сильна кореляція між метриками вказує на узгодженість оцінки якості моделей."
    )

    # Висновок
    st.markdown("---")
    st.subheader("📊 Висновки")

    f1_col = col_map["f1"]
    model_col = "model" if "model" in df.columns else df.columns[0]

    best_model = df.loc[df[f1_col].idxmax()]

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            f"""
        **🏆 Найкраща модель за якістю:**
        **{best_model[model_col]}**
        - F1-Score: {best_model[f1_col]:.3f}
        - ROC AUC: {best_model[col_map['roc_auc']]:.3f}
        - Accuracy: {best_model[col_map['accuracy']]:.3f}
        """
        )

    with col2:
        if "train_time_s" in col_map:
            time_col = col_map["train_time_s"]
            fastest_model = df.loc[df[time_col].idxmin()]
            st.info(
                f"""
            **⚡ Найшвидша модель:**
            **{fastest_model[model_col]}**
            - Час навчання: {fastest_model[time_col]:.2f}s
            - F1-Score: {fastest_model[f1_col]:.3f}
            - ROC AUC: {fastest_model[col_map['roc_auc']]:.3f}
            """
            )
        else:
            st.info(
                """
            **ℹ️ Інформація про швидкість:**
            Дані про час навчання відсутні у файлі результатів.
            """
            )

    # Додаткова статистика
    st.markdown("---")
    st.subheader("📈 Загальна Статистика")

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    col_stat1.metric(
        "Середній F1-Score", f"{df[f1_col].mean():.3f}", f"±{df[f1_col].std():.3f}"
    )

    col_stat2.metric(
        "Середній ROC AUC",
        f"{df[col_map['roc_auc']].mean():.3f}",
        f"±{df[col_map['roc_auc']].std():.3f}",
    )

    col_stat3.metric("Кількість моделей", len(df))

    col_stat4.metric(
        "Найкраща метрика",
        f"{df[f1_col].max():.3f}",
        f"+{(df[f1_col].max() - df[f1_col].min()):.3f}",
    )


# =====================================================================
# 10. ГОЛОВНА НАВІГАЦІЯ
# =====================================================================
if __name__ == "__main__":
    # ЗАГАЛЬНИЙ ЗАГОЛОВОК ДЛЯ ВСІХ ВКЛАДОК
    st.sidebar.markdown("# 📊 Прогноз Відтоку")
    st.sidebar.markdown(
        "### Аналітична система прогнозування поведінки клієнтів телеком-компанії"
    )
    st.sidebar.markdown("---")

    # Меню навігації
    st.sidebar.title("🧭 Меню")
    main_menu = st.sidebar.radio(
        "Оберіть розділ",
        ["Ручний ввід", "Пакетний ввід", "Аналітика та Звіти"],
        index=0,
    )

    # Маршрутизація
    if main_menu == "Ручний ввід":
        manual_input_page(model, scaler)

    elif main_menu == "Пакетний ввід":
        batch_upload_page(model, scaler)

    else:
        st.markdown("## 📈 Аналітичний Портал")
        st.caption("Виберіть тип аналітики для детального огляду")
        st.markdown("---")

        analytic_tab = st.selectbox(
            "Виберіть тип аналітики",
            ["Огляд Даних та Факторів Відтоку", "Порівняння Моделей ML"],
        )

        if analytic_tab == "Огляд Даних та Факторів Відтоку":
            data_analysis_page()
        elif analytic_tab == "Порівняння Моделей ML":
            model_performance_page()

    # Футер
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 Predictive Minds Lab 📡")
    st.sidebar.caption("Версія: 1.0")

    # Інформація про модель
    if MODEL_LOADED:
        st.sidebar.success("🟢 Модель активна")
    else:
        st.sidebar.error("🔴 Модель не завантажена")
