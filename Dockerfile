# ================================================================
# 📦 Dockerfile для Streamlit-додатку "Прогноз Відтоку Клієнтів"
# ================================================================

# 1️⃣ Базовий образ з Python
FROM python:3.11-slim

# 2️⃣ Встановлення системних бібліотек (для pandas, numpy, lightgbm)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Робоча директорія
WORKDIR /app

# 4️⃣ Копіюємо файли проєкту
COPY requirements.txt .
COPY app.py .
COPY models ./models
COPY data ./data

# 5️⃣ Встановлюємо Python-залежності
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Виставляємо порт Streamlit
EXPOSE 8501

# 7️⃣ Запускаємо застосунок
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
