# ----------------------------------------------------------
# 📦 Dockerfile для Streamlit застосунку "Прогноз Відтоку Клієнтів"
# ----------------------------------------------------------
FROM python:3.11-slim

# 1️⃣ Робоча директорія
WORKDIR /app

# 2️⃣ Системні бібліотеки для numpy/pandas/lightgbm
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Копіюємо файли
COPY requirements.txt .
COPY app.py .
COPY models ./models

# 4️⃣ Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Відкриваємо порт Streamlit
EXPOSE 8501

# 6️⃣ Команда запуску
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
