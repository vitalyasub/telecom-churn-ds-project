# Використовуємо офіційний Python образ
FROM python:3.10-slim

# Встановлюємо робочу директорію
WORKDIR /app

# Копіюємо файли у контейнер
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо решту коду
COPY . .

# Відкриваємо порт для Streamlit
EXPOSE 8501

# Команда для запуску Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]