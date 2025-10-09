🧠 Проєкт: Прогнозування відтоку клієнтів для телекомунікаційної компанії
📄 Опис проєкту

Мета цього проєкту — розробити модель машинного навчання для прогнозування ймовірності відтоку клієнтів телекомунікаційної компанії на основі історичних даних.
Проєкт реалізовано в рамках курсу Python Data Science & Machine Learning.

Модель допомагає компанії завчасно визначати клієнтів із високою ймовірністю відтоку та вживати превентивних заходів для їх утримання.

👥 Склад команди
Роль	Ім’я	Завдання
Team Lead	Віталій Субботін	Координація роботи команди, інтеграція результатів, контейнеризація
Scrum Master	Наталія Калашнікова	Організація командної роботи, планування, презентація
Data Analyst (EDA)	Андрій Деренговський	Аналіз даних, виявлення закономірностей
Feature Engineer	Володимир	Підготовка даних, створення ознак
ML Engineer	Михайло Обухов	Розробка та навчання моделей
Visualization Specialist	Анастасія Полякова	Візуалізація результатів, підготовка звіту
🗂️ (Попередня)Структура репозиторію
telecom-churn-ds-project/
├── data/                    # Вхідні дані
│   └── internet_service_churn.csv
├── notebooks/               # Ноутбуки учасників
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Visualization.ipynb
│   └── final_project.ipynb
├── app/                     # Streamlit / FastAPI застосунок
│   └── app.py
├── docker/                  # Файли контейнеризації
│   ├── Dockerfile
│   └── docker-compose.yml
├── reports/                 # Звіти, презентації
│   └── presentation.pptx
├── churn_model.pkl          # Збережена модель
├── requirements.txt          # Залежності
└── README.md                 # Опис проєкту

🔬 Основні етапи реалізації

EDA (Exploratory Data Analysis) — аналіз даних, виявлення пропусків, візуалізація.

Data Preprocessing & Feature Engineering — обробка пропусків, кодування, масштабування.

Model Training & Evaluation — побудова моделей (Logistic Regression, Random Forest, XGBoost), оцінка метрик.

Visualization & Interpretation — створення графіків, пояснення результатів.

Integration & Deployment — інтеграція моделі в застосунок (Streamlit / FastAPI).

Containerization — Dockerfile, docker-compose.

Documentation & Presentation — підготовка звіту та фінальної презентації.

⚙️ Встановлення залежностей
pip install -r requirements.txt

▶️ Запуск проєкту

Запуск застосунку локально:

python app/app.py


Запуск через Docker:

docker-compose up --build

📊 Метрики оцінювання моделі

Accuracy

Precision

Recall

F1-score

ROC-AUC

🧩 Технологічний стек

Python (Pandas, NumPy, Scikit-learn, XGBoost)

Matplotlib, Seaborn

Jupyter Notebook

Streamlit / FastAPI

Docker, GitHub

📈 Результат

Побудована модель передбачення відтоку клієнтів

Веб-застосунок для введення параметрів клієнта та прогнозу відтоку

Повна контейнеризація проєкту

Презентація та технічна документація