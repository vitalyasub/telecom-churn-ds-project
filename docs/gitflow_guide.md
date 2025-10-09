
# 🧭 GitFlow для команди проєкту “Прогнозування відтоку клієнтів”

## 🔹 1. Основна ідея
Гілка **`main`** — стабільна, перевірена версія проєкту.  
Зміни вносяться **тільки через Pull Request (PR)** після рев’ю.  
Кожен учасник працює у **своїй гілці**.

---

## 🔹 2. Створення нової гілки
Після клонування репозиторію:

```bash
git checkout -b feature/назва_задачі_Прізвище
```

📘 Приклади:
```
feature/EDA_Derenhovskyi
feature/Preprocessing_Volodymyr
feature/Model_Obukhov
feature/Visualization_Poliakova
feature/Docs_Kalashnikova
```

---

## 🔹 3. Робота у гілці
1. Зроби зміни у своїх файлах або ноутбуках  
2. Збережи зміни командою:

```bash
git add .
git commit -m "Опис змін (наприклад: Проведено EDA, створено графіки)"
```

3. Відправ гілку на GitHub:
```bash
git push origin feature/EDA_Derenhovskyi
```

---

## 🔹 4. Створення Pull Request (PR)
1. Зайди на GitHub → відкрий репозиторій  
2. Натисни **Compare & pull request**  
3. Вибери:
   - **Base branch:** `main`
   - **Compare:** свою гілку (`feature/EDA_Derenhovskyi`)
4. Напиши короткий опис у полі **Title**  
5. У полі **Reviewers** вибери:  
   🧑‍💼 **Віталій Субботін (Team Lead)** або **Наталія Калашнікова (Scrum Master)**  
6. Натисни **Create pull request**

---

## 🔹 5. Рев’ю та злиття (Merge)
- Після перевірки тімлід натискає **Approve**  
- Потім **Merge pull request**  
- Використовуй **Squash merge** (чистіша історія комітів)

Після merge гілку можна видалити.

---

## 🔹 6. Видалення локальної гілки після merge
```bash
git checkout main
git pull
git branch -d feature/EDA_Derenhovskyi
git push origin --delete feature/EDA_Derenhovskyi
```

---

## 🔹 7. Синхронізація проєкту перед новою роботою
```bash
git checkout main
git pull
```

---

## 💡 Поради
- Один Pull Request = одна логічна задача  
- Не тримай гілку відкритою довго (>2 днів)  
- Додавай зрозумілий опис у комітах  
- Не пуш напряму у `main`  
- Якщо виник конфлікт — повідом тімліда

---

📘 **Рекомендація:**  
Team Lead може додати цей файл у `docs/gitflow_guide.md`,  
щоб кожен мав його під рукою.
