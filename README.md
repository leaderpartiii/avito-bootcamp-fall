# Avito Bot Detection

Проверялись временные и diversity-признаки, LightGBM, CatBoost, логистическая регрессия, TF-IDF по User-Agent, нейросети и различные ансамбли.

Лучший результат показал XGBoost с гиперпараметрами, подобранными Optuna:

Precision@Recall>=0.70 = 0.59375

Константный ответ: 0.09391

Основной код находится в `quickstart.ipynb`. Разведочный анализ данных с графиками находится в ноутбуке `eda.ipynb`. После Run All создаётся submission_xgb.csv и submission.csv.
Для установки достаточно написать poetry install.
