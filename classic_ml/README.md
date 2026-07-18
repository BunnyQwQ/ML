# Classic ML from scratch — модели и бенчмарк

Самописные реализации классических ML-моделей на чистом **NumPy** (без scikit-learn
в самих алгоритмах) в стиле sklearn-API (`fit` / `predict` / `predict_proba`).
Плюс скрипт-бенчмарк, который по очереди обучает каждую модель на одном датасете
и сравнивает метрики и время обучения.

## Модели

Быстрая навигация — каждый файл содержит реализацию алгоритма с нуля:

| Файл | Классы | Задача                                         |
|------|--------|------------------------------------------------|
| [linear_regression.py](models/linear_regression.py) | `LinearRegression` | регрессия (SGD, L1/L2, early stopping)         |
| [softmax_regression.py](models/softmax_regression.py) | `SoftmaxRegression` | мнногоклассовая классификация                  |
| [knn.py](models/knn.py) | `KNNClassifier`, `KNNRegressor` | классификация / регрессия                      |
| [decision_tree.py](models/decision_tree.py) | `DecisionTreeClassifier`, `DecisionTreeRegressor` | классификация / регрессия                      |
| [random_forest.py](models/random_forest.py) | `RandomForestClassifier`, `RandomForestRegressor` | классификация / регрессия (бэггинг)       |
| [svm.py](models/svm.py) | `SVM` | бинарная классификация (hinge / squared-hinge) |
| [gradient_boosting.py](models/gradient_boosting.py) | `GBCustomClassifier`, `GBCustomRegressor` | бинарная классификация / регрессия             |

## Бенчмарк

[benchmark.py](benchmark.py) — обучает все модели на едином датасете и печатает
таблицу метрик и времени обучения (`fit`). Отдельный датасет для классификации,
отдельный для регрессии.

- **Классификация:** `breast_cancer` (569×30, **бинарная**). Датасет именно бинарный,
  потому что `SVM` и `GBCustomClassifier` поддерживают только 2 класса — так все
  6 классификаторов сравниваются на одних и тех же данных.
- **Регрессия:** `diabetes` (442×10).
- **Препроцессинг:** `train_test_split` + `StandardScaler` (обучается только на train,
  чтобы не было утечки из теста).

scikit-learn используется **только** как источник датасетов и метрик — сами модели
берутся из папки [models/](models/).

## Запуск

```bash
python classic_ml/benchmark.py
```

Зависимости: `numpy`, `scikit-learn` (для данных и метрик; `gradient_boosting.py`
внутри тоже использует `sklearn.tree.DecisionTreeRegressor` как слабый ученик).

## Результаты

```
КЛАССИФИКАЦИЯ  (breast_cancer, бинарная)
model                        accuracy    precision       recall           f1
Softmax Regression             0.9825       0.9859       0.9859       0.9859
K-Nearest Neighbors            0.9474       0.9577       0.9577       0.9577
Decision Tree                  0.9298       0.9565       0.9296       0.9429
Random Forest                  0.9649       0.9589       0.9859       0.9722
Support Vector Machine         0.9912       0.9861       1.0000       0.9930
Gradient Boosting              0.9561       0.9583       0.9718       0.9650

РЕГРЕССИЯ  (diabetes)
model                           r2          mae         rmse
Linear Regression           0.4404      43.2030      54.4493
K-Nearest Neighbors         0.4248      42.7775      55.2037
Decision Tree               0.2441      49.9615      63.2844
Random Forest               0.4630      43.3094      53.3402
Gradient Boosting           0.4515      44.7378      53.9098
```

Бенчмарк также выводит колонку `time_s` — время обучения. Полезное наблюдение:
у **KNN** оно ≈ 0 (ленивый алгоритм — `fit` только запоминает данные, вся работа
в `predict`), а **Random Forest** самый медленный (десятки деревьев на чистом NumPy).

## Структура

```
classic_ml/
├── benchmark.py        # обучение всех моделей + таблица метрик
├── models/
│   ├── linear_regression.py
│   ├── softmax_regression.py
│   ├── knn.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── svm.py
│   └── gradient_boosting.py
└── README.md
```
