"""
Бенчмарк самописных моделей из ml_from_scratch/models.

Скрипт по очереди обучает каждую модель на одном и том же датасете
и печатает её метрики. Отдельный датасет для классификации, отдельный —
для регрессии.

Данные и метрики берутся из sklearn, но САМИ МОДЕЛИ — самописные
(из папки models/). sklearn тут только как источник данных и «линейка».

Запуск (нужен интерпретатор с numpy + sklearn):
    python ml_from_scratch/benchmark.py
"""

import importlib.util
import sys
import time
from pathlib import Path

# Windows-консоль по умолчанию не UTF-8, из-за чего кириллица в выводе
# превращается в кракозябры. Переключаем поток вывода на UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

MODELS_DIR = Path(__file__).parent / "models"
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
#  Загрузка самописных моделей                                                #
# --------------------------------------------------------------------------- #
# Папка models/ не является пакетом (нет __init__.py) и не лежит в sys.path,
# поэтому грузим каждый файл вручную через importlib. Регистрируем модули в
# sys.modules под их именами — чтобы перекрёстные импорты между файлами
# (random_forest импортирует decision_tree) находили друг друга.
def _load(module_name: str, filename: str):
    path = MODELS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # регистрируем ДО exec — для cross-import
    spec.loader.exec_module(module)
    return module


# Порядок важен: decision_tree грузим первым, т.к. random_forest на него ссылается.
_dtree = _load("decision_tree", "decision_tree.py")
_forest = _load("random_forest", "random_forest.py")
_linreg = _load("linear_regression", "linear_regression.py")
_softmax = _load("softmax_regression", "softmax_regression.py")
_knn = _load("knn", "knn.py")
_svm = _load("svm", "svm.py")
_gb = _load("gradient_boosting", "gradient_boosting.py")


# --------------------------------------------------------------------------- #
#  Наборы моделей                                                             #
# --------------------------------------------------------------------------- #
# Каждый элемент — (человекочитаемое имя, фабрика, создающая свежую модель).
# Фабрика (lambda), а не готовый объект: так гарантированно обучаем «с нуля».
def build_classifiers():
    return [
        ("Softmax Regression", lambda: _softmax.SoftmaxRegression(
            max_iter=200, random_state=RANDOM_STATE)),
        ("K-Nearest Neighbors", lambda: _knn.KNNClassifier(n_neighbors=5)),
        ("Decision Tree", lambda: _dtree.DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE)),
        ("Random Forest", lambda: _forest.RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=RANDOM_STATE)),
        ("Support Vector Machine", lambda: _svm.SVM(
            max_iter=500, random_state=RANDOM_STATE)),
        ("Gradient Boosting", lambda: _gb.GBCustomClassifier(
            n_estimators=100, max_depth=3, criterion="squared_error",
            random_state=RANDOM_STATE)),
    ]


def build_regressors():
    return [
        ("Linear Regression", lambda: _linreg.LinearRegression(
            max_iter=1000, eta0=0.01, random_state=RANDOM_STATE)),
        ("K-Nearest Neighbors", lambda: _knn.KNNRegressor(n_neighbors=5)),
        ("Decision Tree", lambda: _dtree.DecisionTreeRegressor(
            max_depth=6, random_state=RANDOM_STATE)),
        ("Random Forest", lambda: _forest.RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=RANDOM_STATE)),
        ("Gradient Boosting", lambda: _gb.GBCustomRegressor(
            n_estimators=100, max_depth=3, criterion="squared_error",
            random_state=RANDOM_STATE)),
    ]


# --------------------------------------------------------------------------- #
#  Подготовка данных                                                          #
# --------------------------------------------------------------------------- #
def prepare_data(load_fn):
    """Загрузка → train/test split → стандартизация признаков.

    Scaler обучается ТОЛЬКО на train (fit_transform), а к test применяется
    уже обученный (transform) — иначе статистика теста «протечёт» в обучение.
    """
    data = load_fn()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
#  Обучение и оценка                                                          #
# --------------------------------------------------------------------------- #
def evaluate_classification(models, data):
    X_train, X_test, y_train, y_test = data
    rows = []
    for name, factory in models:
        model = factory()
        start = time.perf_counter()      # монотонный таймер — только вокруг fit
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "time_s": fit_time,
        })
    return rows


def evaluate_regression(models, data):
    X_train, X_test, y_train, y_test = data
    rows = []
    for name, factory in models:
        model = factory()
        start = time.perf_counter()      # монотонный таймер — только вокруг fit
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "time_s": fit_time,
        })
    return rows


# --------------------------------------------------------------------------- #
#  Печать таблицы                                                             #
# --------------------------------------------------------------------------- #
def print_table(title, rows, columns):
    # Ширину колонки с именем считаем динамически — модели переименованы
    # в длинные «продуктовые» названия, фиксированные 20 символов уже малы.
    name_w = max(len("model"), *(len(r["model"]) for r in rows)) + 2
    col_w = 13
    total_w = name_w + col_w * len(columns)

    print(f"\n{title}")
    print("-" * total_w)
    header = f"{'model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in columns)
    print(header)
    print("-" * total_w)
    for row in rows:
        line = f"{row['model']:<{name_w}}"
        for key in columns:
            line += f"{row[key]:>{col_w}.4f}"
        print(line)
    print("-" * total_w)


def main():
    # ---- Классификация ----
    clf_data = prepare_data(load_breast_cancer)
    clf_rows = evaluate_classification(build_classifiers(), clf_data)
    print_table(
        "КЛАССИФИКАЦИЯ  (breast_cancer, бинарная)",
        clf_rows,
        ["accuracy", "precision", "recall", "f1", "time_s"],
    )

    # ---- Регрессия ----
    reg_data = prepare_data(load_diabetes)
    reg_rows = evaluate_regression(build_regressors(), reg_data)
    print_table(
        "РЕГРЕССИЯ  (diabetes)",
        reg_rows,
        ["r2", "mae", "rmse", "time_s"],
    )


if __name__ == "__main__":
    main()
