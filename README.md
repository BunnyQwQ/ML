# ML / Deep Learning — портфолио

Учебное портфолио: реализации ключевых алгоритмов машинного и глубокого обучения
**с нуля** (numpy / PyTorch), плюс прикладные проекты на реальных данных.

## 🧠 Deep Learning с нуля

Сквозная линия обучения — **от скалярного автограда до трансформера**. Каждый
следующий проект опирается на идеи предыдущего:

```
micrograd  →  makemore  →  mlp  →  mini-GPT
(autograd)   (bigram)     (MLP)   (transformer)
```

| Проект | Что реализовано | Ключевой результат |
|--------|-----------------|--------------------|
| [micrograd](DL/micrograd/) | Автоград-движок: класс `Value`, построение вычислительного графа, `backward()` через топологическую сортировку | Backprop с нуля, без фреймворков |
| [makemore](DL/makemore/) | Bigram-модель генерации имён: статистическая (подсчёт частот) и нейросетевая версии | Два подхода к одной задаче |
| [mlp](DL/mlp/) | MLP-языковая модель: обучаемые embeddings, контекст, BatchNorm | Test loss **2.193** (BatchNorm) |
| [mini-GPT](DL/mini_gpt/) | Decoder-only трансформер: self-attention, multi-head, позиционные эмбеддинги — вручную | GPT-архитектура без `nn.Transformer` |
| [res_net](DL/res_net/) | ResNet-18 с нуля для CIFAR-10: residual-блоки, аугментации, cosine LR, early stopping | Test accuracy **93.99%** |

## 🎮 Reinforcement Learning

Обучение с подкреплением — от табличных методов до deep RL. См. индекс: [RL/](RL/)

_Раздел в активной разработке._

## 🤖 LLM

Прикладная работа с большими языковыми моделями (fine-tuning, RAG, агенты) —
развитие темы из `mini_gpt`. См. индекс: [LLM/](LLM/)

_Раздел в активной разработке._

## 📊 Classic ML с нуля

| Проект | Что внутри |
|--------|-----------|
| [ml_from_scratch](ml_from_scratch/) | 7 классических моделей на чистом NumPy (linear/softmax regression, KNN, decision tree, random forest, SVM, gradient boosting) + [бенчмарк](ml_from_scratch/benchmark.py), сравнивающий их метрики и время обучения на едином датасете |

## 📓 Прикладные проекты

Работа с реальными данными — ноутбуки в папке [notebooks/](notebooks/):

| Ноутбук | Задача |
|---------|--------|
| [price_prediction](notebooks/price_prediction.ipynb) | Регрессия — предсказание цены |
| [nlp_content_moderation](notebooks/nlp_content_moderation.ipynb) | Модерация текста: TF-IDF / fastText / логистическая регрессия |
| [pointnet_3d_classification](notebooks/pointnet_3d_classification.ipynb) | Классификация 3D-облаков точек (PointNet-подход, ModelNet, TensorFlow/Keras) |
| [titanic_analysis](notebooks/titanic_analysis.ipynb) | Разведочный анализ и предсказание выживаемости (Titanic) |

## 🛠 Стек

`Python` · `NumPy` · `PyTorch` · `scikit-learn` · `pandas` · `matplotlib` ·
`TensorFlow/Keras` (3D) · `Jupyter`

## 🚀 Запуск

```bash
pip install -r requirements.txt

# пример: бенчмарк классических моделей
python ml_from_scratch/benchmark.py

# пример: обучение ResNet-18 на CIFAR-10
python DL/res_net/res_net18.py
```

У большинства подпроектов есть собственный README с деталями архитектуры,
результатами и ключевыми выводами — переходи по ссылкам в таблицах выше.

## 📄 Лицензия

[MIT](LICENSE)
