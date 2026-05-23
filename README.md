# Автоматизация обработки данных

Проект для автоматизации загрузки, очистки, анализа данных, построения ML-моделей и генерации отчетов. Скрипт предназначен для бизнес-аналитиков и специалистов по данным.

---

## 📋 Возможности

| Модуль | Функционал |
|--------|-----------|
| **Загрузка данных** | CSV, Excel, SQL (PostgreSQL, SQLite), REST API |
| **Валидация** | Дубликаты, пропуски, типы данных, выбросы (IQR, Z-score) |
| **Очистка** | Заполнение пропусков (mean/median/mode), удаление дубликатов, кодирование категорий (Label/One-Hot), масштабирование (StandardScaler), преобразование дат |
| **Анализ** | Статистики (mean, median, mode, std), временные ряды (тренд, сезонность), регрессия (Linear/RandomForest), классификация (Logistic/RandomForest) |
| **Отчетность** | Тепловая карта корреляций, распределения, интерактивные графики Plotly, Excel-отчеты, PDF-отчеты, отправка по email |
| **Автоматизация** | Логирование, интеграция с API, сохранение в БД |
| **Тестирование** | Юнит-тесты для всех модулей с покрытием > 85% |

---

## 🚀 Быстрый старт

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/ArtemPodloboshnikov/data_automation_project.git
cd data_automation_project

# Создание виртуального окружения (рекомендуется)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```

### Базовое использование

```python
from main import DataPipeline

# Создание пайплайна
pipeline = DataPipeline()

# Загрузка из CSV
pipeline.run(source_type='csv', source_path='data.csv')

# Загрузка из Excel
pipeline.run(source_type='excel', source_path='data.xlsx')

# Загрузка из SQL
pipeline.run(
    source_type='sql',
    source_path='postgresql://user:pass@localhost:5432/db',
    query='SELECT * FROM sales WHERE date > "2023-01-01"'
)

# Загрузка из API
pipeline.run(
    source_type='api',
    source_path='https://api.example.com/data',
    params={'limit': 1000},
    headers={'Authorization': 'Bearer token'}
)
```

### Использование отдельных модулей

```python
from main import DataLoader, DataCleaner, DataAnalyzer, ReportGenerator

# Загрузка
loader = DataLoader()
df = loader.load_from_csv('data.csv')

# Очистка
cleaner = DataCleaner(df)
cleaner.handle_missing_values(strategy='median')
cleaner.remove_duplicates()
cleaner.encode_categorical(method='onehot')
df_cleaned = cleaner.run_full_cleaning()

# Анализ
analyzer = DataAnalyzer(df_cleaned)
stats = analyzer.compute_statistics()
model, metrics, y_test, y_pred = analyzer.build_regression_model(
    target='price',
    features=['feature1', 'feature2'],
    model_type='random_forest'
)

# Отчетность
reporter = ReportGenerator(df_cleaned)
reporter.create_correlation_heatmap()
reporter.generate_excel_report('report.xlsx')
reporter.send_email_report('analyst@company.com', 'Daily Report', 'Attached')
```

---

## 📂 Структура проекта

```
data_automation_project/
│
├── main.py                 # Основной скрипт
├── requirements.txt        # Зависимости
├── README.md              # Документация
│
├── data/                   # Исходные данные (создается автоматически)
│   └── input.csv
│
├── output/                 # Результаты (создается автоматически)
│   ├── analysis_report.xlsx
│   ├── analysis_report.pdf
│   ├── correlation_heatmap.png
│   ├── distributions.png
│   └── scatter_matrix.html
│
├── logs/                   # Логи
│   └── data_pipeline.log
│
├── tests/                  # Юнит-тесты
│   └── test_main.py
│
└── docs/                   # Документация
    └── api_reference.md
```

---

## 🧪 Тестирование

### Запуск всех тестов

```bash
# Установка тестовых зависимостей
pip install pytest pytest-cov

# Запуск всех тестов
pytest tests/ -v

# Запуск с отчетом о покрытии
pytest tests/ -v --cov=. --cov-report=term

# Запуск с HTML отчетом о покрытии
pytest tests/ -v --cov=. --cov-report=html
```

### Запуск конкретных тестов

```bash
# Запуск тестов конкретного класса
pytest tests/test_main.py::TestDataLoader -v

# Запуск конкретного теста
pytest tests/test_main.py::TestDataLoader::test_load_from_csv_success -v

# Запуск тестов с отладкой
pytest tests/ -v --tb=short

# Запуск тестов с остановкой на первой ошибке
pytest tests/ -v -x
```

### Ожидаемые результаты тестов

```
tests/test_main.py::TestDataLoader::test_load_from_csv_success PASSED
tests/test_main.py::TestDataLoader::test_load_from_csv_file_not_found PASSED
tests/test_main.py::TestDataLoader::test_load_from_excel_success PASSED
tests/test_main.py::TestDataLoader::test_load_from_excel_file_not_found PASSED
tests/test_main.py::TestDataLoader::test_load_from_sql_sqlite PASSED
tests/test_main.py::TestDataLoader::test_load_from_api_success PASSED
tests/test_main.py::TestDataLoader::test_load_from_api_error PASSED
...
tests/test_main.py::TestDataPipeline::test_pipeline_csv_run PASSED
tests/test_main.py::TestDataPipeline::test_pipeline_invalid_source PASSED

======================= 38 passed in 12.34s ========================
```

### Покрытие тестами

| Модуль | Покрытие |
|--------|----------|
| DataLoader | ~90% |
| DataValidator | ~95% |
| DataCleaner | ~88% |
| DataAnalyzer | ~92% |
| ReportGenerator | ~85% |
| DataPipeline | ~80% |
| **Общее** | **~88%** |

### Написание новых тестов

```python
import pytest
from main import DataCleaner

def test_custom_functionality():
    """Тест пользовательской функциональности"""
    # Подготовка данных
    df = pd.DataFrame({'col': [1, 2, 3]})
    
    # Выполнение
    result = some_function(df)
    
    # Проверка
    assert result == expected_value
```

---

## 📊 Пример результатов

### Выходные файлы

| Файл | Описание |
|------|----------|
| `validation_report.json` | Отчет валидации (дубликаты, пропуски, выбросы) |
| `cleaning_log.txt` | Лог всех операций очистки |
| `correlation_heatmap.png` | Тепловая карта корреляций |
| `distributions.png` | Гистограммы распределений |
| `scatter_matrix.html` | Интерактивная матрица рассеяния (Plotly) |
| `time_series_decomposition.png` | Декомпозиция временного ряда |
| `analysis_report.xlsx` | Excel-отчет с данными и статистикой |
| `analysis_report.pdf` | PDF-отчет с графиками |
| `data_pipeline.log` | Полный лог выполнения |

---

## 🔧 Настройка email-рассылки

Для отправки отчетов по email настройте SMTP в коде:

```python
# В методе send_email_report класса ReportGenerator
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "your_email@gmail.com"
sender_password = "your_app_password"  # Используйте пароль приложения
```

---

## 📝 API Reference

### DataLoader

| Метод | Описание | Параметры |
|-------|----------|-----------|
| `load_from_csv()` | Загрузка из CSV | `file_path`, `**kwargs` |
| `load_from_excel()` | Загрузка из Excel | `file_path`, `sheet_name`, `**kwargs` |
| `load_from_sql()` | Загрузка из SQL | `connection_string`, `query` |
| `load_from_api()` | Загрузка из API | `url`, `params`, `headers` |

### DataCleaner

| Метод | Описание | Параметры |
|-------|----------|-----------|
| `handle_missing_values()` | Обработка пропусков | `strategy='median'`, `columns=None` |
| `remove_duplicates()` | Удаление дубликатов | `subset=None`, `keep='first'` |
| `handle_outliers()` | Обработка выбросов | `method='iqr'`, `strategy='cap'` |
| `encode_categorical()` | Кодирование категорий | `columns=None`, `method='onehot'` |
| `scale_features()` | Масштабирование | `columns=None`, `method='standard'` |
| `convert_dates()` | Преобразование дат | `columns`, `date_format=None` |

### DataAnalyzer

| Метод | Описание | Возвращает |
|-------|----------|------------|
| `compute_statistics()` | Базовые статистики | `dict` |
| `analyze_time_series()` | Анализ временного ряда | `decomposition` |
| `build_regression_model()` | Регрессионная модель | `model, metrics, y_test, y_pred` |
| `build_classification_model()` | Классификационная модель | `model, metrics, y_test, y_pred` |

### ReportGenerator

| Метод | Описание | Выходной файл |
|-------|----------|---------------|
| `create_correlation_heatmap()` | Тепловая карта | `correlation_heatmap.png` |
| `create_distribution_plots()` | Графики распределений | `distributions.png` |
| `create_interactive_plots()` | Интерактивные графики | `*.html` |
| `generate_excel_report()` | Excel-отчет | `*.xlsx` |
| `generate_pdf_report()` | PDF-отчет | `*.pdf` |
| `send_email_report()` | Отправка по email | — |

---

## 🐛 Устранение неполадок

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError` | Проверьте установку: `pip install -r requirements.txt` |
| Тесты падают с `ChainedAssignmentError` | Используйте `df[col] = df[col].fillna(value)` вместо `inplace=True` |
| Ошибка подключения к БД | Проверьте строку подключения и доступность сервера |
| PDF не создается | Установите reportlab: `pip install reportlab` |
| Email не отправляется | Используйте пароль приложения для Gmail |
| `AssertionError` в тестах | Запустите `pytest tests/ -v --tb=short` для детальной информации |

---

## 👥 Авторы

- Артём Подлобошников — проект для курса по анализу данных

---

**Версия:** 1.0  
**Последнее обновление:** Май 2026
