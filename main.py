"""
Автоматизация обработки данных
Версия: 1.0
Автор: Data Analyst
Описание: Скрипт для загрузки, очистки, анализа данных,
         построения ML-моделей и генерации отчетов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import requests
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import os
import sys
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# НАСТРОЙКА ДИРЕКТОРИЙ
# ============================================================

# Создаём необходимые директории
def setup_directories():
    """Создание структуры папок проекта"""
    directories = [
        'data',
        'output',
        'logs',
        'tests',
        'docs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

setup_directories()

# Настройка логирования (файл в папке logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'data_pipeline.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

class DataLoader:
    """Класс для загрузки данных из различных источников"""

    @staticmethod
    def load_from_csv(file_path, **kwargs):
        """Загрузка данных из CSV файла"""
        try:
            # Если файл не в папке data, ищем там
            if not os.path.exists(file_path) and os.path.exists(os.path.join('data', file_path)):
                file_path = os.path.join('data', file_path)
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Загружено {len(df)} строк из CSV: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки CSV: {e}")
            return None

    @staticmethod
    def load_from_excel(file_path, sheet_name=0, **kwargs):
        """Загрузка данных из Excel файла"""
        try:
            if not os.path.exists(file_path) and os.path.exists(os.path.join('data', file_path)):
                file_path = os.path.join('data', file_path)
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Загружено {len(df)} строк из Excel: {file_path}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки Excel: {e}")
            return None

    @staticmethod
    def load_from_sql(connection_string, query):
        """Загрузка данных из SQL базы данных"""
        try:
            # Для SQLite
            if connection_string.startswith('sqlite:///'):
                db_path = connection_string.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
            # Для PostgreSQL, MySQL можно расширить
            else:
                # Для PostgreSQL используем SQLAlchemy
                from sqlalchemy import create_engine
                engine = create_engine(connection_string)
                df = pd.read_sql_query(query, engine)
                engine.dispose()

            logger.info(f"Загружено {len(df)} строк из SQL")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки из SQL: {e}")
            return None

    @staticmethod
    def load_from_api(url, params=None, headers=None):
        """Загрузка данных из REST API"""
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Преобразуем JSON в DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])

            logger.info(f"Загружено {len(df)} строк из API: {url}")
            return df
        except Exception as e:
            logger.error(f"Ошибка загрузки из API: {e}")
            return None


# ============================================================
# 2. ВАЛИДАЦИЯ ДАННЫХ
# ============================================================

class DataValidator:
    """Класс для валидации данных"""

    def __init__(self, df):
        self.df = df
        self.validation_results = {}

    def check_duplicates(self):
        """Проверка на дубликаты"""
        duplicates = self.df.duplicated().sum()
        self.validation_results['duplicates'] = duplicates
        logger.info(f"Найдено дубликатов: {duplicates}")
        return duplicates

    def check_missing_values(self):
        """Проверка пропущенных значений"""
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        self.validation_results['missing'] = missing.to_dict()
        self.validation_results['missing_percent'] = missing_percent.to_dict()

        for col, count in missing.items():
            if count > 0:
                logger.warning(f"Пропуски в колонке '{col}': {count} ({missing_percent[col]:.1f}%)")
        return missing

    def check_data_types(self, expected_types=None):
        """Проверка типов данных"""
        actual_types = self.df.dtypes.to_dict()
        self.validation_results['data_types'] = actual_types

        if expected_types:
            type_errors = {}
            for col, exp_type in expected_types.items():
                if col in actual_types and actual_types[col] != exp_type:
                    type_errors[col] = {'expected': exp_type, 'actual': actual_types[col]}
                    logger.warning(f"Неверный тип в колонке '{col}': ожидался {exp_type}, получен {actual_types[col]}")
            return type_errors
        return actual_types

    def detect_outliers(self, method='iqr', threshold=1.5):
        """Выявление выбросов"""
        outliers = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers_count = (z_scores > threshold).sum()
            else:
                continue

            if outliers_count > 0:
                outliers[col] = outliers_count
                logger.info(f"Выбросы в колонке '{col}': {outliers_count}")

        self.validation_results['outliers'] = outliers
        return outliers

    def run_full_validation(self, expected_types=None):
        """Запуск полной валидации"""
        logger.info("=" * 50)
        logger.info("НАЧАЛО ВАЛИДАЦИИ ДАННЫХ")
        logger.info("=" * 50)

        self.check_duplicates()
        self.check_missing_values()
        self.check_data_types(expected_types)
        self.detect_outliers()

        # Сохраняем отчет валидации в папку output
        validation_path = os.path.join('output', 'validation_report.json')
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        logger.info(f"Валидация завершена. Результат сохранён в {validation_path}")
        return self.validation_results


# ============================================================
# 3. ОЧИСТКА ДАННЫХ
# ============================================================

class DataCleaner:
    """Класс для очистки и предобработки данных"""

    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_log = []

    def handle_missing_values(self, strategy='median', columns=None):
        """Обработка пропущенных значений"""
        if columns is None:
            columns = self.df.columns

        for col in columns:
            if col not in self.df.columns:
                continue

            missing_before = self.df[col].isnull().sum()
            if missing_before == 0:
                continue

            if strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
                self.cleaning_log.append(f"Удалены строки с пропусками в '{col}': {missing_before}")
            elif strategy == 'mean' and pd.api.types.is_numeric_dtype(self.df[col]):
                fill_value = self.df[col].mean()
                self.df[col] = self.df[col].fillna(fill_value)
                self.cleaning_log.append(f"Пропуски в '{col}' заполнены средним ({fill_value:.2f})")
            elif strategy == 'median' and pd.api.types.is_numeric_dtype(self.df[col]):
                fill_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(fill_value)
                self.cleaning_log.append(f"Пропуски в '{col}' заполнены медианой ({fill_value:.2f})")
            elif strategy == 'mode':
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else None
                if fill_value is not None:
                    self.df[col] = self.df[col].fillna(fill_value)
                    self.cleaning_log.append(f"Пропуски в '{col}' заполнены модой ({fill_value})")
            elif strategy == 'forward':
                self.df[col] = self.df[col].fillna(method='ffill')
                self.cleaning_log.append(f"Пропуски в '{col}' заполнены предыдущим значением")

        return self.df

    def remove_duplicates(self, subset=None, keep='first'):
        """Удаление дубликатов"""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)
        self.cleaning_log.append(f"Удалено дубликатов: {before - after}")
        return self.df

    def handle_outliers(self, method='iqr', threshold=1.5, strategy='cap'):
        """Обработка выбросов"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                lower_bound = -threshold
                upper_bound = threshold
            else:
                continue

            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers_count = outliers.sum()

            if outliers_count > 0:
                if strategy == 'remove':
                    self.df = self.df[~outliers]
                    self.cleaning_log.append(f"Удалено выбросов в '{col}': {outliers_count}")
                elif strategy == 'cap':
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.cleaning_log.append(f"Обрезаны выбросы в '{col}': {outliers_count}")

        return self.df

    def encode_categorical(self, columns=None, method='onehot'):
        """Кодирование категориальных признаков"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns

        for col in columns:
            if col not in self.df.columns:
                continue

            if method == 'label':
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.cleaning_log.append(f"Label encoding для '{col}'")
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.cleaning_log.append(f"One-hot encoding для '{col}' → {len(dummies.columns)} новых колонок")

        return self.df

    def scale_features(self, columns=None, method='standard'):
        """Масштабирование числовых признаков"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        scaler = StandardScaler() if method == 'standard' else None

        for col in columns:
            if col in self.df.columns:
                scaled = scaler.fit_transform(self.df[[col]]) if scaler else None
                if scaled is not None:
                    self.df[f'{col}_scaled'] = scaled
                    self.cleaning_log.append(f"Масштабирование '{col}' методом {method}")

        return self.df

    def convert_dates(self, columns, date_format=None):
        """Преобразование строк дат в datetime"""
        for col in columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], format=date_format, errors='coerce')
                self.cleaning_log.append(f"Конвертация '{col}' в datetime")

        return self.df

    def run_full_cleaning(self):
        """Запуск полной очистки данных"""
        logger.info("=" * 50)
        logger.info("НАЧАЛО ОЧИСТКИ ДАННЫХ")
        logger.info("=" * 50)

        # Сохраняем лог очистки в папку output
        cleaning_log_path = os.path.join('output', 'cleaning_log.txt')
        with open(cleaning_log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.cleaning_log))

        logger.info(f"Очистка завершена. Размер данных: {self.df.shape}")
        logger.info(f"Лог очистки сохранён в {cleaning_log_path}")
        return self.df


# ============================================================
# 4. АНАЛИЗ ДАННЫХ
# ============================================================

class DataAnalyzer:
    """Класс для анализа данных и построения ML-моделей"""

    def __init__(self, df):
        self.df = df
        self.statistics = {}
        self.model = None

    def compute_statistics(self):
        """Подсчет базовых статистик"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.statistics[col] = {
                'mean': self.df[col].mean(),
                'median': self.df[col].median(),
                'mode': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None,
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'q1': self.df[col].quantile(0.25),
                'q3': self.df[col].quantile(0.75)
            }

        logger.info("Базовые статистики вычислены")
        return self.statistics

    def analyze_time_series(self, date_col, value_col, period=12):
        """Анализ временного ряда (тренд, сезонность)"""
        try:
            ts_data = self.df.set_index(date_col)[value_col]
            decomposition = seasonal_decompose(ts_data, model='additive', period=period)

            # Создаем визуализацию декомпозиции в папке output
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            axes[0].plot(ts_data, label='Original')
            axes[0].legend()
            axes[1].plot(decomposition.trend, label='Trend', color='orange')
            axes[1].legend()
            axes[2].plot(decomposition.seasonal, label='Seasonal', color='green')
            axes[2].legend()
            axes[3].plot(decomposition.resid, label='Residual', color='red')
            axes[3].legend()
            plt.tight_layout()

            save_path = os.path.join('output', 'time_series_decomposition.png')
            plt.savefig(save_path, dpi=150)
            plt.close()

            logger.info(f"Анализ временного ряда выполнен, график сохранён в {save_path}")
            return decomposition
        except Exception as e:
            logger.error(f"Ошибка анализа временного ряда: {e}")
            return None

    def build_regression_model(self, target, features, test_size=0.2, model_type='random_forest'):
        """Построение регрессионной модели"""
        X = self.df[features]
        y = self.df[target]

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Выбор модели
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()

        # Обучение
        self.model.fit(X_train, y_train)

        # Предсказания
        y_pred = self.model.predict(X_test)

        # Метрики
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        logger.info(f"Регрессионная модель ({model_type}): RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.3f}")

        return self.model, metrics, y_test, y_pred

    def build_classification_model(self, target, features, test_size=0.2, model_type='random_forest'):
        """Построение классификационной модели"""
        X = self.df[features]
        y = self.df[target]

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Выбор модели
        if model_type == 'logistic':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Обучение
        self.model.fit(X_train, y_train)

        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }

        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Классификационная модель ({model_type}): Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

        return self.model, metrics, y_test, y_pred


# ============================================================
# 5. ОТЧЕТНОСТЬ И ВИЗУАЛИЗАЦИЯ
# ============================================================

class ReportGenerator:
    """Класс для генерации отчетов и визуализации"""

    def __init__(self, df):
        self.df = df
        self.plots = []

    def create_correlation_heatmap(self, save_path=None):
        """Создание тепловой карты корреляций"""
        if save_path is None:
            save_path = os.path.join('output', 'correlation_heatmap.png')

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None

        plt.figure(figsize=(12, 8))
        corr = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
        plt.title('Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        self.plots.append(save_path)
        logger.info(f"Тепловая карта корреляций сохранена в {save_path}")
        return corr

    def create_distribution_plots(self, columns=None, save_path=None):
        """Создание графиков распределений"""
        if save_path is None:
            save_path = os.path.join('output', 'distributions.png')

        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns[:6]

        n_cols = min(len(columns), 3)
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(columns):
            if i < len(axes):
                self.df[col].hist(ax=axes[i], bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')

        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        self.plots.append(save_path)
        logger.info(f"Графики распределений сохранены в {save_path}")

    def create_interactive_plots(self):
        """Создание интерактивных графиков с Plotly"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Scatter plot матрица
        if len(numeric_cols) >= 2:
            fig = px.scatter_matrix(
                self.df,
                dimensions=numeric_cols[:4],
                title='Scatter Matrix',
                opacity=0.6
            )
            save_path = os.path.join('output', 'scatter_matrix.html')
            fig.write_html(save_path)
            self.plots.append(save_path)
            logger.info(f"Scatter matrix сохранён в {save_path}")

        # Линейные графики для временных рядов
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            fig = px.line(
                self.df,
                x=date_cols[0],
                y=numeric_cols[0],
                title=f'{numeric_cols[0]} over Time'
            )
            save_path = os.path.join('output', 'time_series.html')
            fig.write_html(save_path)
            self.plots.append(save_path)
            logger.info(f"График временного ряда сохранён в {save_path}")

    def generate_excel_report(self, output_path=None):
        """Генерация Excel-отчета"""
        if output_path is None:
            output_path = os.path.join('output', 'analysis_report.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Основные данные
            self.df.to_excel(writer, sheet_name='Data', index=False)

            # Статистика
            stats = self.df.describe()
            stats.to_excel(writer, sheet_name='Statistics')

            # Информация о колонках
            info_df = pd.DataFrame({
                'Column': self.df.columns,
                'Type': self.df.dtypes.values,
                'Non-Null': self.df.count().values,
                'Null': self.df.isnull().sum().values,
                'Unique': self.df.nunique().values
            })
            info_df.to_excel(writer, sheet_name='Column Info', index=False)

        logger.info(f"Excel отчет сохранен: {output_path}")
        return output_path

    def generate_pdf_report(self, output_path=None):
        """Генерация PDF-отчета"""
        if output_path is None:
            output_path = os.path.join('output', 'analysis_report.pdf')

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.utils import ImageReader

            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4

            # Заголовок
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, "Data Analysis Report")
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            c.drawString(50, height - 100, f"Dataset shape: {self.df.shape}")
            c.drawString(50, height - 120, f"Columns: {len(self.df.columns)}")
            c.drawString(50, height - 140, f"Rows: {len(self.df)}")

            # Добавляем изображения, если есть
            y_position = height - 180
            for plot in self.plots[:2]:
                if plot.endswith('.png') and os.path.exists(plot):
                    try:
                        img = ImageReader(plot)
                        c.drawImage(img, 50, y_position - 200, width=500, height=300)
                        y_position -= 320
                    except:
                        pass

            c.save()
            logger.info(f"PDF отчет сохранен: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка создания PDF: {e}")

    def send_email_report(self, recipient_email, subject, body, attachments=None):
        """Отправка отчета по email"""
        try:
            # Настройки SMTP (пример для Gmail)
            smtp_server = "smtp.gmail.com"
            smtp_port = 587
            sender_email = "your_email@gmail.com"
            sender_password = "your_password"

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
                            msg.attach(part)

            # Отправка
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            logger.info(f"Отчет отправлен на {recipient_email}")
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки email: {e}")
            return False


# ============================================================
# 6. ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================

class DataPipeline:
    """Основной класс пайплайна обработки данных"""

    def __init__(self):
        self.df = None
        self.loader = DataLoader()
        self.validator = None
        self.cleaner = None
        self.analyzer = None
        self.reporter = None

    def run(self, source_type='csv', source_path=None, **kwargs):
        """Запуск полного пайплайна"""
        logger.info("=" * 60)
        logger.info("ЗАПУСК ПАЙПЛАЙНА ОБРАБОТКИ ДАННЫХ")
        logger.info("=" * 60)

        # 1. Загрузка данных
        if source_type == 'csv':
            self.df = self.loader.load_from_csv(source_path, **kwargs)
        elif source_type == 'excel':
            self.df = self.loader.load_from_excel(source_path, **kwargs)
        elif source_type == 'sql':
            self.df = self.loader.load_from_sql(source_path, kwargs.get('query', ''))
        elif source_type == 'api':
            self.df = self.loader.load_from_api(source_path, **kwargs)

        if self.df is None or self.df.empty:
            logger.error("Не удалось загрузить данные")
            return False

        # 2. Валидация
        self.validator = DataValidator(self.df)
        self.validator.run_full_validation()

        # 3. Очистка
        self.cleaner = DataCleaner(self.df)
        self.cleaner.handle_missing_values(strategy='median')
        self.cleaner.remove_duplicates()
        self.cleaner.handle_outliers(strategy='cap')
        self.cleaner.convert_dates(self.df.select_dtypes(include=['object']).columns.tolist())
        self.df = self.cleaner.run_full_cleaning()

        # 4. Анализ
        self.analyzer = DataAnalyzer(self.df)
        self.analyzer.compute_statistics()

        # 5. Отчетность
        self.reporter = ReportGenerator(self.df)
        self.reporter.create_correlation_heatmap()
        self.reporter.create_distribution_plots()
        self.reporter.create_interactive_plots()
        self.reporter.generate_excel_report()
        self.reporter.generate_pdf_report()

        logger.info("=" * 60)
        logger.info("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН")
        logger.info("=" * 60)

        return True


# ============================================================
# 7. ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================

if __name__ == "__main__":
    # Создаём папку data для входных данных
    os.makedirs('data', exist_ok=True)

    # Пример создания тестовых данных
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

    test_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 300,
        'customers': np.random.poisson(50, len(dates)),
        'price': np.random.uniform(10, 50, len(dates)),
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    })

    # Сохраняем тестовые данные в папку data
    test_data_path = os.path.join('data', 'test_data.csv')
    test_data.to_csv(test_data_path, index=False)
    logger.info(f"Тестовые данные сохранены в {test_data_path}")

    # Запускаем пайплайн
    pipeline = DataPipeline()
    pipeline.run(source_type='csv', source_path='test_data.csv')

    print("\n" + "=" * 60)
    print("ПРОЕКТ ЗАВЕРШЕН")
    print("Структура сгенерированных файлов:")
    print("=" * 60)
    print("data/")
    print("  └── test_data.csv (исходные данные)")
    print("output/")
    print("  ├── validation_report.json (отчет валидации)")
    print("  ├── cleaning_log.txt (лог очистки)")
    print("  ├── correlation_heatmap.png (корреляционная матрица)")
    print("  ├── distributions.png (распределения)")
    print("  ├── time_series_decomposition.png (декомпозиция)")
    print("  ├── scatter_matrix.html (интерактивный график)")
    print("  ├── time_series.html (временной ряд)")
    print("  ├── analysis_report.xlsx (Excel отчет)")
    print("  └── analysis_report.pdf (PDF отчет)")
    print("logs/")
    print("  └── data_pipeline.log (лог выполнения)")
    print("=" * 60)
