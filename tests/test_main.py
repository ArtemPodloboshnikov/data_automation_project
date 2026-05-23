# tests/test_main.py
"""
Юнит-тесты для модуля main.py
Запуск: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
import requests
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Импортируем классы из main.py
import sys
sys.path.append('.')
from main import (
    DataLoader, DataValidator, DataCleaner,
    DataAnalyzer, ReportGenerator, DataPipeline
)


# ============================================================
# Фикстуры для тестов
# ============================================================

@pytest.fixture
def sample_dataframe():
    """Создает тестовый DataFrame для тестов"""
    np.random.seed(42)
    df = pd.DataFrame({
        'id': range(1, 101),
        'numeric_col': np.random.normal(100, 15, 100),
        'category_col': np.random.choice(['A', 'B', 'C'], 100),
        'date_col': pd.date_range('2023-01-01', periods=100, freq='D'),
        'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    # Добавляем пропуски для тестов очистки
    df.loc[5:10, 'numeric_col'] = np.nan
    df.loc[20:25, 'category_col'] = np.nan
    return df


@pytest.fixture
def sample_dataframe_with_duplicates(sample_dataframe):
    """Добавляет дубликаты в DataFrame"""
    df = sample_dataframe.copy()
    duplicates = df.iloc[-5:].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    return df


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Создает временный CSV файл в папке data"""
    # Создаём папку data если её нет
    os.makedirs('data', exist_ok=True)
    temp_path = os.path.join('data', 'test_data.csv')
    sample_dataframe.to_csv(temp_path, index=False)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_excel_file(sample_dataframe):
    """Создает временный Excel файл в папке data"""
    os.makedirs('data', exist_ok=True)
    temp_path = os.path.join('data', 'test_data.xlsx')
    sample_dataframe.to_excel(temp_path, index=False)
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================
# Тесты для DataLoader
# ============================================================

class TestDataLoader:
    """Тесты класса DataLoader"""

    def test_load_from_csv_success(self, sample_dataframe, temp_csv_file):
        """Успешная загрузка из CSV"""
        loader = DataLoader()
        df = loader.load_from_csv(temp_csv_file)

        assert df is not None
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_from_csv_file_not_found(self):
        """Загрузка из несуществующего файла"""
        loader = DataLoader()
        df = loader.load_from_csv('nonexistent_file.csv')

        assert df is None

    def test_load_from_excel_success(self, sample_dataframe, temp_excel_file):
        """Успешная загрузка из Excel"""
        loader = DataLoader()
        df = loader.load_from_excel(temp_excel_file)

        assert df is not None
        assert len(df) == len(sample_dataframe)

    def test_load_from_excel_file_not_found(self):
        """Загрузка из несуществующего Excel файла"""
        loader = DataLoader()
        df = loader.load_from_excel('nonexistent.xlsx')

        assert df is None

    @patch('main.pd.read_sql_query')
    def test_load_from_sql_sqlite(self, mock_read_sql):
        """Загрузка из SQLite базы данных"""
        import tempfile
        import os

        # Создаём временный файл базы данных в папке data
        os.makedirs('data', exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False, dir='data') as tmp:
            db_path = tmp.name

        mock_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        mock_read_sql.return_value = mock_df

        loader = DataLoader()
        df = loader.load_from_sql(f'sqlite:///{db_path}', 'SELECT * FROM table')

        assert df is not None
        assert len(df) == 3

        # Очистка
        if os.path.exists(db_path):
            os.remove(db_path)

    @patch('main.requests.get')
    def test_load_from_api_success(self, mock_get, sample_dataframe):
        """Успешная загрузка из API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = sample_dataframe.to_dict('records')
        mock_get.return_value = mock_response

        loader = DataLoader()
        df = loader.load_from_api('https://api.example.com/data')

        assert df is not None
        assert len(df) == len(sample_dataframe)

    @patch('main.requests.get')
    def test_load_from_api_error(self, mock_get):
        """Ошибка при загрузке из API"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Error")
        mock_get.return_value = mock_response

        loader = DataLoader()
        df = loader.load_from_api('https://api.example.com/data')

        assert df is None


# ============================================================
# Тесты для DataValidator
# ============================================================

class TestDataValidator:
    """Тесты класса DataValidator"""

    def test_check_duplicates(self, sample_dataframe_with_duplicates):
        """Проверка обнаружения дубликатов"""
        validator = DataValidator(sample_dataframe_with_duplicates)
        duplicates = validator.check_duplicates()

        assert duplicates == 5

    def test_check_missing_values(self, sample_dataframe):
        """Проверка обнаружения пропусков"""
        validator = DataValidator(sample_dataframe)
        missing = validator.check_missing_values()

        assert missing['numeric_col'] == 6
        assert missing['category_col'] == 6
        assert missing['date_col'] == 0

    def test_check_data_types(self, sample_dataframe):
        """Проверка типов данных"""
        validator = DataValidator(sample_dataframe)
        types = validator.check_data_types()

        assert types['id'] == 'int64'
        assert types['numeric_col'] == 'float64'
        assert pd.api.types.is_datetime64_any_dtype(types['date_col'])

    def test_check_data_types_with_expected(self, sample_dataframe):
        """Проверка типов с ожидаемыми значениями"""
        validator = DataValidator(sample_dataframe)
        expected = {'id': 'int64', 'numeric_col': 'float64', 'target': 'int64'}
        errors = validator.check_data_types(expected)

        assert errors == {}

    def test_detect_outliers_iqr(self, sample_dataframe):
        """Обнаружение выбросов методом IQR"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
        validator = DataValidator(df)
        outliers = validator.detect_outliers(method='iqr')

        assert 'value' in outliers
        assert outliers['value'] == 1

    def test_detect_outliers_zscore(self, sample_dataframe):
        """Обнаружение выбросов методом Z-score"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
        validator = DataValidator(df)
        outliers = validator.detect_outliers(method='zscore', threshold=2)

        assert 'value' in outliers

    def test_run_full_validation(self, sample_dataframe_with_duplicates):
        """Полная валидация"""
        validator = DataValidator(sample_dataframe_with_duplicates)
        results = validator.run_full_validation()

        assert 'duplicates' in results
        assert 'missing' in results
        assert 'data_types' in results
        assert results['duplicates'] == 5

        # Проверяем, что файл создан в папке output
        assert os.path.exists(os.path.join('output', 'validation_report.json'))

        # Очистка
        if os.path.exists(os.path.join('output', 'validation_report.json')):
            os.remove(os.path.join('output', 'validation_report.json'))


# ============================================================
# Тесты для DataCleaner
# ============================================================

class TestDataCleaner:
    """Тесты класса DataCleaner"""

    def test_handle_missing_values_median(self, sample_dataframe):
        """Заполнение пропусков медианой"""
        cleaner = DataCleaner(sample_dataframe)
        original_median = sample_dataframe['numeric_col'].median()

        cleaner.handle_missing_values(strategy='median', columns=['numeric_col'])

        assert cleaner.df['numeric_col'].isnull().sum() == 0
        assert cleaner.df['numeric_col'].iloc[5] == original_median

    def test_handle_missing_values_mean(self, sample_dataframe):
        """Заполнение пропусков средним"""
        cleaner = DataCleaner(sample_dataframe)
        original_mean = sample_dataframe['numeric_col'].mean()

        cleaner.handle_missing_values(strategy='mean', columns=['numeric_col'])

        assert cleaner.df['numeric_col'].isnull().sum() == 0
        assert abs(cleaner.df['numeric_col'].iloc[5] - original_mean) < 0.01

    def test_handle_missing_values_drop(self, sample_dataframe):
        """Удаление строк с пропусками"""
        cleaner = DataCleaner(sample_dataframe)
        rows_before = len(cleaner.df)

        cleaner.handle_missing_values(strategy='drop', columns=['numeric_col'])

        assert len(cleaner.df) == rows_before - 6

    def test_remove_duplicates(self, sample_dataframe_with_duplicates):
        """Удаление дубликатов"""
        cleaner = DataCleaner(sample_dataframe_with_duplicates)
        rows_before = len(cleaner.df)

        cleaner.remove_duplicates()

        assert len(cleaner.df) == rows_before - 5
        assert cleaner.df.duplicated().sum() == 0

    def test_handle_outliers_remove(self):
        """Удаление выбросов"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
        cleaner = DataCleaner(df)
        rows_before = len(cleaner.df)

        cleaner.handle_outliers(method='iqr', strategy='remove')

        assert len(cleaner.df) == rows_before - 1

    def test_handle_outliers_cap(self):
        """Обрезание выбросов"""
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5, 100]})
        cleaner = DataCleaner(df)

        cleaner.handle_outliers(method='iqr', strategy='cap')

        assert cleaner.df['value'].max() < 100

    def test_encode_categorical_label(self, sample_dataframe):
        """Label encoding категорий"""
        cleaner = DataCleaner(sample_dataframe)

        cleaner.encode_categorical(columns=['category_col'], method='label')

        assert 'category_col_encoded' in cleaner.df.columns
        assert cleaner.df['category_col_encoded'].dtype in ['int64', 'float64']

    def test_encode_categorical_onehot(self, sample_dataframe):
        """One-Hot encoding категорий"""
        cleaner = DataCleaner(sample_dataframe)
        unique_categories = cleaner.df['category_col'].nunique()

        cleaner.encode_categorical(columns=['category_col'], method='onehot')

        new_cols = [c for c in cleaner.df.columns if c.startswith('category_col_')]
        assert len(new_cols) == unique_categories

    def test_scale_features(self, sample_dataframe):
        """Масштабирование числовых признаков"""
        cleaner = DataCleaner(sample_dataframe)

        cleaner.scale_features(columns=['numeric_col'], method='standard')

        assert 'numeric_col_scaled' in cleaner.df.columns
        scaled_values = cleaner.df['numeric_col_scaled'].dropna()
        assert abs(scaled_values.mean()) < 0.1
        assert abs(scaled_values.std() - 1) < 0.1

    def test_convert_dates(self, sample_dataframe):
        """Преобразование строк в даты"""
        df = pd.DataFrame({'date_str': ['2023-01-01', '2023-01-02', '2023-01-03']})
        cleaner = DataCleaner(df)

        cleaner.convert_dates(columns=['date_str'])

        assert pd.api.types.is_datetime64_any_dtype(cleaner.df['date_str'])

    def test_run_full_cleaning(self, sample_dataframe_with_duplicates):
        """Полный цикл очистки"""
        cleaner = DataCleaner(sample_dataframe_with_duplicates)
        cleaned_df = cleaner.run_full_cleaning()

        assert cleaned_df is not None
        # Проверяем, что файл создан в папке output
        assert os.path.exists(os.path.join('output', 'cleaning_log.txt'))

        # Очистка
        if os.path.exists(os.path.join('output', 'cleaning_log.txt')):
            os.remove(os.path.join('output', 'cleaning_log.txt'))


# ============================================================
# Тесты для DataAnalyzer
# ============================================================

class TestDataAnalyzer:
    """Тесты класса DataAnalyzer"""

    def test_compute_statistics(self, sample_dataframe):
        """Вычисление статистик"""
        analyzer = DataAnalyzer(sample_dataframe)
        stats = analyzer.compute_statistics()

        assert 'numeric_col' in stats
        assert 'mean' in stats['numeric_col']
        assert 'median' in stats['numeric_col']
        assert 'std' in stats['numeric_col']
        assert 'min' in stats['numeric_col']
        assert 'max' in stats['numeric_col']

        assert stats['numeric_col']['mean'] == sample_dataframe['numeric_col'].mean()
        assert stats['numeric_col']['median'] == sample_dataframe['numeric_col'].median()

    def test_analyze_time_series(self, sample_dataframe):
        """Анализ временного ряда"""
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        values = np.random.normal(100, 10, 365) + np.sin(np.arange(365) * 2 * np.pi / 30) * 20
        df = pd.DataFrame({'date': dates, 'sales': values})

        analyzer = DataAnalyzer(df)
        decomposition = analyzer.analyze_time_series('date', 'sales', period=30)

        assert decomposition is not None
        # Проверяем, что файл создан в папке output
        assert os.path.exists(os.path.join('output', 'time_series_decomposition.png'))

        # Очистка
        if os.path.exists(os.path.join('output', 'time_series_decomposition.png')):
            os.remove(os.path.join('output', 'time_series_decomposition.png'))

    def test_build_regression_model(self, sample_dataframe):
        """Построение регрессионной модели"""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(100) * 0.1
        df = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'target': y
        })

        analyzer = DataAnalyzer(df)
        model, metrics, y_test, y_pred = analyzer.build_regression_model(
            target='target',
            features=['feature1', 'feature2', 'feature3'],
            model_type='linear'
        )

        assert model is not None
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] > 0.9
        assert len(y_pred) == len(y_test)

    def test_build_regression_model_random_forest(self, sample_dataframe):
        """Построение регрессионной модели Random Forest"""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(100) * 0.1
        df = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'target': y
        })

        analyzer = DataAnalyzer(df)
        model, metrics, y_test, y_pred = analyzer.build_regression_model(
            target='target',
            features=['feature1', 'feature2', 'feature3'],
            model_type='random_forest'
        )

        assert model is not None
        assert 'rmse' in metrics
        assert metrics['r2'] > 0.8

    def test_build_classification_model(self, sample_dataframe):
        """Построение классификационной модели"""
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        df = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'feature4': X[:, 3],
            'target': y
        })

        analyzer = DataAnalyzer(df)
        model, metrics, y_test, y_pred = analyzer.build_classification_model(
            target='target',
            features=['feature1', 'feature2', 'feature3', 'feature4'],
            model_type='logistic'
        )

        assert model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert metrics['accuracy'] > 0.8

    def test_build_classification_model_random_forest(self, sample_dataframe):
        """Построение классификационной модели Random Forest"""
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        df = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'feature4': X[:, 3],
            'target': y
        })

        analyzer = DataAnalyzer(df)
        model, metrics, y_test, y_pred = analyzer.build_classification_model(
            target='target',
            features=['feature1', 'feature2', 'feature3', 'feature4'],
            model_type='random_forest'
        )

        assert model is not None
        assert metrics['accuracy'] > 0.8


# ============================================================
# Тесты для ReportGenerator
# ============================================================

class TestReportGenerator:
    """Тесты класса ReportGenerator"""

    def test_create_correlation_heatmap(self, sample_dataframe):
        """Создание тепловой карты корреляций"""
        reporter = ReportGenerator(sample_dataframe)
        corr = reporter.create_correlation_heatmap()

        assert corr is not None
        # Проверяем, что файл создан в папке output
        assert os.path.exists(os.path.join('output', 'correlation_heatmap.png'))

        # Очистка
        if os.path.exists(os.path.join('output', 'correlation_heatmap.png')):
            os.remove(os.path.join('output', 'correlation_heatmap.png'))

    def test_create_distribution_plots(self, sample_dataframe):
        """Создание графиков распределений"""
        reporter = ReportGenerator(sample_dataframe)
        reporter.create_distribution_plots()

        # Проверяем, что файл создан в папке output
        assert os.path.exists(os.path.join('output', 'distributions.png'))

        # Очистка
        if os.path.exists(os.path.join('output', 'distributions.png')):
            os.remove(os.path.join('output', 'distributions.png'))

    def test_create_interactive_plots(self, sample_dataframe):
        """Создание интерактивных графиков"""
        reporter = ReportGenerator(sample_dataframe)
        reporter.create_interactive_plots()

        # Проверяем создание HTML файлов в папке output
        has_scatter = os.path.exists(os.path.join('output', 'scatter_matrix.html'))
        has_timeseries = os.path.exists(os.path.join('output', 'time_series.html'))

        # Должен быть хотя бы один файл (зависит от данных)
        assert has_scatter or has_timeseries

        # Очистка
        if has_scatter:
            os.remove(os.path.join('output', 'scatter_matrix.html'))
        if has_timeseries:
            os.remove(os.path.join('output', 'time_series.html'))

    def test_generate_excel_report(self, sample_dataframe):
        """Генерация Excel отчета"""
        reporter = ReportGenerator(sample_dataframe)
        output_path = reporter.generate_excel_report()

        assert os.path.exists(output_path)
        assert output_path == os.path.join('output', 'analysis_report.xlsx')

        # Проверяем, что файл действительно Excel
        df_loaded = pd.read_excel(output_path, sheet_name='Data')
        assert len(df_loaded) == len(sample_dataframe)

        # Очистка
        if os.path.exists(output_path):
            os.remove(output_path)

    def test_generate_pdf_report(self, sample_dataframe):
        """Генерация PDF отчета"""
        reporter = ReportGenerator(sample_dataframe)
        # Сначала создадим хотя бы один график для PDF
        reporter.create_correlation_heatmap()
        reporter.generate_pdf_report()

        pdf_path = os.path.join('output', 'analysis_report.pdf')
        assert os.path.exists(pdf_path)

        # Очистка
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(os.path.join('output', 'correlation_heatmap.png')):
            os.remove(os.path.join('output', 'correlation_heatmap.png'))

    @patch('main.smtplib.SMTP')
    def test_send_email_report(self, mock_smtp, sample_dataframe):
        """Отправка email отчета"""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        reporter = ReportGenerator(sample_dataframe)

        # Создаем временный файл для вложения
        test_file = os.path.join('output', 'test_report.xlsx')
        os.makedirs('output', exist_ok=True)
        with open(test_file, 'w') as f:
            f.write('test')

        result = reporter.send_email_report(
            recipient_email='test@example.com',
            subject='Test Report',
            body='Test body',
            attachments=[test_file]
        )

        assert result is not None

        # Очистка
        if os.path.exists(test_file):
            os.remove(test_file)


# ============================================================
# Тесты для DataPipeline (интеграционные)
# ============================================================

class TestDataPipeline:
    """Интеграционные тесты DataPipeline"""

    def test_pipeline_csv_run(self, temp_csv_file):
        """Запуск пайплайна с CSV файлом"""
        pipeline = DataPipeline()
        result = pipeline.run(source_type='csv', source_path=os.path.basename(temp_csv_file))

        assert result is True
        assert pipeline.df is not None

        # Проверяем создание файлов в правильных папках
        assert os.path.exists(os.path.join('output', 'validation_report.json'))
        assert os.path.exists(os.path.join('output', 'analysis_report.xlsx'))
        assert os.path.exists(os.path.join('output', 'analysis_report.pdf'))

        # Очистка после теста
        for file in ['validation_report.json', 'cleaning_log.txt', 'analysis_report.xlsx',
                     'analysis_report.pdf', 'correlation_heatmap.png', 'distributions.png']:
            file_path = os.path.join('output', file)
            if os.path.exists(file_path):
                os.remove(file_path)

        if os.path.exists(os.path.join('output', 'scatter_matrix.html')):
            os.remove(os.path.join('output', 'scatter_matrix.html'))
        if os.path.exists(os.path.join('output', 'time_series.html')):
            os.remove(os.path.join('output', 'time_series.html'))

    def test_pipeline_invalid_source(self):
        """Пайплайн с неверным источником"""
        pipeline = DataPipeline()
        result = pipeline.run(source_type='invalid', source_path='dummy')

        assert result is False


# ============================================================
# Запуск тестов
# ============================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
