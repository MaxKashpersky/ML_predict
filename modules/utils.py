"""
Вспомогательные утилиты
"""

import os
import sys
import logging
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from config import config


class Logger:
    """Универсальный логгер"""

    @staticmethod
    def setup_logger(name: str, log_file: str = None, level: int = logging.INFO,
                     console: bool = True, file: bool = True) -> logging.Logger:
        """
        Настройка логгера

        Args:
            name: Имя логгера
            log_file: Путь к файлу лога
            level: Уровень логирования
            console: Логировать в консоль
            file: Логировать в файл

        Returns:
            Настроенный логгер
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Очистка существующих обработчиков
        logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if file and log_file:
            # Создание директории для логов
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def log_function_call(logger: logging.Logger, enabled: bool = True):
        """
        Декоратор для логирования вызовов функций

        Args:
            logger: Логгер
            enabled: Включить логирование
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                if enabled:
                    logger.debug(f"Вызов функции {func.__name__} с args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                if enabled:
                    logger.debug(f"Функция {func.__name__} завершилась")
                return result

            return wrapper

        return decorator


class DataValidator:
    """Валидатор данных"""

    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame, verbose: bool = True) -> bool:
        """
        Валидация OHLCV данных

        Args:
            df: DataFrame с данными
            verbose: Флаг логирования

        Returns:
            True если данные валидны
        """
        try:
            if df.empty:
                if verbose:
                    print("DataFrame пустой")
                return False

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                if verbose:
                    print(f"Отсутствуют колонки: {missing_columns}")
                return False

            # Проверка на NaN
            nan_count = df[required_columns].isna().sum().sum()
            if nan_count > 0:
                if verbose:
                    print(f"Найдено {nan_count} NaN значений")
                return False

            # Проверка корректности значений
            if (df['high'] < df['low']).any():
                if verbose:
                    print("Найдены строки где high < low")
                return False

            if (df['close'] > df['high']).any() or (df['close'] < df['low']).any():
                if verbose:
                    print("Найдены строки где close вне диапазона high-low")
                return False

            if (df['open'] > df['high']).any() or (df['open'] < df['low']).any():
                if verbose:
                    print("Найдены строки где open вне диапазона high-low")
                return False

            if (df['volume'] < 0).any():
                if verbose:
                    print("Найдены отрицательные объемы")
                return False

            # Проверка временных меток
            if df.index.duplicated().any():
                if verbose:
                    print("Найдены дублирующиеся временные метки")
                return False

            if not df.index.is_monotonic_increasing:
                if verbose:
                    print("Временные метки не отсортированы")
                return False

            return True

        except Exception as e:
            if verbose:
                print(f"Ошибка валидации данных: {e}")
            return False

    @staticmethod
    def detect_anomalies(df: pd.DataFrame, threshold: float = 3.0,
                         verbose: bool = True) -> pd.DataFrame:
        """
        Обнаружение аномалий в данных

        Args:
            df: DataFrame с данными
            threshold: Порог для обнаружения выбросов
            verbose: Флаг логирования

        Returns:
            DataFrame с аномалиями
        """
        try:
            anomalies = pd.DataFrame()

            for column in ['open', 'high', 'low', 'close', 'volume']:
                if column in df.columns:
                    # Расчет z-score
                    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())

                    # Поиск аномалий
                    column_anomalies = df[z_scores > threshold].copy()
                    column_anomalies['z_score'] = z_scores[z_scores > threshold]
                    column_anomalies['anomaly_column'] = column

                    anomalies = pd.concat([anomalies, column_anomalies])

            if verbose and not anomalies.empty:
                print(f"Найдено {len(anomalies)} аномалий")

            return anomalies

        except Exception as e:
            if verbose:
                print(f"Ошибка обнаружения аномалий: {e}")
            return pd.DataFrame()


class FileManager:
    """Менеджер файлов"""

    @staticmethod
    def save_object(obj: Any, filepath: str, verbose: bool = True) -> bool:
        """
        Сохранение объекта в файл

        Args:
            obj: Объект для сохранения
            filepath: Путь к файлу
            verbose: Флаг логирования

        Returns:
            True если успешно
        """
        try:
            # Создание директории если не существует
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)

            if verbose:
                print(f"Объект сохранен: {filepath}")

            return True

        except Exception as e:
            if verbose:
                print(f"Ошибка сохранения объекта: {e}")
            return False

    @staticmethod
    def load_object(filepath: str, verbose: bool = True) -> Any:
        """
        Загрузка объекта из файла

        Args:
            filepath: Путь к файлу
            verbose: Флаг логирования

        Returns:
            Загруженный объект или None
        """
        try:
            if not os.path.exists(filepath):
                if verbose:
                    print(f"Файл не существует: {filepath}")
                return None

            with open(filepath, 'rb') as f:
                obj = pickle.load(f)

            if verbose:
                print(f"Объект загружен: {filepath}")

            return obj

        except Exception as e:
            if verbose:
                print(f"Ошибка загрузки объекта: {e}")
            return None

    @staticmethod
    def get_file_hash(filepath: str, verbose: bool = True) -> str:
        """
        Получение хеша файла

        Args:
            filepath: Путь к файлу
            verbose: Флаг логирования

        Returns:
            Хеш файла
        """
        try:
            if not os.path.exists(filepath):
                if verbose:
                    print(f"Файл не существует: {filepath}")
                return ""

            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                buf = f.read()
                hasher.update(buf)

            file_hash = hasher.hexdigest()

            if verbose:
                print(f"Хеш файла {filepath}: {file_hash}")

            return file_hash

        except Exception as e:
            if verbose:
                print(f"Ошибка расчета хеша: {e}")
            return ""


class PerformanceMonitor:
    """Монитор производительности"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.metrics = {}

    def start_timer(self, name: str):
        """Старт таймера"""
        self.metrics[name] = {
            'start': datetime.now(),
            'end': None,
            'duration': None
        }

    def stop_timer(self, name: str):
        """Стоп таймера"""
        if name in self.metrics and self.metrics[name]['end'] is None:
            self.metrics[name]['end'] = datetime.now()
            self.metrics[name]['duration'] = (
                    self.metrics[name]['end'] - self.metrics[name]['start']
            ).total_seconds()

            if self.verbose:
                print(f"{name}: {self.metrics[name]['duration']:.2f} секунд")

    def get_performance_report(self) -> Dict:
        """Получение отчета о производительности"""
        report = {}

        for name, data in self.metrics.items():
            if data['duration'] is not None:
                report[name] = {
                    'duration_seconds': data['duration'],
                    'start_time': data['start'].isoformat(),
                    'end_time': data['end'].isoformat() if data['end'] else None
                }

        return report

    def clear_metrics(self):
        """Очистка метрик"""
        self.metrics.clear()


class ConfigManager:
    """Менеджер конфигурации"""

    @staticmethod
    def save_config(config_data: Dict, filepath: str, verbose: bool = True) -> bool:
        """
        Сохранение конфигурации в файл

        Args:
            config_data: Данные конфигурации
            filepath: Путь к файлу
            verbose: Флаг логирования

        Returns:
            True если успешно
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)

            if verbose:
                print(f"Конфигурация сохранена: {filepath}")

            return True

        except Exception as e:
            if verbose:
                print(f"Ошибка сохранения конфигурации: {e}")
            return False

    @staticmethod
    def load_config(filepath: str, verbose: bool = True) -> Dict:
        """
        Загрузка конфигурации из файла

        Args:
            filepath: Путь к файлу
            verbose: Флаг логирования

        Returns:
            Словарь с конфигурацией
        """
        try:
            if not os.path.exists(filepath):
                if verbose:
                    print(f"Файл конфигурации не существует: {filepath}")
                return {}

            with open(filepath, 'r') as f:
                config_data = json.load(f)

            if verbose:
                print(f"Конфигурация загружена: {filepath}")

            return config_data

        except Exception as e:
            if verbose:
                print(f"Ошибка загрузки конфигурации: {e}")
            return {}

    @staticmethod
    def validate_config(config_data: Dict, required_keys: List[str],
                        verbose: bool = True) -> bool:
        """
        Валидация конфигурации

        Args:
            config_data: Данные конфигурации
            required_keys: Обязательные ключи
            verbose: Флаг логирования

        Returns:
            True если конфигурация валидна
        """
        try:
            missing_keys = [key for key in required_keys if key not in config_data]

            if missing_keys:
                if verbose:
                    print(f"Отсутствуют обязательные ключи: {missing_keys}")
                return False

            return True

        except Exception as e:
            if verbose:
                print(f"Ошибка валидации конфигурации: {e}")
            return False


# Создание экземпляров утилит для удобного импорта
logger = Logger()
validator = DataValidator()
file_manager = FileManager()
performance_monitor = PerformanceMonitor()
config_manager = ConfigManager()