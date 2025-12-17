"""
Модуль для работы с базой данных
"""

import sqlite3
import pandas as pd
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from config import config


class Database:
    def __init__(self, verbose: bool = True):
        """Инициализация базы данных"""
        self.verbose = verbose
        self.setup_logging()
        self.connection = None
        self.cursor = None
        self.connect()
        self.init_tables()

    def setup_logging(self):
        """Настройка логирования"""
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def log(self, message: str, level: str = 'info'):
        """Логирование сообщений"""
        if self.verbose:
            if level == 'info':
                self.logger.info(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'warning':
                self.logger.warning(message)

    def connect(self):
        """Подключение к базе данных"""
        try:
            # Создаем директорию для базы данных если ее нет
            db_dir = os.path.dirname(config.DB_PATH)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)

            self.connection = sqlite3.connect(config.DB_PATH)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            self.log("Database connection established")
        except Exception as e:
            self.log(f"Error connecting to database: {str(e)}", 'error')
            raise

    def init_tables(self):
        """Инициализация таблиц"""
        try:
            # Таблица для хранения исторических данных
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')

            # Таблица для метаданных (последнее обновление)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    last_update DATETIME,
                    last_candle_timestamp DATETIME,
                    PRIMARY KEY (symbol, timeframe)
                )
            ''')

            # Таблица для моделей
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    parameters TEXT,
                    metrics TEXT,
                    model_path TEXT NOT NULL,
                    feature_importance TEXT,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Таблица для результатов бэктеста
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    test_date DATETIME NOT NULL,
                    start_date DATETIME NOT NULL,
                    end_date DATETIME NOT NULL,
                    initial_balance REAL NOT NULL,
                    final_balance REAL NOT NULL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    avg_win REAL,
                    avg_loss REAL,
                    details TEXT
                )
            ''')

            # Индексы для оптимизации
            self.cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_historical_data ON historical_data(symbol, timeframe, timestamp)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_models ON models(symbol, timeframe, is_active)')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest ON backtest_results(model_id, symbol)')

            self.connection.commit()
            self.log("Database tables initialized")

        except Exception as e:
            self.log(f"Error initializing tables: {str(e)}", 'error')
            raise

    def store_historical_data(self, symbol: str, timeframe: str, data: pd.DataFrame, verbose: bool = True) -> bool:
        """
        Сохранение исторических данных в базу
        Возвращает True если сохранение успешно
        """
        try:
            if data.empty:
                self.log(f"No data to save for {symbol} {timeframe}", 'warning')
                return False

            # Создаем копию данных
            df = data.copy()
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'timestamp'}, inplace=True)

            # Убедимся, что timestamp в правильном формате
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Добавляем символ и таймфрейм
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            # Проверяем наличие необходимых колонок
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.log(f"Missing required column: {col}", 'error')
                    return False

            # Подготавливаем данные для вставки
            records = df[['symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

            if not records:
                self.log("No records to save", 'warning')
                return False

            # Используем bulk insert
            query = """
            INSERT OR REPLACE INTO historical_data 
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
            """

            self.cursor.executemany(query, records)

            # Обновление метаданных
            last_timestamp = data.index.max()
            last_timestamp_str = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')

            self.cursor.execute('''
                INSERT OR REPLACE INTO metadata (symbol, timeframe, last_update, last_candle_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (symbol, timeframe, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), last_timestamp_str))

            self.connection.commit()

            added_count = len(records)
            if verbose:
                print(f"✅ Данные сохранены: {added_count} записей")
                self.log(f"Data saved: {added_count} records for {symbol} {timeframe}")

            return True

        except Exception as e:
            self.log(f"Error saving data: {str(e)}", 'error')
            print(f"❌ Ошибка сохранения данных: {e}")
            return False

    def get_historical_data(self, symbol: str, timeframe: str,
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            verbose: bool = True) -> pd.DataFrame:
        """
        Получение исторических данных из базы
        """
        try:
            query = '''
                SELECT timestamp, open, high, low, close, volume 
                FROM historical_data 
                WHERE symbol = ? AND timeframe = ?
            '''
            params = [symbol, timeframe]

            if start_date:
                query += ' AND timestamp >= ?'
                params.append(start_date.strftime('%Y-%m-%d %H:%M:%S'))
            if end_date:
                query += ' AND timestamp <= ?'
                params.append(end_date.strftime('%Y-%m-%d %H:%M:%S'))

            query += ' ORDER BY timestamp ASC'

            df = pd.read_sql_query(query, self.connection, params=params)

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                if verbose:
                    self.log(f"Retrieved {len(df)} records for {symbol} {timeframe}")
            else:
                if verbose:
                    self.log(f"No data found for {symbol} {timeframe}", 'warning')

            return df

        except Exception as e:
            self.log(f"Error retrieving data: {str(e)}", 'error')
            return pd.DataFrame()

    def get_last_timestamp(self, symbol: str, timeframe: str, verbose: bool = True) -> Optional[datetime]:
        """Получение времени последней свечи"""
        try:
            self.cursor.execute('''
                SELECT last_candle_timestamp FROM metadata 
                WHERE symbol = ? AND timeframe = ?
            ''', (symbol, timeframe))

            result = self.cursor.fetchone()
            if result and result[0]:
                last_timestamp = pd.to_datetime(result[0])
                if verbose:
                    self.log(f"Last candle: {last_timestamp}")
                return last_timestamp
            return None

        except Exception as e:
            if verbose:
                self.log(f"Error getting last candle: {str(e)}", 'error')
            return None

    def save_model_info(self, model_id: str, symbol: str, timeframe: str,
                        model_type: str, parameters: str, metrics: str,
                        model_path: str, feature_importance: Optional[str] = None,
                        verbose: bool = True) -> bool:
        """Сохранение информации о модели"""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO models 
                (model_id, symbol, timeframe, model_type, created_at, 
                 parameters, metrics, model_path, feature_importance, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, symbol, timeframe, model_type, datetime.now(),
                  parameters, metrics, model_path, feature_importance, 1))

            self.connection.commit()
            if verbose:
                self.log(f"Model info saved for {model_id}")
                print(f"✅ Информация о модели сохранена в базу")

            return True

        except Exception as e:
            if verbose:
                self.log(f"Error saving model info: {str(e)}", 'error')
                print(f"❌ Ошибка сохранения информации о модели: {e}")
            return False

    def get_available_models(self, symbol: Optional[str] = None,
                             timeframe: Optional[str] = None,
                             model_type: Optional[str] = None,
                             active_only: bool = True,
                             verbose: bool = True) -> pd.DataFrame:
        """Получение списка доступных моделей"""
        try:
            query = 'SELECT * FROM models'
            conditions = []
            params = []

            if active_only:
                conditions.append('is_active = 1')

            if symbol:
                conditions.append('symbol = ?')
                params.append(symbol)

            if timeframe:
                conditions.append('timeframe = ?')
                params.append(timeframe)

            if model_type:
                conditions.append('model_type = ?')
                params.append(model_type)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            query += ' ORDER BY created_at DESC'

            df = pd.read_sql_query(query, self.connection, params=params)

            # Парсим даты если есть колонка created_at
            if 'created_at' in df.columns and not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])

            if verbose:
                self.log(f"Found {len(df)} models")
                if len(df) > 0:
                    print(f"📊 Найдено моделей: {len(df)}")

            return df

        except Exception as e:
            if verbose:
                self.log(f"Error getting models: {str(e)}", 'error')
            return pd.DataFrame()

    def save_backtest_result(self, result_data: Dict, verbose: bool = True) -> bool:
        """Сохранение результата бэктеста в БД"""
        try:
            columns = ', '.join(result_data.keys())
            placeholders = ', '.join(['?'] * len(result_data))

            query = f'INSERT INTO backtest_results ({columns}) VALUES ({placeholders})'
            self.cursor.execute(query, list(result_data.values()))

            self.connection.commit()
            if verbose:
                self.log(f"Backtest result saved")
                print(f"✅ Результат бэктеста сохранен")

            return True

        except Exception as e:
            if verbose:
                self.log(f"Error saving backtest result: {str(e)}", 'error')
                print(f"❌ Ошибка сохранения результата бэктеста: {e}")
            return False

    def get_backtest_results(self, model_id: Optional[str] = None,
                             symbol: Optional[str] = None,
                             limit: int = 10,
                             verbose: bool = True) -> pd.DataFrame:
        """Получение результатов бэктеста"""
        try:
            query = 'SELECT * FROM backtest_results'
            conditions = []
            params = []

            if model_id:
                conditions.append('model_id = ?')
                params.append(model_id)

            if symbol:
                conditions.append('symbol = ?')
                params.append(symbol)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            query += ' ORDER BY test_date DESC LIMIT ?'
            params.append(limit)

            df = pd.read_sql_query(query, self.connection, params=params)

            # Парсим даты
            date_columns = ['test_date', 'start_date', 'end_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            if verbose:
                self.log(f"Retrieved {len(df)} backtest results")

            return df

        except Exception as e:
            if verbose:
                self.log(f"Error getting backtest results: {str(e)}", 'error')
            return pd.DataFrame()

    def delete_model(self, model_id: str, verbose: bool = True) -> bool:
        """Удаление модели из базы данных"""
        try:
            # Проверяем существование модели
            self.cursor.execute('SELECT model_id FROM models WHERE model_id = ?', (model_id,))
            if not self.cursor.fetchone():
                if verbose:
                    self.log(f"Model {model_id} not found", 'warning')
                    print(f"❌ Модель {model_id} не найдена")
                return False

            # Удаляем модель
            self.cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
            self.connection.commit()

            deleted_rows = self.cursor.rowcount

            if verbose:
                self.log(f"Model {model_id} deleted. Rows affected: {deleted_rows}")
                print(f"✅ Модель {model_id} удалена")

            return deleted_rows > 0

        except Exception as e:
            if verbose:
                self.log(f"Error deleting model {model_id}: {str(e)}", 'error')
                print(f"❌ Ошибка удаления модели {model_id}: {e}")
            return False

    def delete_all_models(self, symbol: Optional[str] = None,
                          model_type: Optional[str] = None,
                          verbose: bool = True) -> int:
        """Удаление всех моделей или по фильтру"""
        try:
            query = 'DELETE FROM models'
            conditions = []
            params = []

            if symbol:
                conditions.append('symbol = ?')
                params.append(symbol)

            if model_type:
                conditions.append('model_type = ?')
                params.append(model_type)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            # Выполняем удаление
            self.cursor.execute(query, params)
            self.connection.commit()

            deleted_rows = self.cursor.rowcount

            if verbose:
                self.log(f"Deleted {deleted_rows} models")
                print(f"✅ Удалено моделей: {deleted_rows}")

            return deleted_rows

        except Exception as e:
            if verbose:
                self.log(f"Error deleting models: {str(e)}", 'error')
                print(f"❌ Ошибка удаления моделей: {e}")
            return 0

    def update_model_state(self, model_id: str, is_active: bool, verbose: bool = True) -> bool:
        """Обновление состояния модели (активна/неактивна)"""
        try:
            self.cursor.execute('''
                UPDATE models 
                SET is_active = ? 
                WHERE model_id = ?
            ''', (1 if is_active else 0, model_id))

            self.connection.commit()

            updated_rows = self.cursor.rowcount

            if verbose:
                status = "активирована" if is_active else "деактивирована"
                self.log(f"Model {model_id} {status}")
                print(f"✅ Модель {model_id} {status}")

            return updated_rows > 0

        except Exception as e:
            if verbose:
                self.log(f"Error updating model state: {str(e)}", 'error')
                print(f"❌ Ошибка обновления состояния модели: {e}")
            return False

    def get_model_state(self, model_id: str) -> Optional[bool]:
        """Получение текущего состояния модели"""
        try:
            self.cursor.execute('SELECT is_active FROM models WHERE model_id = ?', (model_id,))
            result = self.cursor.fetchone()

            if result:
                return bool(result[0])
            return None

        except Exception as e:
            self.log(f"Error getting model state: {str(e)}", 'error')
            return None

    def get_system_stats(self) -> Dict[str, Any]:
        """Получение статистики системы"""
        try:
            stats = {}

            # Количество моделей
            self.cursor.execute('SELECT COUNT(*) FROM models')
            stats['total_models'] = self.cursor.fetchone()[0]

            self.cursor.execute('SELECT COUNT(*) FROM models WHERE is_active = 1')
            stats['active_models'] = self.cursor.fetchone()[0]

            # Количество записей данных
            self.cursor.execute('SELECT COUNT(*) FROM historical_data')
            stats['total_data_records'] = self.cursor.fetchone()[0]

            # Размер базы данных
            import os
            if os.path.exists(config.DB_PATH):
                stats['db_size_mb'] = os.path.getsize(config.DB_PATH) / (1024 * 1024)
            else:
                stats['db_size_mb'] = 0

            return stats

        except Exception as e:
            self.log(f"Error getting system stats: {str(e)}", 'error')
            return {}

    def test_connection(self) -> bool:
        """Тестирование подключения к базе данных"""
        try:
            self.cursor.execute('SELECT 1')
            result = self.cursor.fetchone()
            return result is not None and result[0] == 1
        except:
            return False

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.connection:
            self.connection.close()
            self.log("Database connection closed")