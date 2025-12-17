"""
Модуль для работы с базой данных
"""

import sqlite3
import pandas as pd
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config import config


class Database:
    def __init__(self, verbose: bool = True):
        """Инициализация базы данных"""
        self.verbose = verbose
        self.setup_logging()
        self.conn = None
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
            self.conn = sqlite3.connect(config.DB_PATH)
            self.conn.row_factory = sqlite3.Row
            self.log("Database connection established")
        except Exception as e:
            self.log(f"Error connecting to database: {str(e)}", 'error')

    def init_tables(self):
        """Инициализация таблиц"""
        try:
            cursor = self.conn.cursor()

            # Таблица для хранения исторических данных
            cursor.execute('''
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    last_update DATETIME,
                    last_candle_timestamp DATETIME,
                    PRIMARY KEY (symbol, timeframe)
                )
            ''')

            # Таблица для моделей
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    parameters TEXT,
                    metrics TEXT,
                    model_path TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')

            # Таблица для результатов бэктеста
            cursor.execute('''
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
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS idx_historical_data ON historical_data(symbol, timeframe, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models ON models(symbol, timeframe, is_active)')

            self.conn.commit()
            self.log("Database tables initialized")

        except Exception as e:
            self.log(f"Error initializing tables: {str(e)}", 'error')

    def store_historical_data(self, symbol: str, timeframe: str, data: pd.DataFrame, verbose: bool = True):
        """
        Сохранение исторических данных в базу
        """
        try:
            if data.empty:
                self.log(f"No data to save for {symbol} {timeframe}", 'warning')
                return

            cursor = self.conn.cursor()
            added_count = 0
            updated_count = 0

            for idx, row in data.iterrows():
                try:
                    timestamp_str = idx.strftime('%Y-%m-%d %H:%M:%S') if hasattr(idx, 'strftime') else str(idx)

                    cursor.execute('''
                        INSERT OR IGNORE INTO historical_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, timeframe, timestamp_str, row['open'], row['high'],
                          row['low'], row['close'], row['volume']))

                    if cursor.rowcount > 0:
                        added_count += 1
                    else:
                        cursor.execute('''
                            UPDATE historical_data 
                            SET open=?, high=?, low=?, close=?, volume=?
                            WHERE symbol=? AND timeframe=? AND timestamp=?
                        ''', (row['open'], row['high'], row['low'],
                              row['close'], row['volume'], symbol, timeframe, timestamp_str))
                        updated_count += 1

                except Exception as e:
                    if verbose:
                        self.log(f"Error saving data: {str(e)}", 'error')
                    continue

            # Обновление метаданных
            last_timestamp = data.index.max()
            last_timestamp_str = last_timestamp.strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute('''
                INSERT OR REPLACE INTO metadata (symbol, timeframe, last_update, last_candle_timestamp)
                VALUES (?, ?, ?, ?)
            ''', (symbol, timeframe, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), last_timestamp_str))

            self.conn.commit()

            if verbose:
                self.log(
                    f"Data saved: added {added_count}, updated {updated_count} records")

        except Exception as e:
            self.log(f"Error saving data: {str(e)}", 'error')

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

            query += ' ORDER BY timestamp'

            df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['timestamp'])

            if not df.empty:
                df.set_index('timestamp', inplace=True)
                if verbose:
                    self.log(f"Retrieved {len(df)} records")
            else:
                if verbose:
                    self.log(f"No data found", 'warning')

            return df

        except Exception as e:
            self.log(f"Error retrieving data: {str(e)}", 'error')
            return pd.DataFrame()

    def get_last_timestamp(self, symbol: str, timeframe: str, verbose: bool = True) -> Optional[datetime]:
        """Получение времени последней свечи"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT last_candle_timestamp FROM metadata 
                WHERE symbol = ? AND timeframe = ?
            ''', (symbol, timeframe))

            result = cursor.fetchone()
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
                        model_path: str, verbose: bool = True):
        """Сохранение информации о модели"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO models 
                (model_id, symbol, timeframe, model_type, created_at, 
                 parameters, metrics, model_path, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (model_id, symbol, timeframe, model_type, datetime.now(),
                  parameters, metrics, model_path, 1))

            self.conn.commit()
            if verbose:
                self.log(f"Model info saved")

        except Exception as e:
            if verbose:
                self.log(f"Error saving model info: {str(e)}", 'error')

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

            df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['created_at'])

            if verbose:
                self.log(f"Found {len(df)} models")

            return df

        except Exception as e:
            if verbose:
                self.log(f"Error getting models: {str(e)}", 'error')
            return pd.DataFrame()

    def save_backtest_result(self, result_data: Dict, verbose: bool = True):
        """Сохранение результата бэктеста в БД"""
        try:
            cursor = self.conn.cursor()
            columns = ', '.join(result_data.keys())
            placeholders = ', '.join(['?'] * len(result_data))

            query = f'INSERT INTO backtest_results ({columns}) VALUES ({placeholders})'
            cursor.execute(query, list(result_data.values()))

            self.conn.commit()
            if verbose:
                self.log(f"Backtest result saved")

        except Exception as e:
            if verbose:
                self.log(f"Error saving backtest result: {str(e)}", 'error')

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

            df = pd.read_sql_query(query, self.conn, params=params, parse_dates=['test_date', 'start_date', 'end_date'])

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
            cursor = self.conn.cursor()

            # Проверяем существование модели
            cursor.execute('SELECT model_id FROM models WHERE model_id = ?', (model_id,))
            if not cursor.fetchone():
                if verbose:
                    self.log(f"Model {model_id} not found", 'warning')
                return False

            # Удаляем модель
            cursor.execute('DELETE FROM models WHERE model_id = ?', (model_id,))
            self.conn.commit()

            deleted_rows = cursor.rowcount

            if verbose:
                self.log(f"Model {model_id} deleted. Rows affected: {deleted_rows}")

            return deleted_rows > 0

        except Exception as e:
            if verbose:
                self.log(f"Error deleting model {model_id}: {str(e)}", 'error')
            return False

    def delete_all_models(self, symbol: Optional[str] = None,
                          model_type: Optional[str] = None,
                          verbose: bool = True) -> int:
        """Удаление всех моделей или по фильтру"""
        try:
            cursor = self.conn.cursor()

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
            cursor.execute(query, params)
            self.conn.commit()

            deleted_rows = cursor.rowcount

            if verbose:
                self.log(f"Deleted {deleted_rows} models")

            return deleted_rows

        except Exception as e:
            if verbose:
                self.log(f"Error deleting models: {str(e)}", 'error')
            return 0

    def close(self):
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            self.log("Database connection closed")