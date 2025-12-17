"""
Конфигурационный файл торгового бота
"""

import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TimeframeConfig:
    """Конфигурация таймфреймов"""
    TRAINING_TIMEFRAMES = ['5m', '15m']  # Для обучения
    TRADING_TIMEFRAME = '5m'  # Для торговли (скальпинг на 1h - не совсем скальпинг, но ладно)
    BACKTEST_TIMEFRAME = '5m'  # Для бэктеста - ДОЛЖЕН СОВПАДАТЬ С ОБУЧЕНИЕМ!


@dataclass
class TradingConfig:
    """Конфигурация торговли"""
    # Криптопары
    SYMBOLS = ['DOGEUSDT',] #, 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']
    MAIN_SYMBOL = 'BTCUSDT'

    # Риск-менеджмент
    STOP_LOSS_PCT = 2.0  # 2%
    TAKE_PROFIT_PCT = 4.0  # 4%
    POSITION_SIZE_PCT = 1.0  # % от депозита на сделку
    MAX_POSITIONS = 3  # Максимальное количество одновременных позиций

    # Торговые параметры
    COMMISSION = 0.001  # 0.1% комиссия
    LEVERAGE = 3  # Кредитное плечо


@dataclass
class ModelConfig:
    """Конфигурация моделей"""
    # Параметры LSTM
    LSTM_UNITS = [128, 64, 32]
    LSTM_DROPOUT = 0.2
    LSTM_EPOCHS = 100
    LSTM_BATCH_SIZE = 32
    LSTM_PATIENCE = 10

    # Параметры XGBoost
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.01
    XGB_N_ESTIMATORS = 500
    XGB_EARLY_STOPPING_ROUNDS = 50

    # Общие параметры
    TRAIN_TEST_SPLIT = 0.8
    LOOKBACK_WINDOW = 60  # Окно для анализа
    PREDICTION_HORIZON = 10  # Прогноз на N свечей вперед

    # НОВЫЕ ПАРАМЕТРЫ: Пороги для создания меток
    TARGET_THRESHOLD_UP = 0.005  # 0.5% для LONG
    TARGET_THRESHOLD_DOWN = -0.005  # -0.5% для SHORT
    USE_DYNAMIC_THRESHOLDS = True  # Использовать динамические пороги на основе волатильности

@dataclass
class DataConfig:
    """Конфигурация данных"""
    # Периоды
    TRAINING_PERIOD_DAYS = 365 * 2  # 2 года для обучения
    BACKTEST_PERIOD_DAYS = 30  # 30 дней для бэктеста
    UPDATE_INTERVAL_HOURS = 1  # Обновление данных каждые N часов

    # Технические индикаторы
    INDICATORS = [
        'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi', 'macd', 'bb_upper', 'bb_middle', 'bb_lower',
        'atr', 'volume_sma', 'obv'
    ]

    # Фичи
    FEATURES = [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'volatility', 'spread'
    ]


@dataclass
class BacktestConfig:
    """Конфигурация бэктеста"""
    INITIAL_BALANCE = 10000.0  # Начальный баланс
    TRADE_FEE = 0.001  # 0.1%
    SLIPPAGE = 0.0005  # Проскальзывание

    # Критерии оценки
    METRICS = [
        'total_return', 'sharpe_ratio', 'max_drawdown',
        'win_rate', 'profit_factor', 'total_trades'
    ]


class Config:
    """Основной класс конфигурации"""

    def __init__(self):
        # API ключи
        self.API_KEY = os.getenv('BINANCE_API_KEY', '')
        self.API_SECRET = os.getenv('BINANCE_API_SECRET', '')

        # Пути
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DB_PATH = os.getenv('DB_PATH', 'data/crypto_data.db')
        self.MODEL_DIR = os.getenv('MODEL_PATH', 'models/')
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')

        # Создание директорий
        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # Конфигурации
        self.timeframe = TimeframeConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.backtest = BacktestConfig()

        # Временные метки - ИСПРАВЛЕНО
        self.NOW = datetime.now()

        # Обучение: данные за последние 2 года (730 дней)
        self.TRAIN_END_DATE = self.NOW - timedelta(days=self.data.BACKTEST_PERIOD_DAYS)
        self.TRAIN_START_DATE = self.TRAIN_END_DATE - timedelta(days=self.data.TRAINING_PERIOD_DAYS)

        # Бэктест: данные за 30 дней перед тренировочным набором
        self.BACKTEST_END_DATE = self.TRAIN_END_DATE  # То же самое что конец обучения
        self.BACKTEST_START_DATE = self.BACKTEST_END_DATE - timedelta(days=self.data.BACKTEST_PERIOD_DAYS)

        # Для отладки
        print(f"Training period: {self.TRAIN_START_DATE.date()} to {self.TRAIN_END_DATE.date()}")
        print(f"Backtest period: {self.BACKTEST_START_DATE.date()} to {self.BACKTEST_END_DATE.date()}")
        print(f"Using timeframe for backtest: {self.timeframe.BACKTEST_TIMEFRAME}")


# Создаем экземпляр конфигурации
config = Config()