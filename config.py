"""
Конфигурационный файл торгового бота
"""

import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TimeframeConfig:
    """Конфигурация таймфреймов"""
    AVAILABLE_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']  # Все доступные таймфреймы
    TRAINING_TIMEFRAMES = ['5m']  # По умолчанию для обучения
    TRADING_TIMEFRAME = '5m'  # Для торговли
    BACKTEST_TIMEFRAME = '5m'  # Для бэктеста


@dataclass
class TradingConfig:
    """Конфигурация торговли"""
    # Криптопары (все доступные)
    ALL_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'XRPUSDT', 'DOTUSDT']
    SYMBOLS = ['DOGEUSDT']  # По умолчанию
    MAIN_SYMBOL = 'BTCUSDT'

    # Риск-менеджмент (значения по умолчанию)
    _stop_loss_pct: float = 2.0
    _take_profit_pct: float = 4.0
    _position_size_pct: float = 1.0
    _max_positions: int = 3
    _commission: float = 0.001
    _leverage: int = 3

    @property
    def STOP_LOSS_PCT(self) -> float:
        return self._stop_loss_pct

    @STOP_LOSS_PCT.setter
    def STOP_LOSS_PCT(self, value: float):
        self._stop_loss_pct = value

    @property
    def TAKE_PROFIT_PCT(self) -> float:
        return self._take_profit_pct

    @TAKE_PROFIT_PCT.setter
    def TAKE_PROFIT_PCT(self, value: float):
        self._take_profit_pct = value

    @property
    def POSITION_SIZE_PCT(self) -> float:
        return self._position_size_pct

    @POSITION_SIZE_PCT.setter
    def POSITION_SIZE_PCT(self, value: float):
        self._position_size_pct = value

    @property
    def MAX_POSITIONS(self) -> int:
        return self._max_positions

    @MAX_POSITIONS.setter
    def MAX_POSITIONS(self, value: int):
        self._max_positions = value

    @property
    def COMMISSION(self) -> float:
        return self._commission

    @COMMISSION.setter
    def COMMISSION(self, value: float):
        self._commission = value

    @property
    def LEVERAGE(self) -> int:
        return self._leverage

    @LEVERAGE.setter
    def LEVERAGE(self, value: int):
        self._leverage = value


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

    # Пороги для создания меток
    TARGET_THRESHOLD_UP = 0.005  # 0.5% для LONG
    TARGET_THRESHOLD_DOWN = -0.005  # -0.5% для SHORT
    USE_DYNAMIC_THRESHOLDS = True  # Использовать динамические пороги на основе волатильности


@dataclass
class DataConfig:
    """Конфигурация данных"""
    # Периоды (по умолчанию)
    TRAINING_PERIOD_DAYS = 365 * 2  # 2 года для обучения
    BACKTEST_PERIOD_DAYS = 30  # 30 дней для бэктеста
    UPDATE_INTERVAL_HOURS = 1  # Обновление данных каждые N часов
    DEFAULT_UPDATE_DAYS = 120  # По умолчанию загружаем 120 дней

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
    # Значения по умолчанию
    _initial_balance: float = 10000.0
    _trade_fee: float = 0.001
    _slippage: float = 0.0005

    @property
    def INITIAL_BALANCE(self) -> float:
        return self._initial_balance

    @INITIAL_BALANCE.setter
    def INITIAL_BALANCE(self, value: float):
        self._initial_balance = value

    @property
    def TRADE_FEE(self) -> float:
        return self._trade_fee

    @TRADE_FEE.setter
    def TRADE_FEE(self, value: float):
        self._trade_fee = value

    @property
    def SLIPPAGE(self) -> float:
        return self._slippage

    @SLIPPAGE.setter
    def SLIPPAGE(self, value: float):
        self._slippage = value

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
        self.CONFIG_FILE = os.path.join(self.BASE_DIR, 'data', 'user_config.json')

        # Создание директорий
        os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.BASE_DIR, 'data'), exist_ok=True)

        # Конфигурации
        self.timeframe = TimeframeConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.data = DataConfig()
        self.backtest = BacktestConfig()

        # Загрузка пользовательских настроек
        self.load_user_config()

        # Временные метки
        self.NOW = datetime.now()

        # Для отладки
        print(f"Config loaded successfully")

    def load_user_config(self):
        """Загрузка пользовательских настроек из файла"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    user_config = json.load(f)

                # Применяем настройки
                if 'trading' in user_config:
                    trading = user_config['trading']
                    self.trading.STOP_LOSS_PCT = trading.get('STOP_LOSS_PCT', self.trading.STOP_LOSS_PCT)
                    self.trading.TAKE_PROFIT_PCT = trading.get('TAKE_PROFIT_PCT', self.trading.TAKE_PROFIT_PCT)
                    self.trading.POSITION_SIZE_PCT = trading.get('POSITION_SIZE_PCT', self.trading.POSITION_SIZE_PCT)
                    self.trading.MAX_POSITIONS = trading.get('MAX_POSITIONS', self.trading.MAX_POSITIONS)
                    self.trading.COMMISSION = trading.get('COMMISSION', self.trading.COMMISSION)
                    self.trading.LEVERAGE = trading.get('LEVERAGE', self.trading.LEVERAGE)

                if 'backtest' in user_config:
                    backtest = user_config['backtest']
                    self.backtest.INITIAL_BALANCE = backtest.get('INITIAL_BALANCE', self.backtest.INITIAL_BALANCE)
                    self.backtest.TRADE_FEE = backtest.get('TRADE_FEE', self.trading.COMMISSION)  # Синхронизация с комиссией
                    self.backtest.SLIPPAGE = backtest.get('SLIPPAGE', self.backtest.SLIPPAGE)

                print(f"User configuration loaded from {self.CONFIG_FILE}")
        except Exception as e:
            print(f"Error loading user config: {e}")

    def save_user_config(self):
        """Сохранение пользовательских настроек в файл"""
        try:
            user_config = {
                'trading': {
                    'STOP_LOSS_PCT': self.trading.STOP_LOSS_PCT,
                    'TAKE_PROFIT_PCT': self.trading.TAKE_PROFIT_PCT,
                    'POSITION_SIZE_PCT': self.trading.POSITION_SIZE_PCT,
                    'MAX_POSITIONS': self.trading.MAX_POSITIONS,
                    'COMMISSION': self.trading.COMMISSION,
                    'LEVERAGE': self.trading.LEVERAGE
                },
                'backtest': {
                    'INITIAL_BALANCE': self.backtest.INITIAL_BALANCE,
                    'TRADE_FEE': self.trading.COMMISSION,  # Всегда синхронизируем с комиссией
                    'SLIPPAGE': self.backtest.SLIPPAGE
                }
            }

            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(user_config, f, indent=2)

            print(f"User configuration saved to {self.CONFIG_FILE}")

        except Exception as e:
            print(f"Error saving user config: {e}")


# Создаем экземпляр конфигурации
config = Config()