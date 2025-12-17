"""
Модуль управления состоянием программы
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config import config


class StateManager:
    """Управление состоянием программы"""

    def __init__(self):
        """Инициализация менеджера состояния"""
        self.state_file = os.path.join(config.BASE_DIR, 'data', 'session_state.json')
        self.state = self.load_state()

        # Параметры по умолчанию
        if 'selected_symbol' not in self.state:
            self.state['selected_symbol'] = None
        if 'training_days' not in self.state:
            self.state['training_days'] = config.data.TRAINING_PERIOD_DAYS
        if 'backtest_days' not in self.state:
            self.state['backtest_days'] = config.data.BACKTEST_PERIOD_DAYS
        if 'selected_timeframe' not in self.state:
            self.state['selected_timeframe'] = config.timeframe.TRADING_TIMEFRAME

    def load_state(self) -> Dict:
        """Загрузка состояния из файла"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save_state(self):
        """Сохранение состояния в файл"""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"Error saving state: {e}")

    def set_selected_symbol(self, symbol: str):
        """Установка выбранного символа"""
        self.state['selected_symbol'] = symbol
        self.save_state()

    def get_selected_symbol(self) -> Optional[str]:
        """Получение выбранного символа"""
        return self.state.get('selected_symbol')

    def set_training_period(self, days: int):
        """Установка периода обучения (дней)"""
        self.state['training_days'] = days
        self.save_state()

    def get_training_period(self) -> int:
        """Получение периода обучения"""
        return self.state.get('training_days', config.data.TRAINING_PERIOD_DAYS)

    def set_backtest_period(self, days: int):
        """Установка периода бэктеста (дней)"""
        self.state['backtest_days'] = days
        self.save_state()

    def get_backtest_period(self) -> int:
        """Получение периода бэктеста"""
        return self.state.get('backtest_days', config.data.BACKTEST_PERIOD_DAYS)

    def set_selected_timeframe(self, timeframe: str):
        """Установка выбранного таймфрейма"""
        self.state['selected_timeframe'] = timeframe
        self.save_state()

    def get_selected_timeframe(self) -> str:
        """Получение выбранного таймфрейма"""
        return self.state.get('selected_timeframe', config.timeframe.TRADING_TIMEFRAME)

    def get_training_dates(self) -> tuple:
        """
        Получение дат для обучения с учетом периода бэктеста

        Returns:
            tuple: (start_date, end_date) для обучения
        """
        now = datetime.now()

        # Общая длительность: обучение + бэктест
        total_days = self.get_training_period() + self.get_backtest_period()

        # Дата окончания обучения = сейчас - период бэктеста
        train_end = now - timedelta(days=self.get_backtest_period())

        # Дата начала обучения
        train_start = train_end - timedelta(days=self.get_training_period())

        return train_start, train_end

    def get_backtest_dates(self) -> tuple:
        """
        Получение дат для бэктеста

        Returns:
            tuple: (start_date, end_date) для бэктеста
        """
        now = datetime.now()

        # Дата окончания бэктеста = сейчас
        backtest_end = now

        # Дата начала бэктеста
        backtest_start = now - timedelta(days=self.get_backtest_period())

        return backtest_start, backtest_end

    def get_data_fetch_dates(self) -> tuple:
        """
        Получение дат для загрузки данных

        Returns:
            tuple: (start_date, end_date) для загрузки данных
        """
        now = datetime.now()

        # Загружаем данные за период: обучение + бэктест + небольшой запас
        total_days = self.get_training_period() + self.get_backtest_period() + 30

        start_date = now - timedelta(days=total_days)
        end_date = now

        return start_date, end_date

    def reset_state(self):
        """Сброс состояния"""
        self.state = {
            'selected_symbol': None,
            'training_days': config.data.TRAINING_PERIOD_DAYS,
            'backtest_days': config.data.BACKTEST_PERIOD_DAYS,
            'selected_timeframe': config.timeframe.TRADING_TIMEFRAME
        }
        self.save_state()


# Глобальный экземпляр менеджера состояния
state_manager = StateManager()