"""
Модуль для интеллектуального управления историческими данными
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from config import config
from modules.database import Database
from modules.data_fetcher import DataFetcher
from modules.state_manager import state_manager


class DataManager:
    """Менеджер для интеллектуального управления данными"""

    def __init__(self, verbose: bool = True):
        """Инициализация менеджера данных"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.data_fetcher = DataFetcher()

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

    def get_missing_period(self, symbol: str, timeframe: str,
                           start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """
        Определение отсутствующих периодов данных

        Returns:
            Список периодов (start, end) которые нужно загрузить
        """
        try:
            # Получаем существующие данные
            existing_data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                verbose=False
            )

            if existing_data.empty:
                # Нет данных вообще, загружаем весь период
                return [(start_date, end_date)]

            # Определяем пропуски
            missing_periods = []
            current_start = start_date

            # Сортируем по времени
            existing_data = existing_data.sort_index()

            for idx in existing_data.index:
                if idx > current_start:
                    # Нашли пропуск
                    missing_periods.append((current_start, idx - timedelta(minutes=1)))
                current_start = idx + timedelta(minutes=1)

            # Проверяем конец периода
            if current_start < end_date:
                missing_periods.append((current_start, end_date))

            return missing_periods

        except Exception as e:
            self.log(f"Error getting missing periods: {e}", 'error')
            return [(start_date, end_date)]

    def update_data_for_symbol(self, symbol: str, timeframe: str,
                               days_back: int = 120, verbose: bool = True) -> Dict:
        """
        Интеллектуальное обновление данных для символа и таймфрейма

        Returns:
            Словарь с результатами обновления
        """
        try:
            if verbose:
                print(f"\n📊 Обновление данных для {symbol} {timeframe}")
                print(f"   Период: {days_back} дней")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Определяем отсутствующие данные
            missing_periods = self.get_missing_period(symbol, timeframe, start_date, end_date)

            if not missing_periods:
                if verbose:
                    print(f"   ✅ Все данные уже есть в базе")
                return {'symbol': symbol, 'timeframe': timeframe, 'status': 'already_exists', 'loaded': 0}

            total_loaded = 0
            for period_start, period_end in missing_periods:
                period_days = (period_end - period_start).days

                if verbose:
                    print(f"   📥 Загрузка: {period_start.date()} - {period_end.date()} ({period_days} дней)")

                # Загружаем данные
                data = self.data_fetcher.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days_back=period_days + 10  # Немного с запасом
                )

                if data.empty:
                    if verbose:
                        print(f"   ❌ Не удалось загрузить данные")
                    continue

                # Фильтруем по нужному периоду
                data = data[(data.index >= period_start) & (data.index <= period_end)]

                if not data.empty:
                    # Сохраняем в БД
                    self.db.store_historical_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=data,
                        verbose=verbose
                    )

                    total_loaded += len(data)

                    if verbose:
                        print(f"   ✅ Загружено {len(data)} свечей")
                else:
                    if verbose:
                        print(f"   ⚠️  Нет данных для этого периода")

            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'success',
                'loaded': total_loaded,
                'periods_missing': len(missing_periods)
            }

            if verbose:
                if total_loaded > 0:
                    print(f"   🎉 Всего загружено {total_loaded} свечей")
                else:
                    print(f"   ℹ️  Данные не требуют обновления")

            return result

        except Exception as e:
            self.log(f"Error updating data for {symbol} {timeframe}: {e}", 'error')
            return {'symbol': symbol, 'timeframe': timeframe, 'status': 'error', 'error': str(e)}

    def initialize_all_data(self, days_back: int = 365, verbose: bool = True) -> Dict:
        """
        Инициализация всех данных (первичная загрузка)

        Returns:
            Словарь с результатами
        """
        try:
            results = {}
            total_loaded = 0

            print(f"\n🚀 ИНИЦИАЛИЗАЦИЯ ВСЕХ ДАННЫХ")
            print(f"   Период: {days_back} дней")
            print(f"   Символы: {len(config.trading.ALL_SYMBOLS)}")
            print(f"   Таймфреймы: {len(config.timeframe.AVAILABLE_TIMEFRAMES)}")
            print("=" * 60)

            for symbol in config.trading.ALL_SYMBOLS:
                for timeframe in config.timeframe.AVAILABLE_TIMEFRAMES:
                    if verbose:
                        print(f"\n📊 {symbol} {timeframe}")

                    result = self.update_data_for_symbol(
                        symbol=symbol,
                        timeframe=timeframe,
                        days_back=days_back,
                        verbose=verbose
                    )

                    results[f"{symbol}_{timeframe}"] = result
                    total_loaded += result.get('loaded', 0)

            summary = {
                'total_symbols': len(config.trading.ALL_SYMBOLS),
                'total_timeframes': len(config.timeframe.AVAILABLE_TIMEFRAMES),
                'total_pairs': len(config.trading.ALL_SYMBOLS) * len(config.timeframe.AVAILABLE_TIMEFRAMES),
                'total_loaded': total_loaded,
                'results': results
            }

            print(f"\n{'=' * 60}")
            print(f"🎉 ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА")
            print(f"   Всего загружено: {total_loaded} свечей")
            print(f"   Обработано пар: {summary['total_pairs']}")
            print(f"{'=' * 60}")

            return summary

        except Exception as e:
            self.log(f"Error initializing all data: {e}", 'error')
            return {'status': 'error', 'error': str(e)}

    def ensure_training_data(self, symbol: str, timeframe: str,
                             training_days: int, backtest_days: int) -> bool:
        """
        Гарантирует наличие данных для обучения с учетом бэктеста
        """
        try:
            # Рассчитываем период обучения
            end_date = datetime.now() - timedelta(days=backtest_days)
            start_date = end_date - timedelta(days=training_days)

            # Проверяем наличие данных
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                verbose=False
            )

            if len(data) < training_days * 24 * 12:  # Минимальное количество свечей для 5m
                print(f"\n⚠️  Недостаточно данных для обучения {symbol} {timeframe}")
                print(f"   Нужно загрузить данные с {start_date.date()} по {end_date.date()}")

                # Загружаем недостающие данные
                days_to_load = training_days + backtest_days + 30  # С запасом
                result = self.update_data_for_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    days_back=days_to_load,
                    verbose=True
                )

                return result.get('loaded', 0) > 0

            return True

        except Exception as e:
            self.log(f"Error ensuring training data: {e}", 'error')
            return False


# Глобальный экземпляр менеджера данных
data_manager = DataManager()