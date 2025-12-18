"""
Модуль для получения данных с биржи
"""

import ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional
from config import config
from modules.state_manager import state_manager

import sys
import time

logger = logging.getLogger(__name__)

class ProgressBar:
    """Класс для отображения прогресс-бара"""
    def __init__(self, total, prefix='', suffix='', length=50, fill='█'):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.start_time = time.time()

    def update(self, iteration):
        """Обновить прогресс-бар"""
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)

        elapsed_time = time.time() - self.start_time
        if iteration > 0:
            time_per_item = elapsed_time / iteration
            remaining = self.total - iteration
            eta = time_per_item * remaining
            eta_str = f"ETA: {self.format_time(eta)}"
        else:
            eta_str = "ETA: --:--:--"

        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} {eta_str}')
        sys.stdout.flush()

    def finish(self):
        """Завершить прогресс-бар"""
        sys.stdout.write('\n')
        sys.stdout.flush()

    @staticmethod
    def format_time(seconds):
        """Форматирование времени"""
        if seconds < 60:
            return f"{seconds:.0f}с"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes:.0f}м {seconds:.0f}с"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}ч {minutes:.0f}м"



class DataFetcher:
    def __init__(self):
        """Инициализация подключения к бирже"""
        self.exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if config.trading.LEVERAGE > 1 else 'spot'
            }
        })

        # Проверка подключения
        try:
            self.exchange.load_markets()
            logger.info("Connected to Binance")
        except Exception as e:
            logger.error(f"Error connecting to Binance: {e}")

    def fetch_ohlcv(self, symbol: str, timeframe: str,
                    since: Optional[int] = None,
                    limit: Optional[int] = 1000) -> pd.DataFrame:
        """
        Получение OHLCV данных

        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        try:
            # Преобразование символа в формат CCXT
            symbol_ccxt = symbol.replace('USDT', '/USDT')

            # Получение данных
            ohlcv = self.exchange.fetch_ohlcv(
                symbol_ccxt,
                timeframe,
                since=since,
                limit=limit
            )

            # Преобразование в DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            # Преобразование timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.debug(f"Retrieved {len(df)} candles for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    # data_fetcher.py - обновленная функция fetch_historical_data
    def fetch_historical_data(self, symbol: str, timeframe: str,
                              days_back: int = 365,
                              show_progress: bool = True) -> pd.DataFrame:
        """
        Получение исторических данных за указанный период с прогресс-баром
        """
        try:
            # Определяем дату начала
            since_date = datetime.now() - timedelta(days=days_back)
            since = int(since_date.timestamp() * 1000)

            # Рассчитываем примерное количество свечей
            timeframe_minutes = self.get_timeframe_minutes(timeframe)
            total_minutes = days_back * 24 * 60
            estimated_candles = total_minutes // timeframe_minutes

            if show_progress:
                print(f"\n📥 Загрузка данных {symbol} {timeframe}")
                print(f"📅 Период: {since_date.date()} - {datetime.now().date()}")
                print(f"📊 Примерно свечей: {estimated_candles:,}")

                # Создаем прогресс-бар
                progress = ProgressBar(estimated_candles,
                                       prefix='Прогресс:',
                                       suffix='завершено',
                                       length=30)

            all_data = []
            total_candles = 0

            while True:
                try:
                    # Получение данных порциями
                    df = self.fetch_ohlcv(symbol, timeframe, since=since)
                    if df.empty or len(df) == 0:
                        if show_progress:
                            progress.finish()
                        break

                    all_data.append(df)
                    total_candles += len(df)

                    # Обновление времени для следующей порции
                    since = int(df.index[-1].timestamp() * 1000) + 1

                    # Обновляем прогресс-бар
                    if show_progress:
                        progress.update(min(total_candles, estimated_candles))

                    # Пауза для соблюдения лимитов
                    self.exchange.sleep(500)

                    # Проверка достижения текущего времени
                    if len(df) < 1000:
                        if show_progress:
                            progress.update(estimated_candles)  # Завершаем на 100%
                            progress.finish()
                        break

                except Exception as e:
                    if show_progress:
                        progress.finish()
                        print(f"⚠️ Ошибка при загрузке: {e}")
                    break

            if all_data:
                full_df = pd.concat(all_data)
                full_df = full_df[~full_df.index.duplicated(keep='first')]
                if show_progress:
                    print(f"✅ Загружено {total_candles:,} свечей")
                    print(f"📈 Первая свеча: {full_df.index[0]}")
                    print(f"📉 Последняя свеча: {full_df.index[-1]}")
                return full_df.sort_index()

            if show_progress:
                print("❌ Не удалось загрузить данные")
            return pd.DataFrame()

        except Exception as e:
            if show_progress:
                print(f"\n❌ Ошибка загрузки данных: {e}")
            return pd.DataFrame()

    def get_timeframe_minutes(self, timeframe: str) -> int:
        """Конвертация таймфрейма в минуты"""
        timeframe_map = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1w': 10080,
            '1M': 43200
        }
        return timeframe_map.get(timeframe, 5)  # по умолчанию 5 минут