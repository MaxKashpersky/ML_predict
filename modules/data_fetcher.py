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


logger = logging.getLogger(__name__)


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

    def fetch_historical_data(self, symbol: str, timeframe: str,
                              days_back: int = 365) -> pd.DataFrame:
        """
        Получение исторических данных за указанный период
        """
        try:
            # Определяем дату начала
            since_date = datetime.now() - timedelta(days=days_back)
            since = int(since_date.timestamp() * 1000)

            all_data = []
            total_candles = 0

            print(f"\n📥 Загрузка данных {symbol} {timeframe}")
            print(f"Период: {since_date.date()} - {datetime.now().date()}")
            print("Прогресс:", end=" ")

            while True:
                try:
                    # Получение данных порциями
                    df = self.fetch_ohlcv(symbol, timeframe, since=since)
                    if df.empty or len(df) == 0:
                        print("\n")
                        break

                    all_data.append(df)
                    total_candles += len(df)

                    # Обновление времени для следующей порции
                    since = int(df.index[-1].timestamp() * 1000) + 1

                    # Отображение прогресса
                    print(f"{total_candles}", end=" ")

                    # Пауза для соблюдения лимитов
                    self.exchange.sleep(1000)

                    # Проверка достижения текущего времени
                    if len(df) < 1000:
                        print("\n")
                        break

                except Exception as e:
                    print(f"\nОшибка при загрузке: {e}")
                    break

            if all_data:
                full_df = pd.concat(all_data)
                full_df = full_df[~full_df.index.duplicated(keep='first')]
                print(f"✅ Загружено {total_candles} свечей")
                return full_df.sort_index()

            print("❌ Не удалось загрузить данные")
            return pd.DataFrame()

        except Exception as e:
            print(f"\n❌ Ошибка загрузки данных: {e}")
            return pd.DataFrame()

    def fetch_data_for_training(self, symbol: str = None, timeframe: str = None) -> pd.DataFrame:
        """
        Получение данных для обучения с учетом периодов из state_manager
        """
        try:
            if symbol is None:
                symbol = state_manager.get_selected_symbol()
                if not symbol:
                    logger.error("No symbol selected")
                    return pd.DataFrame()

            if timeframe is None:
                timeframe = state_manager.get_selected_timeframe()

            # Получаем период для загрузки данных
            start_date, end_date = state_manager.get_data_fetch_dates()
            days_back = (end_date - start_date).days + 10  # +10 дней запаса

            logger.info(f"Fetching {days_back} days of data for {symbol} {timeframe}")

            data = self.fetch_historical_data(symbol, timeframe, days_back)

            if not data.empty:
                # Фильтруем данные по нужному периоду
                data = data[(data.index >= start_date) & (data.index <= end_date)]
                logger.info(f"Retrieved {len(data)} candles for training")

            return data

        except Exception as e:
            logger.error(f"Error fetching data for training: {e}")
            return pd.DataFrame()