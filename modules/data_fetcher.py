"""
Модуль для получения данных с биржи
"""

import ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from config import config

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

        Args:
            symbol: Торговая пара (BTC/USDT)
            timeframe: Таймфрейм ('1m', '5m', '1h', '1d')
            since: Время начала в миллисекундах
            limit: Количество свечей

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

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            days_back: Количество дней назад

        Returns:
            DataFrame с историческими данными
        """
        all_data = []
        since = self.exchange.parse8601(
            (datetime.now() - timedelta(days=days_back)).isoformat()
        )

        while True:
            try:
                # Получение данных порциями
                df = self.fetch_ohlcv(symbol, timeframe, since=since)
                if df.empty:
                    break

                all_data.append(df)

                # Обновление времени для следующей порции
                since = int(df.index[-1].timestamp() * 1000) + 1

                # Пауза для соблюдения лимитов
                self.exchange.sleep(1000)

                # Проверка достижения текущего времени
                if len(df) < 1000:
                    break

            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break

        if all_data:
            full_df = pd.concat(all_data)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            return full_df.sort_index()

        return pd.DataFrame()

    def get_latest_data(self, symbol: str, timeframe: str,
                        lookback: int = 100) -> pd.DataFrame:
        """
        Получение последних данных

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            lookback: Количество свечей

        Returns:
            DataFrame с последними данными
        """
        return self.fetch_ohlcv(symbol, timeframe, limit=lookback)

    def fetch_and_store_historical_data(self, symbol: str, timeframe: str,
                                        days_back: int = 365,
                                        verbose: bool = True):
        """
        Получение и сохранение исторических данных в БД

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            days_back: Количество дней назад
            verbose: Флаг логирования
        """
        try:
            from modules.database import Database

            # Получение данных
            data = self.fetch_historical_data(symbol, timeframe, days_back)

            if data.empty:
                if verbose:
                    logger.warning(f"No data retrieved for {symbol} {timeframe}")
                return

            # Сохранение в БД
            db = Database()
            db.store_historical_data(symbol, timeframe, data, verbose=verbose)

            if verbose:
                logger.info(f"Saved {len(data)} rows for {symbol} {timeframe}")

        except Exception as e:
            logger.error(f"Error fetching and storing data for {symbol}: {e}")