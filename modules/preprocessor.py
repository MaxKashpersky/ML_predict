"""
Модуль предобработки данных и расчета технических индикаторов
"""

import pandas as pd
import numpy as np
import ta
import logging
from typing import Tuple, List, Optional, Dict
from config import config


class DataPreprocessor:
    def __init__(self, verbose: bool = True):
        """Инициализация препроцессора"""
        self.verbose = verbose
        self.setup_logging()
        self.indicators_cache = {}

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
            elif level == 'debug':
                self.logger.debug(message)

    def calculate_all_indicators(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Расчет всех технических индикаторов

        Args:
            df: DataFrame с колонками OHLCV
            verbose: Флаг логирования

        Returns:
            DataFrame с добавленными индикаторами
        """
        try:
            if df.empty:
                self.log("Empty DataFrame for indicator calculation", 'warning')
                return df

            data = df.copy()

            # Проверка наличия необходимых колонок
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.log(f"Missing columns: {missing_columns}", 'error')
                return data

            self.log(f"Calculating indicators for {len(data)} candles", 'debug')

            # 1. Скользящие средние
            data['SMA_20'] = ta.trend.sma_indicator(data['close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['close'], window=50)
            data['SMA_100'] = ta.trend.sma_indicator(data['close'], window=100)
            data['SMA_200'] = ta.trend.sma_indicator(data['close'], window=200)

            data['EMA_12'] = ta.trend.ema_indicator(data['close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['close'], window=26)
            data['EMA_50'] = ta.trend.ema_indicator(data['close'], window=50)

            # 2. Индекс относительной силы (RSI)
            data['RSI_14'] = ta.momentum.rsi(data['close'], window=14)
            data['RSI_7'] = ta.momentum.rsi(data['close'], window=7)

            # 3. Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'],
                                                     window=14, smooth_window=3)
            data['STOCH_K'] = stoch.stoch()
            data['STOCH_D'] = stoch.stoch_signal()

            # 4. MACD
            macd = ta.trend.MACD(data['close'], window_slow=26, window_fast=12, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_SIGNAL'] = macd.macd_signal()
            data['MACD_DIFF'] = macd.macd_diff()

            # 5. Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'], window=20, window_dev=2)
            data['BB_UPPER'] = bb.bollinger_hband()
            data['BB_MIDDLE'] = bb.bollinger_mavg()
            data['BB_LOWER'] = bb.bollinger_lband()
            data['BB_WIDTH'] = (data['BB_UPPER'] - data['BB_LOWER']) / data['BB_MIDDLE']
            data['BB_PCT'] = (data['close'] - data['BB_LOWER']) / (data['BB_UPPER'] - data['BB_LOWER'])

            # 6. Average True Range (ATR)
            data['ATR_14'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

            # 7. Parabolic SAR
            psar = ta.trend.PSARIndicator(data['high'], data['low'], data['close'])
            data['PSAR'] = psar.psar()

            # 8. Commodity Channel Index (CCI)
            data['CCI_20'] = ta.trend.cci(data['high'], data['low'], data['close'], window=20)

            # 9. Awesome Oscillator
            data['AO'] = ta.momentum.awesome_oscillator(data['high'], data['low'])

            # 10. Williams %R
            data['WILLIAMS_R'] = ta.momentum.williams_r(data['high'], data['low'], data['close'], lbp=14)

            # 11. Rate of Change (ROC)
            data['ROC_10'] = ta.momentum.roc(data['close'], window=10)
            data['ROC_20'] = ta.momentum.roc(data['close'], window=20)

            # 12. Money Flow Index (MFI)
            data['MFI_14'] = ta.volume.money_flow_index(data['high'], data['low'], data['close'], data['volume'],
                                                        window=14)

            # 13. On-Balance Volume (OBV)
            data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])

            # 14. Volume Weighted Average Price (VWAP)
            # Для VWAP нужны данные за день, поэтому рассчитываем отдельно если есть дневные данные

            # 15. Ichimoku Cloud (частично)
            ichimoku = ta.trend.IchimokuIndicator(data['high'], data['low'])
            data['ICHIMOKU_CONVERSION'] = ichimoku.ichimoku_conversion_line()
            data['ICHIMOKU_BASE'] = ichimoku.ichimoku_base_line()
            data['ICHIMOKU_A'] = ichimoku.ichimoku_a()
            data['ICHIMOKU_B'] = ichimoku.ichimoku_b()

            # 16. Дополнительные фичи
            data['RETURNS'] = data['close'].pct_change()
            data['LOG_RETURNS'] = np.log(data['close'] / data['close'].shift(1))
            data['VOLATILITY_20'] = data['RETURNS'].rolling(window=20).std()
            data['VOLUME_MA_20'] = data['volume'].rolling(window=20).mean()
            data['PRICE_RANGE'] = (data['high'] - data['low']) / data['close']
            data['BODY_SIZE'] = (data['close'] - data['open']) / data['close']

            # 17. Временные фичи
            data['HOUR'] = data.index.hour
            data['DAY_OF_WEEK'] = data.index.dayofweek
            data['MONTH'] = data.index.month

            # 18. Целевые переменные для классификации
            # Создаем несколько целевых переменных с разными горизонтами

            # ПЕРВОЕ: Проверим волатильность данных
            volatility = data['close'].pct_change().std() * 100  # Волатильность в процентах
            avg_return = abs(data['close'].pct_change().mean()) * 100

            self.log(f"Data statistics:", 'info')
            self.log(f"  Volatility (std of returns): {volatility:.2f}%", 'info')
            self.log(f"  Average absolute return: {avg_return:.4f}%", 'info')

            # Анализ распределения доходностей
            returns = data['close'].pct_change().dropna()
            percentiles = np.percentile(returns * 100, [10, 25, 50, 75, 90])
            self.log(f"  Return percentiles (10%, 25%, 50%, 75%, 90%): {percentiles}", 'info')

            # РЕАЛИСТИЧНЫЕ ПОРОГИ НА ОСНОВЕ РАСПРЕДЕЛЕНИЯ ДАННЫХ
            # Автоматический подбор порогов на основе процентилей
            # Цель: примерно 30% LONG, 40% HOLD, 30% SHORT

            # Классификация: 1 - рост, 0 - боковик, -1 - падение
            for horizon in [1, 3, 5, 10]:  # 1, 3, 5, 10 свечей вперед
                future_return = data['close'].pct_change(horizon).shift(-horizon)
                target_name = f'TARGET_CLASS_{horizon}'

                # Пропускаем если недостаточно данных
                if future_return.dropna().empty:
                    self.log(f"Skipping {target_name}: insufficient data", 'warning')
                    continue

                # СТРАТЕГИЯ 1: Используем процентили для балансировки
                try:
                    # Целевое распределение: 30% LONG, 40% HOLD, 30% SHORT
                    threshold_up = np.percentile(future_return.dropna(), 70)  # Верхние 30%
                    threshold_down = np.percentile(future_return.dropna(), 30)  # Нижние 30%

                    # Гарантируем минимальные пороги
                    min_threshold = max(volatility * 0.5 / 100, 0.001)  # Половина волатильности, минимум 0.1%
                    threshold_up = max(threshold_up, min_threshold)
                    threshold_down = min(threshold_down, -min_threshold)

                    # Ограничиваем максимальные пороги (не более 2% для скальпинга)
                    max_threshold = 0.02  # 2% максимум
                    threshold_up = min(threshold_up, max_threshold)
                    threshold_down = max(threshold_down, -max_threshold)

                    # Логируем выбранные пороги
                    if verbose:
                        self.log(f"Creating target {target_name}:", 'info')
                        self.log(
                            f"  Using percentiles: LONG > {threshold_up * 100:.3f}%, SHORT < {threshold_down * 100:.3f}%",
                            'info')
                        self.log(
                            f"  (70th percentile: {threshold_up * 100:.3f}%, 30th percentile: {threshold_down * 100:.3f}%)",
                            'info')

                    # Создаем метки
                    data[target_name] = 0  # По умолчанию HOLD
                    data.loc[future_return > threshold_up, target_name] = 1  # LONG
                    data.loc[future_return < threshold_down, target_name] = -1  # SHORT

                except Exception as percentile_error:
                    self.log(f"Percentile method failed for {target_name}: {percentile_error}", 'warning')
                    # СТРАТЕГИЯ 2: Используем фиксированные пороги на основе волатильности
                    base_threshold = volatility * 0.8 / 100  # 80% от волатильности
                    threshold_up = max(base_threshold, 0.002)  # Минимум 0.2%
                    threshold_down = -threshold_up

                    if verbose:
                        self.log(f"  Fallback to volatility-based thresholds: ±{threshold_up * 100:.3f}%", 'info')

                    data[target_name] = 0
                    data.loc[future_return > threshold_up, target_name] = 1
                    data.loc[future_return < threshold_down, target_name] = -1

                # Логируем распределение
                class_counts = data[target_name].value_counts()
                total = len(data[target_name].dropna())

                if verbose and total > 0:
                    self.log(f"Target distribution {target_name}:", 'info')
                    for cls in [-1, 0, 1]:
                        count = class_counts.get(cls, 0)
                        percentage = (count / total) * 100
                        self.log(f"  Class {cls}: {count} ({percentage:.1f}%)", 'info')

                    # ПРЕДУПРЕЖДЕНИЕ если распределение слишком несбалансированное
                    hold_percentage = class_counts.get(0, 0) / total * 100
                    if hold_percentage > 90:
                        self.log(f"⚠️  КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Слишком много HOLD меток ({hold_percentage:.1f}%)!",
                                 'error')
                        self.log(f"⚠️  Данные могут быть слишком стабильными или период слишком короткий", 'error')
                        self.log(f"⚠️  Рекомендации:", 'error')
                        self.log(f"⚠️    1. Увеличьте период данных (минимум 6 месяцев)", 'error')
                        self.log(f"⚠️    2. Используйте больший таймфрейм (1h вместо 5m)", 'error')
                        self.log(f"⚠️    3. Выберите более волатильную криптовалюту", 'error')

                        # ДИАГНОСТИКА: сколько реальных движений?
                        large_moves = len(future_return[abs(future_return) > threshold_up])
                        self.log(f"⚠️  Диагностика: всего {large_moves} движений > {threshold_up * 100:.2f}%", 'error')

                    elif hold_percentage > 80:
                        self.log(f"⚠️  Предупреждение: много HOLD меток ({hold_percentage:.1f}%)", 'warning')
                        self.log(f"⚠️  Рассмотрите увеличение периода данных", 'warning')

                    elif hold_percentage < 40:
                        self.log(f"⚠️  Предупреждение: мало HOLD меток ({hold_percentage:.1f}%)", 'warning')
                        self.log(f"⚠️  Пороги могут быть слишком низкими", 'warning')

            # 19. Альтернативные целевые переменные на основе технических индикаторов
            # (если ценовые метки не работают)

            # Создаем технические метки на основе индикаторов
            if verbose:
                self.log(f"Creating technical indicator targets...", 'info')

            # RSI сигналы
            data['RSI_SIGNAL'] = 0
            data.loc[data['RSI_14'] < 30, 'RSI_SIGNAL'] = 1  # Oversold -> потенциальный LONG
            data.loc[data['RSI_14'] > 70, 'RSI_SIGNAL'] = -1  # Overbought -> потенциальный SHORT

            # MACD сигналы (пересечение)
            data['MACD_CROSS'] = 0
            # Быстрое пересечение сверху вниз
            macd_cross_up = (data['MACD'] > data['MACD_SIGNAL']) & (
                        data['MACD'].shift(1) <= data['MACD_SIGNAL'].shift(1))
            macd_cross_down = (data['MACD'] < data['MACD_SIGNAL']) & (
                        data['MACD'].shift(1) >= data['MACD_SIGNAL'].shift(1))
            data.loc[macd_cross_up, 'MACD_CROSS'] = 1
            data.loc[macd_cross_down, 'MACD_CROSS'] = -1

            # Bollinger Bands сигналы
            data['BB_SIGNAL'] = 0
            data.loc[data['close'] < data['BB_LOWER'] * 0.99, 'BB_SIGNAL'] = 1  # Ниже нижней полосы -> LONG
            data.loc[data['close'] > data['BB_UPPER'] * 1.01, 'BB_SIGNAL'] = -1  # Выше верхней полосы -> SHORT

            # Комбинированный технический сигнал
            data['TECH_TARGET'] = 0
            # Для LONG: минимум 2 индикатора говорят LONG
            long_signals = (data['RSI_SIGNAL'] == 1).astype(int) + \
                           (data['MACD_CROSS'] == 1).astype(int) + \
                           (data['BB_SIGNAL'] == 1).astype(int)

            # Для SHORT: минимум 2 индикатора говорят SHORT
            short_signals = (data['RSI_SIGNAL'] == -1).astype(int) + \
                            (data['MACD_CROSS'] == -1).astype(int) + \
                            (data['BB_SIGNAL'] == -1).astype(int)

            data.loc[long_signals >= 2, 'TECH_TARGET'] = 1
            data.loc[short_signals >= 2, 'TECH_TARGET'] = -1

            if verbose:
                tech_counts = data['TECH_TARGET'].value_counts()
                tech_total = len(data['TECH_TARGET'].dropna())
                self.log(f"Technical target distribution (TECH_TARGET):", 'info')
                for cls in [-1, 0, 1]:
                    count = tech_counts.get(cls, 0)
                    percentage = (count / tech_total) * 100 if tech_total > 0 else 0
                    self.log(f"  Class {cls}: {count} ({percentage:.1f}%)", 'info')

            # 20. Целевые переменные для регрессии
            for horizon in [1, 3, 5, 10]:
                future_return = data['close'].pct_change(horizon).shift(-horizon)
                data[f'TARGET_REG_{horizon}'] = future_return

            # Удаляем строки с NaN значениями
            initial_len = len(data)
            data = data.dropna()
            removed_count = initial_len - len(data)

            if verbose:
                self.log(f"Indicators calculated: added {len(data.columns) - len(df.columns)} columns")
                self.log(f"Removed {removed_count} rows with NaN values")
                self.log(f"Remaining {len(data)} rows")

                # Итоговый отчет о распределении меток
                self.log(f"Final target distributions:", 'info')
                for horizon in [1, 3, 5, 10]:
                    target_col = f'TARGET_CLASS_{horizon}'
                    if target_col in data.columns:
                        counts = data[target_col].value_counts()
                        total = len(data[target_col])
                        self.log(f"  {target_col}:", 'info')
                        for cls in sorted(counts.index):
                            percentage = (counts[cls] / total) * 100
                            signal = {1: 'LONG', 0: 'HOLD', -1: 'SHORT'}.get(cls, 'UNKNOWN')
                            self.log(f"    {signal}: {counts[cls]} ({percentage:.1f}%)", 'info')

            temporal_features = ['HOUR', 'DAY_OF_WEEK', 'MONTH']
            for temp_feature in temporal_features:
                if temp_feature in data.columns:
                    data = data.drop(columns=[temp_feature])
                    if verbose:
                        self.log(f"Removed temporal feature: {temp_feature}", 'debug')

            # Кешируем результат
            cache_key = f"{len(df)}_{df.index[-1]}"
            self.indicators_cache[cache_key] = data.copy()

            return data

        except Exception as e:
            self.log(f"Error calculating indicators: {e}", 'error')
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", 'error')
            return df

    def prepare_features_for_training(self, df: pd.DataFrame,
                                      target_column: str = 'TARGET_CLASS_5',
                                      lookback_window: int = 60,
                                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Подготовка фичей для обучения

        Args:
            df: DataFrame с индикаторами
            target_column: Название целевой колонки
            lookback_window: Окно для создания последовательностей
            verbose: Флаг логирования

        Returns:
            X, y для обучения
        """
        try:
            if target_column not in df.columns:
                self.log(f"Target column {target_column} not found", 'error')
                return np.array([]), np.array([])

            # Логируем распределение целевой переменной
            class_counts = df[target_column].value_counts()
            total = len(df[target_column])
            self.log(f"Target distribution for {target_column}:", 'info')
            for cls in [-1, 0, 1]:
                count = class_counts.get(cls, 0)
                percentage = (count / total) * 100
                self.log(f"  Class {cls}: {count} ({percentage:.1f}%)", 'info')

                # ВАЖНОЕ ПРЕДУПРЕЖДЕНИЕ
                if cls == 0 and percentage > 80:
                    self.log(f"⚠️  ВНИМАНИЕ: Слишком много HOLD меток ({percentage:.1f}%)!", 'warning')
                    self.log(f"⚠️  Модель будет предсказывать в основном HOLD!", 'warning')
                    self.log(f"⚠️  Увеличьте пороги в calculate_all_indicators!", 'warning')

            # Отделяем фичи и таргеты
            feature_columns = [col for col in df.columns
                               if not col.startswith('TARGET_')
                               and col not in ['open', 'high', 'low', 'close', 'volume']]

            X_data = df[feature_columns].values
            y_data = df[target_column].values

            # Создаем последовательности для LSTM
            X_sequences = []
            y_sequences = []

            for i in range(lookback_window, len(X_data)):
                X_sequences.append(X_data[i - lookback_window:i])
                y_sequences.append(y_data[i])

            X = np.array(X_sequences)
            y = np.array(y_sequences)

            if verbose:
                self.log(f"Created {len(X)} sequences")
                self.log(f"X shape: {X.shape}, y shape: {y.shape}")

                # Логируем распределение после создания последовательностей
                unique, counts = np.unique(y, return_counts=True)
                self.log(f"Final class distribution in sequences:", 'info')
                for cls, cnt in zip(unique, counts):
                    percentage = (cnt / len(y)) * 100
                    self.log(f"  Class {cls}: {cnt} ({percentage:.1f}%)", 'info')

            return X, y

        except Exception as e:
            self.log(f"Error preparing features: {e}", 'error')
            return np.array([]), np.array([])

    def prepare_features_for_prediction(self, df: pd.DataFrame,
                                        lookback_window: int = 60,
                                        verbose: bool = True) -> np.ndarray:
        """Подготовка фичи для предсказания"""
        try:
            # Для XGBoost используем все фичи кроме целевых и базовых OHLCV
            feature_columns = [col for col in df.columns
                               if not col.startswith('TARGET_')
                               and col not in ['open', 'high', 'low', 'close', 'volume']]

            # Оставляем только существующие колонки
            available_features = [col for col in feature_columns if col in df.columns]

            if verbose:
                print(f"  📊 Используется {len(available_features)} фичей")

            X_data = df[available_features].values

            # Берем последние lookback_window значений
            if len(X_data) >= lookback_window:
                X_sequence = X_data[-lookback_window:].reshape(1, lookback_window, -1)

                if verbose:
                    print(f"  ✅ Создана последовательность для предсказания: {X_sequence.shape}")
                    print(f"  🔢 Всего фичей: {X_sequence.shape[1] * X_sequence.shape[2]}")

                return X_sequence
            else:
                if verbose:
                    print(f"  ❌ Недостаточно данных для создания последовательности")
                return np.array([])

        except Exception as e:
            self.log(f"Error preparing features for prediction: {e}", 'error')
            return np.array([])

    def create_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20],
                                verbose: bool = True) -> pd.DataFrame:
        """
        Создание скользящих статистик

        Args:
            df: Исходный DataFrame
            window_sizes: Размеры окон
            verbose: Флаг логирования

        Returns:
            DataFrame с дополнительными фичами
        """
        try:
            data = df.copy()

            for window in window_sizes:
                # Скользящие статистики для цены
                data[f'ROLLING_MEAN_{window}'] = data['close'].rolling(window=window).mean()
                data[f'ROLLING_STD_{window}'] = data['close'].rolling(window=window).std()
                data[f'ROLLING_MIN_{window}'] = data['low'].rolling(window=window).min()
                data[f'ROLLING_MAX_{window}'] = data['high'].rolling(window=window).max()

                # Скользящие статистики для объема
                data[f'VOLUME_MEAN_{window}'] = data['volume'].rolling(window=window).mean()
                data[f'VOLUME_STD_{window}'] = data['volume'].rolling(window=window).std()

                # Отношение цены к скользящим средним
                data[f'PRICE_TO_MA_{window}'] = data['close'] / data[f'ROLLING_MEAN_{window}']

            if verbose:
                self.log(f"Added {len(window_sizes) * 7} rolling features")

            return data

        except Exception as e:
            self.log(f"Error creating rolling features: {e}", 'error')
            return df

    def normalize_features(self, X: np.ndarray, fit: bool = True,
                           scaler=None, verbose: bool = True) -> Tuple[np.ndarray, any]:
        """
        Нормализация фичей

        Args:
            X: Входные данные
            fit: Флаг обучения скейлера
            scaler: Существующий скейлер
            verbose: Флаг логирования

        Returns:
            Нормализованные данные и скейлер
        """
        try:
            from sklearn.preprocessing import StandardScaler

            if fit or scaler is None:
                scaler = StandardScaler()
                # Для 3D данных (LSTM) преобразуем в 2D для обучения скейлера
                if len(X.shape) == 3:
                    X_2d = X.reshape(-1, X.shape[2])
                    scaler.fit(X_2d)
                    X_normalized = scaler.transform(X_2d).reshape(X.shape)
                else:
                    scaler.fit(X)
                    X_normalized = scaler.transform(X)
            else:
                if len(X.shape) == 3:
                    X_2d = X.reshape(-1, X.shape[2])
                    X_normalized = scaler.transform(X_2d).reshape(X.shape)
                else:
                    X_normalized = scaler.transform(X)

            if verbose:
                self.log(f"Data normalized. Shape: {X_normalized.shape}")

            return X_normalized, scaler

        except Exception as e:
            self.log(f"Error normalizing features: {e}", 'error')
            return X, scaler

    def split_train_test(self, X: np.ndarray, y: np.ndarray,
                         test_size: float = 0.2, verbose: bool = True) -> Tuple:
        """
        Разделение на тренировочную и тестовую выборки

        Args:
            X: Признаки
            y: Целевые значения
            test_size: Размер тестовой выборки
            verbose: Флаг логирования

        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            split_idx = int(len(X) * (1 - test_size))

            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]

            if verbose:
                self.log(f"Data split: train={len(X_train)}, test={len(X_test)}")
                self.log(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

                # Логируем распределение в train/test
                unique_train, counts_train = np.unique(y_train, return_counts=True)
                unique_test, counts_test = np.unique(y_test, return_counts=True)

                self.log(f"Train class distribution:", 'info')
                for cls, cnt in zip(unique_train, counts_train):
                    percentage = (cnt / len(y_train)) * 100
                    self.log(f"  Class {cls}: {cnt} ({percentage:.1f}%)", 'info')

                self.log(f"Test class distribution:", 'info')
                for cls, cnt in zip(unique_test, counts_test):
                    percentage = (cnt / len(y_test)) * 100
                    self.log(f"  Class {cls}: {cnt} ({percentage:.1f}%)", 'info')

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.log(f"Error splitting data: {e}", 'error')
            return X, np.array([]), y, np.array([])

    def balance_classes(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Балансировка классов путем oversampling меньших классов

        Args:
            X: Признаки (3D для LSTM)
            y: Метки (должны быть -1, 0, 1)
            verbose: Флаг логирования

        Returns:
            Сбалансированные X, y
        """
        try:
            unique, counts = np.unique(y, return_counts=True)

            if verbose:
                self.log(f"Original class distribution:", 'info')
                for cls, cnt in zip(unique, counts):
                    percentage = (cnt / len(y)) * 100
                    self.log(f"  Class {cls}: {cnt} samples ({percentage:.1f}%)", 'info')

            # Определяем целевые количества для каждого класса
            # Цель: сделать распределение примерно 40% HOLD, 30% LONG, 30% SHORT
            total_samples = len(y)
            target_counts = {
                -1: int(total_samples * 0.3),  # SHORT: 30%
                0: int(total_samples * 0.4),   # HOLD: 40%
                1: int(total_samples * 0.3)    # LONG: 30%
            }

            X_balanced = []
            y_balanced = []

            for cls in [-1, 0, 1]:
                # Индексы для текущего класса
                idx = np.where(y == cls)[0]
                cls_count = len(idx)
                target_count = target_counts[cls]

                if verbose:
                    self.log(f"Processing class {cls}: {cls_count} samples -> target: {target_count}", 'info')

                if cls_count == 0:
                    self.log(f"Warning: No samples for class {cls}", 'warning')
                    continue

                if cls_count < target_count:
                    # Oversampling: повторяем существующие примеры
                    repeat_times = target_count // cls_count
                    remainder = target_count % cls_count

                    # Повторяем существующие примеры
                    for _ in range(repeat_times):
                        X_balanced.append(X[idx])
                        y_balanced.append(y[idx])

                    # Добавляем остаток случайно
                    if remainder > 0:
                        selected_idx = np.random.choice(idx, size=remainder, replace=True)
                        X_balanced.append(X[selected_idx])
                        y_balanced.append(y[selected_idx])

                    if verbose:
                        self.log(f"  Oversampled: {cls_count} -> {repeat_times * cls_count + remainder}", 'info')
                else:
                    # Undersampling: берем случайную выборку
                    selected_idx = np.random.choice(idx, size=target_count, replace=False)
                    X_balanced.append(X[selected_idx])
                    y_balanced.append(y[selected_idx])

                    if verbose:
                        self.log(f"  Undersampled: {cls_count} -> {target_count}", 'info')

            # Объединяем
            if X_balanced:
                X_balanced = np.concatenate(X_balanced, axis=0)
                y_balanced = np.concatenate(y_balanced, axis=0)
            else:
                X_balanced = X.copy()
                y_balanced = y.copy()

            # Перемешиваем
            shuffle_idx = np.random.permutation(len(X_balanced))
            X_balanced = X_balanced[shuffle_idx]
            y_balanced = y_balanced[shuffle_idx]

            # Логируем итоговое распределение
            unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
            if verbose:
                self.log(f"Balanced class distribution:", 'info')
                for cls, cnt in zip(unique_bal, counts_bal):
                    percentage = (cnt / len(y_balanced)) * 100
                    self.log(f"  Class {cls}: {cnt} samples ({percentage:.1f}%)", 'info')

            return X_balanced, y_balanced

        except Exception as e:
            self.log(f"Error balancing classes: {e}", 'error')
            return X, y

    def add_advanced_features(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Добавление расширенных фичей к данным

        Args:
            df: DataFrame с базовыми индикаторами
            verbose: Флаг логирования

        Returns:
            DataFrame с расширенными фичами
        """
        try:
            if df.empty:
                return df

            data = df.copy()

            if verbose:
                self.log("Adding advanced features...", 'info')

            # Создаем копию для безопасных операций
            result = data.copy()

            # 1. Взаимодействия между основными индикаторами
            try:
                if 'RSI_14' in result.columns and 'MACD' in result.columns:
                    result['RSI_MACD_INTERACTION'] = result['RSI_14'] * result['MACD']
            except:
                pass

            try:
                if 'BB_WIDTH' in result.columns and 'ATR_14' in result.columns:
                    # Избегаем деления на ноль
                    result['BB_ATR_RATIO'] = result['BB_WIDTH'] / (result['ATR_14'].replace(0, np.nan) + 1e-6)
            except:
                pass

            try:
                if 'VOLUME_MA_20' in result.columns and 'volume' in result.columns:
                    result['VOLUME_RATIO'] = result['volume'] / (result['VOLUME_MA_20'].replace(0, np.nan) + 1e-6)
            except:
                pass

            # 2. Скользящие статистики для ключевых индикаторов
            key_indicators = ['RSI_14', 'MACD', 'ATR_14', 'OBV']
            for indicator in key_indicators:
                if indicator in result.columns:
                    try:
                        # Скользящее среднее
                        result[f'{indicator}_MA_10'] = result[indicator].rolling(window=10, min_periods=3).mean()
                        result[f'{indicator}_MA_20'] = result[indicator].rolling(window=20, min_periods=5).mean()

                        # Скользящее стандартное отклонение
                        result[f'{indicator}_STD_10'] = result[indicator].rolling(window=10, min_periods=3).std()

                        # Z-score (нормализованное отклонение)
                        rolling_mean = result[indicator].rolling(window=20, min_periods=5).mean()
                        rolling_std = result[indicator].rolling(window=20, min_periods=5).std()
                        result[f'{indicator}_ZSCORE_20'] = (result[indicator] - rolling_mean) / (
                                    rolling_std.replace(0, np.nan) + 1e-6)
                    except:
                        continue

            # 3. Моменты распределения для доходностей
            if 'close' in result.columns:
                try:
                    returns = result['close'].pct_change()
                    for window in [10, 20]:
                        # Используем kurt() вместо kurtosis()
                        result[f'RETURNS_SKEW_{window}'] = returns.rolling(window=window, min_periods=5).skew()
                        result[f'RETURNS_KURTOSIS_{window}'] = returns.rolling(window=window,
                                                                               min_periods=5).kurt()  # Изменено с kurtosis()
                except Exception as e:
                    if verbose:
                        self.log(f"Ошибка при расчете моментов распределения: {e}", 'warning')

            # 4. Ценовые паттерны (упрощенные)
            if all(col in result.columns for col in ['high', 'low', 'close', 'open']):
                try:
                    # Волатильность внутри свечи
                    result['CANDLE_BODY'] = abs(result['close'] - result['open'])
                    result['CANDLE_RANGE'] = result['high'] - result['low']
                    result['BODY_TO_RANGE_RATIO'] = result['CANDLE_BODY'] / (
                                result['CANDLE_RANGE'].replace(0, np.nan) + 1e-6)
                except:
                    pass

            # 5. Volume profile features (безопасная версия)
            if 'volume' in result.columns and 'close' in result.columns:
                try:
                    # Накопление/распределение (Accumulation/Distribution Line)
                    clv = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (
                                result['high'] - result['low'])
                    clv = clv.replace([np.inf, -np.inf], np.nan).fillna(0)
                    result['ADL'] = (clv * result['volume']).cumsum()
                except:
                    pass

            # 6. Разности и производные
            for window in [1, 2, 3, 5]:
                for indicator in ['RSI_14', 'MACD', 'ATR_14', 'close']:
                    if indicator in result.columns:
                        try:
                            result[f'{indicator}_DIFF_{window}'] = result[indicator].diff(window)
                        except:
                            pass

            # 7. Временные фичи (если индекс - datetime)
            try:
                if hasattr(result.index, 'hour'):
                    result['HOUR_OF_DAY'] = result.index.hour
                    result['DAY_OF_WEEK'] = result.index.dayofweek
                    result['DAY_OF_MONTH'] = result.index.day

                    # Циклическое кодирование времени
                    result['HOUR_SIN'] = np.sin(2 * np.pi * result['HOUR_OF_DAY'] / 24)
                    result['HOUR_COS'] = np.cos(2 * np.pi * result['HOUR_OF_DAY'] / 24)
                    result['DAY_SIN'] = np.sin(2 * np.pi * result['DAY_OF_WEEK'] / 7)
                    result['DAY_COS'] = np.cos(2 * np.pi * result['DAY_OF_WEEK'] / 7)
            except:
                pass

            # Удаляем временные колонки если они создавались
            temp_cols = ['CANDLE_BODY', 'CANDLE_RANGE']
            for col in temp_cols:
                if col in result.columns:
                    result = result.drop(columns=[col])

            # Удаляем строки с NaN значениями, которые появились из-за скользящих окон
            initial_len = len(result)
            result = result.dropna()
            removed_count = initial_len - len(result)

            if verbose:
                self.log(f"Added {len(result.columns) - len(df.columns)} advanced features", 'info')
                self.log(f"Removed {removed_count} rows with NaN values from advanced features", 'info')
                self.log(f"Total features now: {len(result.columns)}", 'info')

                # Показать несколько новых фичей
                new_features = [col for col in result.columns if col not in df.columns]
                if new_features:
                    self.log(f"New features (first 10): {new_features[:10]}", 'debug')

            return result

        except Exception as e:
            self.log(f"Error adding advanced features: {e}", 'error')
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", 'error')
            return df