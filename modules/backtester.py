"""
Модуль для бэктестирования торговых стратегий
"""

# ===== ДОБАВЬТЕ ЭТОТ БЛОК ПОСЛЕ ИМПОРТОВ =====
import os
import warnings
import sys
import io
import contextlib
import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from config import config
from modules.database import Database
from modules.predictor import SignalPredictor
from modules.preprocessor import DataPreprocessor
from modules.state_manager import state_manager

# Отключить логирование TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ABS_SUPPRESS_LOGGING'] = '1'

# Отключить предупреждения
warnings.filterwarnings('ignore')

# Отключить логирование TensorFlow и abseil
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except:
    pass

try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
except:
    pass


class Backtester:
    def __init__(self, verbose: bool = True):
        """Инициализация бэктестера"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.predictor = SignalPredictor(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.state_manager = state_manager

        # Параметры по умолчанию
        self.initial_balance = config.backtest.INITIAL_BALANCE
        self.commission = config.trading.COMMISSION
        self.stop_loss_pct = config.trading.STOP_LOSS_PCT / 100
        self.take_profit_pct = config.trading.TAKE_PROFIT_PCT / 100
        self.slippage = getattr(config.backtest, 'SLIPPAGE', 0.0005)

        # Константы для пакетной обработки
        self.LSTM_BATCH_SIZE = getattr(config.backtest, 'LSTM_BATCH_SIZE', 256)
        self.PROGRESS_UPDATE_INTERVAL = getattr(config.backtest, 'PROGRESS_UPDATE_INTERVAL', 100)

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

    def get_best_model(self, symbol: str, model_id: Optional[str] = None, verbose: bool = True) -> Tuple[Any, Any]:
        """
        Получение лучшей или указанной модели
        """
        try:
            from modules.trainer import ModelTrainer
            trainer = ModelTrainer(verbose=verbose)

            if model_id:
                # Загружаем указанную модель
                model, scaler = trainer.load_model(model_id, verbose=verbose)
                if model:
                    return model, scaler
                else:
                    self.log(f"Failed to load specified model {model_id}", 'warning')

            # Ищем лучшую активную модель
            models_df = self.db.get_available_models(
                symbol=symbol,
                active_only=True,
                verbose=verbose
            )

            if models_df.empty:
                self.log(f"No active models found for {symbol}", 'warning')
                return None, None

            # Выбираем модель с лучшей accuracy
            best_model_row = None
            best_accuracy = -1

            for _, row in models_df.iterrows():
                try:
                    metrics = row['metrics']
                    if isinstance(metrics, str):
                        import json
                        metrics = json.loads(metrics)

                    accuracy = metrics.get('accuracy', metrics.get('val_accuracy', 0))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_row = row
                except:
                    continue

            if best_model_row is None:
                self.log(f"No models with valid metrics for {symbol}", 'warning')
                return None, None

            # Загружаем лучшую модель
            model, scaler = trainer.load_model(best_model_row['model_id'], verbose=verbose)
            return model, scaler

        except Exception as e:
            self.log(f"Error getting best model: {str(e)}", 'error')
            return None, None

    def prepare_backtest_data(self, data: pd.DataFrame, model_type: str, verbose: bool = True) -> pd.DataFrame:
        """
        Подготовка данных для бэктеста
        """
        try:
            if data.empty:
                return pd.DataFrame()

            if verbose:
                print(f"  📊 Исходные данные: {len(data)} строк, {len(data.columns)} колонок")

            # Расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            if data_with_indicators.empty:
                self.log("Failed to calculate indicators", 'warning')
                return pd.DataFrame()

            if verbose:
                print(
                    f"  📈 Данные с индикаторами: {len(data_with_indicators)} строк, {len(data_with_indicators.columns)} колонок")
                print(f"  🔤 Пример колонок: {list(data_with_indicators.columns[:10])}...")

            # Добавляем расширенные фичи если это LSTM модель
            if 'lstm' in model_type.lower():
                data_with_indicators = self.preprocessor.add_advanced_features(
                    data_with_indicators, verbose=verbose
                )

                if verbose:
                    print(
                        f"  🔧 Добавлены расширенные фичи: {len(data_with_indicators)} строк, {len(data_with_indicators.columns)} колонок")

            return data_with_indicators

        except Exception as e:
            self.log(f"Error preparing backtest data: {str(e)}", 'error')
            return pd.DataFrame()

    class ProgressBar:
        """Универсальный прогресс-бар для различных операций"""

        def __init__(self, total, prefix='Прогресс', suffix='завершено', length=30, fill='█', verbose=True):
            self.total = total
            self.prefix = prefix
            self.suffix = suffix
            self.length = length
            self.fill = fill
            self.start_time = time.time()
            self.current = 0
            self.verbose = verbose
            self.last_update_time = time.time()
            self.update_interval = 0.5  # Обновлять не чаще чем раз в 0.5 секунд

        def update(self, iteration=None, force=False):
            """Обновить прогресс-бар"""
            if not self.verbose:
                return

            current_time = time.time()
            if not force and current_time - self.last_update_time < self.update_interval:
                return

            self.last_update_time = current_time

            if iteration is not None:
                self.current = iteration
            else:
                self.current += 1

            percent = ("{0:.1f}").format(100 * (self.current / float(self.total)))
            filled_length = int(self.length * self.current // self.total)
            bar = self.fill * filled_length + '─' * (self.length - filled_length)

            elapsed_time = time.time() - self.start_time
            if self.current > 0:
                time_per_item = elapsed_time / self.current
                remaining = self.total - self.current
                eta = time_per_item * remaining
                eta_str = f"ETA: {self._format_time(eta)}"
            else:
                eta_str = "ETA: --:--:--"

            # Очистить строку и вывести прогресс
            sys.stdout.write(f'\r{self.prefix} │{bar}│ {percent}% {self.suffix} {eta_str}')
            sys.stdout.flush()

        def finish(self, message=""):
            """Завершить прогресс-бар"""
            if not self.verbose:
                return

            elapsed_time = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed_time)

            if message:
                sys.stdout.write(f'\r{message} │ Время: {elapsed_str}\n')
            else:
                sys.stdout.write(f'\r{self.prefix} завершено │ Время: {elapsed_str}\n')
            sys.stdout.flush()

        @staticmethod
        def _format_time(seconds):
            """Форматирование времени"""
            if seconds < 60:
                return f"{seconds:.1f}с"
            elif seconds < 3600:
                minutes = seconds // 60
                seconds = seconds % 60
                return f"{minutes:.0f}м {seconds:.0f}с"
            else:
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                return f"{hours:.0f}ч {minutes:.0f}м"

    def generate_lstm_signals_batch(self, data: pd.DataFrame, model: Any, scaler: Any,
                                    feature_columns: List[str], lookback_window: int,
                                    verbose: bool = True) -> pd.DataFrame:
        """
        Генерация сигналов LSTM с пакетной обработкой
        """
        try:
            if data.empty or model is None:
                return pd.DataFrame()

            total_points = len(data) - lookback_window
            if total_points <= 0:
                return pd.DataFrame()

            if verbose:
                print(f"  🤖 LSTM: Генерация сигналов для {total_points} точек")
                print(f"  📊 Используется {len(feature_columns)} фичей")
                print(f"  📦 Размер пакета: {self.LSTM_BATCH_SIZE}")

            # Подготавливаем массив для всех предсказаний
            all_predictions = np.zeros(total_points)
            all_confidences = np.zeros(total_points)

            # Подготавливаем пакеты данных
            num_batches = (total_points + self.LSTM_BATCH_SIZE - 1) // self.LSTM_BATCH_SIZE

            if verbose:
                progress = self.ProgressBar(
                    total=num_batches,
                    prefix='🤖 LSTM пакетная обработка',
                    suffix='пакетов обработано',
                    verbose=verbose
                )

            for batch_idx in range(num_batches):
                try:
                    start_idx = batch_idx * self.LSTM_BATCH_SIZE
                    end_idx = min(start_idx + self.LSTM_BATCH_SIZE, total_points)
                    batch_size = end_idx - start_idx

                    # Подготавливаем пакет данных
                    batch_data = np.zeros((batch_size, lookback_window, len(feature_columns)))

                    for i in range(batch_size):
                        window_start = start_idx + i
                        window_end = window_start + lookback_window

                        # Извлекаем окно данных
                        window_data = data.iloc[window_start:window_end][feature_columns].values
                        batch_data[i] = window_data

                    # Нормализуем пакет если есть скейлер
                    if scaler is not None:
                        try:
                            # Преобразуем в 2D для нормализации
                            batch_2d = batch_data.reshape(batch_size, -1)
                            batch_norm_2d = scaler.transform(batch_2d)
                            batch_norm = batch_norm_2d.reshape(batch_size, lookback_window, -1)
                        except Exception as e:
                            if verbose and batch_idx == 0:
                                print(f"  ⚠️ Ошибка нормализации: {e}")
                            batch_norm = batch_data
                    else:
                        batch_norm = batch_data

                    # Предсказание для всего пакета
                    batch_predictions = model.predict(batch_norm, verbose=0)

                    # Обрабатываем предсказания
                    for i in range(batch_size):
                        prediction = batch_predictions[i]

                        if len(prediction.shape) == 0 or prediction.shape[0] == 1:
                            # Бинарная классификация или регрессия
                            predicted_class = int(round(prediction[0])) if hasattr(prediction, '__len__') else int(round(prediction))
                            confidence = abs(prediction[0] - 0.5) * 2 if hasattr(prediction, '__len__') else 0.5
                            predicted_class = predicted_class - 1  # Преобразуем [0,1,2] -> [-1,0,1]
                        else:
                            # Многоклассовая классификация
                            predicted_class = np.argmax(prediction) - 1
                            confidence = np.max(prediction)

                        all_predictions[start_idx + i] = predicted_class
                        all_confidences[start_idx + i] = confidence

                    if verbose:
                        progress.update(batch_idx + 1)

                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ Ошибка в пакете {batch_idx}: {e}")
                    continue

            if verbose:
                progress.finish("✅ LSTM пакетная обработка завершена")

            # Создаем DataFrame с результатами
            result_df = data.iloc[lookback_window:].copy()
            result_df = result_df.iloc[:len(all_predictions)].copy()

            result_df['signal'] = all_predictions
            result_df['confidence'] = all_confidences
            result_df['prediction_time'] = result_df.index

            # Фильтруем только сигналы с предсказаниями
            signals_df = result_df[result_df['signal'] != 0].copy()

            if verbose:
                print(f"  ✅ Сгенерировано {len(signals_df)} LSTM сигналов")
                if len(signals_df) > 0:
                    long_count = len(signals_df[signals_df['signal'] > 0])
                    short_count = len(signals_df[signals_df['signal'] < 0])
                    print(f"  📈 LONG: {long_count}, SHORT: {short_count}")

            return signals_df

        except Exception as e:
            if verbose:
                print(f"  ❌ Ошибка пакетной обработки LSTM: {e}")
                import traceback
                traceback.print_exc()
            return pd.DataFrame()

    def generate_xgboost_signals(self, data: pd.DataFrame, model: Any, scaler: Any,
                                 feature_columns: List[str], lookback_window: int,
                                 verbose: bool = True) -> pd.DataFrame:
        """
        Генерация сигналов XGBoost
        """
        try:
            if data.empty or model is None:
                return pd.DataFrame()

            total_points = len(data) - lookback_window
            if total_points <= 0:
                return pd.DataFrame()

            if verbose:
                print(f"  🌳 XGBoost: Генерация сигналов для {total_points} точек")
                progress = self.ProgressBar(
                    total=total_points,
                    prefix='🌳 XGBoost предсказания',
                    suffix='точек обработано',
                    verbose=verbose
                )

            # Подготавливаем массивы для результатов
            signals = np.zeros(len(data))
            confidences = np.zeros(len(data))

            for i in range(lookback_window, len(data)):
                try:
                    # Извлекаем окно данных
                    window_data = data.iloc[i-lookback_window:i][feature_columns].values

                    # Преобразуем в формат для XGBoost
                    X_window_flat = window_data.flatten().reshape(1, -1)

                    # Нормализуем если есть скейлер
                    if scaler is not None:
                        try:
                            X_norm = scaler.transform(X_window_flat)
                        except Exception as e:
                            X_norm = X_window_flat
                    else:
                        X_norm = X_window_flat

                    # Предсказание
                    prediction = model.predict(X_norm)
                    predicted_class = int(prediction[0]) - 1  # Преобразуем [0,1,2] -> [-1,0,1]

                    # Получаем вероятность если возможно
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_norm)
                        confidence = np.max(proba[0])
                    else:
                        confidence = 0.5

                    signals[i] = predicted_class
                    confidences[i] = confidence

                    if verbose and i % self.PROGRESS_UPDATE_INTERVAL == 0:
                        progress.update(i)

                except Exception as e:
                    if verbose and i == lookback_window:
                        print(f"  ⚠️ Ошибка предсказания: {e}")
                    continue

            if verbose:
                progress.finish("✅ XGBoost предсказания завершены")

            # Создаем DataFrame с результатами
            signals_df = data.copy()
            signals_df['signal'] = signals
            signals_df['confidence'] = confidences
            signals_df['prediction_time'] = signals_df.index

            # Фильтруем только сигналы с предсказаниями
            valid_signals = signals_df[signals_df['signal'] != 0].copy()

            if verbose:
                print(f"  ✅ Сгенерировано {len(valid_signals)} XGBoost сигналов")
                if len(valid_signals) > 0:
                    long_count = len(valid_signals[valid_signals['signal'] > 0])
                    short_count = len(valid_signals[valid_signals['signal'] < 0])
                    print(f"  📈 LONG: {long_count}, SHORT: {short_count}")

            return valid_signals

        except Exception as e:
            if verbose:
                print(f"  ❌ Ошибка генерации XGBoost сигналов: {e}")
            return pd.DataFrame()

    def generate_backtest_signals_optimized(self, data: pd.DataFrame, model: Any, scaler: Any,
                                            model_type: str, verbose: bool = True) -> pd.DataFrame:
        """
        Оптимизированная генерация торговых сигналов
        """
        try:
            if data.empty or model is None:
                return pd.DataFrame()

            if verbose:
                print(f"\n🎯 ГЕНЕРАЦИЯ СИГНАЛОВ ({model_type.upper()})")
                print(f"  📊 Данные: {len(data)} строк")

            # Определяем фичи для модели
            feature_columns = self.get_model_features(model, data, verbose)
            if not feature_columns:
                if verbose:
                    print("  ❌ Не удалось определить фичи для модели")
                return pd.DataFrame()

            # Получаем lookback_window из конфига
            lookback_window = config.model.LOOKBACK_WINDOW

            if verbose:
                print(f"  🔍 Используется {len(feature_columns)} фичей")
                print(f"  📐 Lookback window: {lookback_window}")

            # Выбираем метод генерации в зависимости от типа модели
            if 'lstm' in model_type.lower():
                return self.generate_lstm_signals_batch(
                    data, model, scaler, feature_columns, lookback_window, verbose
                )
            else:  # XGBoost и другие
                return self.generate_xgboost_signals(
                    data, model, scaler, feature_columns, lookback_window, verbose
                )

        except Exception as e:
            if verbose:
                print(f"  ❌ Ошибка генерации сигналов: {e}")
                import traceback
                traceback.print_exc()
            return pd.DataFrame()

    def get_model_features(self, model: Any, data: pd.DataFrame, verbose: bool = True) -> List[str]:
        """
        Получение фичей из модели
        """
        try:
            # Пытаемся получить фичи из атрибутов модели
            if hasattr(model, 'base_feature_names'):
                feature_columns = model.base_feature_names
            elif hasattr(model, '_features'):
                feature_columns = model._features
            elif hasattr(model, 'feature_names'):
                feature_columns = model.feature_names

                # Если это расширенные фичи с лагами, извлекаем базовые
                if feature_columns and any('_t-' in str(f) for f in feature_columns[:10]):
                    base_features = set()
                    for feature in feature_columns:
                        if isinstance(feature, str) and '_t-' in feature:
                            base_feature = feature.split('_t-')[0]
                            base_features.add(base_feature)
                        else:
                            base_features.add(str(feature))
                    feature_columns = list(base_features)
            else:
                # Дефолтный набор фичей
                base_features = ['close', 'volume', 'returns']
                tech_indicators = [col for col in data.columns
                                  if any(indicator in col.lower() for indicator in
                                        ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'obv', 'adx'])]
                feature_columns = base_features + tech_indicators

            # Фильтруем только существующие в данных
            feature_columns = [col for col in feature_columns if col in data.columns]

            # Сортируем для consistency
            feature_columns = sorted(feature_columns)

            if verbose:
                print(f"  📋 Найдено {len(feature_columns)} фичей")
                if len(feature_columns) <= 10:
                    print(f"  📋 Фичи: {feature_columns}")
                else:
                    print(f"  📋 Первые 10 фичей: {feature_columns[:10]}...")

            return feature_columns

        except Exception as e:
            if verbose:
                print(f"  ⚠️ Ошибка получения фичей из модели: {e}")
            return []

    def execute_backtest(self, signals: pd.DataFrame, initial_balance: float,
                        commission: float, verbose: bool = True) -> Dict[str, Any]:
        """
        Выполнение бэктеста на основе сигналов
        """
        try:
            if signals.empty:
                return {'error': 'No signals to backtest'}

            # Инициализация переменных
            balance = initial_balance
            position = 0.0  # 0 = нет позиции, >0 = LONG, <0 = SHORT
            entry_price = 0.0
            trade_history = []
            peak_balance = initial_balance
            max_drawdown = 0.0

            if verbose:
                print(f"\n💼 ВЫПОЛНЕНИЕ БЭКТЕСТА")
                print(f"  💰 Начальный баланс: ${initial_balance:,.2f}")
                print(f"  📊 Всего сигналов: {len(signals)}")
                progress = self.ProgressBar(
                    total=len(signals),
                    prefix='💼 Выполнение сделок',
                    suffix='сделок обработано',
                    verbose=verbose
                )

            for i, (timestamp, row) in enumerate(signals.iterrows()):
                try:
                    current_price = row['close']
                    signal = int(row['signal'])
                    confidence = row['confidence']

                    # Логика торговли
                    if position == 0 and signal != 0:  # Открытие позиции
                        position = signal  # 1 для LONG, -1 для SHORT
                        entry_price = current_price

                        trade = {
                            'timestamp': timestamp,
                            'type': 'LONG' if signal == 1 else 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': None,
                            'entry_balance': balance,
                            'exit_balance': None,
                            'pnl': None,
                            'pnl_pct': None,
                            'duration': None,
                            'result': 'OPEN',
                            'confidence': confidence
                        }
                        trade_history.append(trade)

                        if verbose and len(trade_history) <= 5:
                            print(f"  📈 Открыта {trade['type']} позиция по ${entry_price:.4f}")

                    elif position != 0:  # Есть открытая позиция
                        # Расчет P&L
                        if position == 1:  # LONG позиция
                            pnl_pct = (current_price - entry_price) / entry_price
                        else:  # SHORT позиция
                            pnl_pct = (entry_price - current_price) / entry_price

                        pnl = balance * pnl_pct

                        # Проверка стоп-лосса и тейк-профита
                        close_trade = False
                        close_reason = ""

                        if pnl_pct <= -self.stop_loss_pct:
                            close_trade = True
                            close_reason = "STOP LOSS"
                        elif pnl_pct >= self.take_profit_pct:
                            close_trade = True
                            close_reason = "TAKE PROFIT"
                        elif signal == -position:  # Противоположный сигнал
                            close_trade = True
                            close_reason = "REVERSE SIGNAL"

                        if close_trade:
                            # Закрытие позиции
                            exit_balance = balance + pnl

                            # Учитываем комиссию
                            commission_fee = exit_balance * commission
                            exit_balance -= commission_fee

                            # Обновляем баланс
                            balance = exit_balance

                            # Обновляем историю сделки
                            trade = trade_history[-1]
                            trade['exit_price'] = current_price
                            trade['exit_balance'] = exit_balance
                            trade['pnl'] = pnl
                            trade['pnl_pct'] = pnl_pct * 100
                            trade['duration'] = (timestamp - trade['timestamp']).total_seconds() / 3600  # в часах
                            trade['result'] = 'WIN' if pnl > 0 else 'LOSS'
                            trade['close_reason'] = close_reason

                            # Сбрасываем позицию
                            position = 0
                            entry_price = 0.0

                            if verbose and len(trade_history) <= 5:
                                result_emoji = "✅" if pnl > 0 else "❌"
                                print(f"  {result_emoji} Закрыта позиция: P&L ${pnl:+.2f} ({pnl_pct*100:+.2f}%) - {close_reason}")

                    # Обновляем максимальную просадку
                    if balance > peak_balance:
                        peak_balance = balance

                    current_drawdown = (peak_balance - balance) / peak_balance * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown

                    if verbose and i % self.PROGRESS_UPDATE_INTERVAL == 0:
                        progress.update(i)

                except Exception as e:
                    if verbose:
                        self.log(f"Error processing signal at {timestamp}: {str(e)}", 'warning')
                    continue

            # Закрываем последнюю позицию если она осталась открытой
            if position != 0 and len(trade_history) > 0:
                last_price = signals.iloc[-1]['close']
                trade = trade_history[-1]

                if position == 1:  # LONG
                    pnl_pct = (last_price - entry_price) / entry_price
                else:  # SHORT
                    pnl_pct = (entry_price - last_price) / entry_price

                pnl = balance * pnl_pct
                exit_balance = balance + pnl
                commission_fee = exit_balance * commission
                exit_balance -= commission_fee
                balance = exit_balance

                trade['exit_price'] = last_price
                trade['exit_balance'] = exit_balance
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct * 100
                trade['duration'] = (signals.index[-1] - trade['timestamp']).total_seconds() / 3600
                trade['result'] = 'WIN' if pnl > 0 else 'LOSS'
                trade['close_reason'] = 'END OF PERIOD'

                if verbose:
                    result_emoji = "✅" if pnl > 0 else "❌"
                    print(f"  {result_emoji} Позиция закрыта в конце периода: P&L ${pnl:+.2f} ({pnl_pct*100:+.2f}%)")

            if verbose:
                progress.finish("✅ Бэктест выполнен")

            # Расчет итоговых метрик
            total_trades = len([t for t in trade_history if t['result'] in ['WIN', 'LOSS']])
            winning_trades = len([t for t in trade_history if t['result'] == 'WIN'])
            losing_trades = len([t for t in trade_history if t['result'] == 'LOSS'])

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            total_pnl = sum([t['pnl'] or 0 for t in trade_history])
            total_return = (balance - initial_balance) / initial_balance * 100

            winning_pnl = sum([t['pnl'] or 0 for t in trade_history if t['result'] == 'WIN'])
            losing_pnl = sum([t['pnl'] or 0 for t in trade_history if t['result'] == 'LOSS'])

            profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')

            avg_win = np.mean([t['pnl'] or 0 for t in trade_history if t['result'] == 'WIN']) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] or 0 for t in trade_history if t['result'] == 'LOSS']) if losing_trades > 0 else 0

            # Подготовка результатов
            results = {
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'trade_history': trade_history,
                'summary': {
                    'aggregated': {
                        'total_return': total_return,
                        'final_balance': balance,
                        'total_pnl': total_pnl,
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': losing_trades,
                        'avg_win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'max_drawdown': max_drawdown,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss
                    }
                }
            }

            if verbose:
                print(f"\n📊 РЕЗУЛЬТАТЫ БЭКТЕСТА:")
                print(f"  💰 Конечный баланс: ${balance:,.2f}")
                print(f"  📈 Общая доходность: {total_return:.2f}%")
                print(f"  🎯 Win Rate: {win_rate:.1f}% ({winning_trades}/{total_trades})")
                print(f"  📊 Макс. просадка: {max_drawdown:.2f}%")
                print(f"  ⚖️  Profit Factor: {profit_factor:.2f}")
                print(f"  📊 Всего сделок: {total_trades}")

            return results

        except Exception as e:
            self.log(f"Error executing backtest: {str(e)}", 'error')
            return {'error': str(e)}

    def save_backtest_results(self, results: Dict[str, Any], symbol: str, model_id: Optional[str] = None):
        """
        Сохранение результатов бэктеста в базу данных
        """
        try:
            if 'error' in results:
                return False

            # Получаем информацию о модели если не указана
            if not model_id:
                models_df = self.db.get_available_models(
                    symbol=symbol,
                    active_only=True,
                    verbose=False
                )
                if not models_df.empty:
                    model_id = models_df.iloc[0]['model_id']

            # Подготавливаем данные для сохранения
            result_data = {
                'model_id': model_id or 'unknown',
                'symbol': symbol,
                'timeframe': self.state_manager.get_selected_timeframe(),
                'test_date': datetime.now(),
                'start_date': self.state_manager.get_backtest_dates()[0],
                'end_date': self.state_manager.get_backtest_dates()[1],
                'initial_balance': results['initial_balance'],
                'final_balance': results['final_balance'],
                'total_return': results['total_return'],
                'sharpe_ratio': 0,
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'total_trades': results['total_trades'],
                'winning_trades': results['winning_trades'],
                'losing_trades': results['losing_trades'],
                'avg_win': results['avg_win'],
                'avg_loss': results['avg_loss'],
                'details': '{}'
            }

            # Сохраняем в базу
            self.db.save_backtest_result(result_data, verbose=self.verbose)

            return True

        except Exception as e:
            self.log(f"Error saving backtest results: {str(e)}", 'error')
            return False

    def run_comprehensive_backtest(self, symbol: str,
                                 initial_balance: float = 10000.0,
                                 commission: float = None,
                                 model_id: str = None,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Комплексный бэктест для модели
        """
        try:
            if verbose:
                print(f"\n🚀 ЗАПУСК КОМПЛЕКСНОГО БЭКТЕСТА")
                print(f"  📊 Символ: {symbol}")
                print(f"  💰 Начальный баланс: ${initial_balance:,.2f}")
                if commission is not None:
                    print(f"  📈 Комиссия: {commission * 100:.2f}%")
                else:
                    print(f"  📈 Комиссия: {self.commission * 100:.2f}%")

            # Устанавливаем комиссию если предоставлена
            if commission is not None:
                self.commission = commission

            # Получаем даты для бэктеста
            start_date, end_date = self.state_manager.get_backtest_dates()

            if verbose:
                print(f"  📅 Период: {start_date} - {end_date}")

            # Получаем данные для бэктеста
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=self.state_manager.get_selected_timeframe(),
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )

            if data.empty:
                if verbose:
                    print("❌ Нет данных для бэктеста")
                return {'error': 'No data available for backtest'}

            if verbose:
                print(f"  📊 Загружено данных: {len(data)} строк")

            # Получаем модель
            model_info = self.get_best_model(symbol, model_id, verbose=verbose)

            if not model_info:
                if verbose:
                    print("❌ Не найдена подходящая модель для бэктеста")
                return {'error': 'No suitable model found'}

            model, scaler = model_info

            # Определяем тип модели
            model_type = self.determine_model_type(model, verbose)

            if verbose:
                print(f"  🤖 Тип модели: {model_type.upper()}")

            # Подготавливаем данные
            preprocessed_data = self.prepare_backtest_data(
                data, model_type=model_type, verbose=verbose
            )

            if preprocessed_data.empty:
                if verbose:
                    print("❌ Не удалось подготовить данные для бэктеста")
                return {'error': 'Failed to prepare data for backtest'}

            if verbose:
                print(f"  📊 Подготовленные данные: {len(preprocessed_data)} строк")

            # Генерируем сигналы
            signals = self.generate_backtest_signals_optimized(
                preprocessed_data, model, scaler, model_type, verbose=verbose
            )

            if signals.empty:
                if verbose:
                    print("❌ Не удалось сгенерировать сигналы")
                return {'error': 'Failed to generate signals'}

            # Выполняем бэктест
            results = self.execute_backtest(
                signals=signals,
                initial_balance=initial_balance,
                commission=self.commission,
                verbose=verbose
            )

            # Сохраняем результаты
            if results and 'error' not in results:
                self.save_backtest_results(results, symbol, model_id)

                if verbose:
                    print("\n✅ БЭКТЕСТ УСПЕШНО ЗАВЕРШЕН!")

            return results

        except Exception as e:
            error_msg = f"Error in backtest: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return {'error': error_msg}

    def determine_model_type(self, model: Any, verbose: bool = True) -> str:
        """
        Определение типа модели
        """
        model_type = 'unknown'

        try:
            if hasattr(model, 'get_booster'):
                model_type = 'xgb'
            elif hasattr(model, 'name') and 'lstm' in str(model.name).lower():
                model_type = 'lstm'
            elif 'xgb' in str(type(model)).lower():
                model_type = 'xgb'
            elif 'lstm' in str(type(model)).lower():
                model_type = 'lstm'
            else:
                # Пытаемся определить по другим признакам
                try:
                    import xgboost
                    if isinstance(model, xgboost.XGBClassifier) or isinstance(model, xgboost.XGBRegressor):
                        model_type = 'xgb'
                except:
                    pass

                try:
                    import tensorflow as tf
                    if isinstance(model, tf.keras.Model):
                        model_type = 'lstm'
                except:
                    pass

            if verbose and model_type == 'unknown':
                print(f"  ⚠️  Не удалось определить тип модели, используется по умолчанию")

            return model_type

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Ошибка определения типа модели: {e}")
            return 'unknown'

    def debug_model_features(self, model: Any, scaler: Any, verbose: bool = True):
        """
        Диагностика фичей модели и скейлера
        """
        if verbose:
            print(f"\n🔍 ДИАГНОСТИКА МОДЕЛИ:")

            # Информация о модели
            if hasattr(model, 'feature_names'):
                print(f"  📋 Фичи в модели (model.feature_names): {len(model.feature_names)}")
                if isinstance(model.feature_names, list):
                    print(f"  Первые 10: {model.feature_names[:10]}")

            if hasattr(model, 'base_feature_names'):
                print(f"  📋 Базовые фичи (model.base_feature_names): {len(model.base_feature_names)}")
                print(f"  {model.base_feature_names}")

            # Информация о скейлере
            if scaler is not None:
                print(f"  🔢 Информация о скейлере:")
                if hasattr(scaler, 'n_features_in_'):
                    print(f"    Ожидает фичей: {scaler.n_features_in_}")

                # Пытаемся получить фичи скейлера
                if hasattr(scaler, 'feature_names_in_'):
                    print(f"    Фичи скейлера: {len(scaler.feature_names_in_)}")
                    print(f"    Первые 10: {scaler.feature_names_in_[:10]}")

            # Проверяем, совпадает ли количество фичей
            if hasattr(model, 'feature_names') and scaler is not None and hasattr(scaler, 'n_features_in_'):
                model_features_count = len(model.feature_names) if isinstance(model.feature_names, list) else 0
                if model_features_count > 0:
                    print(f"  ⚖️  Сравнение фичей:")
                    print(f"    Модель: {model_features_count} фичей")
                    print(f"    Скейлер: {scaler.n_features_in_} фичей")

                    if model_features_count != scaler.n_features_in_:
                        print(f"  ❌ НЕСОВПАДЕНИЕ! Модель и скейлер обучены на разном количестве фичей!")