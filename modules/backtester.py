"""
Модуль для бэктестирования торговых стратегий
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from config import config
from modules.database import Database
from modules.predictor import SignalPredictor
from modules.preprocessor import DataPreprocessor
from modules.state_manager import state_manager


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

    def generate_backtest_signals(self, data: pd.DataFrame, model: Any, scaler: Any,
                                  model_type: str, verbose: bool = True) -> pd.DataFrame:
        """
        Генерация торговых сигналов для бэктеста
        """
        try:
            if data.empty or model is None:
                return pd.DataFrame()

            # Создаем копию данных для сигналов
            signals_df = data.copy()
            signals_df['signal'] = 0  # 0 = HOLD, 1 = LONG, -1 = SHORT
            signals_df['confidence'] = 0.0
            signals_df['prediction_time'] = signals_df.index

            # Получаем lookback_window из конфига
            lookback_window = config.model.LOOKBACK_WINDOW

            # Получаем feature_names из модели если есть
            feature_names = None
            if hasattr(model, 'feature_names'):
                feature_names = model.feature_names
            elif hasattr(model, 'base_feature_names'):
                # Используем базовые имена фичей для XGBoost
                feature_names = model.base_feature_names

            # Если нет feature_names, получаем их из данных
            if feature_names is None:
                feature_names = [col for col in data.columns
                                 if not col.startswith('TARGET_')
                                 and col not in ['signal', 'confidence', 'prediction_time']
                                 and pd.api.types.is_numeric_dtype(data[col])]

            # Для XGBoost нам нужны правильные фичи
            if 'xgb' in model_type.lower():
                # Для XGBoost, который обучался на 2D данных (lookback_window * features)
                expected_features = lookback_window * len(feature_names)
            else:
                expected_features = len(feature_names)

            for i in range(lookback_window, len(signals_df)):
                try:
                    # Берем окно данных
                    window_data = signals_df.iloc[i - lookback_window:i]

                    # Проверяем, что у нас достаточно фичей
                    available_features = [col for col in feature_names if col in window_data.columns]

                    if len(available_features) != len(feature_names):
                        if verbose and i == lookback_window:  # Показываем только для первого окна
                            self.log(f"Feature mismatch: expected {len(feature_names)}, got {len(available_features)}",
                                     'warning')
                        continue

                    # Готовим данные для предсказания
                    X_window = window_data[available_features].values

                    # Для XGBoost нужно преобразовать в 2D
                    if 'xgb' in model_type.lower():
                        X_window_flat = X_window.flatten().reshape(1, -1)

                        # Проверяем размерность
                        if X_window_flat.shape[1] != expected_features:
                            if verbose and i == lookback_window:
                                self.log(
                                    f"XGBoost feature shape mismatch: expected {expected_features}, got {X_window_flat.shape[1]}",
                                    'warning')
                                self.log(f"Lookback: {lookback_window}, Features: {len(feature_names)}", 'warning')
                            continue

                        X_window_final = X_window_flat
                    else:
                        # Для LSTM оставляем как есть (3D)
                        X_window_final = X_window.reshape(1, lookback_window, -1)

                    # Нормализуем если есть скейлер
                    if scaler is not None:
                        try:
                            if 'xgb' in model_type.lower():
                                X_window_norm = scaler.transform(X_window_final)
                            else:
                                X_window_norm = scaler.transform(X_window_final.reshape(1, -1)).reshape(1,
                                                                                                        lookback_window,
                                                                                                        -1)
                        except Exception as e:
                            if verbose and i == lookback_window:
                                self.log(f"Normalization error: {str(e)}", 'warning')
                            X_window_norm = X_window_final
                    else:
                        X_window_norm = X_window_final

                    # Делаем предсказание
                    if hasattr(model, 'predict'):
                        prediction = model.predict(X_window_norm)

                        if 'lstm' in model_type.lower():
                            # LSTM возвращает вероятности для каждого класса
                            if len(prediction.shape) == 2:
                                predicted_class = np.argmax(prediction[0]) - 1  # Преобразуем [0,1,2] -> [-1,0,1]
                                confidence = np.max(prediction[0])
                            else:
                                predicted_class = int(prediction[0]) - 1
                                confidence = 0.5
                        else:
                            # XGBoost возвращает классы
                            predicted_class = int(prediction[0]) - 1
                            confidence = 0.5

                        signals_df.iloc[i, signals_df.columns.get_loc('signal')] = predicted_class
                        signals_df.iloc[i, signals_df.columns.get_loc('confidence')] = confidence

                except Exception as e:
                    if verbose and i == lookback_window:  # Показываем только для первого окна
                        self.log(f"Error generating signal at index {i}: {str(e)}", 'warning')
                    continue

            # Фильтруем только строки с сигналами
            signals_with_data = signals_df[signals_df['signal'] != 0].copy()

            if verbose:
                self.log(f"Generated {len(signals_with_data)} signals")

            return signals_with_data

        except Exception as e:
            self.log(f"Error generating backtest signals: {str(e)}", 'error')
            return pd.DataFrame()

    def generate_backtest_signals_simple(self, data: pd.DataFrame, model: Any, scaler: Any,
                                         model_type: str, verbose: bool = True) -> pd.DataFrame:
        """
        Упрощенная генерация торговых сигналов для бэктеста
        """
        try:
            if data.empty or model is None:
                return pd.DataFrame()

            # Создаем копию данных для сигналов
            signals_df = data.copy()
            signals_df['signal'] = 0
            signals_df['confidence'] = 0.0

            # Получаем lookback_window из конфига
            lookback_window = config.model.LOOKBACK_WINDOW

            # ВАЖНО: Определяем, какие фичи использовались при обучении
            # Способ 1: Получаем базовые фичи из модели
            base_feature_columns = None

            if hasattr(model, 'base_feature_names'):
                base_feature_columns = model.base_feature_names
                if verbose:
                    print(f"  📋 Базовые фичи из модели: {len(base_feature_columns)} фичей")
                    print(f"  📋 Базовые фичи: {base_feature_columns}")
            elif hasattr(model, 'feature_names'):
                # Проверяем, являются ли фичи расширенными
                feature_names = model.feature_names
                if isinstance(feature_names, list) and len(feature_names) > 0:
                    # Если фичи содержат временные лаги - это расширенные фичи
                    if any('_t-' in str(feature) for feature in feature_names[:10]):
                        # Извлекаем базовые фичи из расширенных
                        base_features = set()
                        for feature in feature_names:
                            if isinstance(feature, str) and '_t-' in feature:
                                base_feature = feature.split('_t-')[0]
                                base_features.add(base_feature)
                            else:
                                base_features.add(str(feature))
                        base_feature_columns = list(base_features)
                        if verbose:
                            print(f"  🔍 Обнаружены расширенные фичи в модели")
                            print(f"  🔄 Извлечено базовых фичей: {len(base_feature_columns)}")
                    else:
                        # Если нет временных лагов - это базовые фичи
                        base_feature_columns = feature_names
                        if verbose:
                            print(f"  📋 Используются фичи из модели как базовые: {len(base_feature_columns)} фичей")

            # Способ 2: Если фичи не найдены, используем дефолтный набор
            if base_feature_columns is None:
                # Базовый набор фичей (как в trainer.py)
                base_features = ['close', 'volume', 'returns']

                # Технические индикаторы
                tech_indicators = [col for col in data.columns
                                   if any(indicator in col.lower() for indicator in
                                          ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'obv', 'adx'])]

                base_feature_columns = base_features + tech_indicators

                # Фильтруем только существующие в данных
                base_feature_columns = [col for col in base_feature_columns if col in data.columns]

                if verbose:
                    print(
                        f"  ⚠️  Фичи не найдены в модели, используется дефолтный набор: {len(base_feature_columns)} фичей")

            # Убеждаемся, что все базовые фичи есть в данных
            missing_features = []
            for feature in base_feature_columns:
                if feature not in signals_df.columns:
                    missing_features.append(feature)

            if missing_features:
                if verbose:
                    print(f"  ⚠️  Создаем недостающие базовые фичи: {len(missing_features)} фичей")
                for feature in missing_features:
                    signals_df[feature] = 0.0

            # Обновляем список базовых фичей только теми, что есть в данных
            base_feature_columns = [col for col in base_feature_columns if col in signals_df.columns]
            base_feature_columns = sorted(base_feature_columns)  # Сортируем для consistency

            if verbose:
                print(f"  🔍 Базовые фичи для XGBoost: {len(base_feature_columns)} фичей")
                print(f"  📊 Базовые фичи: {base_feature_columns}")
                print(f"  📐 Lookback window: {lookback_window}")
                print(f"  🤖 Тип модели: {model_type}")

                # Вычисляем ожидаемое количество фичей
                expected_features = len(base_feature_columns) * lookback_window
                print(f"  🔢 Ожидается XGBoost фичей: {expected_features} (базовые × lookback)")

                # Проверяем скейлер
                if scaler is not None and hasattr(scaler, 'n_features_in_'):
                    print(f"  🔢 Скейлер ожидает: {scaler.n_features_in_} фичей")
                    if scaler.n_features_in_ != expected_features:
                        print(
                            f"  ⚠️  НЕСОВПАДЕНИЕ! Скейлер ожидает {scaler.n_features_in_}, а должно быть {expected_features}")

            # Проверяем, достаточно ли фичей
            if len(base_feature_columns) == 0:
                print(f"  ❌ Нет базовых фичей для предсказания")
                return pd.DataFrame()

            # Генерируем сигналы
            signals_generated = 0

            for i in range(lookback_window, len(signals_df)):
                try:
                    # Берем окно данных
                    window_data = signals_df.iloc[i - lookback_window:i]

                    # Проверяем, что у нас все нужные базовые фичи
                    available_features = [col for col in base_feature_columns if col in window_data.columns]
                    if len(available_features) != len(base_feature_columns):
                        if verbose and i == lookback_window:
                            print(
                                f"  ⚠️  Не все базовые фичи доступны: {len(available_features)} из {len(base_feature_columns)}")
                        continue

                    # Готовим X для предсказания - КРИТИЧЕСКАЯ ЧАСТЬ!
                    X_window = window_data[base_feature_columns].values

                    # Для XGBoost преобразуем в 2D с правильным форматом
                    if 'xgb' in model_type.lower():
                        # Правильное преобразование: (lookback_window, базовые_фичи) -> (1, lookback_window × базовые_фичи)
                        X_window_flat = X_window.flatten().reshape(1, -1)

                        # Проверяем размерность
                        expected_shape = len(base_feature_columns) * lookback_window
                        actual_shape = X_window_flat.shape[1]

                        if verbose and i == lookback_window:
                            print(f"  📊 Окно данных shape: {X_window.shape}")
                            print(f"  📊 После flatten: {X_window_flat.shape}")
                            print(f"  🔍 Ожидается: {expected_shape}, получено: {actual_shape}")

                        if actual_shape != expected_shape:
                            if verbose and i == lookback_window:
                                print(f"  ❌ Размерность не совпадает: {actual_shape} != {expected_shape}")
                                print(f"     Базовые фичи: {len(base_feature_columns)}, lookback: {lookback_window}")
                            continue

                        # Нормализуем если есть скейлер
                        if scaler is not None:
                            try:
                                # ВАЖНО: Проверяем, что скейлер ожидает правильное количество фичей
                                if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != actual_shape:
                                    if verbose and i == lookback_window:
                                        print(
                                            f"  ⚠️  Скейлер ожидает {scaler.n_features_in_} фичей, а получили {actual_shape}")
                                        print(f"  ⚠️  Пытаемся использовать скейлер, но могут быть ошибки...")

                                X_window_norm = scaler.transform(X_window_flat)
                                if verbose and i == lookback_window:
                                    print(f"  ✅ Данные нормализованы успешно")
                            except Exception as scaler_error:
                                if verbose and i == lookback_window:
                                    print(f"  ⚠️  Ошибка нормализации: {scaler_error}")
                                # Продолжаем без нормализации
                                X_window_norm = X_window_flat
                        else:
                            X_window_norm = X_window_flat

                        # Делаем предсказание
                        try:
                            prediction = model.predict(X_window_norm)
                            predicted_class = int(prediction[0]) - 1  # Преобразуем [0,1,2] -> [-1,0,1]

                            # Получаем вероятность если возможно
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(X_window_norm)
                                confidence = np.max(proba[0])
                            else:
                                confidence = 0.5

                            if verbose and i == lookback_window:
                                print(f"  ✅ Предсказание успешно: class={predicted_class}, confidence={confidence:.3f}")
                                signals_generated += 1
                        except Exception as predict_error:
                            if verbose and i == lookback_window:
                                print(f"  ⚠️  Ошибка предсказания: {predict_error}")
                            continue

                    else:  # Для LSTM
                        X_window_3d = X_window.reshape(1, lookback_window, -1)

                        if verbose and i == lookback_window:
                            print(f"  📊 LSTM input shape: {X_window_3d.shape}")

                        # Нормализуем если есть скейлер
                        if scaler is not None:
                            try:
                                # Для LSTM скейлер ожидает 2D данные
                                X_flat = X_window_3d.reshape(1, -1)
                                X_norm_flat = scaler.transform(X_flat)
                                X_window_norm = X_norm_flat.reshape(1, lookback_window, -1)
                            except Exception as e:
                                if verbose and i == lookback_window:
                                    print(f"  ⚠️  Ошибка нормализации LSTM: {e}")
                                X_window_norm = X_window_3d
                        else:
                            X_window_norm = X_window_3d

                        # Делаем предсказание
                        prediction = model.predict(X_window_norm)

                        if len(prediction.shape) == 2:
                            predicted_class = np.argmax(prediction[0]) - 1
                            confidence = np.max(prediction[0])
                        else:
                            predicted_class = int(prediction[0]) - 1
                            confidence = 0.5

                    # Сохраняем сигнал
                    signals_df.iloc[i, signals_df.columns.get_loc('signal')] = predicted_class
                    signals_df.iloc[i, signals_df.columns.get_loc('confidence')] = confidence

                except Exception as e:
                    if verbose and i == lookback_window:
                        print(f"  ⚠️  Ошибка предсказания в точке {i}: {e}")
                        import traceback
                        traceback.print_exc()
                    continue

            # Фильтруем сигналы
            valid_signals = signals_df[signals_df['signal'] != 0].copy()

            if verbose:
                print(f"  ✅ Сгенерировано {len(valid_signals)} сигналов")
                if len(valid_signals) > 0:
                    long_count = len(valid_signals[valid_signals['signal'] == 1])
                    short_count = len(valid_signals[valid_signals['signal'] == -1])
                    hold_count = len(valid_signals[valid_signals['signal'] == 0])
                    print(f"  📈 Сигналы: LONG={long_count}, SHORT={short_count}, HOLD={hold_count}")

            return valid_signals

        except Exception as e:
            if verbose:
                print(f"  ❌ Ошибка генерации сигналов: {e}")
                import traceback
                traceback.print_exc()
            return pd.DataFrame()


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

            for i, (timestamp, row) in enumerate(signals.iterrows()):
                try:
                    current_price = row['close']
                    signal = row['signal']
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
                            'result': 'OPEN'
                        }
                        trade_history.append(trade)

                        if verbose and len(trade_history) <= 10:  # Показываем только первые 10 сделок
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

                            if verbose and len(trade_history) <= 10:
                                result_emoji = "✅" if pnl > 0 else "❌"
                                print(f"  {result_emoji} Закрыта позиция: P&L ${pnl:+.2f} ({pnl_pct*100:+.2f}%) - {close_reason}")

                    # Обновляем максимальную просадку
                    if balance > peak_balance:
                        peak_balance = balance

                    current_drawdown = (peak_balance - balance) / peak_balance * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown

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
                self.log(f"Backtest completed: {total_trades} trades, Return: {total_return:.2f}%")

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
                'sharpe_ratio': 0,  # Можно рассчитать если есть данные о доходности
                'max_drawdown': results['max_drawdown'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'total_trades': results['total_trades'],
                'winning_trades': results['winning_trades'],
                'losing_trades': results['losing_trades'],
                'avg_win': results['avg_win'],
                'avg_loss': results['avg_loss'],
                'details': '{}'  # Можно сохранить детали сделок
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
                print(f"🚀 Запуск бэктеста для {symbol}")
                print(f"   Начальный баланс: ${initial_balance:,.2f}")
                if commission is not None:
                    print(f"   Комиссия: {commission * 100:.2f}%")
                else:
                    print(f"   Комиссия: {self.commission * 100:.2f}%")

            # Устанавливаем комиссию если предоставлена
            if commission is not None:
                self.commission = commission

            # Получаем даты для бэктеста
            start_date, end_date = self.state_manager.get_backtest_dates()

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

            # Получаем модель
            model_info = self.get_best_model(symbol, model_id, verbose=verbose)

            if not model_info:
                if verbose:
                    print("❌ Не найдена подходящая модель для бэктеста")
                return {'error': 'No suitable model found'}

            model, scaler = model_info

            if verbose:
                self.debug_model_features(model, scaler, verbose=verbose)

            model_type = 'unknown'

            # Добавьте:
            # Явно определяем тип модели
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
                    if isinstance(model, xgboost.XGBClassifier):
                        model_type = 'xgb'
                except:
                    pass

                try:
                    import tensorflow as tf
                    if isinstance(model, tf.keras.Model):
                        model_type = 'lstm'
                except:
                    pass

            if verbose:
                print(f"  🤖 Окончательно определен тип модели: {model_type}")

            # Подготавливаем данные
            preprocessed_data = self.prepare_backtest_data(
                data, model_type=model_type, verbose=verbose
            )

            if preprocessed_data.empty:
                if verbose:
                    print("❌ Не удалось подготовить данные для бэктеста")
                return {'error': 'Failed to prepare data for backtest'}

            # Генерируем сигналы
            signals = self.generate_backtest_signals_simple(
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
                    print("✅ Бэктест завершен успешно!")
                    print(f"   Конечный баланс: ${results['final_balance']:,.2f}")
                    print(f"   Общая доходность: {results['total_return']:.2f}%")
                    print(f"   Win Rate: {results['win_rate']:.1f}%")

            return results

        except Exception as e:
            error_msg = f"Error in backtest: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
            return {'error': error_msg}

    def get_model_features(self, model: Any, verbose: bool = True) -> List[str]:
        """
        Получение фичей из модели
        """
        try:
            feature_columns = None

            # Пытаемся получить фичи из атрибутов модели
            if hasattr(model, 'base_feature_names'):
                return model.base_feature_names
            elif hasattr(model, '_features'):
                return model._features
            elif hasattr(model, 'feature_names'):
                # Проверяем, не являются ли это расширенными фичами с лагами
                feature_names = model.feature_names
                if isinstance(feature_names, list) and len(feature_names) > 0:
                    # Если первый фич содержит временной лаг, извлекаем базовые фичи
                    if any('_t-' in feature for feature in feature_names):
                        base_features = set()
                        for feature in feature_names:
                            if '_t-' in feature:
                                base_feature = feature.split('_t-')[0]
                                base_features.add(base_feature)
                        return list(base_features)
                    else:
                        return feature_names

            # Пытаемся получить из метаданных модели
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if hasattr(booster, 'feature_names'):
                    return booster.feature_names

            return None

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Ошибка получения фичей из модели: {e}")
            return None

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
                        print(f"  ⚠️  Это основная причина ошибки!")

    def debug_data_preparation(self, data: pd.DataFrame, feature_columns: List[str],
                               lookback_window: int, model_type: str, verbose: bool = True):
        """
        Детальная диагностика подготовки данных
        """
        if verbose:
            print(f"\n🔬 ДЕТАЛЬНАЯ ДИАГНОСТИКА ПОДГОТОВКИ ДАННЫХ:")
            print(f"  📊 Исходные данные: {len(data)} строк, {len(data.columns)} колонок")
            print(f"  🔍 Используемые фичи: {len(feature_columns)}")
            print(f"  📐 Lookback window: {lookback_window}")
            print(f"  🤖 Тип модели: {model_type}")

            # Показываем первые несколько строк с фичами
            if len(data) > 0 and len(feature_columns) > 0:
                sample_data = data[feature_columns].head(3)
                print(f"  📋 Пример данных (первые 3 строки):")
                for idx, row in sample_data.iterrows():
                    print(f"    {idx}: {[round(val, 4) for val in row.values[:5]]}...")

            # Для XGBoost показываем ожидаемый формат
            if 'xgb' in model_type.lower():
                print(f"\n  🎯 ОЖИДАЕМЫЙ ФОРМАТ ДЛЯ XGBOOST:")
                print(f"    Входные данные: окно {lookback_window} × {len(feature_columns)} фичей")
                print(
                    f"    После flatten: 1 × {lookback_window * len(feature_columns)} = 1 × {lookback_window * len(feature_columns)}")

                # Пример для первой точки
                if len(data) >= lookback_window:
                    window_data = data.iloc[:lookback_window][feature_columns]
                    print(f"\n  📊 ПРИМЕР ПРЕОБРАЗОВАНИЯ:")
                    print(f"    Окно данных shape: {window_data.shape}")
                    print(f"    Flattened shape: {window_data.values.flatten().reshape(1, -1).shape}")