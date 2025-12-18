"""
Модуль для генерации торговых сигналов
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from config import config
from modules.database import Database
from modules.preprocessor import DataPreprocessor
from modules.trainer import ModelTrainer
from modules.state_manager import state_manager


class SignalPredictor:
    def __init__(self, verbose: bool = True):
        """Инициализация предсказателя"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.trainer = ModelTrainer(verbose=verbose)

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

    def get_best_model_id(self, symbol: str, preferred_type: str = None) -> Optional[str]:
        """Получение ID лучшей модели для символа"""
        try:
            models_df = self.db.get_available_models(
                symbol=symbol,
                active_only=True,
                verbose=False
            )

            if not models_df.empty:
                # Если указан предпочтительный тип
                if preferred_type:
                    type_models = models_df[models_df['model_type'].str.contains(preferred_type)]
                    if not type_models.empty:
                        # Берем последнюю модель нужного типа
                        return type_models.iloc[0]['model_id']

                # Иначе берем последнюю XGBoost модель (она быстрее)
                xgb_models = models_df[models_df['model_type'].str.contains('xgb')]
                if not xgb_models.empty:
                    return xgb_models.iloc[0]['model_id']

                # Иначе последнюю модель любого типа
                return models_df.iloc[0]['model_id']

            return None

        except Exception as e:
            self.log(f"Error getting best model: {e}", 'error')
            return None

    def prepare_data_for_xgboost_prediction(self, data: pd.DataFrame, model: Any,
                                          lookback_window: int, verbose: bool = True) -> np.ndarray:
        """
        Подготовка данных для предсказания XGBoost модели
        с правильным количеством фичей
        """
        try:
            if data.empty or model is None:
                return np.array([])

            # Получаем базовые фичи из модели
            if hasattr(model, 'base_feature_names'):
                base_features = model.base_feature_names
            elif hasattr(model, 'feature_names'):
                # Если это расширенные фичи с лагами
                feature_names = model.feature_names
                if feature_names and any('_t-' in str(f) for f in feature_names[:10]):
                    # Извлекаем уникальные базовые фичи
                    base_features = set()
                    for feature in feature_names:
                        if isinstance(feature, str) and '_t-' in feature:
                            base_feature = feature.split('_t-')[0]
                            base_features.add(base_feature)
                        else:
                            base_features.add(str(feature))
                    base_features = list(base_features)
                else:
                    base_features = feature_names
            else:
                # Дефолтный набор фичей
                base_features = ['open', 'high', 'low', 'close', 'volume', 'returns']
                tech_indicators = [col for col in data.columns
                                  if any(indicator in col.lower() for indicator in
                                         ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'obv', 'adx', 'stoch',
                                          'williams'])]
                base_features += tech_indicators

            # Проверяем, какие фичи есть в данных
            available_features = [f for f in base_features if f in data.columns]
            missing_features = [f for f in base_features if f not in data.columns]

            if verbose:
                print(f"  🔧 XGBoost: требуется {len(base_features)} фичей")
                print(f"  📊 Доступно в данных: {len(available_features)} фичей")
                if missing_features:
                    print(f"  ⚠️  Отсутствуют {len(missing_features)} фичей")

            # Создаем недостающие фичи с нулевыми значениями
            for feature in missing_features:
                data[feature] = 0.0

            # Убеждаемся, что у нас достаточно данных
            if len(data) < lookback_window:
                if verbose:
                    print(f"  ❌ Недостаточно данных: {len(data)} < {lookback_window}")
                return np.array([])

            # Берем последние lookback_window значений
            window_data = data.iloc[-lookback_window:][base_features].values

            # Преобразуем в формат для XGBoost (2D)
            X_window_flat = window_data.flatten().reshape(1, -1)

            if verbose:
                print(f"  📏 Размер окна: {window_data.shape} -> {X_window_flat.shape}")
                print(f"  🔢 Всего фичей: {X_window_flat.shape[1]}")

            return X_window_flat

        except Exception as e:
            self.log(f"Error preparing data for XGBoost prediction: {e}", 'error')
            return np.array([])

    def calculate_indicators_for_prediction(self, data: pd.DataFrame, model_type: str,
                                          verbose: bool = True) -> pd.DataFrame:
        """
        Расчет индикаторов специально для предсказания
        с гарантией наличия всех нужных фичей
        """
        try:
            if data.empty:
                return data

            if verbose:
                print(f"  📈 Расчет индикаторов для предсказания...")

            # Используем базовый расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            if verbose:
                print(f"  📊 Базовых индикаторов: {len(data_with_indicators.columns)}")

            # Для LSTM гарантируем наличие всех 55 фичей
            if 'lstm' in model_type.lower():
                if verbose:
                    print(f"  🤖 Подготовка фичей для LSTM (требуется 55)")

                # Список ВСЕХ фичей, которые могут быть в LSTM модели
                all_possible_features = [
                    # Основные OHLCV
                    'open', 'high', 'low', 'close', 'volume',

                    # Скользящие средние
                    'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
                    'EMA_12', 'EMA_26', 'EMA_50',

                    # RSI
                    'RSI_14', 'RSI_7',

                    # Stochastic
                    'STOCH_K', 'STOCH_D',

                    # MACD
                    'MACD', 'MACD_SIGNAL', 'MACD_DIFF',

                    # Bollinger Bands
                    'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_WIDTH', 'BB_PCT',

                    # Другие индикаторы
                    'ATR_14', 'PSAR', 'CCI_20', 'AO', 'WILLIAMS_R',
                    'ROC_10', 'ROC_20', 'MFI_14', 'OBV',

                    # Ichimoku
                    'ICHIMOKU_CONVERSION', 'ICHIMOKU_BASE', 'ICHIMOKU_A', 'ICHIMOKU_B',

                    # Производные фичи
                    'RETURNS', 'LOG_RETURNS', 'VOLATILITY_20', 'VOLUME_MA_20',
                    'PRICE_RANGE', 'BODY_SIZE',

                    # Сигналы индикаторов
                    'RSI_SIGNAL', 'MACD_CROSS', 'BB_SIGNAL', 'TECH_TARGET',

                    # Временные фичи (будут добавлены позже)
                ]

                # Сначала проверяем, что есть базовые 5 фичей
                if not all(col in data_with_indicators.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    if verbose:
                        print(f"  ❌ Отсутствуют базовые OHLCV фичи")
                    return data_with_indicators

                # Проверяем, какие фичи из списка есть в данных
                existing_features = [f for f in all_possible_features if f in data_with_indicators.columns]
                missing_features = [f for f in all_possible_features if f not in data_with_indicators.columns]

                if verbose:
                    print(f"  ✅ Есть фичей: {len(existing_features)}")
                    if missing_features:
                        print(f"  ⚠️  Отсутствуют: {len(missing_features)} фичей")
                        print(f"     Пример: {missing_features[:5]}")

                # Создаем недостающие фичи с нулевыми значениями
                for feature in missing_features:
                    data_with_indicators[feature] = 0.0
                    if verbose and len(missing_features) <= 10:
                        print(f"     ➕ Создана фича: {feature}")

                # Добавляем временные фичи если индекс - datetime
                temporal_features_added = False
                if hasattr(data_with_indicators.index, 'hour'):
                    data_with_indicators['HOUR'] = data_with_indicators.index.hour
                    data_with_indicators['DAY_OF_WEEK'] = data_with_indicators.index.dayofweek
                    data_with_indicators['MONTH'] = data_with_indicators.index.month
                    temporal_features_added = True
                else:
                    # Добавляем фиктивные временные фичи
                    data_with_indicators['HOUR'] = 0
                    data_with_indicators['DAY_OF_WEEK'] = 0
                    data_with_indicators['MONTH'] = 0

                # Формируем итоговый список из 55 фичей
                final_features = all_possible_features + ['HOUR', 'DAY_OF_WEEK', 'MONTH']

                # Убеждаемся, что у нас все фичи
                for feature in final_features:
                    if feature not in data_with_indicators.columns:
                        data_with_indicators[feature] = 0.0

                # Оставляем только нужные фичи и в правильном порядке
                data_with_indicators = data_with_indicators[final_features]

                if verbose:
                    print(f"  ✅ Итоговое количество фичей для LSTM: {len(data_with_indicators.columns)}")
                    print(f"  📊 Первые 10 фичей: {list(data_with_indicators.columns)[:10]}")

            return data_with_indicators

        except Exception as e:
            self.log(f"Error calculating indicators for prediction: {e}", 'error')
            return data

    def _get_lstm_signal(self, symbol: str, model, scaler, model_id: str, model_type: str,
                        data_with_indicators: pd.DataFrame, verbose: bool = True) -> Dict:
        """Получение сигнала для LSTM модели"""
        try:
            if verbose:
                print(f"  🤖 Генерация сигнала LSTM для {symbol}")
                print(f"  📊 Данные: {len(data_with_indicators)} строк, {len(data_with_indicators.columns)} колонок")

            # Проверяем количество фичей
            current_features = len(data_with_indicators.columns)

            if verbose:
                print(f"  🔍 Текущее количество фичей: {current_features}")
                print(f"  🎯 Целевое количество фичей: 55")

            # Если фичей недостаточно, добавляем недостающие
            if current_features < 55:
                missing = 55 - current_features
                if verbose:
                    print(f"  ⚠️  Недостаточно фичей: нужно добавить {missing}")

                # Добавляем фиктивные фичи
                for i in range(missing):
                    col_name = f'MISSING_{i}'
                    data_with_indicators[col_name] = 0.0

                if verbose:
                    print(f"  ✅ Добавлено {missing} фиктивных фичей")

            # Если фичей слишком много, обрезаем
            elif current_features > 55:
                if verbose:
                    print(f"  ⚠️  Слишком много фичей: {current_features}, обрезаем до 55")
                # Оставляем первые 55 колонок
                cols_to_keep = list(data_with_indicators.columns)[:55]
                data_with_indicators = data_with_indicators[cols_to_keep]

            # Подготовка последовательности
            if verbose:
                print(f"  🔧 Подготовка последовательности...")

            X_sequence = self.preprocessor.prepare_features_for_prediction(
                df=data_with_indicators,
                lookback_window=config.model.LOOKBACK_WINDOW,
                verbose=verbose
            )

            if len(X_sequence) == 0:
                if verbose:
                    print(f"  ❌ Не удалось создать последовательность")
                return {'signal': 'HOLD', 'reason': 'Insufficient data for prediction'}

            # Проверяем финальное количество фичей
            final_features = X_sequence.shape[-1]

            if verbose:
                print(f"  ✅ Создана последовательность: {X_sequence.shape}")
                print(f"  🔢 Финальное количество фичей: {final_features}")

            # Двойная проверка - если все еще не 55, корректируем
            if final_features != 55:
                if verbose:
                    print(f"  ⚠️  Все еще несоответствие: {final_features} != 55")
                    print(f"  🔧 Применяем корректировку...")

                if final_features < 55:
                    # Добавляем нулевые фичи
                    diff = 55 - final_features
                    zeros = np.zeros((X_sequence.shape[0], X_sequence.shape[1], diff))
                    X_sequence = np.concatenate([X_sequence, zeros], axis=-1)
                    if verbose:
                        print(f"  ✅ Добавлено {diff} нулевых фичей")
                else:
                    # Обрезаем лишние фичи
                    X_sequence = X_sequence[:, :, :55]
                    if verbose:
                        print(f"  ✅ Обрезано до 55 фичей")

            # Нормализация
            if scaler is not None:
                try:
                    X_normalized, _ = self.preprocessor.normalize_features(
                        X_sequence, fit=False, scaler=scaler, verbose=verbose
                    )
                    if verbose:
                        print(f"  ✅ Данные нормализованы")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Ошибка нормализации: {e}")
                    X_normalized = X_sequence
            else:
                X_normalized = X_sequence
                if verbose:
                    print(f"  ⚠️  Скалер не найден, используем ненормализованные данные")

            # Предсказание
            if verbose:
                print(f"  🤖 Выполнение предсказания LSTM...")

            predictions = model.predict(X_normalized, verbose=0)

            if verbose:
                print(f"  📊 Получены предсказания: {predictions.shape}")

            # Извлекаем вероятности классов
            if len(predictions.shape) == 2 and predictions.shape[1] == 3:
                # Мультиклассовая классификация с 3 классами
                predicted_class = np.argmax(predictions[0]) - 1  # -1, 0, 1
                confidence = np.max(predictions[0])

                probabilities = {
                    'SHORT': float(predictions[0][0]),
                    'HOLD': float(predictions[0][1]),
                    'LONG': float(predictions[0][2])
                }
            elif len(predictions.shape) == 2 and predictions.shape[1] == 1:
                # Бинарная классификация или регрессия
                predicted_value = predictions[0][0]
                if predicted_value > 0.5:
                    predicted_class = 1  # LONG
                    confidence = predicted_value
                elif predicted_value < -0.5:
                    predicted_class = -1  # SHORT
                    confidence = abs(predicted_value)
                else:
                    predicted_class = 0  # HOLD
                    confidence = 0.5

                probabilities = {
                    'SHORT': max(0, 1 - predicted_value) / 2,
                    'HOLD': 0.5,
                    'LONG': max(0, predicted_value) / 2
                }
            else:
                # Неизвестный формат
                predicted_class = 0
                confidence = 0.0
                probabilities = {'SHORT': 0, 'HOLD': 1, 'LONG': 0}

            if verbose:
                print(f"  🎯 Предсказание: класс {predicted_class}, уверенность {confidence:.2%}")
                print(f"  📊 Вероятности: SHORT={probabilities['SHORT']:.2%}, "
                      f"HOLD={probabilities['HOLD']:.2%}, LONG={probabilities['LONG']:.2%}")

            # Преобразование класса в сигнал
            signal_map = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}
            signal = signal_map.get(predicted_class, 'HOLD')

            # Создание результата
            result = {
                'symbol': symbol,
                'signal': signal,
                'timestamp': datetime.now(),
                'price': float(data_with_indicators['close'].iloc[-1]) if 'close' in data_with_indicators.columns else 0.0,
                'model_id': model_id,
                'model_type': model_type,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'reason': 'AI model prediction'
            }

            self.log(f"Generated LSTM signal for {symbol}: {signal} (confidence: {confidence:.2f})")
            return result

        except Exception as e:
            self.log(f"Error in LSTM signal generation: {e}", 'error')
            return {'symbol': symbol, 'signal': 'HOLD', 'reason': f'LSTM Error: {str(e)}'}

    def _get_xgboost_signal(self, symbol: str, model, scaler, model_id: str, model_type: str,
                           data_with_indicators: pd.DataFrame, verbose: bool = True) -> Dict:
        """Получение сигнала для XGBoost модели"""
        try:
            if verbose:
                print(f"  🤖 Генерация сигнала XGBoost для {symbol}")

            # Для XGBoost - специальная подготовка данных
            X_window_flat = self.prepare_data_for_xgboost_prediction(
                data_with_indicators, model, config.model.LOOKBACK_WINDOW, verbose
            )

            if X_window_flat.size == 0:
                return {'signal': 'HOLD', 'reason': 'Failed to prepare data for XGBoost prediction'}

            # Проверяем совпадение количества фичей
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
            elif hasattr(model, 'feature_names'):
                expected_features = len(model.feature_names)
            else:
                expected_features = X_window_flat.shape[1]

            if X_window_flat.shape[1] != expected_features:
                if verbose:
                    print(f"  ⚠️  Несоответствие фичей: данные {X_window_flat.shape[1]}, модель {expected_features}")
                    print(f"  🔧 Корректируем размерность...")

                # Корректируем размерность
                if X_window_flat.shape[1] < expected_features:
                    # Дополняем нулями
                    diff = expected_features - X_window_flat.shape[1]
                    zeros = np.zeros((1, diff))
                    X_window_flat = np.hstack([X_window_flat, zeros])
                    if verbose:
                        print(f"  ✅ Дополнено нулями: +{diff} фичей")
                else:
                    # Обрезаем
                    X_window_flat = X_window_flat[:, :expected_features]
                    if verbose:
                        print(f"  ✅ Обрезано: {X_window_flat.shape[1]} фичей")

            # Нормализация если есть скейлер
            if scaler is not None:
                try:
                    X_normalized = scaler.transform(X_window_flat)
                    if verbose:
                        print(f"  ✅ Данные нормализованы")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Ошибка нормализации: {e}")
                    X_normalized = X_window_flat
            else:
                X_normalized = X_window_flat

            # Предсказание
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_normalized)[0]
                predicted_class = model.predict(X_normalized)[0]
                confidence = np.max(proba)

                # Преобразуем класс [0,1,2] -> [-1,0,1]
                predicted_class = int(predicted_class) - 1

                probabilities = {
                    'SHORT': float(proba[0]),
                    'HOLD': float(proba[1]),
                    'LONG': float(proba[2])
                }
            else:
                predicted_class = model.predict(X_normalized)[0]
                predicted_class = int(predicted_class) - 1
                confidence = 1.0
                probabilities = {'SHORT': 0, 'HOLD': 0, 'LONG': 0}
                probabilities[['SHORT', 'HOLD', 'LONG'][predicted_class + 1]] = 1.0

            if verbose:
                print(f"  🤖 Предсказание: класс {predicted_class}, уверенность {confidence:.2%}")

            # Преобразование класса в сигнал
            signal_map = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}
            signal = signal_map.get(predicted_class, 'HOLD')

            # Создание результата
            result = {
                'symbol': symbol,
                'signal': signal,
                'timestamp': datetime.now(),
                'price': float(data_with_indicators['close'].iloc[-1]) if 'close' in data_with_indicators.columns else 0.0,
                'model_id': model_id,
                'model_type': model_type,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'reason': 'AI model prediction'
            }

            self.log(f"Generated XGBoost signal for {symbol}: {signal} (confidence: {confidence:.2f})")
            return result

        except Exception as e:
            self.log(f"Error in XGBoost signal generation: {e}", 'error')
            return {'symbol': symbol, 'signal': 'HOLD', 'reason': f'XGBoost Error: {str(e)}'}

    def get_signal(self, symbol: str = None, model_id: str = None, verbose: bool = True) -> Dict:
        """
        Получение торгового сигнала для указанной пары
        """
        try:
            if symbol is None:
                symbol = state_manager.get_selected_symbol()
                if not symbol:
                    return {'signal': 'HOLD', 'reason': 'No symbol selected'}

            self.log(f"Generating signal for {symbol}")

            # Загрузка модели
            if model_id:
                # Загрузка конкретной модели
                model, scaler, model_type = self.load_specific_model_by_id(model_id, symbol)
                if model is None:
                    return {'signal': 'HOLD', 'reason': f'Failed to load model {model_id}'}
            else:
                # Загрузка лучшей модели
                model_id = self.get_best_model_id(symbol)
                if model_id is None:
                    return {'signal': 'HOLD', 'reason': 'No trained models available'}

                model, scaler = self.trainer.load_model(model_id, verbose=verbose)
                if model is None:
                    return {'signal': 'HOLD', 'reason': 'Failed to load model'}

                # Получаем информацию о модели
                all_models_df = self.db.get_available_models(
                    symbol=symbol,
                    active_only=True,
                    verbose=False
                )
                if all_models_df.empty:
                    return {'signal': 'HOLD', 'reason': 'No models found for symbol'}

                model_row = all_models_df[all_models_df['model_id'] == model_id]
                if model_row.empty:
                    return {'signal': 'HOLD', 'reason': f'Model {model_id} not found'}

                model_type = model_row.iloc[0]['model_type']

            # Получение достаточного количества данных для корректного расчета индикаторов
            end_date = datetime.now()
            # Для генерации сигналов берем БОЛЬШЕ данных, чтобы все индикаторы успели рассчитаться
            start_date = end_date - timedelta(days=180)  # 6 месяцев данных

            if verbose:
                print(f"  📅 Загрузка данных с {start_date.date()} по {end_date.date()}")

            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=state_manager.get_selected_timeframe(),
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )

            if data.empty or len(data) < config.model.LOOKBACK_WINDOW:
                if verbose:
                    print(f"  ❌ Недостаточно данных: {len(data)} строк < {config.model.LOOKBACK_WINDOW}")
                return {'signal': 'HOLD', 'reason': 'Insufficient data'}

            if verbose:
                print(f"  📊 Загружено данных: {len(data)} строк")
                print(f"  🤖 Тип модели: {model_type}")

            # Расчет индикаторов с гарантией наличия всех нужных фичей
            data_with_indicators = self.calculate_indicators_for_prediction(
                data, model_type, verbose=verbose
            )

            if data_with_indicators.empty:
                return {'signal': 'HOLD', 'reason': 'Failed to calculate indicators'}

            # Подготовка данных для предсказания в зависимости от типа модели
            if 'lstm' in model_type.lower():
                return self._get_lstm_signal(symbol, model, scaler, model_id, model_type,
                                            data_with_indicators, verbose)
            elif 'xgb' in model_type.lower():
                return self._get_xgboost_signal(symbol, model, scaler, model_id, model_type,
                                               data_with_indicators, verbose)
            else:
                return {'signal': 'HOLD', 'reason': 'Unknown model type'}

        except Exception as e:
            self.log(f"Error generating signal for {symbol}: {e}", 'error')
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", 'error')
            return {'symbol': symbol, 'signal': 'HOLD', 'reason': f'Error: {str(e)}'}

    def load_specific_model_by_id(self, model_id: str, symbol: str = None) -> Tuple[Any, Any, str]:
        """Загрузка конкретной модели по ID"""
        try:
            # Получаем все модели для символа
            if symbol:
                all_models_df = self.db.get_available_models(
                    symbol=symbol,
                    active_only=True,
                    verbose=False
                )
            else:
                all_models_df = self.db.get_available_models(
                    active_only=True,
                    verbose=False
                )

            if all_models_df.empty:
                return None, None, None

            # Находим конкретную модель по ID
            model_row = all_models_df[all_models_df['model_id'] == model_id]
            if model_row.empty:
                return None, None, None

            # Загружаем модель
            model, scaler = self.trainer.load_model(model_id, verbose=self.verbose)
            model_type = model_row.iloc[0]['model_type']

            return model, scaler, model_type

        except Exception as e:
            self.log(f"Error loading specific model {model_id}: {e}", 'error')
            return None, None, None