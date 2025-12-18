"""
Модуль для генерации торговых сигналов
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
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

    def get_signal(self, symbol: str = None, model_id: str = None, verbose: bool = True) -> Dict:
        """
        Получение торгового сигнала для указанной пары

        Args:
            symbol: Торговый символ
            model_id: ID конкретной модели (опционально)
            verbose: Вывод подробной информации

        Returns:
            dict: Словарь с сигналом и метаданными
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

            # Получение последних данных
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=state_manager.get_selected_timeframe(),
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )

            if data.empty or len(data) < config.model.LOOKBACK_WINDOW:
                return {'signal': 'HOLD', 'reason': 'Insufficient data'}

            # Расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            # Подготовка данных для предсказания
            X_sequence = self.preprocessor.prepare_features_for_prediction(
                df=data_with_indicators,
                lookback_window=config.model.LOOKBACK_WINDOW,
                verbose=verbose
            )

            if len(X_sequence) == 0:
                return {'signal': 'HOLD', 'reason': 'Insufficient data for prediction'}

            # Проверяем количество фичей
            num_features = X_sequence.shape[-1]

            # Проверяем совместимость фичей с моделью
            # Если это LSTM и количество фичей не соответствует, попробуем использовать базовые фичи
            if 'lstm' in model_type and num_features != 55:
                self.log(f"Model expects 55 features but data has {num_features}. Trying to adjust...", 'warning')

                # Используем только базовые фичи
                basic_columns = ['open', 'high', 'low', 'close', 'volume']
                basic_data = data[basic_columns].copy()

                # Пересчитываем индикаторы
                basic_with_indicators = self.preprocessor.calculate_all_indicators(
                    basic_data, verbose=verbose
                )

                X_sequence = self.preprocessor.prepare_features_for_prediction(
                    df=basic_with_indicators,
                    lookback_window=config.model.LOOKBACK_WINDOW,
                    verbose=verbose
                )

                if X_sequence.shape[-1] != 55:
                    return {'signal': 'HOLD',
                            'reason': f'Feature mismatch: model expects 55 features, got {X_sequence.shape[-1]}'}

            # Предсказание
            if 'lstm' in model_type:
                # Нормализация
                X_normalized, _ = self.preprocessor.normalize_features(
                    X_sequence, fit=False, scaler=scaler, verbose=verbose
                )

                # Предсказание
                predictions = model.predict(X_normalized, verbose=0)
                predicted_class = np.argmax(predictions[0]) - 1
                confidence = np.max(predictions[0])

                probabilities = {
                    'SHORT': float(predictions[0][0]),
                    'HOLD': float(predictions[0][1]),
                    'LONG': float(predictions[0][2])
                }

            elif 'xgb' in model_type:
                # Преобразование 3D -> 2D для XGBoost
                X_2d = X_sequence.reshape(1, -1)

                # Нормализация
                X_normalized, _ = self.preprocessor.normalize_features(
                    X_2d, fit=False, scaler=scaler, verbose=verbose
                )

                # Предсказание
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_normalized)[0]
                    predicted_class = model.predict(X_normalized)[0]
                    confidence = np.max(proba)

                    probabilities = {
                        'SHORT': float(proba[0]),
                        'HOLD': float(proba[1]),
                        'LONG': float(proba[2])
                    }
                else:
                    predicted_class = model.predict(X_normalized)[0]
                    confidence = 1.0
                    probabilities = {'SHORT': 0, 'HOLD': 0, 'LONG': 0}
                    probabilities[['SHORT', 'HOLD', 'LONG'][predicted_class + 1]] = 1.0
            else:
                return {'signal': 'HOLD', 'reason': 'Unknown model type'}

            # Преобразование класса в сигнал
            signal_map = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}
            signal = signal_map.get(predicted_class, 'HOLD')

            # Создание результата
            result = {
                'symbol': symbol,
                'signal': signal,
                'timestamp': datetime.now(),
                'price': float(data['close'].iloc[-1]),
                'model_id': model_id,
                'model_type': model_type,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'reason': 'AI model prediction'
            }

            self.log(f"Generated signal for {symbol}: {signal} (confidence: {confidence:.2f})")
            return result

        except Exception as e:
            self.log(f"Error generating signal for {symbol}: {e}", 'error')
            return {'symbol': symbol, 'signal': 'HOLD', 'reason': f'Error: {str(e)}'}

    def load_specific_model_by_id(self, model_id: str, symbol: str = None):
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
                return None, None

            # Находим конкретную модель по ID
            model_row = all_models_df[all_models_df['model_id'] == model_id]
            if model_row.empty:
                return None, None

            # Загружаем модель
            model, scaler = self.trainer.load_model(model_id, verbose=self.verbose)
            model_type = model_row.iloc[0]['model_type']

            return model, scaler, model_type

        except Exception as e:
            self.log(f"Error loading specific model {model_id}: {e}", 'error')
            return None, None, None