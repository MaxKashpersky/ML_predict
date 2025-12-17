"""
Модуль для обучения моделей машинного обучения
"""

import numpy as np
import pandas as pd
import logging
import json
import os
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from config import config

# Импорты для машинного обучения
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("Tensorflow/Keras not installed. Some features will be unavailable.")

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split
except ImportError:
    print("XGBoost/scikit-learn not installed. Some features will be unavailable.")

from modules.database import Database
from modules.preprocessor import DataPreprocessor


class ModelTrainer:
    def __init__(self, verbose: bool = True):
        """Инициализация тренера моделей"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.model_cache = {}

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

    def generate_model_id(self, symbol: str, model_type: str) -> str:
        """
        Генерация уникального ID для модели

        Args:
            symbol: Торговая пара
            model_type: Тип модели

        Returns:
            Уникальный ID модели
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.md5(f"{symbol}_{model_type}_{timestamp}".encode()).hexdigest()[:8]
        return f"{symbol}_{model_type}_{timestamp}_{random_hash}"

    def train_lstm_classifier(self, symbol: str, timeframe: str = '5m',
                              target_column: str = 'TARGET_CLASS_5',
                              verbose: bool = True) -> Tuple[Any, Dict]:
        """
        Обучение LSTM классификатора

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            target_column: Целевая колонка
            verbose: Флаг логирования

        Returns:
            Обученная модель и метрики
        """
        try:
            self.log(f"Training LSTM classifier for {symbol} {timeframe}...")

            # Получение данных
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=config.TRAIN_START_DATE,
                end_date=config.TRAIN_END_DATE,
                verbose=verbose
            )

            if data.empty:
                self.log(f"No data for {symbol} {timeframe}", 'error')
                return None, {}

            # Расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            if data_with_indicators.empty:
                self.log("Failed to calculate indicators", 'error')
                return None, {}

            # Подготовка данных для обучения
            X, y = self.preprocessor.prepare_features_for_training(
                df=data_with_indicators,
                target_column=target_column,
                lookback_window=config.model.LOOKBACK_WINDOW,
                verbose=verbose
            )

            if len(X) == 0 or len(y) == 0:
                self.log("No data for training after preprocessing", 'error')
                return None, {}

            # ДИАГНОСТИКА: логируем распределение классов
            unique, counts = np.unique(y, return_counts=True)
            self.log(f"Original class distribution:", 'info')
            total = len(y)
            for cls, cnt in zip(unique, counts):
                percentage = (cnt / total) * 100
                self.log(f"  Class {cls}: {cnt} samples ({percentage:.1f}%)", 'info')

            # ПРОВЕРКА: если слишком много HOLD (класс 0), предлагаем решение
            hold_percentage = (counts[unique == 0][0] / total * 100) if 0 in unique else 0
            if hold_percentage > 80:
                self.log(f"WARNING: Too many HOLD samples ({hold_percentage:.1f}%)! Consider adjusting thresholds.", 'warning')

            # Преобразование меток для классификации (3 класса: -1, 0, 1 -> 0, 1, 2)
            y_categorical = y + 1  # Преобразуем [-1, 0, 1] -> [0, 1, 2]
            y_categorical = to_categorical(y_categorical, num_classes=3)

            # Разделение на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
            )

            # Нормализация
            X_train_norm, scaler = self.preprocessor.normalize_features(
                X_train, fit=True, verbose=verbose
            )
            X_val_norm, _ = self.preprocessor.normalize_features(
                X_val, fit=False, scaler=scaler, verbose=verbose
            )

            # Создание модели LSTM
            model = Sequential([
                Input(shape=(X_train_norm.shape[1], X_train_norm.shape[2])),
                LSTM(config.model.LSTM_UNITS[0], return_sequences=True,
                     dropout=config.model.LSTM_DROPOUT),
                LSTM(config.model.LSTM_UNITS[1], return_sequences=True,
                     dropout=config.model.LSTM_DROPOUT),
                LSTM(config.model.LSTM_UNITS[2], dropout=config.model.LSTM_DROPOUT),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')  # 3 класса
            ])

            # Компиляция модели
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'Precision', 'Recall']
            )

            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.model.LSTM_PATIENCE,
                    restore_best_weights=True,
                    verbose=1 if verbose else 0
                ),
                ModelCheckpoint(
                    filepath=os.path.join(config.MODEL_DIR, f'{symbol}_lstm_best.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1 if verbose else 0
                )
            ]

            # Обучение модели
            history = model.fit(
                X_train_norm, y_train,
                epochs=config.model.LSTM_EPOCHS,
                batch_size=config.model.LSTM_BATCH_SIZE,
                validation_data=(X_val_norm, y_val),
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )

            # Оценка модели
            val_loss, val_accuracy, val_precision, val_recall = model.evaluate(
                X_val_norm, y_val, verbose=0
            )

            # Предсказания для дополнительных метрик
            y_pred_proba = model.predict(X_val_norm, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_val, axis=1)

            # Преобразуем обратно: [0, 1, 2] -> [-1, 0, 1]
            y_pred_original = y_pred - 1
            y_true_original = y_true - 1

            # Расчет метрик
            f1 = f1_score(y_true_original, y_pred_original, average='weighted')
            conf_matrix = confusion_matrix(y_true_original, y_pred_original)

            # Подготовка метрик
            metrics = {
                'val_loss': float(val_loss),
                'val_accuracy': float(val_accuracy),
                'val_precision': float(val_precision),
                'val_recall': float(val_recall),
                'f1_score': float(f1),
                'confusion_matrix': conf_matrix.tolist(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'class_distribution_train': {
                    int(cls-1): int(count) for cls, count in zip(*np.unique(y_train.argmax(axis=1)-1, return_counts=True))
                },
                'class_distribution_val': {
                    int(cls-1): int(count) for cls, count in zip(*np.unique(y_true_original, return_counts=True))
                }
            }

            # Логирование результатов
            if verbose:
                self.log(f"LSTM training completed:")
                self.log(f"  Validation Loss: {val_loss:.4f}")
                self.log(f"  Validation Accuracy: {val_accuracy:.4f}")
                self.log(f"  Validation Precision: {val_precision:.4f}")
                self.log(f"  Validation Recall: {val_recall:.4f}")
                self.log(f"  F1 Score: {f1:.4f}")
                self.log(f"  Confusion Matrix:\n{conf_matrix}")

            return model, metrics

        except Exception as e:
            self.log(f"Error training LSTM classifier: {e}", 'error')
            return None, {}

    def train_xgboost_classifier(self, symbol: str, timeframe: str = '5m',
                                 target_column: str = 'TARGET_CLASS_5',
                                 verbose: bool = True) -> Tuple[Any, Dict]:
        """
        Обучение XGBoost классификатора

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            target_column: Целевая колонка
            verbose: Флаг логирования

        Returns:
            Обученная модель и метрики
        """
        try:
            self.log(f"Training XGBoost classifier for {symbol} {timeframe}...")

            # Получение данных
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=config.TRAIN_START_DATE,
                end_date=config.TRAIN_END_DATE,
                verbose=verbose
            )

            if data.empty:
                self.log(f"No data for {symbol} {timeframe}", 'error')
                return None, {}

            # Расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            if data_with_indicators.empty:
                self.log("Failed to calculate indicators", 'error')
                return None, {}

            # Подготовка данных для XGBoost (2D вместо 3D для LSTM)
            feature_columns = [col for col in data_with_indicators.columns
                               if not col.startswith('TARGET_')
                               and col not in ['open', 'high', 'low', 'close', 'volume']]

            X = data_with_indicators[feature_columns].values
            y = data_with_indicators[target_column].values

            # ДИАГНОСТИКА: логируем распределение классов
            unique, counts = np.unique(y, return_counts=True)
            self.log(f"Class distribution for XGBoost:", 'info')
            total = len(y)
            for cls, cnt in zip(unique, counts):
                percentage = (cnt / total) * 100
                self.log(f"  Class {cls}: {cnt} samples ({percentage:.1f}%)", 'info')

            # Преобразование меток для XGBoost (-1, 0, 1 -> 0, 1, 2)
            y_xgb = y + 1

            # Разделение на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
            )

            # Нормализация
            X_train_norm, scaler = self.preprocessor.normalize_features(
                X_train, fit=True, verbose=verbose
            )
            X_val_norm, _ = self.preprocessor.normalize_features(
                X_val, fit=False, scaler=scaler, verbose=verbose
            )

            # Создание и обучение XGBoost модели
            model = xgb.XGBClassifier(
                n_estimators=config.model.XGB_N_ESTIMATORS,
                max_depth=config.model.XGB_MAX_DEPTH,
                learning_rate=config.model.XGB_LEARNING_RATE,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            # Обучение с early stopping
            model.fit(
                X_train_norm, y_train,
                eval_set=[(X_val_norm, y_val)],
                early_stopping_rounds=config.model.XGB_EARLY_STOPPING_ROUNDS,
                verbose=False
            )

            # Оценка модели
            y_pred = model.predict(X_val_norm)
            y_pred_proba = model.predict_proba(X_val_norm)

            # Преобразуем обратно: [0, 1, 2] -> [-1, 0, 1]
            y_pred_original = y_pred - 1
            y_val_original = y_val - 1

            # Расчет метрик
            accuracy = accuracy_score(y_val_original, y_pred_original)
            precision = precision_score(y_val_original, y_pred_original, average='weighted')
            recall = recall_score(y_val_original, y_pred_original, average='weighted')
            f1 = f1_score(y_val_original, y_pred_original, average='weighted')
            conf_matrix = confusion_matrix(y_val_original, y_pred_original)

            # Подготовка метрик
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': conf_matrix.tolist(),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else config.model.XGB_N_ESTIMATORS,
                'class_distribution_train': {
                    int(cls-1): int(count) for cls, count in zip(*np.unique(y_train-1, return_counts=True))
                },
                'class_distribution_val': {
                    int(cls-1): int(count) for cls, count in zip(*np.unique(y_val_original, return_counts=True))
                }
            }

            # Логирование результатов
            if verbose:
                self.log(f"XGBoost training completed:")
                self.log(f"  Accuracy: {accuracy:.4f}")
                self.log(f"  Precision: {precision:.4f}")
                self.log(f"  Recall: {recall:.4f}")
                self.log(f"  F1 Score: {f1:.4f}")
                self.log(f"  Confusion Matrix:\n{conf_matrix}")
                self.log(f"  Best iteration: {metrics['best_iteration']}")

            return model, metrics

        except Exception as e:
            self.log(f"Error training XGBoost classifier: {e}", 'error')
            return None, {}

    def save_model(self, model, scaler, model_id: str, symbol: str,
                   model_type: str, metrics: Dict, verbose: bool = True) -> bool:
        """
        Сохранение модели и метаданных

        Args:
            model: Обученная модель
            scaler: Скейлер для нормализации
            model_id: ID модели
            symbol: Торговая пара
            model_type: Тип модели
            metrics: Метрики модели
            verbose: Флаг логирования

        Returns:
            True если успешно
        """
        try:
            # Сохранение модели
            model_path = os.path.join(config.MODEL_DIR, f"{model_id}.h5")
            scaler_path = os.path.join(config.MODEL_DIR, f"{model_id}_scaler.pkl")

            if 'lstm' in model_type:
                model.save(model_path)
            elif 'xgb' in model_type:
                with open(model_path.replace('.h5', '.pkl'), 'wb') as f:
                    pickle.dump(model, f)

            # Сохранение скейлера
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            # Подготовка параметров для сохранения
            parameters = {
                'lookback_window': config.model.LOOKBACK_WINDOW,
                'target_column': 'TARGET_CLASS_5',
                'training_period': {
                    'start': config.TRAIN_START_DATE.isoformat(),
                    'end': config.TRAIN_END_DATE.isoformat()
                }
            }

            if 'lstm' in model_type:
                parameters.update({
                    'lstm_units': config.model.LSTM_UNITS,
                    'lstm_dropout': config.model.LSTM_DROPOUT,
                    'epochs': config.model.LSTM_EPOCHS,
                    'batch_size': config.model.LSTM_BATCH_SIZE
                })
            elif 'xgb' in model_type:
                parameters.update({
                    'max_depth': config.model.XGB_MAX_DEPTH,
                    'learning_rate': config.model.XGB_LEARNING_RATE,
                    'n_estimators': config.model.XGB_N_ESTIMATORS
                })

            # Сохранение в базу данных
            success = self.db.save_model_info(
                model_id=model_id,
                symbol=symbol,
                timeframe='5m',
                model_type=model_type,
                parameters=json.dumps(parameters),
                metrics=json.dumps(metrics),
                model_path=model_path,
                verbose=verbose
            )

            if success and verbose:
                self.log(f"Model {model_id} saved successfully")
                self.log(f"  Model file: {model_path}")
                self.log(f"  Scaler file: {scaler_path}")

            return success

        except Exception as e:
            self.log(f"Error saving model: {e}", 'error')
            return False

    def load_model(self, model_id: str, verbose: bool = True) -> Tuple[Any, Any]:
        """
        Загрузка модели и скейлера

        Args:
            model_id: ID модели
            verbose: Флаг логирования

        Returns:
            Модель и скейлер
        """
        try:
            # Проверка кеша
            if model_id in self.model_cache:
                if verbose:
                    self.log(f"Loading model {model_id} from cache")
                return self.model_cache[model_id]

            # Получение информации о модели из БД
            models_df = self.db.get_available_models(model_id=model_id, verbose=verbose)

            if models_df.empty:
                self.log(f"Model {model_id} not found in database", 'error')
                return None, None

            model_info = models_df.iloc[0]
            model_path = model_info['model_path']
            model_type = model_info['model_type']

            # Загрузка скейлера
            scaler_path = model_path.replace('.h5', '_scaler.pkl').replace('.pkl', '_scaler.pkl')

            if not os.path.exists(scaler_path):
                self.log(f"Scaler file not found: {scaler_path}", 'error')
                return None, None

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            # Загрузка модели
            model = None
            if 'lstm' in model_type:
                if os.path.exists(model_path):
                    try:
                        model = load_model(model_path)
                    except:
                        # Попробуем загрузить как .keras
                        keras_path = model_path.replace('.h5', '.keras')
                        if os.path.exists(keras_path):
                            model = load_model(keras_path)
            elif 'xgb' in model_type:
                pkl_path = model_path.replace('.h5', '.pkl')
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        model = pickle.load(f)

            if model is None:
                self.log(f"Failed to load model from {model_path}", 'error')
                return None, None

            # Кеширование
            self.model_cache[model_id] = (model, scaler)

            if verbose:
                self.log(f"Model {model_id} loaded successfully")
                self.log(f"  Type: {model_type}")
                self.log(f"  Path: {model_path}")

            return model, scaler

        except Exception as e:
            self.log(f"Error loading model: {e}", 'error')
            return None, None

    def train_models(self, symbol: str = None, verbose: bool = True):
        """
        Обучение моделей для указанного символа или всех символов

        Args:
            symbol: Торговая пара (если None - все символы)
            verbose: Флаг логирования
        """
        try:
            symbols = [symbol] if symbol else config.trading.SYMBOLS

            for sym in symbols:
                self.log(f"Training models for {sym}...")

                # Обучение LSTM
                lstm_model, lstm_metrics = self.train_lstm_classifier(
                    symbol=sym,
                    verbose=verbose
                )

                if lstm_model is not None:
                    lstm_model_id = self.generate_model_id(sym, 'lstm_class')
                    self.save_model(
                        model=lstm_model,
                        scaler=None,  # Скейлер сохраняется внутри train_lstm_classifier
                        model_id=lstm_model_id,
                        symbol=sym,
                        model_type='lstm_class',
                        metrics=lstm_metrics,
                        verbose=verbose
                    )

                # Обучение XGBoost
                xgb_model, xgb_metrics = self.train_xgboost_classifier(
                    symbol=sym,
                    verbose=verbose
                )

                if xgb_model is not None:
                    xgb_model_id = self.generate_model_id(sym, 'xgb_class')
                    self.save_model(
                        model=xgb_model,
                        scaler=None,  # Скейлер сохраняется внутри train_xgboost_classifier
                        model_id=xgb_model_id,
                        symbol=sym,
                        model_type='xgb_class',
                        metrics=xgb_metrics,
                        verbose=verbose
                    )

                self.log(f"Models trained for {sym}")

        except Exception as e:
            self.log(f"Error training models: {e}", 'error')

    def compare_models(self, symbol: str, verbose: bool = True) -> pd.DataFrame:
        """
        Сравнение моделей для символа

        Args:
            symbol: Торговая пара
            verbose: Флаг логирования

        Returns:
            DataFrame с сравнением моделей
        """
        try:
            models_df = self.db.get_available_models(
                symbol=symbol,
                active_only=True,
                verbose=verbose
            )

            if models_df.empty:
                self.log(f"No models found for {symbol}", 'warning')
                return pd.DataFrame()

            # Парсинг метрик
            comparison_data = []
            for _, row in models_df.iterrows():
                metrics = json.loads(row['metrics'])

                if 'lstm' in row['model_type']:
                    accuracy = metrics.get('val_accuracy', 0)
                else:
                    accuracy = metrics.get('accuracy', 0)

                f1 = metrics.get('f1_score', 0)

                comparison_data.append({
                    'model_id': row['model_id'],
                    'model_type': row['model_type'],
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'created_at': row['created_at'],
                    'training_samples': metrics.get('training_samples', 0),
                    'class_distribution': metrics.get('class_distribution_train', {})
                })

            comparison_df = pd.DataFrame(comparison_data)

            if not comparison_df.empty:
                comparison_df = comparison_df.sort_values('f1_score', ascending=False)

                if verbose:
                    self.log(f"Model comparison for {symbol}:")
                    for _, row in comparison_df.iterrows():
                        self.log(f"  {row['model_id']}: {row['model_type']}, "
                               f"Accuracy: {row['accuracy']:.4f}, F1: {row['f1_score']:.4f}")

            return comparison_df

        except Exception as e:
            self.log(f"Error comparing models: {e}", 'error')
            return pd.DataFrame()

    def batch_convert_models(self, symbol: str = None, verbose: bool = True) -> Dict:
        """
        Конвертация моделей H5 в Keras формат

        Args:
            symbol: Торговая пара
            verbose: Флаг логирования

        Returns:
            Словарь с результатами конвертации
        """
        try:
            import tensorflow as tf

            if symbol:
                models_df = self.db.get_available_models(symbol=symbol, verbose=verbose)
            else:
                models_df = self.db.get_available_models(verbose=verbose)

            results = {'converted': 0, 'failed': 0, 'already_converted': 0}

            for _, row in models_df.iterrows():
                if 'lstm' not in row['model_type']:
                    continue

                model_path = row['model_path']
                keras_path = model_path.replace('.h5', '.keras')

                # Проверка существования
                if os.path.exists(keras_path):
                    if verbose:
                        self.log(f"Keras model already exists: {keras_path}")
                    results['already_converted'] += 1
                    continue

                # Конвертация
                try:
                    if os.path.exists(model_path):
                        model = load_model(model_path)
                        model.save(keras_path)

                        if verbose:
                            self.log(f"Converted: {model_path} -> {keras_path}")
                        results['converted'] += 1
                    else:
                        if verbose:
                            self.log(f"Source model not found: {model_path}", 'warning')
                        results['failed'] += 1
                except Exception as e:
                    if verbose:
                        self.log(f"Conversion failed for {model_path}: {e}", 'error')
                    results['failed'] += 1

            return results

        except Exception as e:
            self.log(f"Error converting models: {e}", 'error')
            return {'converted': 0, 'failed': 0, 'already_converted': 0}