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
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, Any, List
from config import config
from modules.database import Database
from modules.preprocessor import DataPreprocessor
from modules.state_manager import state_manager


# Импорты для машинного обучения
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("Tensorflow/Keras not installed. LSTM features will be unavailable.")

try:
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif, RFE
except ImportError:
    print("XGBoost/scikit-learn not installed. XGBoost features will be unavailable.")


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

    def generate_model_id(self, symbol: str, model_type: str) -> str:
        """
        Генерация уникального ID для модели
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_hash = hashlib.md5(f"{symbol}_{model_type}_{timestamp}".encode()).hexdigest()[:8]
        return f"{symbol}_{model_type}_{timestamp}_{random_hash}"

    def ensure_training_data(self, symbol: str, timeframe: str, training_days: int) -> bool:
        """
        Гарантирует наличие данных для обучения
        Возвращает True если данные доступны или успешно загружены
        """
        try:
            from modules.data_fetcher import DataFetcher

            print(f"   🔍 Проверка данных для обучения {symbol} ({timeframe})...")

            # Получаем даты для обучения из state_manager
            train_start, train_end = state_manager.get_training_dates()
            days_back = max(training_days, (train_end - train_start).days)

            print(f"   📅 Проверка данных за последние {days_back} дней...")

            # Проверяем наличие данных в базе
            existing_data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=train_start,
                end_date=train_end,
                verbose=False
            )

            min_samples_needed = config.model.LOOKBACK_WINDOW * 10  # Минимум 10 последовательностей
            if len(existing_data) >= min_samples_needed:
                print(f"   ✅ Данные уже есть: {len(existing_data)} свечей")
                return True

            print(f"   ⚠️  Недостаточно данных: {len(existing_data)} из {min_samples_needed} нужных")
            print(f"   📥 Загрузка данных...")

            # Загружаем данные
            data_fetcher = DataFetcher()
            data = data_fetcher.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                days_back=days_back
            )

            if data.empty:
                print(f"   ❌ Не удалось загрузить данные для {symbol}")
                return False

            # Сохраняем в базу
            success = self.db.store_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                data=data,
                verbose=True
            )

            if success:
                print(f"   ✅ Данные загружены и сохранены: {len(data)} свечей")
                return True
            else:
                print(f"   ❌ Ошибка сохранения данных")
                return False

        except Exception as e:
            print(f"   ❌ Ошибка проверки данных: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_training_data(self, symbol: str, timeframe: str,
                              use_advanced_features: bool = True,
                              verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Подготовка данных для обучения
        Возвращает X, y и список фичей
        """
        try:
            # Получаем даты для обучения из state_manager
            train_start, train_end = state_manager.get_training_dates()

            print(f"   📅 Получение данных с {train_start.date()} по {train_end.date()}...")

            # Получение данных
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=train_start,
                end_date=train_end,
                verbose=verbose
            )

            if data.empty:
                print("❌ Нет данных для указанного периода")
                return np.array([]), np.array([]), []

            print(f"   ✅ Получено {len(data)} свечей")

            # Расчет индикаторов
            print(f"   📊 Расчет индикаторов...")
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            if data_with_indicators.empty:
                print("❌ Не удалось рассчитать индикаторы")
                return np.array([]), np.array([]), []

            # Добавляем расширенные фичи если нужно
            if use_advanced_features:
                print(f"   🔧 Добавление расширенных фич...")
                data_with_indicators = self.preprocessor.add_advanced_features(
                    data_with_indicators, verbose=verbose
                )

            print(f"   ✅ Всего фичей: {len(data_with_indicators.columns)}")

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Сохраняем список всех фичей для последующего использования
            all_features = list(data_with_indicators.columns)

            # ИСКЛЮЧАЕМ ВРЕМЕННЫЕ ФИЧИ И ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ для обучения
            exclude_patterns = ['TARGET_', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 'WEEK', '_SIN', '_COS']

            feature_columns_for_training = []
            for feature in all_features:
                exclude = False
                for pattern in exclude_patterns:
                    if pattern in feature:
                        exclude = True
                        break
                if not exclude:
                    feature_columns_for_training.append(feature)

            print(f"   📋 Фичи для обучения: {len(feature_columns_for_training)}")
            if verbose and len(feature_columns_for_training) <= 20:
                print(f"   📋 Список фичей: {feature_columns_for_training}")

            # Проверяем какие целевые переменные есть
            target_columns = [col for col in data_with_indicators.columns if col.startswith('TARGET_')]

            if not target_columns:
                print("❌ Не найдены целевые переменные")
                return np.array([]), np.array([]), []

            # Используем TARGET_CLASS_5 по умолчанию
            target_column = 'TARGET_CLASS_5'
            if target_column not in data_with_indicators.columns:
                target_column = target_columns[0]  # Используем первую доступную

            print(f"   🎯 Используется целевая переменная: {target_column}")

            # Проверяем распределение классов
            if target_column in data_with_indicators.columns:
                class_dist = data_with_indicators[target_column].value_counts()
                print(f"   📈 Распределение классов:")
                for cls, count in class_dist.items():
                    percentage = (count / len(data_with_indicators)) * 100
                    print(f"      Класс {cls}: {count} ({percentage:.1f}%)")

                # Проверяем баланс классов
                min_class = class_dist.min()
                max_class = class_dist.max()
                if min_class > 0:
                    imbalance_ratio = max_class / min_class
                    if imbalance_ratio > 3:
                        print(f"   ⚠️  Сильный дисбаланс классов: соотношение {imbalance_ratio:.1f}:1")

            # Удаляем строки с NaN в целевой переменной
            initial_len = len(data_with_indicators)
            data_with_indicators = data_with_indicators.dropna(subset=[target_column])
            if len(data_with_indicators) < initial_len:
                print(f"   🧹 Удалено {initial_len - len(data_with_indicators)} строк с NaN в целевой переменной")

            # Подготовка данных для обучения
            print(f"   🔄 Подготовка последовательностей...")
            X, y, feature_names = self.prepare_sequences_with_features(
                df=data_with_indicators,
                target_column=target_column,
                lookback_window=config.model.LOOKBACK_WINDOW,
                use_advanced_features=use_advanced_features,
                feature_columns=feature_columns_for_training,  # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: передаем фичи
                verbose=verbose
            )

            if len(X) == 0 or len(y) == 0:
                print("❌ Не удалось создать последовательности для обучения")
                return np.array([]), np.array([]), []

            print(f"   ✅ Создано {len(X)} последовательностей")
            print(f"   📐 Размерность X: {X.shape}")
            print(f"   📐 Размерность y: {y.shape}")
            print(f"   🔤 Количество фичей: {len(feature_names)}")

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Сохраняем фичи в preprocessor для последующего использования
            self.preprocessor.last_training_features = feature_names.copy()
            print(f"   💾 Сохранено {len(feature_names)} фичей в preprocessor")

            return X, y, feature_names

        except Exception as e:
            print(f"❌ Ошибка подготовки данных: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), []

    def train_lstm_classifier(self, symbol: str, timeframe: str = '5m',
                            use_advanced_features: bool = True,
                            verbose: bool = True) -> Dict[str, Any]:
        """
        Обучение LSTM классификатора
        """
        try:
            print(f"\n🚀 НАЧИНАЕМ ОБУЧЕНИЕ LSTM МОДЕЛИ")
            print(f"   Криптовалюта: {symbol}")
            print(f"   Таймфрейм: {timeframe}")
            print(f"   Расширенные фичи: {'Да' if use_advanced_features else 'Нет'}")
            print("=" * 70)

            # Гарантируем наличие данных для обучения
            training_days = state_manager.get_training_period()
            if not self.ensure_training_data(symbol, timeframe, training_days):
                print("❌ Не удалось обеспечить данные для обучения LSTM")
                return {'model': None, 'metrics': {}, 'feature_importance': None}

            # Подготовка данных
            X, y, feature_names = self.prepare_training_data(
                symbol, timeframe, use_advanced_features, verbose
            )

            if len(X) == 0:
                print("❌ Нет данных для обучения")
                return {'model': None, 'metrics': {}, 'feature_importance': None}

            print(f"✅ Данные подготовлены: {len(X)} последовательностей")

            # Преобразование меток для классификации
            y_categorical = y + 1  # Преобразуем [-1, 0, 1] -> [0, 1, 2]
            y_categorical = to_categorical(y_categorical, num_classes=3)

            # Разделение на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
            )

            print(f"📈 Разделение данных:")
            print(f"   Обучающая выборка: {len(X_train)} последовательностей")
            print(f"   Валидационная выборка: {len(X_val)} последовательностей")

            # Нормализация
            print(f"🔢 Нормализация данных...")
            X_train_norm, scaler = self.preprocessor.normalize_features(
                X_train, fit=True, verbose=verbose
            )
            X_val_norm, _ = self.preprocessor.normalize_features(
                X_val, fit=False, scaler=scaler, verbose=verbose
            )

            # Создание модели LSTM
            print(f"🏗️  Создание архитектуры LSTM...")
            input_shape = (X_train_norm.shape[1], X_train_norm.shape[2])

            model = Sequential([
                Input(shape=input_shape),
                LSTM(config.model.LSTM_UNITS[0], return_sequences=True,
                     dropout=config.model.LSTM_DROPOUT, recurrent_dropout=config.model.LSTM_DROPOUT),
                BatchNormalization(),
                LSTM(config.model.LSTM_UNITS[1], return_sequences=True,
                     dropout=config.model.LSTM_DROPOUT, recurrent_dropout=config.model.LSTM_DROPOUT),
                BatchNormalization(),
                LSTM(config.model.LSTM_UNITS[2], dropout=config.model.LSTM_DROPOUT),
                BatchNormalization(),
                Dense(128, activation='relu'),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])

            # Компиляция модели
            print(f"⚙️  Компиляция модели...")
            model.compile(
                optimizer=Adam(learning_rate=config.model.LSTM_LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc')]
            )

            # Показываем архитектуру модели
            print(f"\n🏛️  Архитектура модели:")
            model.summary(print_fn=lambda x: print(f"   {x}"))

            # Callbacks
            print(f"\n⏱️  Начинаем обучение...")
            print(f"   Эпох: {config.model.LSTM_EPOCHS}")
            print(f"   Размер батча: {config.model.LSTM_BATCH_SIZE}")
            print(f"   Learning rate: {config.model.LSTM_LEARNING_RATE}")
            print(f"   Early stopping patience: {config.model.LSTM_PATIENCE}")

            # Создаем каталог для логирования TensorBoard
            log_dir = os.path.join(config.LOG_DIR, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
            os.makedirs(log_dir, exist_ok=True)

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=config.model.LSTM_PATIENCE,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=os.path.join(config.MODEL_DIR, f"lstm_best_{symbol}_{timeframe}.h5"),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=0
                )
            ]

            # Обучение модели
            print(f"\n🎓 Обучение началось:")
            print("=" * 70)

            history = model.fit(
                X_train_norm, y_train,
                epochs=config.model.LSTM_EPOCHS,
                batch_size=config.model.LSTM_BATCH_SIZE,
                validation_data=(X_val_norm, y_val),
                callbacks=callbacks,
                verbose=1
            )

            print("=" * 70)
            print(f"✅ Обучение завершено!")
            print(f"   Количество эпох: {len(history.history['loss'])}")

            # Оценка модели
            print(f"\n📊 Оценка модели на валидационной выборке...")
            eval_results = model.evaluate(X_val_norm, y_val, verbose=0)

            # Расчет дополнительных метрик
            y_pred_proba = model.predict(X_val_norm, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_val, axis=1)
            y_pred_original = y_pred - 1
            y_true_original = y_true - 1

            # Подробные метрики
            accuracy = accuracy_score(y_true_original, y_pred_original)
            precision = precision_score(y_true_original, y_pred_original, average='weighted')
            recall = recall_score(y_true_original, y_pred_original, average='weighted')
            f1 = f1_score(y_true_original, y_pred_original, average='weighted')
            conf_matrix = confusion_matrix(y_true_original, y_pred_original)

            # Classification report
            class_report = classification_report(y_true_original, y_pred_original,
                                                target_names=['DOWN', 'HOLD', 'UP'],
                                                output_dict=True)

            # Подготовка метрик
            metrics = {
                'val_loss': float(eval_results[0]),
                'val_accuracy': float(eval_results[1]),
                'val_precision': float(eval_results[2]),
                'val_recall': float(eval_results[3]),
                'val_auc': float(eval_results[4]),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'feature_count': X_train_norm.shape[2],
                'sequence_length': X_train_norm.shape[1],
                'training_period': {
                    'start': state_manager.get_training_dates()[0].isoformat(),
                    'end': state_manager.get_training_dates()[1].isoformat()
                },
                'training_history': {
                    'loss': [float(x) for x in history.history.get('loss', [])],
                    'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                    'accuracy': [float(x) for x in history.history.get('accuracy', [])],
                    'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
                }
            }

            print(f"\n🎯 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ LSTM:")
            print(f"   Validation Loss: {metrics['val_loss']:.4f}")
            print(f"   Validation Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
            print(f"   AUC: {metrics['val_auc']:.4f}")

            # Показываем confusion matrix
            print(f"\n📊 CONFUSION MATRIX:")
            print("   DOWN  HOLD  UP")
            for i, row in enumerate(conf_matrix):
                class_name = ['DOWN', 'HOLD', 'UP'][i]
                print(f"   {class_name} {row}")

            # Feature importance для LSTM (средние веса)
            feature_importance = None
            try:
                # Простой способ оценки важности фичей для LSTM
                layer_weights = []
                for layer in model.layers:
                    if isinstance(layer, LSTM):
                        layer_weights.append(layer.get_weights()[0])  # Веса ячеек

                if layer_weights:
                    avg_weights = np.mean([np.mean(np.abs(w), axis=1) for w in layer_weights], axis=0)
                    if len(avg_weights) == len(feature_names):
                        feature_importance = dict(zip(feature_names, avg_weights.tolist()))
                        print(f"\n📈 Рассчитана важность фичей для LSTM")
            except:
                pass

            # Сохраняем скейлер в атрибуте модели
            model.scaler = scaler
            model.feature_names = feature_names

            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'scaler': scaler
            }

        except Exception as e:
            print(f"\n❌ ОШИБКА ОБУЧЕНИЯ LSTM: {e}")
            import traceback
            traceback.print_exc()
            return {'model': None, 'metrics': {}, 'feature_importance': None}

    def train_xgboost_classifier(self, symbol: str, timeframe: str = '5m',
                                 use_advanced_features: bool = True,
                                 verbose: bool = True) -> Dict[str, Any]:
        """
        Обучение XGBoost классификатора
        """
        try:
            print(f"\n🚀 НАЧИНАЕМ ОБУЧЕНИЕ XGBOOST МОДЕЛИ")
            print(f"   Криптовалюта: {symbol}")
            print(f"   Таймфрейм: {timeframe}")
            print(f"   Расширенные фичи: {'Да' if use_advanced_features else 'Нет'}")
            print("=" * 70)

            # Гарантируем наличие данных для обучения
            training_days = state_manager.get_training_period()
            if not self.ensure_training_data(symbol, timeframe, training_days):
                print("❌ Не удалось обеспечить данные для обучения XGBoost")
                return {'model': None, 'metrics': {}, 'feature_importance': None}

            # Подготовка данных
            X, y, feature_names = self.prepare_training_data(
                symbol, timeframe, use_advanced_features, verbose
            )

            if len(X) == 0:
                print("❌ Нет данных для обучения")
                return {'model': None, 'metrics': {}, 'feature_importance': None}

            print(f"✅ Данные подготовлены: {len(X)} последовательностей")
            print(f"   Преобразование 3D -> 2D для XGBoost...")

            # Преобразование 3D -> 2D для XGBoost
            X_2d = X.reshape(X.shape[0], -1)
            y_xgb = y + 1  # Преобразуем [-1, 0, 1] -> [0, 1, 2]

            # Проверяем и создаем новые имена фичей для 2D представления
            expanded_feature_names = []
            for i in range(X.shape[1]):  # Для каждого временного шага
                for feature_name in feature_names:
                    expanded_feature_names.append(f"{feature_name}_t-{X.shape[1] - i - 1}")

            print(f"   📐 Размерность X_2d: {X_2d.shape}")
            print(f"   🔤 Количество фичей в 2D: {len(expanded_feature_names)}")

            # Разделение на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_2d, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
            )

            print(f"📈 Разделение данных:")
            print(f"   Обучающая выборка: {len(X_train)} образцов")
            print(f"   Валидационная выборка: {len(X_val)} образцов")

            # Нормализация
            print(f"🔢 Нормализация данных...")
            X_train_norm, scaler = self.preprocessor.normalize_features(
                X_train, fit=True, verbose=verbose
            )
            X_val_norm, _ = self.preprocessor.normalize_features(
                X_val, fit=False, scaler=scaler, verbose=verbose
            )

            # ВАЖНОЕ ИСПРАВЛЕНИЕ: Сохраняем информацию о фичах в скейлере
            if hasattr(scaler, 'feature_names_in_'):
                scaler.feature_names_in_ = expanded_feature_names
            elif hasattr(scaler, 'feature_names'):
                scaler.feature_names = expanded_feature_names

            # Создание и обучение XGBoost модели
            print(f"\n🌲 Создание XGBoost модели...")
            print(f"   n_estimators: {config.model.XGB_N_ESTIMATORS}")
            print(f"   max_depth: {config.model.XGB_MAX_DEPTH}")
            print(f"   learning_rate: {config.model.XGB_LEARNING_RATE}")
            print(f"   subsample: {config.model.XGB_SUBSAMPLE}")
            print(f"   colsample_bytree: {config.model.XGB_COLSAMPLE_BYTREE}")
            print(f"   early_stopping_rounds: {config.model.XGB_EARLY_STOPPING_ROUNDS}")

            model = xgb.XGBClassifier(
                n_estimators=config.model.XGB_N_ESTIMATORS,
                max_depth=config.model.XGB_MAX_DEPTH,
                learning_rate=config.model.XGB_LEARNING_RATE,
                subsample=config.model.XGB_SUBSAMPLE,
                colsample_bytree=config.model.XGB_COLSAMPLE_BYTREE,
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                enable_categorical=False,
                tree_method='hist',
                eval_metric=['merror', 'mlogloss'],
                early_stopping_rounds=config.model.XGB_EARLY_STOPPING_ROUNDS
            )

            # Обучение с early stopping
            print(f"\n🎓 Начинаем обучение XGBoost...")

            eval_set = [(X_train_norm, y_train), (X_val_norm, y_val)]
            eval_metric = ["merror", "mlogloss"]

            model.fit(
                X_train_norm, y_train,
                eval_set=eval_set,
                verbose=10
            )

            # Оценка модели
            print(f"\n📊 Оценка модели...")
            y_pred = model.predict(X_val_norm)
            y_pred_proba = model.predict_proba(X_val_norm)
            y_pred_original = y_pred - 1
            y_val_original = y_val - 1

            accuracy = accuracy_score(y_val_original, y_pred_original)
            precision = precision_score(y_val_original, y_pred_original, average='weighted')
            recall = recall_score(y_val_original, y_pred_original, average='weighted')
            f1 = f1_score(y_val_original, y_pred_original, average='weighted')
            conf_matrix = confusion_matrix(y_val_original, y_pred_original)

            # Classification report
            class_report = classification_report(y_val_original, y_pred_original,
                                                 target_names=['DOWN', 'HOLD', 'UP'],
                                                 output_dict=True)

            # Feature importance
            feature_importance_dict = {}
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                if len(importance_values) == len(expanded_feature_names):
                    for i, feature_name in enumerate(expanded_feature_names):
                        feature_importance_dict[feature_name] = float(importance_values[i])

                # Группируем по основным фичам (без временных лагов)
                aggregated_importance = {}
                for feature_name, importance in feature_importance_dict.items():
                    base_feature = feature_name.split('_t-')[0]  # Убираем временной лаг
                    aggregated_importance[base_feature] = aggregated_importance.get(base_feature, 0) + importance

                # Сортируем по важности
                sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
                feature_importance_dict = dict(sorted_features[:20])  # Топ-20 фичей

            # Подготовка метрик
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'confusion_matrix': conf_matrix.tolist(),
                'classification_report': class_report,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'best_iteration': int(model.best_iteration) if hasattr(model,
                                                                       'best_iteration') else config.model.XGB_N_ESTIMATORS,
                'feature_count': X_train_norm.shape[1],
                'training_period': {
                    'start': state_manager.get_training_dates()[0].isoformat(),
                    'end': state_manager.get_training_dates()[1].isoformat()
                },
                'eval_results': {
                    'train_merror': model.evals_result()['validation_0']['merror'][-1] if hasattr(model,
                                                                                                  'evals_result') else 0,
                    'train_mlogloss': model.evals_result()['validation_0']['mlogloss'][-1] if hasattr(model,
                                                                                                      'evals_result') else 0,
                    'val_merror': model.evals_result()['validation_1']['merror'][-1] if hasattr(model,
                                                                                                'evals_result') else 0,
                    'val_mlogloss': model.evals_result()['validation_1']['mlogloss'][-1] if hasattr(model,
                                                                                                    'evals_result') else 0
                }
            }

            print(f"\n🎯 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ XGBOOST:")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1 Score: {f1:.4f}")
            print(f"   Best iteration: {metrics['best_iteration']}")

            if 'eval_results' in metrics:
                print(f"   Train merror: {metrics['eval_results']['train_merror']:.4f}")
                print(f"   Validation merror: {metrics['eval_results']['val_merror']:.4f}")

            # Показываем confusion matrix
            print(f"\n📊 CONFUSION MATRIX:")
            print("   DOWN  HOLD  UP")
            for i, row in enumerate(conf_matrix):
                class_name = ['DOWN', 'HOLD', 'UP'][i]
                print(f"   {class_name} {row}")

            # Показываем топ фичей по важности
            if feature_importance_dict:
                print(f"\n🏆 ТОП-10 ВАЖНЕЙШИХ ФИЧ:")
                top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"   {i:2d}. {feature:<30} {importance:.4f}")

            # ВАЖНОЕ ИСПРАВЛЕНИЕ: Сохраняем ПРАВИЛЬНЫЕ фичи в атрибутах модели
            model.scaler = scaler
            model.feature_names = expanded_feature_names  # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: используем расширенные фичи (1500), а не базовые
            model.expanded_feature_names = expanded_feature_names  # Расширенные фичи (1500)
            model.base_feature_names = feature_names  # Базовые фичи (25) - для обратной совместимости
            model._features = expanded_feature_names  # Для обратной совместимости - используем расширенные

            # ДОБАВЛЯЕМ КРИТИЧЕСКУЮ ИНФОРМАЦИЮ:
            model._lookback_window = config.model.LOOKBACK_WINDOW
            model._base_features_count = len(feature_names)
            model._expanded_features_count = len(expanded_feature_names)
            model._model_type = 'xgb_class'  # Явно указываем тип модели

            print(f"\n💾 ИНФОРМАЦИЯ О ФИЧАХ ДЛЯ ПРЕДСКАЗАНИЯ:")
            print(f"   Базовые фичи: {len(feature_names)}")
            print(f"   Расширенные фичи (2D): {len(expanded_feature_names)}")
            print(f"   Lookback window: {config.model.LOOKBACK_WINDOW}")
            print(f"   Формула: {len(feature_names)} × {config.model.LOOKBACK_WINDOW} = {len(expanded_feature_names)}")
            print(f"   📋 Первые 10 расширенных фичей: {expanded_feature_names[:10]}")

            # Проверяем, что formula работает
            expected_expanded = len(feature_names) * config.model.LOOKBACK_WINDOW
            if expected_expanded != len(expanded_feature_names):
                print(f"  ⚠️  ВНИМАНИЕ: Формула не совпадает!")
                print(f"     Ожидалось: {expected_expanded}, получилось: {len(expanded_feature_names)}")

            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance_dict,
                'feature_names': expanded_feature_names,  # Теперь возвращаем расширенные фичи
                'base_feature_names': feature_names,
                'expanded_feature_names': expanded_feature_names,
                'scaler': scaler
            }

        except Exception as e:
            print(f"\n❌ ОШИБКА ОБУЧЕНИЯ XGBOOST: {e}")
            import traceback
            traceback.print_exc()
            return {'model': None, 'metrics': {}, 'feature_importance': None}

    def compare_models(self, comparison_results: Dict[str, Dict]):
        """
        Сравнение результатов разных моделей
        """
        print(f"\n🔬 СРАВНЕНИЕ МОДЕЛЕЙ:")
        print("=" * 80)

        if not comparison_results:
            print("   ❌ Нет данных для сравнения")
            return

        print(f"{'Модель':<15} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10} {'Samples':<10}")
        print("-" * 80)

        best_model = None
        best_score = -1

        for model_name, results in comparison_results.items():
            metrics = results.get('metrics', {})
            accuracy = metrics.get('accuracy', metrics.get('val_accuracy', 0))
            f1 = metrics.get('f1_score', 0)
            precision = metrics.get('precision', metrics.get('val_precision', 0))
            recall = metrics.get('recall', metrics.get('val_recall', 0))
            samples = metrics.get('training_samples', 0) + metrics.get('validation_samples', 0)

            print(f"{model_name:<15} {accuracy:.4f}      {f1:.4f}      {precision:.4f}      {recall:.4f}      {samples:<10}")

            # Определяем лучшую модель по F1 score
            if f1 > best_score:
                best_score = f1
                best_model = model_name

        print("-" * 80)
        print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model} (F1 Score: {best_score:.4f})")

        # Дополнительные рекомендации
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        if best_score > 0.6:
            print(f"   ✅ Отличные результаты! Модель {best_model} показывает высокую точность")
        elif best_score > 0.5:
            print(f"   👍 Хорошие результаты, можно использовать {best_model} для торговли")
        else:
            print(f"   ⚠️  Результаты ниже среднего, рекомендуется:")
            print(f"      1. Увеличить период обучения")
            print(f"      2. Добавить больше индикаторов")
            print(f"      3. Попробовать другие параметры моделей")

    def save_model(self, model: Any, model_id: str, symbol: str,
                   model_type: str, metrics: Dict,
                   feature_importance: Dict = None,
                   verbose: bool = True) -> bool:
        """
        Сохранение модели и метаданных
        """
        try:
            # Пути для сохранения
            model_filename = f"{model_id}.h5" if 'lstm' in model_type else f"{model_id}.pkl"
            model_path = os.path.join(config.MODEL_DIR, model_filename)

            scaler_filename = f"{model_id}_scaler.pkl"
            scaler_path = os.path.join(config.MODEL_DIR, scaler_filename)

            # Сохранение модели
            if 'lstm' in model_type:
                model.save(model_path)
            elif 'xgb' in model_type:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            else:
                self.log(f"Unknown model type: {model_type}", 'error')
                return False

            # Сохранение скейлера
            if hasattr(model, 'scaler') and model.scaler is not None:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(model.scaler, f)
            else:
                # Если скейлера нет, создаем пустой файл
                with open(scaler_path, 'wb') as f:
                    pickle.dump(None, f)

            # Подготовка параметров для сохранения
            parameters = {
                'lookback_window': config.model.LOOKBACK_WINDOW,
                'prediction_horizon': config.model.PREDICTION_HORIZON,
                'timeframe': state_manager.get_selected_timeframe(),
                'training_period': {
                    'days': state_manager.get_training_period(),
                    'start': state_manager.get_training_dates()[0].isoformat(),
                    'end': state_manager.get_training_dates()[1].isoformat()
                },
                'feature_count': metrics.get('feature_count', 0),
                'use_advanced_features': True  # По умолчанию включено
            }

            if 'lstm' in model_type:
                parameters.update({
                    'lstm_units': config.model.LSTM_UNITS,
                    'lstm_dropout': config.model.LSTM_DROPOUT,
                    'lstm_learning_rate': config.model.LSTM_LEARNING_RATE,
                    'epochs': config.model.LSTM_EPOCHS,
                    'batch_size': config.model.LSTM_BATCH_SIZE,
                    'patience': config.model.LSTM_PATIENCE
                })
            elif 'xgb' in model_type:
                parameters.update({
                    'max_depth': config.model.XGB_MAX_DEPTH,
                    'learning_rate': config.model.XGB_LEARNING_RATE,
                    'n_estimators': config.model.XGB_N_ESTIMATORS,
                    'subsample': config.model.XGB_SUBSAMPLE,
                    'colsample_bytree': config.model.XGB_COLSAMPLE_BYTREE,
                    'early_stopping_rounds': config.model.XGB_EARLY_STOPPING_ROUNDS
                })

            # Добавляем feature importance в метрики
            if feature_importance:
                metrics['feature_importance'] = feature_importance

            # ВАЖНОЕ ИСПРАВЛЕНИЕ: Сохраняем правильные фичи в метриках
            # Для XGBoost сохраняем расширенные фичи
            if 'xgb' in model_type:
                if hasattr(model, 'expanded_feature_names'):
                    metrics['feature_names'] = model.expanded_feature_names
                    metrics['base_feature_names'] = model.base_feature_names if hasattr(model,
                                                                                        'base_feature_names') else []
                    if verbose:
                        print(f"  💾 Для XGBoost сохранены расширенные фичи: {len(model.expanded_feature_names)} фичей")
                        print(
                            f"  💾 Базовые фичи: {len(model.base_feature_names) if hasattr(model, 'base_feature_names') else 0} фичей")
                elif hasattr(model, 'feature_names'):
                    # Проверяем, расширенные ли это фичи
                    if len(model.feature_names) > 100:  # Много фичей = расширенные
                        metrics['feature_names'] = model.feature_names
                        metrics['base_feature_names'] = model.base_feature_names if hasattr(model,
                                                                                            'base_feature_names') else []
                    else:
                        # Базовые фичи - сохраняем как есть
                        metrics['feature_names'] = model.feature_names
                        if verbose:
                            print(f"  ⚠️  Для XGBoost сохранены базовые фичи: {len(model.feature_names)} фичей")
            else:
                # Для LSTM сохраняем базовые фичи
                if hasattr(model, 'feature_names'):
                    metrics['feature_names'] = model.feature_names

            # Сохранение в базу данных
            success = self.db.save_model_info(
                model_id=model_id,
                symbol=symbol,
                timeframe=state_manager.get_selected_timeframe(),
                model_type=model_type,
                parameters=json.dumps(parameters),
                metrics=json.dumps(metrics),
                model_path=model_path,
                feature_importance=json.dumps(feature_importance) if feature_importance else None,
                verbose=verbose
            )

            if success:
                if verbose:
                    print(f"✅ Модель {model_id} сохранена успешно")
                    print(f"   Файл модели: {model_path}")
                    print(f"   Файл скейлера: {scaler_path}")
                    print(f"   Метрики: accuracy={metrics.get('accuracy', metrics.get('val_accuracy', 0)):.4f}")
                    if 'feature_names' in metrics:
                        print(f"   Сохранено фичей: {len(metrics['feature_names'])}")
                        if 'xgb' in model_type and len(metrics['feature_names']) > 100:
                            print(
                                f"   ⚠️  ВНИМАНИЕ: XGBoost модель сохранила {len(metrics['feature_names'])} расширенных фичей")
                            print(f"   ⚠️  При предсказании нужно использовать эти же фичи!")
                return True
            else:
                if verbose:
                    print(f"❌ Ошибка сохранения информации о модели в базу")
                return False

        except Exception as e:
            print(f"❌ Ошибка сохранения модели: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model(self, model_id: str, verbose: bool = True) -> Tuple[Any, Any]:
        """
        Загрузка модели и скейлера
        """
        try:
            # Проверка кеша
            if model_id in self.model_cache:
                if verbose:
                    self.log(f"Loading model {model_id} from cache")
                return self.model_cache[model_id]

            # Получение информации о модели из БД
            models_df = self.db.get_available_models(active_only=False, verbose=verbose)

            if models_df.empty:
                self.log(f"Model {model_id} not found in database", 'error')
                return None, None

            model_info = models_df[models_df['model_id'] == model_id]
            if model_info.empty:
                self.log(f"Model {model_id} not found in database", 'error')
                return None, None

            model_info = model_info.iloc[0]
            model_path = model_info['model_path']
            model_type = model_info['model_type']

            # Загрузка скейлера
            scaler_path = model_path.replace('.h5', '_scaler.pkl').replace('.pkl', '_scaler.pkl')

            scaler = None
            if os.path.exists(scaler_path):
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
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

            if model is None:
                self.log(f"Failed to load model from {model_path}", 'error')
                return None, None

            # ВАЖНОЕ ИСПРАВЛЕНИЕ: Восстанавливаем фичи из метаданных
            if 'metrics' in model_info and model_info['metrics']:
                try:
                    metrics = json.loads(model_info['metrics'])

                    # Для всех моделей
                    if 'feature_names' in metrics:
                        model.feature_names = metrics['feature_names']
                        if verbose:
                            print(f"  💾 Загружены фичи из метаданных: {len(model.feature_names)} фичей")

                    # Для XGBoost дополнительная информация
                    if 'xgb' in model_type:
                        if 'base_feature_names' in metrics:
                            model.base_feature_names = metrics['base_feature_names']
                            if verbose:
                                print(f"  💾 Базовые фичи: {len(model.base_feature_names)} фичей")

                        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем и исправляем фичи для XGBoost
                        if hasattr(model, 'feature_names'):
                            current_feature_count = len(model.feature_names)

                            # Определяем ожидаемое количество фичей
                            lookback_window = config.model.LOOKBACK_WINDOW
                            if hasattr(model, 'base_feature_names'):
                                base_count = len(model.base_feature_names)
                                expected_expanded = base_count * lookback_window
                            else:
                                # Пытаемся вычислить
                                base_count = len([f for f in model.feature_names if '_t-' not in str(f)])
                                expected_expanded = base_count * lookback_window

                            if verbose:
                                print(f"  🔍 Диагностика фичей XGBoost:")
                                print(f"     Текущие фичи: {current_feature_count}")
                                print(f"     Ожидается расширенных: {expected_expanded}")
                                print(f"     Lookback window: {lookback_window}")

                            # Если фичей мало (базовые), создаем расширенные
                            if current_feature_count < 100 and current_feature_count * lookback_window == expected_expanded:
                                if verbose:
                                    print(f"  🔧 Обнаружены базовые фичи, создаем расширенные...")

                                # Получаем базовые фичи
                                base_features = []
                                if hasattr(model, 'base_feature_names'):
                                    base_features = model.base_feature_names
                                elif hasattr(model, 'feature_names'):
                                    base_features = model.feature_names

                                if base_features:
                                    # Создаем расширенные фичи
                                    expanded_features = []
                                    for i in range(lookback_window):
                                        for feature in base_features:
                                            expanded_features.append(f"{feature}_t-{lookback_window - i - 1}")

                                    # Сохраняем оба набора
                                    model.expanded_feature_names = expanded_features
                                    model.feature_names = expanded_features  # Основные фичи = расширенные
                                    model._features = base_features  # Для обратной совместимости

                                    if verbose:
                                        print(f"  ✅ Создано {len(expanded_features)} расширенных фичей")
                                        print(
                                            f"  📋 Формула: {len(base_features)} × {lookback_window} = {len(expanded_features)}")
                            else:
                                # Проверяем, расширенные ли это уже фичи
                                if current_feature_count > 100:
                                    if verbose:
                                        print(f"  ✅ Похоже на расширенные фичи ({current_feature_count} фичей)")
                                    model.expanded_feature_names = model.feature_names
                                else:
                                    if verbose:
                                        print(f"  ⚠️  Непонятный формат фичей: {current_feature_count} фичей")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️  Ошибка загрузки метаданных фичей: {e}")

            # Дополнительная диагностика для XGBoost
            if 'xgb' in model_type and verbose:
                print(f"  🔧 Финальная проверка XGBoost модели:")
                if hasattr(model, 'feature_names'):
                    print(f"     feature_names: {len(model.feature_names)} фичей")
                if hasattr(model, 'expanded_feature_names'):
                    print(f"     expanded_feature_names: {len(model.expanded_feature_names)} фичей")
                if hasattr(model, 'base_feature_names'):
                    print(f"     base_feature_names: {len(model.base_feature_names)} фичей")

                # Проверяем, соответствует ли количество фичей ожиданиям модели
                if hasattr(model, 'feature_names') and hasattr(scaler, 'n_features_in_'):
                    model_features = len(model.feature_names)
                    scaler_features = scaler.n_features_in_
                    if model_features != scaler_features:
                        print(f"  ⚠️  ВНИМАНИЕ: Модель и скейлер имеют разное количество фичей!")
                        print(f"     Модель: {model_features} фичей")
                        print(f"     Скейлер: {scaler_features} фичей")

            # Кеширование
            self.model_cache[model_id] = (model, scaler)

            if verbose:
                print(f"✅ Модель {model_id} загружена успешно")
                print(f"   Тип: {model_type}")
                print(f"   Путь: {model_path}")
                if hasattr(model, 'feature_names'):
                    print(f"   Фичей в модели: {len(model.feature_names)}")
                    if 'xgb' in model_type and len(model.feature_names) < 100:
                        print(f"   ⚠️  ВНИМАНИЕ: XGBoost модель имеет только {len(model.feature_names)} фичей")
                        print(
                            f"   ⚠️  Модель ожидает {len(model.feature_names) * config.model.LOOKBACK_WINDOW} фичей при предсказании!")

            return model, scaler

        except Exception as e:
            self.log(f"Error loading model: {e}", 'error')
            import traceback
            traceback.print_exc()
            return None, None

    def prepare_sequences_with_features(self, df: pd.DataFrame, target_column: str,
                                        lookback_window: int = 60,
                                        use_advanced_features: bool = True,
                                        feature_columns: List[str] = None,  # Новый параметр
                                        verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Создание последовательностей для обучения с возвратом имен фичей
        """
        try:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Если переданы фичи, используем их
            if feature_columns is not None:
                feature_columns_to_use = feature_columns
            else:
                # Старая логика (для обратной совместимости)
                base_features = ['close', 'volume', 'returns']
                tech_indicators = [col for col in df.columns
                                   if any(indicator in col for indicator in
                                          ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'OBV', 'ADX'])]
                advanced_features = []
                if use_advanced_features:
                    advanced_features = [col for col in df.columns
                                         if col.startswith('FEATURE_') or
                                         any(x in col for x in
                                             ['volatility', 'spread', 'skew', 'kurtosis', 'volume_profile'])]

                feature_columns_to_use = base_features + tech_indicators + advanced_features

                # ИСКЛЮЧАЕМ ВРЕМЕННЫЕ ФИЧИ ДЛЯ СОВМЕСТИМОСТИ
                temporal_features = ['HOUR', 'DAY_OF_WEEK', 'MONTH', 'HOUR_OF_DAY', 'DAY', 'WEEK', '_SIN', '_COS']
                feature_columns_to_use = [col for col in feature_columns_to_use
                                          if not any(temp in col for temp in temporal_features)]

            # Сохраняем список фичей для последующего использования
            self.last_feature_columns = feature_columns_to_use.copy()

            # Оставляем только существующие колонки
            feature_columns_to_use = [col for col in feature_columns_to_use if col in df.columns]

            missing_features = [col for col in self.last_feature_columns if col not in df.columns]
            if missing_features and verbose:
                print(f"   ⚠️  Отсутствуют фичи: {missing_features[:5]}...")

            if len(feature_columns_to_use) == 0:
                print("   ❌ Нет фичей для обучения")
                return np.array([]), np.array([]), []

            print(f"   🔍 Используется {len(feature_columns_to_use)} фичей")
            if verbose:
                print(f"   📋 Фичи: {', '.join(feature_columns_to_use[:10])}" +
                      ("..." if len(feature_columns_to_use) > 10 else ""))

            # Создаем массивы
            X = []
            y = []

            data_features = df[feature_columns_to_use].values
            data_target = df[target_column].values

            for i in range(lookback_window, len(df)):
                X.append(data_features[i - lookback_window:i])
                y.append(data_target[i])

            if len(X) == 0:
                print("   ❌ Не удалось создать ни одной последовательности")
                return np.array([]), np.array([]), []

            X_array = np.array(X)
            y_array = np.array(y)

            print(f"   📐 Размерность X: {X_array.shape}")
            print(f"   📐 Размерность y: {y_array.shape}")

            return X_array, y_array, feature_columns_to_use

        except Exception as e:
            print(f"   ❌ Ошибка создания последовательностей: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), []