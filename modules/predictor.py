"""
Модуль для генерации торговых сигналов
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from config import config
from modules.database import Database
from modules.preprocessor import DataPreprocessor
from modules.trainer import ModelTrainer


class SignalPredictor:
    def __init__(self, verbose: bool = True):
        """Инициализация предсказателя"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.trainer = ModelTrainer(verbose=verbose)
        self.current_signals = {}

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

    def get_latest_data(self, symbol: str, timeframe: str,
                        lookback_window: int = None,
                        verbose: bool = True) -> pd.DataFrame:
        """
        Получение последних данных для предсказания

        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            lookback_window: Окно для LSTM
            verbose: Флаг логирования

        Returns:
            DataFrame с последними данными
        """
        try:
            if lookback_window is None:
                lookback_window = config.model.LOOKBACK_WINDOW

            # Получение данных за последние N+100 свечей (с запасом для индикаторов)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_window * 2)

            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )

            if data.empty:
                self.log(f"No data available for {symbol} {timeframe}", 'error')
                return pd.DataFrame()

            # Расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            # Оставляем только последние lookback_window + 10 строк
            data_for_prediction = data_with_indicators.tail(lookback_window + 10)

            self.log(f"Prepared {len(data_for_prediction)} candles for prediction of {symbol}")

            return data_for_prediction

        except Exception as e:
            self.log(f"Error fetching data for prediction: {e}", 'error')
            return pd.DataFrame()

    def prepare_for_prediction(self, data: pd.DataFrame,
                               lookback_window: int = None,
                               verbose: bool = True) -> np.ndarray:
        """
        Подготовка данных для предсказания

        Args:
            data: DataFrame с индикаторами
            lookback_window: Окно для LSTM
            verbose: Флаг логирования

        Returns:
            Подготовленные данные для модели
        """
        try:
            if lookback_window is None:
                lookback_window = config.model.LOOKBACK_WINDOW

            # Подготовка фичей
            X_sequence = self.preprocessor.prepare_features_for_prediction(
                df=data,
                lookback_window=lookback_window,
                verbose=verbose
            )

            return X_sequence

        except Exception as e:
            self.log(f"Error preparing data for prediction: {e}", 'error')
            return np.array([])

    def predict_with_lstm(self, model, scaler, X_sequence: np.ndarray,
                          verbose: bool = True) -> Tuple[int, Dict]:
        """
        Предсказание с использованием LSTM модели

        Args:
            model: LSTM модель
            scaler: Скейлер
            X_sequence: Входные данные
            verbose: Флаг логирования

        Returns:
            Класс предсказания и вероятности
        """
        try:
            if len(X_sequence) == 0:
                return 0, {}

            # Нормализация
            X_normalized, _ = self.preprocessor.normalize_features(
                X_sequence, fit=False, scaler=scaler, verbose=verbose
            )

            # Предсказание
            predictions = model.predict(X_normalized, verbose=0)

            # Получение класса и вероятностей
            predicted_class = np.argmax(predictions[0]) - 1  # -1, 0, 1
            probabilities = {
                'SHORT': float(predictions[0][0]),
                'HOLD': float(predictions[0][1]),
                'LONG': float(predictions[0][2])
            }

            confidence = np.max(predictions[0])

            if verbose:
                self.log(f"LSTM prediction: class={predicted_class}, "
                         f"confidence={confidence:.4f}, probabilities={probabilities}")

            return predicted_class, {
                'probabilities': probabilities,
                'confidence': float(confidence),
                'model_type': 'lstm'
            }

        except Exception as e:
            self.log(f"LSTM prediction error: {e}", 'error')
            return 0, {}

    def predict_with_xgboost(self, model, scaler, X_sequence: np.ndarray,
                             verbose: bool = True) -> Tuple[int, Dict]:
        """
        Предсказание с использованием XGBoost модели

        Args:
            model: XGBoost модель
            scaler: Скейлер
            X_sequence: Входные данные
            verbose: Флаг логирования

        Returns:
            Класс предсказания и вероятности
        """
        try:
            if len(X_sequence) == 0:
                return 0, {}

            # Преобразование 3D -> 2D для XGBoost
            X_2d = X_sequence.reshape(1, -1)

            # Нормализация
            X_normalized, _ = self.preprocessor.normalize_features(
                X_2d, fit=False, scaler=scaler, verbose=verbose
            )

            # Предсказание
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_normalized)[0]
                predicted_class = model.predict(X_normalized)[0]
            else:
                predicted_class = model.predict(X_normalized)[0]
                probabilities = [0, 0, 0]
                probabilities[predicted_class + 1] = 1.0

            confidence = np.max(probabilities)

            prob_dict = {
                'SHORT': float(probabilities[0]),
                'HOLD': float(probabilities[1]),
                'LONG': float(probabilities[2])
            }

            if verbose:
                self.log(f"XGBoost prediction: class={predicted_class}, "
                         f"confidence={confidence:.4f}, probabilities={prob_dict}")

            return int(predicted_class), {
                'probabilities': prob_dict,
                'confidence': float(confidence),
                'model_type': 'xgb'
            }

        except Exception as e:
            self.log(f"XGBoost prediction error: {e}", 'error')
            return 0, {}

    def get_best_model(self, symbol: str, verbose: bool = True) -> Tuple[str, str]:
        """
        Получение лучшей модели для символа

        Args:
            symbol: Торговая пара
            verbose: Флаг логирования

        Returns:
            ID лучшей модели и её тип
        """
        try:
            # Сравнение моделей
            comparison_df = self.trainer.compare_models(symbol, verbose=False)

            if not comparison_df.empty:
                best_model = comparison_df.iloc[0]
                return best_model['model_id'], best_model['model_type']

            # Если сравнения нет, берем последнюю активную модель
            models_df = self.db.get_available_models(
                symbol=symbol,
                active_only=True,
                verbose=verbose
            )

            if not models_df.empty:
                latest_model = models_df.iloc[0]
                return latest_model['model_id'], latest_model['model_type']

            self.log(f"No models available for {symbol}", 'warning')
            return None, None

        except Exception as e:
            self.log(f"Error getting best model: {e}", 'error')
            return None, None

    def get_signal(self, symbol: str, model_id: str = None,
                   verbose: bool = True) -> Dict:
        """
        Получение торгового сигнала для указанной пары

        Args:
            symbol: Торговая пара
            model_id: ID модели (если None - используется лучшая)
            verbose: Флаг логирования

        Returns:
            Словарь с сигналом и метаданными
        """
        try:
            self.log(f"Generating signal for {symbol}")

            # Получение лучшей модели, если не указана
            if model_id is None:
                model_id, model_type = self.get_best_model(symbol, verbose=verbose)
                if model_id is None:
                    return self.get_technical_signal(symbol, verbose=verbose)
            else:
                # Получение типа модели по ID
                models_df = self.db.get_available_models(model_id=model_id, verbose=verbose)
                if models_df.empty:
                    self.log(f"Model {model_id} not found", 'error')
                    return self.get_technical_signal(symbol, verbose=verbose)
                model_type = models_df.iloc[0]['model_type']

            # Загрузка модели и скейлера
            model, scaler = self.trainer.load_model(model_id, verbose=verbose)
            if model is None:
                self.log(f"Failed to load model {model_id}", 'warning')
                return self.get_technical_signal(symbol, verbose=verbose)

            # Получение данных для предсказания
            data = self.get_latest_data(
                symbol=symbol,
                timeframe=config.timeframe.TRADING_TIMEFRAME,
                verbose=verbose
            )

            if data.empty:
                return {'signal': 'HOLD', 'reason': 'No data'}

            # Подготовка данных для предсказания
            X_sequence = self.prepare_for_prediction(data, verbose=verbose)

            if len(X_sequence) == 0:
                return {'signal': 'HOLD', 'reason': 'Insufficient data'}

            # Предсказание в зависимости от типа модели
            if 'lstm' in model_type:
                predicted_class, details = self.predict_with_lstm(
                    model, scaler, X_sequence, verbose=verbose
                )
            elif 'xgb' in model_type:
                predicted_class, details = self.predict_with_xgboost(
                    model, scaler, X_sequence, verbose=verbose
                )
            else:
                predicted_class, details = 0, {}

            # Преобразование класса в сигнал
            signal_map = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}
            signal = signal_map.get(predicted_class, 'HOLD')

            # Дополнительный технический анализ
            tech_signal, tech_details = self.get_technical_signal(symbol, verbose=False)

            # Комбинирование сигналов (если AI сигнал слабый, используем технический)
            final_signal = signal
            reason = 'AI model'

            if details.get('confidence', 0) < 0.6:  # Низкая уверенность AI
                if tech_signal != 'HOLD':
                    final_signal = tech_signal
                    reason = 'Technical analysis (low AI confidence)'
                else:
                    reason = 'AI model (low confidence)'

            # Создание результата
            result = {
                'symbol': symbol,
                'signal': final_signal,
                'ai_signal': signal,
                'tech_signal': tech_signal,
                'timestamp': datetime.now(),
                'price': float(data['close'].iloc[-1]),
                'model_id': model_id,
                'model_type': model_type,
                'confidence': details.get('confidence', 0),
                'probabilities': details.get('probabilities', {}),
                'reason': reason,
                'technical_indicators': tech_details
            }

            # Сохранение сигнала
            self.current_signals[symbol] = result

            self.log(f"Generated signal for {symbol}: {final_signal} "
                     f"(AI: {signal}, Tech: {tech_signal}, Confidence: {details.get('confidence', 0):.2f})")

            return result

        except Exception as e:
            self.log(f"Error generating signal for {symbol}: {e}", 'error')
            return {'symbol': symbol, 'signal': 'HOLD', 'reason': f'Error: {str(e)}'}

    def get_technical_signal(self, symbol: str, verbose: bool = True) -> Tuple[str, Dict]:
        """
        Генерация сигнала на основе технического анализа

        Args:
            symbol: Торговая пара
            verbose: Флаг логирования

        Returns:
            Сигнал и детали технического анализа
        """
        try:
            # Получение последних данных
            data = self.get_latest_data(
                symbol=symbol,
                timeframe=config.timeframe.TRADING_TIMEFRAME,
                lookback_window=100,
                verbose=verbose
            )

            if data.empty or len(data) < 50:
                return 'HOLD', {'error': 'Insufficient data'}

            last_row = data.iloc[-1]
            prev_row = data.iloc[-2]

            details = {}
            signals = []

            # 1. Анализ RSI
            if 'RSI_14' in data.columns:
                rsi = last_row['RSI_14']
                details['rsi'] = float(rsi)

                if rsi < 30:
                    signals.append(('LONG', 'RSI oversold'))
                elif rsi > 70:
                    signals.append(('SHORT', 'RSI overbought'))

            # 2. Анализ MACD
            if 'MACD' in data.columns and 'MACD_SIGNAL' in data.columns:
                macd = last_row['MACD']
                macd_signal = last_row['MACD_SIGNAL']
                details['macd'] = float(macd)
                details['macd_signal'] = float(macd_signal)

                if macd > macd_signal and prev_row['MACD'] <= prev_row['MACD_SIGNAL']:
                    signals.append(('LONG', 'MACD bullish crossover'))
                elif macd < macd_signal and prev_row['MACD'] >= prev_row['MACD_SIGNAL']:
                    signals.append(('SHORT', 'MACD bearish crossover'))

            # 3. Анализ Bollinger Bands
            if 'BB_UPPER' in data.columns and 'BB_LOWER' in data.columns:
                price = last_row['close']
                bb_upper = last_row['BB_UPPER']
                bb_lower = last_row['BB_LOWER']
                details['bb_position'] = float((price - bb_lower) / (bb_upper - bb_lower))

                if price < bb_lower:
                    signals.append(('LONG', 'Price below BB lower'))
                elif price > bb_upper:
                    signals.append(('SHORT', 'Price above BB upper'))

            # 4. Анализ скользящих средних
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                sma_20 = last_row['SMA_20']
                sma_50 = last_row['SMA_50']
                details['sma_20'] = float(sma_20)
                details['sma_50'] = float(sma_50)

                if sma_20 > sma_50 and prev_row['SMA_20'] <= prev_row['SMA_50']:
                    signals.append(('LONG', 'Golden cross'))
                elif sma_20 < sma_50 and prev_row['SMA_20'] >= prev_row['SMA_50']:
                    signals.append(('SHORT', 'Death cross'))

            # 5. Анализ объема
            if 'volume' in data.columns and 'VOLUME_MA_20' in data.columns:
                volume = last_row['volume']
                volume_ma = last_row['VOLUME_MA_20']
                volume_ratio = volume / volume_ma if volume_ma > 0 else 1
                details['volume_ratio'] = float(volume_ratio)

                if volume_ratio > 1.5 and last_row['close'] > prev_row['close']:
                    signals.append(('LONG', 'High volume with price increase'))
                elif volume_ratio > 1.5 and last_row['close'] < prev_row['close']:
                    signals.append(('SHORT', 'High volume with price decrease'))

            # Определение итогового сигнала
            if not signals:
                return 'HOLD', details

            # Подсчет сигналов
            long_count = sum(1 for s in signals if s[0] == 'LONG')
            short_count = sum(1 for s in signals if s[0] == 'SHORT')

            if long_count >= 2 and long_count > short_count:
                signal = 'LONG'
                reason = f'Multiple indicators ({long_count} signals)'
            elif short_count >= 2 and short_count > long_count:
                signal = 'SHORT'
                reason = f'Multiple indicators ({short_count} signals)'
            else:
                signal = 'HOLD'
                reason = 'Conflicting or weak signals'

            details['signals_found'] = signals
            details['long_count'] = long_count
            details['short_count'] = short_count

            if verbose:
                self.log(f"Technical analysis {symbol}: {signal} - {reason}")

            return signal, details

        except Exception as e:
            self.log(f"Technical analysis error {symbol}: {e}", 'error')
            return 'HOLD', {'error': str(e)}

    def get_all_signals(self, model_ids: Dict = None, verbose: bool = True) -> Dict:
        """
        Получение сигналов для всех торговых пар

        Args:
            model_ids: Словарь symbol->model_id (если None - используются лучшие модели)
            verbose: Флаг логирования

        Returns:
            Словарь сигналов по всем парам
        """
        try:
            all_signals = {}

            for symbol in config.trading.SYMBOLS:
                model_id = model_ids.get(symbol) if model_ids else None
                signal = self.get_signal(symbol, model_id, verbose=verbose)
                all_signals[symbol] = signal

            # Логирование сводки
            if verbose:
                summary = {
                    'LONG': [],
                    'SHORT': [],
                    'HOLD': []
                }

                for symbol, signal_data in all_signals.items():
                    summary[signal_data['signal']].append(symbol)

                self.log(f"Signals summary: LONG={summary['LONG']}, "
                         f"SHORT={summary['SHORT']}, HOLD={summary['HOLD']}")

            return all_signals

        except Exception as e:
            self.log(f"Error getting all signals: {e}", 'error')
            return {}

    def validate_signal(self, signal_data: Dict, verbose: bool = True) -> bool:
        """
        Валидация сигнала

        Args:
            signal_data: Данные сигнала
            verbose: Флаг логирования

        Returns:
            True если сигнал валиден
        """
        try:
            required_fields = ['symbol', 'signal', 'confidence', 'price', 'timestamp']

            # Проверка наличия полей
            for field in required_fields:
                if field not in signal_data:
                    if verbose:
                        self.log(f"Missing field {field} in signal", 'warning')
                    return False

            # Проверка значений
            if signal_data['signal'] not in ['LONG', 'SHORT', 'HOLD']:
                if verbose:
                    self.log(f"Invalid signal: {signal_data['signal']}", 'warning')
                return False

            if not 0 <= signal_data['confidence'] <= 1:
                if verbose:
                    self.log(f"Invalid confidence: {signal_data['confidence']}", 'warning')
                return False

            if signal_data['price'] <= 0:
                if verbose:
                    self.log(f"Invalid price: {signal_data['price']}", 'warning')
                return False

            # Проверка свежести сигнала
            signal_time = pd.to_datetime(signal_data['timestamp'])
            age_minutes = (datetime.now() - signal_time).total_seconds() / 60

            if age_minutes > 30:  # Сигнал старше 30 минут
                if verbose:
                    self.log(f"Signal is outdated: {age_minutes:.1f} minutes", 'warning')
                return False

            return True

        except Exception as e:
            if verbose:
                self.log(f"Signal validation error: {e}", 'error')
            return False