"""
Модуль для бэктестирования торговых стратегий
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from config import config
from modules.database import Database
from modules.preprocessor import DataPreprocessor
from modules.predictor import SignalPredictor


class Backtester:
    def __init__(self, verbose: bool = True):
        """Инициализация бэктестера"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)
        self.predictor = SignalPredictor(verbose=verbose)
        self.results_cache = {}

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

    def select_model_interactive(self, symbol: str) -> Optional[str]:
        """
        Интерактивный выбор модели для бэктеста

        Args:
            symbol: Торговая пара

        Returns:
            ID выбранной модели или None
        """
        try:
            # Получение доступных моделей
            models_df = self.db.get_available_models(
                symbol=symbol,
                active_only=True,
                verbose=False
            )

            if models_df.empty:
                self.log(f"No models available for {symbol}", 'warning')
                return None

            print(f"\nAvailable models for {symbol}:")
            print("=" * 80)

            for i, (_, row) in enumerate(models_df.iterrows()):
                metrics = json.loads(row['metrics'])
                accuracy = metrics.get('accuracy') or metrics.get('val_accuracy', 0)
                f1_score = metrics.get('f1_score', 0)

                print(f"{i + 1}. {row['model_id']}")
                print(f"   Type: {row['model_type']}")
                print(f"   Created: {row['created_at']}")
                print(f"   Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")

                # Показываем распределение классов
                if 'class_distribution_train' in metrics:
                    dist = metrics['class_distribution_train']
                    print(f"   Class distribution: {dist}")

                print()

            print("=" * 80)

            # Выбор модели
            while True:
                try:
                    choice = input(f"Select model (1-{len(models_df)}) or 'q' to exit: ")

                    if choice.lower() == 'q':
                        return None

                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(models_df):
                        selected_model = models_df.iloc[choice_idx]
                        self.log(f"Selected model: {selected_model['model_id']}")
                        return selected_model['model_id']
                    else:
                        print(f"Invalid choice. Enter a number from 1 to {len(models_df)}")

                except ValueError:
                    print("Please enter a number")

        except Exception as e:
            self.log(f"Model selection error: {e}", 'error')
            return None

    def run_backtest(self, symbol: str, model_id: str,
                     start_date: datetime = None,
                     end_date: datetime = None,
                     initial_balance: float = 10000.0,
                     verbose: bool = True) -> Dict:
        """
        Запуск бэктеста для одной модели

        Args:
            symbol: Торговая пара
            model_id: ID модели
            start_date: Дата начала
            end_date: Дата окончания
            initial_balance: Начальный баланс
            verbose: Флаг логирования

        Returns:
            Словарь с результатами бэктеста
        """
        try:
            # Установка дат по умолчанию
            if start_date is None:
                start_date = datetime.now() - timedelta(days=60)

            if end_date is None:
                end_date = datetime.now()

            if verbose:
                self.log(f"Starting backtest for {symbol} with model {model_id}")
                self.log(f"Period: {start_date} - {end_date}")
                self.log(f"Initial balance: ${initial_balance:,.2f}")

            # Получение данных
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=config.timeframe.BACKTEST_TIMEFRAME,
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )

            if data.empty:
                self.log(f"No data for backtest {symbol} in period {start_date} to {end_date}", 'error')
                # Попробуем получить все доступные данные
                self.log("Trying to get all available data...", 'warning')
                data = self.db.get_historical_data(
                    symbol=symbol,
                    timeframe=config.timeframe.BACKTEST_TIMEFRAME,
                    start_date=None,  # Все данные
                    end_date=None,
                    verbose=verbose
                )

                if data.empty:
                    self.log(f"No data available at all for {symbol}", 'error')
                    return {'error': 'No data'}
                else:
                    self.log(f"Using all available data: {len(data)} candles", 'info')
                    # Обновляем даты на фактический диапазон данных
                    start_date = data.index[0]
                    end_date = data.index[-1]

            # Детальное логирование входных данных
            self.log(f"Retrieved {len(data)} candles for backtest", 'info')
            self.log(f"Data columns: {data.columns.tolist()}", 'debug')
            self.log(f"Date range: {data.index[0]} to {data.index[-1]}", 'debug')
            self.log(f"Sample data:\n{data[['open', 'high', 'low', 'close', 'volume']].head()}", 'debug')

            # Загрузка модели
            from modules.trainer import ModelTrainer
            trainer = ModelTrainer(verbose=verbose)
            model, scaler = trainer.load_model(model_id, verbose=verbose)

            if model is None:
                self.log(f"Failed to load model {model_id}", 'error')
                return {'error': 'Failed to load model'}

            self.log(f"Model loaded successfully. Model type: {type(model)}", 'info')

            # Расчет индикаторов
            data_with_indicators = self.preprocessor.calculate_all_indicators(
                data, verbose=verbose
            )

            if data_with_indicators.empty:
                self.log("Failed to calculate indicators", 'error')
                return {'error': 'Failed to calculate indicators'}

            # Логирование данных с индикаторами
            self.log(f"Data with indicators shape: {data_with_indicators.shape}", 'info')
            indicator_count = len(data_with_indicators.columns) - len(data.columns)
            self.log(f"Added {indicator_count} indicator columns", 'info')

            # Проверка распределения целевых переменных
            target_columns = [col for col in data_with_indicators.columns if 'TARGET_' in col]
            for target_col in target_columns:
                if target_col in data_with_indicators.columns:
                    dist = data_with_indicators[target_col].value_counts().to_dict()
                    total = len(data_with_indicators[target_col].dropna())
                    self.log(f"Target distribution for {target_col}:", 'info')
                    for cls in [-1, 0, 1]:
                        count = dist.get(cls, 0)
                        percentage = (count / total) * 100 if total > 0 else 0
                        self.log(f"  Class {cls}: {count} ({percentage:.1f}%)", 'info')

            # Симуляция торговли
            trades = []
            balance = initial_balance
            position = None  # None, 'LONG', 'SHORT'
            entry_price = 0
            position_size = 0
            equity_curve = [initial_balance]

            # Статистика сигналов для анализа
            signal_stats = {'LONG': 0, 'SHORT': 0, 'HOLD': 0}
            confidence_stats = []
            prediction_stats = []

            # Основной цикл бэктеста
            total_candles = len(data_with_indicators)
            self.log(f"Starting backtest loop with {total_candles} candles", 'info')
            self.log(f"Lookback window: {config.model.LOOKBACK_WINDOW}", 'debug')

            for i in range(config.model.LOOKBACK_WINDOW, total_candles):
                current_time = data_with_indicators.index[i]
                current_price = data_with_indicators['close'].iloc[i]

                # Получение сигнала для текущего момента
                lookback_data = data_with_indicators.iloc[i - config.model.LOOKBACK_WINDOW:i + 1]

                # Подготовка данных для предсказания
                X_sequence = self.preprocessor.prepare_features_for_prediction(
                    lookback_data, verbose=False
                )

                if len(X_sequence) == 0:
                    if verbose and i % 500 == 0:
                        self.log(f"Step {i}: No sequence generated", 'debug')
                    continue

                # Получение предсказания
                models_df = self.db.get_available_models(
                    symbol=symbol,
                    active_only=True,
                    verbose=False
                )

                if models_df.empty:
                    continue

                # Фильтруем по model_id
                model_info = models_df[models_df['model_id'] == model_id]
                if model_info.empty:
                    continue

                model_type = model_info.iloc[0]['model_type']
                signal = 'HOLD'
                confidence = 0
                predicted_class = 0
                probabilities = {'SHORT': 0, 'HOLD': 0, 'LONG': 0}

                try:
                    if 'lstm' in model_type:
                        # Нормализация
                        X_normalized, _ = self.preprocessor.normalize_features(
                            X_sequence, fit=False, scaler=scaler, verbose=False
                        )

                        # Предсказание
                        predictions = model.predict(X_normalized, verbose=0)
                        predicted_class = np.argmax(predictions[0]) - 1
                        confidence = np.max(predictions[0])

                        # Сохраняем вероятности для анализа
                        probabilities = {
                            'SHORT': float(predictions[0][0]),
                            'HOLD': float(predictions[0][1]),
                            'LONG': float(predictions[0][2])
                        }

                        # Детальное логирование каждые 100 свечей
                        if verbose and i % 100 == 0:
                            self.log(f"Step {i}/{total_candles}: Predictions: {predictions[0]}", 'debug')
                            self.log(f"Step {i}: Class: {predicted_class}, Confidence: {confidence:.4f}", 'debug')
                            self.log(f"Step {i}: Probabilities: SHORT={probabilities['SHORT']:.4f}, "
                                     f"HOLD={probabilities['HOLD']:.4f}, LONG={probabilities['LONG']:.4f}", 'debug')

                    elif 'xgb' in model_type:
                        # Преобразование 3D -> 2D
                        X_2d = X_sequence.reshape(1, -1)
                        # Нормализация
                        X_normalized, _ = self.preprocessor.normalize_features(
                            X_2d, fit=False, scaler=scaler, verbose=False
                        )

                        # Предсказание
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X_normalized)[0]
                            predicted_class = model.predict(X_normalized)[0]
                            confidence = np.max(proba)

                            # Для XGBoost классы могут быть в другом формате
                            if len(proba) == 3:
                                probabilities = {
                                    'SHORT': float(proba[0]),
                                    'HOLD': float(proba[1]),
                                    'LONG': float(proba[2])
                                }
                        else:
                            predicted_class = model.predict(X_normalized)[0]
                            confidence = 1.0

                    # Преобразование класса в сигнал
                    signal_map = {-1: 'SHORT', 0: 'HOLD', 1: 'LONG'}
                    signal = signal_map.get(predicted_class, 'HOLD')

                    # Собираем статистику
                    signal_stats[signal] += 1
                    confidence_stats.append(confidence)
                    prediction_stats.append({
                        'step': i,
                        'time': current_time,
                        'signal': signal,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'price': current_price
                    })

                    # Детальное логирование сигналов
                    if verbose and i % 100 == 0:
                        self.log(f"Step {i}: Signal: {signal}, Confidence: {confidence:.4f}", 'info')
                        self.log(f"Step {i}: Current position: {position}", 'info')

                except Exception as e:
                    if verbose and i % 500 == 0:
                        self.log(f"Prediction error on step {i}: {e}", 'debug')
                    continue

                # Торговая логика
                trade_result = None
                open_trade = False
                close_trade = False

                # ВАЖНОЕ ИСПРАВЛЕНИЕ: УСЛОВИЯ ОТКРЫТИЯ ПОЗИЦИИ
                # Вместо жесткого порога confidence > 0.6, используем более гибкие условия

                # Условия для открытия позиции
                if position is None and signal != 'HOLD':
                    # Разные стратегии в зависимости от confidence
                    if confidence > 0.7:  # Высокая уверенность
                        should_open = True
                        reason = "high_confidence"
                    elif confidence > 0.55 and signal != 'HOLD':  # Средняя уверенность
                        # Проверяем, что это не случайный сигнал
                        if i > config.model.LOOKBACK_WINDOW + 10:
                            # Смотрим на предыдущие сигналы
                            recent_signals = [pred['signal'] for pred in prediction_stats[-5:] if 'signal' in pred]
                            if len(recent_signals) >= 3:
                                same_signal_count = sum(1 for s in recent_signals if s == signal)
                                if same_signal_count >= 2:  # Сигнал подтверждается
                                    should_open = True
                                    reason = "confirmed_signal"
                                else:
                                    should_open = False
                                    reason = "unconfirmed"
                            else:
                                should_open = True
                                reason = "first_signal"
                        else:
                            should_open = True
                            reason = "early_signal"
                    else:
                        should_open = False
                        reason = "low_confidence"

                    # ДОПОЛНИТЕЛЬНО: принудительное открытие для тестирования каждые N свечей
                    # Раскомментируйте для тестирования бэктестера:
                    # if i % 200 == 0:  # Открываем каждую 200ю свечу
                    #     should_open = True
                    #     signal = 'LONG' if i % 400 == 0 else 'SHORT'  # Чередуем
                    #     reason = "test_signal"

                    if should_open:
                        # Открытие позиции
                        position = signal
                        entry_price = current_price
                        position_size = (balance * config.trading.POSITION_SIZE_PCT / 100) / current_price

                        # Минимальный размер позиции (защита от микро-позиций)
                        min_position_value = 10.0  # Минимум $10
                        if position_size * current_price < min_position_value:
                            position_size = min_position_value / current_price

                        trade_result = {
                            'type': 'OPEN',
                            'time': current_time,
                            'position': position,
                            'price': current_price,
                            'size': position_size,
                            'confidence': confidence,
                            'reason': reason,
                            'balance_before': balance
                        }

                        open_trade = True

                        if verbose:
                            self.log(f"🔵 OPEN {position} at ${current_price:.2f}", 'info')
                            self.log(f"   Size: {position_size:.6f} ({position_size * current_price:.2f} USD)", 'info')
                            self.log(f"   Confidence: {confidence:.4f}, Reason: {reason}", 'info')

                elif position is not None:
                    # Проверка стоп-лосса и тейк-профита
                    price_change_pct = (current_price - entry_price) / entry_price
                    if position == 'SHORT':
                        price_change_pct = -price_change_pct

                    stop_loss_hit = price_change_pct <= -config.trading.STOP_LOSS_PCT / 100
                    take_profit_hit = price_change_pct >= config.trading.TAKE_PROFIT_PCT / 100

                    # Сигнал на закрытие
                    close_signal = False
                    if (position == 'LONG' and signal == 'SHORT') or \
                            (position == 'SHORT' and signal == 'LONG'):
                        close_signal = True

                    # Проверка на разворот сигнала с высокой уверенностью
                    if close_signal and confidence > 0.7:
                        close_trade = True
                        reason = 'REVERSE_SIGNAL'
                    elif stop_loss_hit:
                        close_trade = True
                        reason = 'STOP_LOSS'
                    elif take_profit_hit:
                        close_trade = True
                        reason = 'TAKE_PROFIT'

                    # Дополнительное условие: закрытие при слабом сигнале в противоположном направлении
                    elif signal != 'HOLD' and signal != position and confidence > 0.6:
                        close_trade = True
                        reason = 'OPPOSITE_SIGNAL'

                    # Закрытие позиции
                    if close_trade:
                        # Расчет P&L
                        if position == 'LONG':
                            pnl = (current_price - entry_price) * position_size
                        else:  # SHORT
                            pnl = (entry_price - current_price) * position_size

                        # Вычет комиссии
                        commission = (entry_price * position_size * config.trading.COMMISSION +
                                      current_price * position_size * config.trading.COMMISSION)
                        pnl -= commission

                        # Расчет процента прибыли/убытка
                        pnl_pct = (pnl / (entry_price * position_size)) * 100

                        balance += pnl

                        trade_result = {
                            'type': 'CLOSE',
                            'time': current_time,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'size': position_size,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'commission': commission,
                            'reason': reason,
                            'balance_after': balance,
                            'price_change_pct': price_change_pct * 100
                        }

                        if verbose:
                            color = "🟢" if pnl > 0 else "🔴"
                            self.log(f"{color} CLOSE {position} at ${current_price:.2f}", 'info')
                            self.log(f"   Entry: ${entry_price:.2f}, Exit: ${current_price:.2f}", 'info')
                            self.log(f"   P&L: ${pnl:.2f} ({pnl_pct:.2f}%)", 'info')
                            self.log(f"   Commission: ${commission:.2f}", 'info')
                            self.log(f"   Reason: {reason}", 'info')
                            self.log(f"   Balance: ${balance:,.2f}", 'info')

                        # Сброс позиции
                        position = None
                        entry_price = 0
                        position_size = 0

                # Сохраняем сделку
                if trade_result:
                    trades.append(trade_result)

                # Обновление кривой баланса
                equity_curve.append(balance)

                # Логирование прогресса
                if verbose and i % 500 == 0:
                    self.log(f"Progress: {i}/{total_candles} candles ({i / total_candles * 100:.1f}%)", 'info')
                    self.log(f"  Balance: ${balance:,.2f}, Trades: {len(trades)}", 'info')
                    self.log(f"  Signal stats: {signal_stats}", 'info')
                    if confidence_stats:
                        avg_conf = np.mean(confidence_stats)
                        self.log(f"  Avg confidence: {avg_conf:.4f}", 'info')

            # Закрытие открытой позиции в конце периода
            if position is not None:
                last_price = data_with_indicators['close'].iloc[-1]

                if position == 'LONG':
                    pnl = (last_price - entry_price) * position_size
                else:  # SHORT
                    pnl = (entry_price - last_price) * position_size

                # Вычет комиссии
                commission = (entry_price * position_size * config.trading.COMMISSION +
                              last_price * position_size * config.trading.COMMISSION)
                pnl -= commission

                balance += pnl

                trades.append({
                    'type': 'CLOSE',
                    'time': data_with_indicators.index[-1],
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': last_price,
                    'size': position_size,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (entry_price * position_size)) * 100,
                    'commission': commission,
                    'reason': 'END_OF_PERIOD',
                    'balance_after': balance
                })

                if verbose:
                    self.log(f"🔚 CLOSED final position at end of period", 'info')
                    self.log(f"   Final P&L: ${pnl:.2f}", 'info')

            # Расчет метрик
            metrics = self.calculate_metrics(
                trades=trades,
                equity_curve=equity_curve,
                initial_balance=initial_balance,
                final_balance=balance,
                verbose=verbose
            )

            # Формирование статистики сигналов
            signal_summary = {
                'total_predictions': sum(signal_stats.values()),
                'signal_distribution': signal_stats,
                'signal_percentages': {
                    sig: (count / sum(signal_stats.values()) * 100)
                    for sig, count in signal_stats.items()
                },
                'avg_confidence': float(np.mean(confidence_stats)) if confidence_stats else 0,
                'median_confidence': float(np.median(confidence_stats)) if confidence_stats else 0,
                'confidence_distribution': {
                    'low': len([c for c in confidence_stats if c < 0.5]),
                    'medium': len([c for c in confidence_stats if 0.5 <= c < 0.7]),
                    'high': len([c for c in confidence_stats if c >= 0.7])
                }
            }

            # Формирование результата
            result = {
                'symbol': symbol,
                'model_id': model_id,
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': ((balance - initial_balance) / initial_balance) * 100,
                'total_trades': len([t for t in trades if t['type'] == 'CLOSE']),
                'winning_trades': len([t for t in trades if t['type'] == 'CLOSE' and t['pnl'] > 0]),
                'losing_trades': len([t for t in trades if t['type'] == 'CLOSE' and t['pnl'] <= 0]),
                'trades': trades[:100],  # Сохраняем только первые 100 сделок
                'equity_curve': equity_curve,
                'metrics': metrics,
                'signal_statistics': signal_summary,
                'prediction_samples': prediction_stats[:20] if prediction_stats else []  # Примеры предсказаний
            }

            # Детальный вывод результатов
            if verbose:
                self.log(f"═══════════════════════════════════════════", 'info')
                self.log(f"📊 BACKTEST COMPLETED", 'info')
                self.log(f"═══════════════════════════════════════════", 'info')
                self.log(f"Symbol: {symbol}", 'info')
                self.log(f"Model: {model_id}", 'info')
                self.log(f"Period: {start_date.date()} to {end_date.date()}", 'info')
                self.log(f"", 'info')
                self.log(f"💰 FINANCIAL RESULTS", 'info')
                self.log(f"  Initial Balance: ${initial_balance:,.2f}", 'info')
                self.log(f"  Final Balance:   ${balance:,.2f}", 'info')
                self.log(f"  Total Return:    {result['total_return']:.2f}%", 'info')
                self.log(f"", 'info')
                self.log(f"📈 TRADE STATISTICS", 'info')
                self.log(f"  Total Trades:    {result['total_trades']}", 'info')
                self.log(f"  Winning Trades:  {result['winning_trades']}", 'info')
                self.log(f"  Losing Trades:   {result['losing_trades']}", 'info')
                self.log(f"  Win Rate:        {metrics.get('win_rate', 0):.1f}%", 'info')
                self.log(f"", 'info')
                self.log(f"🎯 SIGNAL STATISTICS", 'info')
                for sig, count in signal_stats.items():
                    pct = (count / sum(signal_stats.values())) * 100
                    self.log(f"  {sig}: {count} ({pct:.1f}%)", 'info')
                self.log(
                    f"  Avg Confidence: {np.mean(confidence_stats):.4f}" if confidence_stats else "  No confidence data",
                    'info')
                self.log(f"═══════════════════════════════════════════", 'info')

            # Сохранение результата
            self.save_backtest_result(result, verbose=verbose)

            return result

        except Exception as e:
            self.log(f"Backtest error: {e}", 'error')
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", 'error')
            return {'error': str(e)}

    def calculate_metrics(self, trades: List, equity_curve: List,
                          initial_balance: float, final_balance: float,
                          verbose: bool = True) -> Dict:
        """
        Расчет метрик производительности

        Args:
            trades: Список сделок
            equity_curve: Кривая баланса
            initial_balance: Начальный баланс
            final_balance: Конечный баланс
            verbose: Флаг логирования

        Returns:
            Словарь с метриками
        """
        try:
            closed_trades = [t for t in trades if t['type'] == 'CLOSE']

            if not closed_trades:
                return {
                    'sharpe_ratio': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'avg_trade': 0,
                    'expectancy': 0
                }

            # Расчет доходностей
            returns = np.diff(equity_curve) / equity_curve[:-1]

            # Коэффициент Шарпа (предполагаем безрисковую ставку 0%)
            sharpe_ratio = 0
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365)

            # Максимальная просадка
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) * 100

            # Статистика сделок
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

            win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0

            total_profit = sum(t['pnl'] for t in winning_trades)
            total_loss = abs(sum(t['pnl'] for t in losing_trades))

            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            avg_trade = np.mean([t['pnl'] for t in closed_trades]) if closed_trades else 0

            # Математическое ожидание
            expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * abs(avg_loss))

            metrics = {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss),
                'avg_trade': float(avg_trade),
                'expectancy': float(expectancy),
                'total_profit': float(total_profit),
                'total_loss': float(total_loss),
                'largest_win': float(max([t['pnl'] for t in winning_trades])) if winning_trades else 0,
                'largest_loss': float(min([t['pnl'] for t in losing_trades])) if losing_trades else 0
            }

            if verbose:
                self.log(f"Metrics calculated: Sharpe={sharpe_ratio:.2f}, "
                         f"MaxDD={max_drawdown:.2f}%, WinRate={win_rate:.1f}%")

            return metrics

        except Exception as e:
            self.log(f"Error calculating metrics: {e}", 'error')
            return {}

    def save_backtest_result(self, result: Dict, verbose: bool = True):
        """Сохранение результата бэктеста в БД"""
        try:
            # Подготовка данных для сохранения
            backtest_data = {
                'model_id': result['model_id'],
                'symbol': result['symbol'],
                'timeframe': config.timeframe.BACKTEST_TIMEFRAME,
                'test_date': datetime.now(),
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'initial_balance': result['initial_balance'],
                'final_balance': result['final_balance'],
                'total_return': result['total_return'],
                'sharpe_ratio': result['metrics']['sharpe_ratio'],
                'max_drawdown': result['metrics']['max_drawdown'],
                'win_rate': result['metrics']['win_rate'],
                'profit_factor': result['metrics']['profit_factor'],
                'total_trades': result['total_trades'],
                'winning_trades': result['winning_trades'],
                'losing_trades': result['losing_trades'],
                'avg_win': result['metrics']['avg_win'],
                'avg_loss': result['metrics']['avg_loss'],
                'details': json.dumps({
                    'trades_count': len(result.get('trades', [])),
                    'equity_points': len(result.get('equity_curve', []))
                })
            }

            self.db.save_backtest_result(backtest_data, verbose=verbose)

            if verbose:
                self.log(f"Backtest result saved to database")

        except Exception as e:
            if verbose:
                self.log(f"Error saving backtest result: {e}", 'error')

    def run_comprehensive_backtest(self, symbol: str = None,
                                   model_ids: Dict = None,
                                   initial_balance: float = None,
                                   verbose: bool = True) -> Dict:
        """
        Комплексный бэктест для нескольких моделей/символов

        Args:
            symbol: Торговая пара (если None - все пары)
            model_ids: Словарь symbol->model_id
            initial_balance: Начальный баланс
            verbose: Флаг логирования

        Returns:
            Словарь с результатами
        """
        try:
            if initial_balance is None:
                initial_balance = 10000.0

            symbols = [symbol] if symbol else config.trading.SYMBOLS

            all_results = {}

            for sym in symbols:
                self.log(f"Comprehensive backtest for {sym}")

                # Выбор модели для бэктеста
                if model_ids and sym in model_ids:
                    model_id = model_ids[sym]
                else:
                    model_id = self.select_model_interactive(sym)
                    if model_id is None:
                        self.log(f"No model selected for {sym}", 'warning')
                        continue

                # Запуск бэктеста с правильными датами
                result = self.run_backtest(
                    symbol=sym,
                    model_id=model_id,
                    start_date=None,  # Используем умолчания внутри метода
                    end_date=None,
                    initial_balance=initial_balance,
                    verbose=verbose
                )

                if 'error' not in result:
                    all_results[sym] = result
                else:
                    self.log(f"Backtest failed for {sym}: {result.get('error', 'Unknown error')}", 'warning')

            # Сводный отчет
            if all_results:
                summary = self.generate_backtest_summary(all_results, verbose=verbose)
            else:
                summary = {'error': 'No successful backtests'}
                self.log("No backtests were successful", 'warning')

            return {
                'individual_results': all_results,
                'summary': summary
            }

        except Exception as e:
            self.log(f"Comprehensive backtest error: {e}", 'error')
            return {'error': str(e)}

    def generate_backtest_summary(self, results: Dict, verbose: bool = True) -> Dict:
        """
        Генерация сводного отчета по бэктестам

        Args:
            results: Результаты бэктестов
            verbose: Флаг логирования

        Returns:
            Сводный отчет
        """
        try:
            if not results:
                return {'error': 'No results'}

            summary_data = []

            for symbol, result in results.items():
                summary_data.append({
                    'symbol': symbol,
                    'model_id': result['model_id'],
                    'initial_balance': result['initial_balance'],
                    'final_balance': result['final_balance'],
                    'total_return': result['total_return'],
                    'total_trades': result['total_trades'],
                    'win_rate': result['metrics']['win_rate'],
                    'sharpe_ratio': result['metrics']['sharpe_ratio'],
                    'max_drawdown': result['metrics']['max_drawdown'],
                    'profit_factor': result['metrics']['profit_factor']
                })

            summary_df = pd.DataFrame(summary_data)

            # Агрегированные метрики
            aggregated = {
                'avg_return': summary_df['total_return'].mean(),
                'total_return': (summary_df['final_balance'].sum() -
                                 summary_df['initial_balance'].sum()) /
                                summary_df['initial_balance'].sum() * 100,
                'avg_win_rate': summary_df['win_rate'].mean(),
                'avg_sharpe': summary_df['sharpe_ratio'].mean(),
                'avg_max_drawdown': summary_df['max_drawdown'].mean(),
                'best_performer': summary_df.loc[summary_df['total_return'].idxmax()]['symbol'],
                'worst_performer': summary_df.loc[summary_df['total_return'].idxmin()]['symbol'],
                'total_trades': summary_df['total_trades'].sum()
            }

            # Сохранение отчета
            report = {
                'summary_df': summary_df.to_dict('records'),
                'aggregated': aggregated,
                'timestamp': datetime.now().isoformat()
            }

            # Сохранение в файл
            os.makedirs(config.LOG_DIR, exist_ok=True)
            report_path = os.path.join(config.LOG_DIR,
                                       f"backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)

            if verbose:
                self.log(f"Summary report:")
                self.log(f"Average return: {aggregated['avg_return']:.2f}%")
                self.log(f"Total return: {aggregated['total_return']:.2f}%")
                self.log(f"Average win rate: {aggregated['avg_win_rate']:.1f}%")
                self.log(f"Average Sharpe ratio: {aggregated['avg_sharpe']:.2f}")
                self.log(f"Best performer: {aggregated['best_performer']}")
                self.log(f"Worst performer: {aggregated['worst_performer']}")
                self.log(f"Report saved: {report_path}")

            return report

        except Exception as e:
            self.log(f"Error generating summary report: {e}", 'error')
            return {'error': str(e)}

    def plot_results(self, result: Dict, save_path: str = None, verbose: bool = True):
        """
        Визуализация результатов бэктеста

        Args:
            result: Результаты бэктеста
            save_path: Путь для сохранения графика
            verbose: Флаг логирования
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if 'error' in result:
                self.log("No data for visualization", 'warning')
                return

            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # 1. Кривая баланса
            ax1 = axes[0]
            ax1.plot(result['equity_curve'], label='Balance', color='blue', linewidth=2)
            ax1.axhline(y=result['initial_balance'], color='red', linestyle='--',
                        label='Initial balance')
            ax1.set_title(f'Balance curve - {result["symbol"]}')
            ax1.set_ylabel('Balance ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Доходность по сделкам
            ax2 = axes[1]
            if result['trades']:
                close_trades = [t for t in result['trades'] if t['type'] == 'CLOSE']
                trade_pnls = [t['pnl'] for t in close_trades]
                trade_numbers = range(1, len(trade_pnls) + 1)

                colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
                ax2.bar(trade_numbers, trade_pnls, color=colors, alpha=0.7)
                ax2.axhline(y=0, color='black', linewidth=0.5)
                ax2.set_title('Trade P&L')
                ax2.set_xlabel('Trade number')
                ax2.set_ylabel('P&L ($)')
                ax2.grid(True, alpha=0.3)

            # 3. Гистограмма доходностей
            ax3 = axes[2]
            if result['trades']:
                close_trades = [t for t in result['trades'] if t['type'] == 'CLOSE']
                trade_returns = [t['pnl_pct'] for t in close_trades]

                ax3.hist(trade_returns, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
                ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
                ax3.set_title('Trade returns distribution')
                ax3.set_xlabel('Return (%)')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                if verbose:
                    self.log(f"Chart saved: {save_path}")

            plt.show()

        except ImportError:
            self.log("Matplotlib not installed. Skipping visualization.", 'warning')
        except Exception as e:
            self.log(f"Visualization error: {e}", 'error')