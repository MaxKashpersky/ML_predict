"""
Модуль для бэктестирования торговых стратегий
"""

import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import config
from modules.database import Database
from modules.preprocessor import DataPreprocessor
from modules.state_manager import state_manager


class Backtester:
    def __init__(self, verbose: bool = True):
        """Инициализация бэктестера"""
        self.verbose = verbose
        self.setup_logging()
        self.db = Database(verbose=verbose)
        self.preprocessor = DataPreprocessor(verbose=verbose)

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

    def run_simple_backtest(self, symbol: str = None,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           initial_balance: float = 10000.0,
                           verbose: bool = True) -> Dict:
        """
        Простой бэктест для выбранной криптовалюты
        """
        try:
            if symbol is None:
                symbol = state_manager.get_selected_symbol()
                if not symbol:
                    return {'error': 'No symbol selected'}

            # Используем даты из state_manager если не указаны
            if start_date is None or end_date is None:
                start_date, end_date = state_manager.get_backtest_dates()

            if verbose:
                self.log(f"Starting backtest for {symbol}")
                self.log(f"Period: {start_date.date()} - {end_date.date()}")
                self.log(f"Initial balance: ${initial_balance:,.2f}")

            # Получение данных
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe=state_manager.get_selected_timeframe(),
                start_date=start_date,
                end_date=end_date,
                verbose=verbose
            )

            if data.empty:
                self.log(f"No data for {symbol}", 'error')
                return {'error': 'No data available'}

            # Простая стратегия: SMA crossover
            data['SMA_10'] = data['close'].rolling(window=10).mean()
            data['SMA_30'] = data['close'].rolling(window=30).mean()

            balance = initial_balance
            position = None
            entry_price = 0
            trades = []
            equity_curve = [balance]

            for i in range(30, len(data)):
                current_price = data['close'].iloc[i]
                sma_10 = data['SMA_10'].iloc[i]
                sma_30 = data['SMA_30'].iloc[i]

                # Сигнал: SMA crossover
                if sma_10 > sma_30 and (i == 30 or data['SMA_10'].iloc[i-1] <= data['SMA_30'].iloc[i-1]):
                    signal = 'LONG'
                elif sma_10 < sma_30 and (i == 30 or data['SMA_10'].iloc[i-1] >= data['SMA_30'].iloc[i-1]):
                    signal = 'SHORT'
                else:
                    signal = 'HOLD'

                # Торговая логика
                if position is None and signal == 'LONG':
                    position = 'LONG'
                    entry_price = current_price
                    position_size = (balance * config.trading.POSITION_SIZE_PCT / 100) / current_price
                    trades.append({
                        'type': 'OPEN',
                        'position': 'LONG',
                        'price': current_price,
                        'size': position_size,
                        'time': data.index[i]
                    })

                elif position == 'LONG' and signal == 'SHORT':
                    pnl = (current_price - entry_price) * position_size
                    pnl -= (entry_price * position_size * config.trading.COMMISSION +
                           current_price * position_size * config.trading.COMMISSION)

                    balance += pnl
                    trades.append({
                        'type': 'CLOSE',
                        'position': 'LONG',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': (pnl / (entry_price * position_size)) * 100,
                        'time': data.index[i]
                    })
                    position = None

                equity_curve.append(balance)

            # Закрытие открытой позиции
            if position is not None:
                last_price = data['close'].iloc[-1]
                pnl = (last_price - entry_price) * position_size
                pnl -= (entry_price * position_size * config.trading.COMMISSION +
                       last_price * position_size * config.trading.COMMISSION)

                balance += pnl
                trades.append({
                    'type': 'CLOSE',
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': last_price,
                    'pnl': pnl,
                    'pnl_pct': (pnl / (entry_price * position_size)) * 100,
                    'reason': 'END_OF_PERIOD',
                    'time': data.index[-1]
                })

            # Расчет метрик
            closed_trades = [t for t in trades if t['type'] == 'CLOSE']

            if closed_trades:
                winning_trades = [t for t in closed_trades if t['pnl'] > 0]
                losing_trades = [t for t in closed_trades if t['pnl'] <= 0]

                win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
                total_profit = sum(t['pnl'] for t in winning_trades)
                total_loss = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            else:
                win_rate = 0
                profit_factor = 0

            # Расчет просадки
            equity_series = pd.Series(equity_curve)
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) * 100

            result = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': initial_balance,
                'final_balance': balance,
                'total_return': ((balance - initial_balance) / initial_balance) * 100,
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_trades) if closed_trades else 0,
                'losing_trades': len(losing_trades) if closed_trades else 0,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor if profit_factor != float('inf') else 0,
                'strategy': 'SMA Crossover (10/30)'
            }

            if verbose:
                self.log(f"Backtest completed:")
                self.log(f"  Initial: ${initial_balance:,.2f}")
                self.log(f"  Final: ${balance:,.2f}")
                self.log(f"  Return: {result['total_return']:.2f}%")
                self.log(f"  Trades: {result['total_trades']}")
                self.log(f"  Win Rate: {result['win_rate']:.1f}%")
                self.log(f"  Max Drawdown: {result['max_drawdown']:.2f}%")

            return result

        except Exception as e:
            self.log(f"Backtest error: {e}", 'error')
            return {'error': str(e)}

    def run_comprehensive_backtest(self, symbol: str = None,
                                  initial_balance: float = None,
                                  verbose: bool = True) -> Dict:
        """
        Комплексный бэктест для выбранной криптовалюты
        """
        try:
            if symbol is None:
                symbol = state_manager.get_selected_symbol()
                if not symbol:
                    return {'error': 'No symbol selected'}

            if initial_balance is None:
                initial_balance = config.backtest.INITIAL_BALANCE

            self.log(f"Comprehensive backtest for {symbol}")

            # Запуск простого бэктеста
            result = self.run_simple_backtest(
                symbol=symbol,
                initial_balance=initial_balance,
                verbose=verbose
            )

            # Сохранение результата
            if 'error' not in result:
                self.save_backtest_result(result, verbose=verbose)

            return {
                'individual_results': {symbol: result},
                'summary': result
            }

        except Exception as e:
            self.log(f"Comprehensive backtest error: {e}", 'error')
            return {'error': str(e)}

    def save_backtest_result(self, result: Dict, verbose: bool = True):
        """Сохранение результата бэктеста в БД"""
        try:
            backtest_data = {
                'model_id': 'SIMPLE_SMA',
                'symbol': result['symbol'],
                'timeframe': state_manager.get_selected_timeframe(),
                'test_date': datetime.now(),
                'start_date': result['start_date'],
                'end_date': result['end_date'],
                'initial_balance': result['initial_balance'],
                'final_balance': result['final_balance'],
                'total_return': result['total_return'],
                'sharpe_ratio': 0,  # Можно добавить расчет
                'max_drawdown': result['max_drawdown'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'total_trades': result['total_trades'],
                'winning_trades': result['winning_trades'],
                'losing_trades': result['losing_trades'],
                'avg_win': 0,  # Можно добавить расчет
                'avg_loss': 0,  # Можно добавить расчет
                'details': json.dumps({'strategy': result.get('strategy', 'Simple SMA')})
            }

            self.db.save_backtest_result(backtest_data, verbose=verbose)

            if verbose:
                self.log(f"Backtest result saved to database")

        except Exception as e:
            if verbose:
                self.log(f"Error saving backtest result: {e}", 'error')