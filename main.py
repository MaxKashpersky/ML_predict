"""
Точка входа в программу
"""

import schedule
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from modules.data_fetcher import DataFetcher
from modules.database import Database
from modules.trainer import ModelTrainer
from modules.predictor import SignalPredictor
from modules.backtester import Backtester
from config import config

# Настройка кодировки UTF-8 для Windows
if sys.platform == "win32":
    import io
    # Устанавливаем UTF-8 как стандартную кодировку
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # Для консоли Windows
    os.system('chcp 65001 > nul')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{config.LOG_DIR}/trading_bot_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        """Инициализация торгового бота"""
        self.db = Database()
        self.data_fetcher = DataFetcher()
        self.trainer = ModelTrainer()
        self.predictor = SignalPredictor()
        self.backtester = Backtester()

        logger.info("Trading bot initialized")

    def update_data(self, days_back: int = 30):
        """Обновление исторических данных"""
        try:
            logger.info("Starting data update...")

            for symbol in config.trading.SYMBOLS:
                for tf in config.timeframe.TRAINING_TIMEFRAMES:
                    logger.info(f"Fetching data for {symbol} {tf}...")

                    # Получение данных
                    data = self.data_fetcher.fetch_historical_data(
                        symbol=symbol,
                        timeframe=tf,
                        days_back=days_back
                    )

                    if data.empty:
                        logger.warning(f"No data retrieved for {symbol} {tf}")
                        continue

                    # Сохранение в БД
                    self.db.store_historical_data(
                        symbol=symbol,
                        timeframe=tf,
                        data=data,
                        verbose=True
                    )

                    logger.info(f"Saved {len(data)} rows for {symbol} {tf}")

            logger.info("Data updated successfully")

        except Exception as e:
            logger.error(f"Error updating data: {e}")

    def train_models(self, symbol: str = None):
        """Обучение моделей"""
        try:
            logger.info("Starting model training...")

            if symbol:
                # Обучение только для указанного символа
                self.trainer.train_models(symbol=symbol)
            else:
                # Обучение для всех символов
                for sym in config.trading.SYMBOLS:
                    logger.info(f"Training models for {sym}...")
                    self.trainer.train_models(symbol=sym)

            logger.info("Models trained successfully")

        except Exception as e:
            logger.error(f"Error training models: {e}")

    def run_backtest(self, symbol: str = None):
        """Запуск бэктеста"""
        try:
            logger.info("Running backtest...")

            if symbol:
                results = self.backtester.run_comprehensive_backtest(
                    symbol=symbol,
                    initial_balance=config.backtest.INITIAL_BALANCE
                )
            else:
                results = self.backtester.run_comprehensive_backtest(
                    symbol=config.trading.MAIN_SYMBOL,
                    initial_balance=config.backtest.INITIAL_BALANCE
                )

            logger.info("Backtest completed")
            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None

    def generate_signals(self):
        """Генерация торговых сигналов"""
        try:
            logger.info("Generating signals...")

            signals = {}
            for symbol in config.trading.SYMBOLS:
                signal = self.predictor.get_signal(symbol)
                signals[symbol] = signal
                logger.info(f"Signal for {symbol}: {signal.get('signal', 'ERROR')}")

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {}

    def run_pipeline(self):
        """Запуск полного пайплайна"""
        logger.info("=" * 50)
        logger.info("Running full pipeline")
        logger.info("=" * 50)

        # 1. Обновление данных
        print("\nStep 1: Updating data...")
        self.update_data()

        # 2. Обучение моделей
        print("\nStep 2: Training models...")
        self.train_models()

        # 3. Бэктест
        print("\nStep 3: Running backtest...")
        results = self.run_backtest()

        # 4. Генерация сигналов
        print("\nStep 4: Generating signals...")
        signals = self.generate_signals()

        logger.info("Pipeline completed successfully")
        return {"backtest_results": results, "signals": signals}

    def test_backtester_with_simple_strategy(self, symbol: str = None):
        """Тестирование бэктестера с простой стратегией"""
        try:
            if symbol is None:
                symbol = config.trading.MAIN_SYMBOL

            self.log(f"Testing backtester with simple strategy for {symbol}")

            # Получаем данные
            data = self.db.get_historical_data(
                symbol=symbol,
                timeframe='5m',
                start_date=datetime.now() - timedelta(days=30),
                end_date=datetime.now(),
                verbose=True
            )

            if data.empty:
                self.log("No data for test", 'error')
                return

            # Простая стратегия: SMA crossover
            data['SMA_10'] = data['close'].rolling(window=10).mean()
            data['SMA_30'] = data['close'].rolling(window=30).mean()

            balance = 10000.0
            position = None
            entry_price = 0
            trades = []

            for i in range(30, len(data)):
                current_price = data['close'].iloc[i]
                sma_10 = data['SMA_10'].iloc[i]
                sma_30 = data['SMA_30'].iloc[i]

                # Простой сигнал: SMA crossover
                if sma_10 > sma_30 and (i == 30 or data['SMA_10'].iloc[i - 1] <= data['SMA_30'].iloc[i - 1]):
                    signal = 'LONG'
                elif sma_10 < sma_30 and (i == 30 or data['SMA_10'].iloc[i - 1] >= data['SMA_30'].iloc[i - 1]):
                    signal = 'SHORT'
                else:
                    signal = 'HOLD'

                # Торговая логика
                if position is None and signal == 'LONG':
                    position = 'LONG'
                    entry_price = current_price
                    trades.append({'type': 'OPEN', 'position': 'LONG', 'price': current_price})

                elif position == 'LONG' and signal == 'SHORT':
                    pnl = (current_price - entry_price) * (balance * 0.01 / entry_price)
                    balance += pnl
                    trades.append({'type': 'CLOSE', 'position': 'LONG', 'pnl': pnl})
                    position = None

            self.log(f"Test completed: {len(trades)} trades, Final balance: ${balance:,.2f}")
            return trades

        except Exception as e:
            self.log(f"Test error: {e}", 'error')

def display_menu():
    """Отображение главного меню"""
    print("\n" + "=" * 50)
    print("AI TRADING BOT")
    print("=" * 50)
    print("1. Update data")
    print("2. Train models")
    print("3. Run backtest")
    print("4. Generate signals")
    print("5. Run full pipeline")
    print("6. Manage models (view/delete)")
    print("7. Start scheduler mode")
    print("8. Exit")
    print("=" * 50)

def get_user_choice():
    """Получение выбора пользователя"""
    while True:
        try:
            choice = input("\nSelect an option (1-8): ")
            if choice.isdigit() and 1 <= int(choice) <= 8:
                return int(choice)
            else:
                print("Invalid choice. Please enter a number from 1 to 8.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")

def manage_models_menu(bot):
    """Меню управления моделями"""
    while True:
        print("\n" + "=" * 50)
        print("MANAGE MODELS")
        print("=" * 50)
        print("1. View all models")
        print("2. View models by symbol")
        print("3. View models by type")
        print("4. Delete specific model")
        print("5. Delete all models for symbol")
        print("6. Delete all models of type")
        print("7. Delete ALL models (careful!)")
        print("8. Back to main menu")
        print("9. Convert H5 models to Keras format")
        print("=" * 50)

        try:
            choice = input("Select option (1-8): ")

            if choice == '1':
                # Просмотр всех моделей
                models_df = bot.db.get_available_models(active_only=False, verbose=False)
                if models_df.empty:
                    print("\nNo models found in database.")
                else:
                    print(f"\nFound {len(models_df)} models:")
                    print("-" * 80)
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        print(f"{i+1}. ID: {row['model_id']}")
                        print(f"   Symbol: {row['symbol']}, Type: {row['model_type']}")
                        print(f"   Created: {row['created_at']}")
                        print(f"   Active: {'Yes' if row['is_active'] else 'No'}")
                        print()

            elif choice == '2':
                # Просмотр моделей по символу
                symbol = input("Enter symbol (e.g., BTCUSDT): ").strip().upper()
                if symbol:
                    models_df = bot.db.get_available_models(symbol=symbol, active_only=False, verbose=False)
                    if models_df.empty:
                        print(f"\nNo models found for {symbol}.")
                    else:
                        print(f"\nFound {len(models_df)} models for {symbol}:")
                        print("-" * 80)
                        for i, (_, row) in enumerate(models_df.iterrows()):
                            print(f"{i+1}. ID: {row['model_id']}")
                            print(f"   Type: {row['model_type']}")
                            print(f"   Created: {row['created_at']}")
                            print(f"   Active: {'Yes' if row['is_active'] else 'No'}")
                            print()

            elif choice == '3':
                # Просмотр моделей по типу
                print("\nAvailable model types:")
                print("1. lstm_class (LSTM classification)")
                print("2. xgb_class (XGBoost classification)")
                print("3. ensemble (Ensemble model)")
                type_choice = input("\nSelect type (1-3) or enter custom type: ").strip()

                if type_choice == '1':
                    model_type = 'lstm_class'
                elif type_choice == '2':
                    model_type = 'xgb_class'
                elif type_choice == '3':
                    model_type = 'ensemble'
                else:
                    model_type = type_choice

                models_df = bot.db.get_available_models(model_type=model_type, active_only=False, verbose=False)
                if models_df.empty:
                    print(f"\nNo models found of type '{model_type}'.")
                else:
                    print(f"\nFound {len(models_df)} models of type '{model_type}':")
                    print("-" * 80)
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        print(f"{i+1}. ID: {row['model_id']}")
                        print(f"   Symbol: {row['symbol']}")
                        print(f"   Created: {row['created_at']}")
                        print(f"   Active: {'Yes' if row['is_active'] else 'No'}")
                        print()

            elif choice == '4':
                # Удаление конкретной модели
                model_id = input("Enter model ID to delete: ").strip()
                if model_id:
                    print(f"\nWARNING: This will delete model '{model_id}' from database!")
                    confirm = input("Are you sure? Type 'DELETE' to confirm: ")

                    if confirm == 'DELETE':
                        # Получаем информацию о модели перед удалением
                        models_df = bot.db.get_available_models(active_only=False, verbose=False)
                        model_info = models_df[models_df['model_id'] == model_id]

                        if not model_info.empty:
                            row = model_info.iloc[0]
                            print(f"\nDeleting model:")
                            print(f"  ID: {row['model_id']}")
                            print(f"  Symbol: {row['symbol']}")
                            print(f"  Type: {row['model_type']}")
                            print(f"  Created: {row['created_at']}")

                            confirm2 = input("\nFinal confirmation (y/n): ")
                            if confirm2.lower() == 'y':
                                success = bot.db.delete_model(model_id, verbose=True)
                                if success:
                                    print(f"\n✓ Model '{model_id}' deleted successfully!")

                                    # Также удалим файлы модели
                                    model_path = row['model_path']
                                    scaler_path = model_path.replace('.h5', '_scaler.pkl').replace('.pkl', '_scaler.pkl')

                                    if os.path.exists(model_path):
                                        try:
                                            os.remove(model_path)
                                            print(f"✓ Model file '{os.path.basename(model_path)}' deleted")
                                        except Exception as e:
                                            print(f"Warning: Could not delete model file: {e}")

                                    if os.path.exists(scaler_path):
                                        try:
                                            os.remove(scaler_path)
                                            print(f"✓ Scaler file '{os.path.basename(scaler_path)}' deleted")
                                        except Exception as e:
                                            print(f"Warning: Could not delete scaler file: {e}")
                                else:
                                    print(f"\n✗ Failed to delete model '{model_id}'")
                            else:
                                print("\nDeletion cancelled.")
                        else:
                            print(f"\nModel '{model_id}' not found.")
                    else:
                        print("\nDeletion cancelled.")

            elif choice == '5':
                # Удаление всех моделей для символа
                symbol = input("Enter symbol to delete all models for: ").strip().upper()
                if symbol:
                    models_df = bot.db.get_available_models(symbol=symbol, active_only=False, verbose=False)

                    if models_df.empty:
                        print(f"\nNo models found for {symbol}.")
                    else:
                        count = len(models_df)
                        print(f"\nWARNING: This will delete ALL {count} models for {symbol}!")
                        print("\nModels to be deleted:")
                        print("-" * 80)
                        for i, (_, row) in enumerate(models_df.iterrows()):
                            print(f"{i+1}. {row['model_id']} ({row['model_type']}, {row['created_at']})")

                        confirm = input(f"\nDelete ALL {count} models for {symbol}? Type 'DELETE ALL' to confirm: ")

                        if confirm == 'DELETE ALL':
                            confirm2 = input("\nFinal confirmation - this cannot be undone (y/n): ")
                            if confirm2.lower() == 'y':
                                deleted_count = bot.db.delete_all_models(symbol=symbol, verbose=True)
                                print(f"\n✓ Deleted {deleted_count} models for {symbol}!")

                                # Удалим файлы моделей для этого символа
                                import glob
                                model_files = glob.glob(os.path.join(config.MODEL_DIR, f"*{symbol}*.*"))
                                file_count = 0
                                for file in model_files:
                                    try:
                                        os.remove(file)
                                        file_count += 1
                                    except Exception as e:
                                        print(f"Warning: Could not delete file {file}: {e}")

                                if file_count > 0:
                                    print(f"✓ Deleted {file_count} model files from disk")
                            else:
                                print("\nDeletion cancelled.")
                        else:
                            print("\nDeletion cancelled.")

            elif choice == '6':
                # Удаление всех моделей определенного типа
                print("\nAvailable model types:")
                print("1. lstm_class (LSTM classification)")
                print("2. xgb_class (XGBoost classification)")
                print("3. ensemble (Ensemble model)")
                type_choice = input("\nSelect type to delete (1-3) or enter custom type: ").strip()

                if type_choice == '1':
                    model_type = 'lstm_class'
                elif type_choice == '2':
                    model_type = 'xgb_class'
                elif type_choice == '3':
                    model_type = 'ensemble'
                else:
                    model_type = type_choice

                models_df = bot.db.get_available_models(model_type=model_type, active_only=False, verbose=False)

                if models_df.empty:
                    print(f"\nNo models found of type '{model_type}'.")
                else:
                    count = len(models_df)
                    print(f"\nWARNING: This will delete ALL {count} models of type '{model_type}'!")
                    print(f"\nModels to be deleted:")
                    print("-" * 80)
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        print(f"{i+1}. {row['model_id']} ({row['symbol']}, {row['created_at']})")

                    confirm = input(f"\nDelete ALL {count} models of type '{model_type}'? Type 'DELETE TYPE' to confirm: ")

                    if confirm == 'DELETE TYPE':
                        confirm2 = input("\nFinal confirmation - this cannot be undone (y/n): ")
                        if confirm2.lower() == 'y':
                            deleted_count = bot.db.delete_all_models(model_type=model_type, verbose=True)
                            print(f"\n✓ Deleted {deleted_count} models of type '{model_type}'!")

                            # Удалим файлы моделей этого типа
                            import glob
                            if model_type == 'lstm_class':
                                patterns = ['*.h5', '*.keras']
                            elif model_type == 'xgb_class':
                                patterns = ['*.pkl']
                            else:
                                patterns = ['*ensemble*']

                            file_count = 0
                            for pattern in patterns:
                                model_files = glob.glob(os.path.join(config.MODEL_DIR, pattern))
                                for file in model_files:
                                    try:
                                        os.remove(file)
                                        file_count += 1
                                    except Exception as e:
                                        print(f"Warning: Could not delete file {file}: {e}")

                            if file_count > 0:
                                print(f"✓ Deleted {file_count} model files from disk")
                        else:
                            print("\nDeletion cancelled.")
                    else:
                        print("\nDeletion cancelled.")

            elif choice == '7':
                # Удаление ВСЕХ моделей
                models_df = bot.db.get_available_models(active_only=False, verbose=False)
                total_count = len(models_df)

                if total_count == 0:
                    print("\nNo models in database.")
                else:
                    print(f"\n⚠️  ⚠️  CRITICAL WARNING ⚠️  ⚠️")
                    print(f"This will delete ALL {total_count} models from the database!")
                    print("This action cannot be undone!")
                    print("\nModels to be deleted:")
                    print("-" * 80)

                    # Группируем по символам для отображения
                    symbols = models_df['symbol'].unique()
                    for symbol in symbols:
                        symbol_count = len(models_df[models_df['symbol'] == symbol])
                        print(f"{symbol}: {symbol_count} models")

                    confirm = input(f"\nDelete ALL {total_count} models? Type 'DELETE EVERYTHING' to confirm: ")

                    if confirm == 'DELETE EVERYTHING':
                        confirm2 = input("\nAre you absolutely sure? Type 'YES I AM SURE' to confirm: ")

                        if confirm2 == 'YES I AM SURE':
                            confirm3 = input("\nLast chance to cancel. Type 'FINAL CONFIRMATION' to proceed: ")

                            if confirm3 == 'FINAL CONFIRMATION':
                                deleted_count = bot.db.delete_all_models(verbose=True)
                                print(f"\n✓ Deleted ALL {deleted_count} models from database!")

                                # Также удалим файлы моделей из папки models
                                import glob
                                model_files = glob.glob(os.path.join(config.MODEL_DIR, "*.h5")) + \
                                            glob.glob(os.path.join(config.MODEL_DIR, "*.pkl")) + \
                                            glob.glob(os.path.join(config.MODEL_DIR, "*.keras")) + \
                                            glob.glob(os.path.join(config.MODEL_DIR, "*_scaler.pkl"))

                                file_count = 0
                                for file in model_files:
                                    try:
                                        os.remove(file)
                                        file_count += 1
                                    except Exception as e:
                                        print(f"Warning: Could not delete file {file}: {e}")

                                if file_count > 0:
                                    print(f"✓ Deleted {file_count} model files from disk")
                            else:
                                print("\nDeletion cancelled.")
                        else:
                            print("\nDeletion cancelled.")
                    else:
                        print("\nDeletion cancelled.")

            elif choice == '8':
                # Возврат в главное меню
                print("\nReturning to main menu...")
                break

            elif choice == '9':
                print("\nConverting H5 models to Keras format...")
                symbol = input("Enter symbol to convert (or press Enter for all): ").strip().upper() or None
                results = bot.trainer.batch_convert_models(symbol=symbol, verbose=True)
                print(f"\nConversion results: {results['converted']} converted, {results['failed']} failed")

            else:
                print("Invalid choice. Please select 1-8.")

        except KeyboardInterrupt:
            print("\n\nReturning to main menu...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

def run_interactive_mode(bot):
    """Запуск интерактивного режима"""
    while True:
        display_menu()
        choice = get_user_choice()

        if choice == 1:
            print("\n" + "=" * 50)
            print("UPDATING DATA")
            print("=" * 50)
            try:
                # Спросить сколько дней данных загружать
                days_back = input("How many days of data to fetch? (default: 30): ")
                if not days_back.strip():
                    days_back = 30
                else:
                    days_back = int(days_back)

                print(f"\nFetching {days_back} days of data...")
                bot.update_data(days_back=days_back)
                print("\nData update completed successfully!")
            except Exception as e:
                print(f"\nError updating data: {e}")

        elif choice == 2:
            print("\n" + "=" * 50)
            print("TRAINING MODELS")
            print("=" * 50)
            try:
                # Пользователь может выбрать символ для обучения
                print(f"\nAvailable symbols: {config.trading.SYMBOLS}")
                symbol_choice = input("Enter symbol to train (or press Enter for all): ")

                if symbol_choice.strip() and symbol_choice in config.trading.SYMBOLS:
                    print(f"\nTraining models for {symbol_choice}...")
                    bot.train_models(symbol=symbol_choice)
                else:
                    print("\nTraining models for all symbols...")
                    bot.train_models()

                print("\nModel training completed successfully!")
            except Exception as e:
                print(f"\nError training models: {e}")

        elif choice == 3:
            print("\n" + "=" * 50)
            print("RUNNING BACKTEST")
            print("=" * 50)
            try:
                # Пользователь может выбрать символ для бэктеста
                print(f"\nAvailable symbols: {config.trading.SYMBOLS}")
                symbol_choice = input("Enter symbol to backtest (or press Enter for main symbol): ")

                if symbol_choice.strip() and symbol_choice in config.trading.SYMBOLS:
                    print(f"\nRunning backtest for {symbol_choice}...")
                    results = bot.run_backtest(symbol=symbol_choice)
                else:
                    print(f"\nRunning backtest for {config.trading.MAIN_SYMBOL}...")
                    results = bot.run_backtest()

                if results:
                    print("\nBacktest completed successfully!")
                    # Показать основные результаты
                    if 'summary' in results:
                        summary = results['summary']
                        if 'aggregated' in summary:
                            agg = summary['aggregated']
                            print(f"\nSummary results:")
                            print(f"  Average return: {agg.get('avg_return', 0):.2f}%")
                            print(f"  Total return: {agg.get('total_return', 0):.2f}%")
                            print(f"  Average win rate: {agg.get('avg_win_rate', 0):.1f}%")
                            print(f"  Best performer: {agg.get('best_performer', 'N/A')}")
                else:
                    print("\nBacktest failed or no results returned.")
            except Exception as e:
                print(f"\nError running backtest: {e}")

        elif choice == 4:
            print("\n" + "=" * 50)
            print("GENERATING SIGNALS")
            print("=" * 50)
            try:
                print("\nGenerating signals...")
                signals = bot.generate_signals()

                if signals:
                    print("\n" + "-" * 50)
                    print("CURRENT TRADING SIGNALS")
                    print("-" * 50)

                    for symbol, signal in signals.items():
                        if isinstance(signal, dict):
                            signal_str = signal.get('signal', 'UNKNOWN')
                            confidence = signal.get('confidence', 0)
                            price = signal.get('price', 0)
                            reason = signal.get('reason', '')

                            # Цвет для сигнала
                            if signal_str == 'LONG':
                                signal_color = "\033[92m"  # Зеленый
                            elif signal_str == 'SHORT':
                                signal_color = "\033[91m"  # Красный
                            else:
                                signal_color = "\033[93m"  # Желтый

                            reset_color = "\033[0m"

                            print(f"{symbol}: {signal_color}{signal_str}{reset_color}")
                            print(f"  Price: ${price:.2f}")
                            print(f"  Confidence: {confidence:.2%}")
                            print(f"  Reason: {reason}")
                            print()
                        else:
                            print(f"{symbol}: {signal}")

                    print("-" * 50)
                else:
                    print("\nNo signals generated or error occurred.")
            except Exception as e:
                print(f"\nError generating signals: {e}")

        elif choice == 5:
            print("\n" + "=" * 50)
            print("RUNNING FULL PIPELINE")
            print("=" * 50)
            print("\nWarning: This will run all steps and may take a long time.")
            print("Steps:")
            print("  1. Update historical data")
            print("  2. Train ML models")
            print("  3. Run backtest")
            print("  4. Generate trading signals")

            confirm = input("\nContinue? (y/n): ")

            if confirm.lower() == 'y':
                try:
                    print("\nStarting pipeline...")
                    results = bot.run_pipeline()
                    print("\n" + "=" * 50)
                    print("PIPELINE COMPLETED SUCCESSFULLY!")
                    print("=" * 50)

                    if results and 'signals' in results:
                        print("\nFinal trading signals:")
                        print("-" * 30)
                        for symbol, signal in results['signals'].items():
                            if isinstance(signal, dict):
                                print(f"{symbol}: {signal.get('signal', 'UNKNOWN')}")
                except Exception as e:
                    print(f"\nError running pipeline: {e}")
            else:
                print("\nPipeline cancelled.")

        elif choice == 6:
            print("\n" + "=" * 50)
            print("MANAGE MODELS")
            print("=" * 50)
            print("\nWarning: Be careful when deleting models!")
            print("Deleted models cannot be recovered.")

            manage_models_menu(bot)

        elif choice == 7:
            print("\n" + "=" * 50)
            print("STARTING SCHEDULER MODE")
            print("=" * 50)
            print("\nBot will run in background with scheduled tasks.")
            print(f"Data update every {config.data.UPDATE_INTERVAL_HOURS} hours")
            print("Model training daily at 00:00")
            print("Signal generation every hour")
            print("\nPress Ctrl+C to stop the scheduler.")

            confirm = input("\nStart scheduler? (y/n): ")
            if confirm.lower() == 'y':
                try:
                    start_scheduler_mode(bot)
                except KeyboardInterrupt:
                    print("\n\nScheduler stopped by user.")
                except Exception as e:
                    print(f"\nError in scheduler mode: {e}")
            else:
                print("\nScheduler mode cancelled.")

        elif choice == 8:
            print("\nExiting...")
            sys.exit(0)

def start_scheduler_mode(bot):
    """Запуск режима планировщика"""
    # Настройка расписания
    schedule.every(config.data.UPDATE_INTERVAL_HOURS).hours.do(
        lambda: bot.update_data(days_back=7)  # Загружаем только 7 дней для обновлений
    )
    schedule.every().day.at("00:00").do(
        lambda: bot.train_models()
    )
    schedule.every().hour.do(
        lambda: bot.generate_signals()
    )

    logger.info("Bot running in scheduler mode")
    print("\nBot is now running in scheduler mode...")
    print("Press Ctrl+C to stop.")

    # Бесконечный цикл для выполнения запланированных задач
    while True:
        schedule.run_pending()
        time.sleep(60)

def main():
    """Основная функция"""
    print("Initializing Trading Bot...")

    try:
        bot = TradingBot()
        print("Trading Bot initialized successfully!")

        # Проверка аргументов командной строки
        import sys
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            if mode == "update":
                if len(sys.argv) > 2:
                    days_back = int(sys.argv[2])
                    bot.update_data(days_back=days_back)
                else:
                    bot.update_data()
            elif mode == "train":
                if len(sys.argv) > 2:
                    bot.train_models(symbol=sys.argv[2])
                else:
                    bot.train_models()
            elif mode == "backtest":
                if len(sys.argv) > 2:
                    bot.run_backtest(symbol=sys.argv[2])
                else:
                    bot.run_backtest()
            elif mode == "signal":
                bot.generate_signals()
            elif mode == "pipeline":
                bot.run_pipeline()
            elif mode == "manage_models":
                manage_models_menu(bot)
            elif mode == "scheduler":
                start_scheduler_mode(bot)
            else:
                print("Available modes: update, train, backtest, signal, pipeline, manage_models, scheduler")
                print("Or run without arguments for interactive menu.")
        else:
            # Запуск интерактивного меню
            run_interactive_mode(bot)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError initializing Trading Bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()