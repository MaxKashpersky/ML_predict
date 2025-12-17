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
from modules.state_manager import state_manager
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

        # Проверяем, выбрана ли криптовалюта
        self.selected_symbol = state_manager.get_selected_symbol()

        logger.info("Trading bot initialized")

    def select_cryptocurrency(self):
        """Выбор криптовалюты для работы"""
        print("\n" + "=" * 50)
        print("ВЫБОР КРИПТОВАЛЮТЫ")
        print("=" * 50)
        print("Доступные криптовалюты:")

        for i, symbol in enumerate(config.trading.ALL_SYMBOLS, 1):
            print(f"{i}. {symbol}")

        print(f"{len(config.trading.ALL_SYMBOLS) + 1}. Ввести вручную")
        print(f"{len(config.trading.ALL_SYMBOLS) + 2}. Список популярных криптовалют")

        while True:
            try:
                choice = input("\nВыберите опцию (1-11): ")

                if choice.isdigit():
                    choice_num = int(choice)

                    if 1 <= choice_num <= len(config.trading.ALL_SYMBOLS):
                        selected_symbol = config.trading.ALL_SYMBOLS[choice_num - 1]
                        break
                    elif choice_num == len(config.trading.ALL_SYMBOLS) + 1:
                        selected_symbol = input("Введите символ криптовалюты (например, BTCUSDT): ").strip().upper()
                        if selected_symbol:
                            break
                        else:
                            print("Пожалуйста, введите корректный символ")
                    elif choice_num == len(config.trading.ALL_SYMBOLS) + 2:
                        self.show_popular_cryptocurrencies()
                        continue
                    else:
                        print(f"Пожалуйста, введите число от 1 до {len(config.trading.ALL_SYMBOLS) + 2}")
                else:
                    print("Пожалуйста, введите число")

            except KeyboardInterrupt:
                print("\n\nВыход из программы...")
                sys.exit(0)
            except Exception as e:
                print(f"Ошибка: {e}")

        # Устанавливаем выбранную криптовалюту
        state_manager.set_selected_symbol(selected_symbol)
        self.selected_symbol = selected_symbol

        print(f"\nВыбрана криптовалюта: {selected_symbol}")
        print(f"Все операции будут выполняться только с {selected_symbol}")

        # Устанавливаем таймфрейм по умолчанию
        state_manager.set_selected_timeframe('5m')

        return selected_symbol

    def show_popular_cryptocurrencies(self):
        """Показать список популярных криптовалют"""
        print("\n📊 Популярные криптовалюты (топ-20):")
        print("-" * 50)

        # Топ-20 криптовалют по рыночной капитализации (пример)
        popular_crypto = [
            "BTCUSDT",  # Bitcoin
            "ETHUSDT",  # Ethereum
            "BNBUSDT",  # Binance Coin
            "SOLUSDT",  # Solana
            "XRPUSDT",  # Ripple
            "ADAUSDT",  # Cardano
            "DOGEUSDT", # Dogecoin
            "AVAXUSDT", # Avalanche
            "DOTUSDT",  # Polkadot
            "TRXUSDT",  # TRON
            "LINKUSDT", # Chainlink
            "MATICUSDT", # Polygon
            "SHIBUSDT", # Shiba Inu
            "LTCUSDT",  # Litecoin
            "UNIUSDT",  # Uniswap
            "ATOMUSDT", # Cosmos
            "ETCUSDT",  # Ethereum Classic
            "XLMUSDT",  # Stellar
            "ICPUSDT",  # Internet Computer
            "FILUSDT",  # Filecoin
        ]

        for i, symbol in enumerate(popular_crypto, 1):
            print(f"{i:2d}. {symbol}")

        print("\n💡 Совет: Для лучших результатов выбирайте криптовалюты с высокой ликвидностью")
        print("   (BTCUSDT, ETHUSDT, BNBUSDT и т.д.)")

    def configure_periods(self):
        """Настройка периодов обучения и бэктеста"""
        print("\n" + "=" * 50)
        print("НАСТРОЙКА ПЕРИОДОВ")
        print("=" * 50)

        try:
            # Период обучения
            print(f"\n📚 ПЕРИОД ОБУЧЕНИЯ:")
            print(f"   По умолчанию: {config.data.TRAINING_PERIOD_DAYS} дней")
            print(f"   Рекомендуется: 30-180 дней для стабильных моделей")

            training_days = input(f"\nПериод обучения в днях (Enter для {config.data.TRAINING_PERIOD_DAYS}): ").strip()
            if training_days:
                training_days = int(training_days)
                if training_days < 7:
                    print("⚠️  Внимание: период менее 7 дней может быть недостаточным для обучения!")
                    confirm = input("Продолжить? (y/n): ")
                    if confirm.lower() != 'y':
                        print("Используется значение по умолчанию")
                        training_days = config.data.TRAINING_PERIOD_DAYS
                elif training_days > 365:
                    print("⚠️  Внимание: период более 365 дней может привести к устаревшим паттернам!")
                    confirm = input("Продолжить? (y/n): ")
                    if confirm.lower() != 'y':
                        print("Используется значение по умолчанию")
                        training_days = config.data.TRAINING_PERIOD_DAYS
                state_manager.set_training_period(training_days)
                print(f"✅ Установлен период обучения: {training_days} дней")
            else:
                training_days = config.data.TRAINING_PERIOD_DAYS
                print(f"✅ Используется период обучения по умолчанию: {training_days} дней")

            # Период бэктеста
            print(f"\n📊 ПЕРИОД БЭКТЕСТА:")
            print(f"   По умолчанию: {config.data.BACKTEST_PERIOD_DAYS} дней")
            print(f"   Рекомендуется: 7-30 дней для актуальной проверки")

            backtest_days = input(f"\nПериод бэктеста в днях (Enter для {config.data.BACKTEST_PERIOD_DAYS}): ").strip()
            if backtest_days:
                backtest_days = int(backtest_days)
                if backtest_days < 3:
                    print("⚠️  Внимание: период менее 3 дней может быть недостаточным для оценки!")
                    confirm = input("Продолжить? (y/n): ")
                    if confirm.lower() != 'y':
                        print("Используется значение по умолчанию")
                        backtest_days = config.data.BACKTEST_PERIOD_DAYS
                state_manager.set_backtest_period(backtest_days)
                print(f"✅ Установлен период бэктеста: {backtest_days} дней")
            else:
                backtest_days = config.data.BACKTEST_PERIOD_DAYS
                print(f"✅ Используется период бэктеста по умолчанию: {backtest_days} дней")

            # Показываем расчетные даты
            train_start, train_end = state_manager.get_training_dates()
            backtest_start, backtest_end = state_manager.get_backtest_dates()

            print(f"\n📅 РАСЧЕТНЫЕ ПЕРИОДЫ:")
            print(f"   Обучение: {train_start.date()} - {train_end.date()}")
            print(f"   Бэктест:  {backtest_start.date()} - {backtest_end.date()}")

            # Проверка перекрытия периодов
            if train_end >= backtest_start:
                print("\n⚠️  ВНИМАНИЕ: Периоды обучения и бэктеста пересекаются!")
                print("   Это может привести к переобучению моделей.")
                confirm = input("   Продолжить? (y/n): ")
                if confirm.lower() != 'y':
                    print("Настройка отменена")
                    return

        except ValueError:
            print("❌ Ошибка: пожалуйста, введите целое число")
        except Exception as e:
            print(f"❌ Ошибка настройки периодов: {e}")

    def update_data(self):
        """Обновление исторических данных для выбранной криптовалюты"""
        try:
            if not self.selected_symbol:
                print("❌ Сначала выберите криптовалюту!")
                return

            logger.info(f"Starting data update for {self.selected_symbol}...")

            # Получаем период для загрузки данных
            start_date, end_date = state_manager.get_data_fetch_dates()
            days_back = (end_date - start_date).days

            print(f"\n📥 ЗАГРУЗКА ДАННЫХ")
            print(f"   Криптовалюта: {self.selected_symbol}")
            print(f"   Таймфрейм:    {state_manager.get_selected_timeframe()}")
            print(f"   Период:       {start_date.date()} - {end_date.date()}")
            print(f"   Дней:         {days_back}")
            print("=" * 50)

            # Получаем таймфрейм для обучения
            timeframe = state_manager.get_selected_timeframe()

            # Проверка доступности данных
            print("🔍 Проверка доступности данных на Binance...")

            # Получение данных
            data = self.data_fetcher.fetch_historical_data(
                symbol=self.selected_symbol,
                timeframe=timeframe,
                days_back=days_back
            )

            if data.empty:
                logger.warning(f"No data retrieved for {self.selected_symbol} {timeframe}")
                print("❌ Не удалось получить данные")

                # Предложения по решению проблемы
                print("\n💡 Возможные решения:")
                print("   1. Проверьте корректность символа")
                print("   2. Попробуйте другой таймфрейм")
                print("   3. Уменьшите период загрузки")
                print("   4. Проверьте подключение к интернету")
                return

            # Анализ полученных данных
            print(f"\n✅ Данные получены: {len(data)} свечей")
            print(f"   Первая свеча: {data.index[0]}")
            print(f"   Последняя свеча: {data.index[-1]}")
            print(f"   Пропущенных значений: {data.isnull().sum().sum()}")

            # Сохранение в БД
            print(f"\n💾 СОХРАНЕНИЕ ДАННЫХ В БАЗУ...")
            success = self.db.store_historical_data(
                symbol=self.selected_symbol,
                timeframe=timeframe,
                data=data,
                verbose=True
            )

            if success:
                logger.info(f"Saved {len(data)} rows for {self.selected_symbol} {timeframe}")
                print(f"✅ Данные сохранены: {len(data)} свечей")

                # Дополнительная статистика
                if len(data) > 0:
                    print(f"\n📊 СТАТИСТИКА ДАННЫХ:")
                    print(f"   Средний объем: {data['volume'].mean():.2f}")
                    print(f"   Волатильность (ATR): {data['high'].std():.2f}")
            else:
                print("❌ Ошибка сохранения данных в базу")

        except Exception as e:
            logger.error(f"Error updating data: {e}")
            print(f"❌ Ошибка загрузки данных: {e}")

    def train_specific_model(self):
        """Обучение конкретной модели с выбором таймфрейма"""
        try:
            if not self.selected_symbol:
                print("❌ Сначала выберите криптовалюту!")
                return

            print("\n" + "=" * 50)
            print("ОБУЧЕНИЕ МОДЕЛИ")
            print("=" * 50)

            # Выбор типа модели
            print("\n🤖 ВЫБОР ТИПА МОДЕЛИ:")
            print("   1. LSTM (нейронная сеть) - точнее, но требует GPU")
            print("   2. XGBoost (градиентный бустинг) - быстрее, работает на CPU")
            print("   3. Обе модели - комплексный подход")
            print("   4. Сравнение моделей - тестирование обеих")

            model_choice = input("\nВыбор (1-4): ").strip()

            if model_choice == '1':
                model_types = ['lstm_class']
                print("✅ Выбрана LSTM модель")
            elif model_choice == '2':
                model_types = ['xgb_class']
                print("✅ Выбрана XGBoost модель")
            elif model_choice == '3':
                model_types = ['lstm_class', 'xgb_class']
                print("✅ Выбраны обе модели")
            elif model_choice == '4':
                print("🔬 Режим сравнения моделей активирован")
                model_types = ['lstm_class', 'xgb_class']
            else:
                print("⚠️  Неверный выбор, используются обе модели")
                model_types = ['lstm_class', 'xgb_class']

            # Выбор таймфрейма
            print("\n⏱️  ВЫБОР ТАЙМФРЕЙМА:")
            for i, tf in enumerate(config.timeframe.AVAILABLE_TIMEFRAMES, 1):
                print(f"   {i}. {tf}")

            tf_choice = input(f"\nВыберите таймфрейм (1-{len(config.timeframe.AVAILABLE_TIMEFRAMES)}): ").strip()

            if tf_choice.isdigit() and 1 <= int(tf_choice) <= len(config.timeframe.AVAILABLE_TIMEFRAMES):
                selected_timeframe = config.timeframe.AVAILABLE_TIMEFRAMES[int(tf_choice) - 1]
                state_manager.set_selected_timeframe(selected_timeframe)
                print(f"✅ Выбран таймфрейм: {selected_timeframe}")
            else:
                selected_timeframe = '5m'
                print(f"✅ Используется таймфрейм по умолчанию: {selected_timeframe}")

            # Дополнительные настройки
            print("\n⚙️  ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ:")
            use_advanced_features = input("Использовать расширенные фичи? (y/n): ").strip().lower() == 'y'

            if use_advanced_features:
                print("✅ Будут использованы: технические индикаторы, volume profile")
            else:
                print("✅ Будут использованы базовые фичи")

            print(f"\n📋 СВОДКА НАСТРОЕК:")
            print(f"   Криптовалюта: {self.selected_symbol}")
            print(f"   Таймфрейм:    {selected_timeframe}")
            print(f"   Модели:       {', '.join(model_types)}")
            print(f"   Расширенные фичи: {'Да' if use_advanced_features else 'Нет'}")

            train_start, train_end = state_manager.get_training_dates()
            print(f"   Период обучения: {train_start.date()} - {train_end.date()}")
            print(f"   Длительность: {state_manager.get_training_period()} дней")

            print(f"\n⚠️  ВНИМАНИЕ:")
            print(f"   Обучение LSTM может занять 10-30 минут")
            print(f"   Обучение XGBoost обычно занимает 1-5 минут")
            print(f"   Для ускорения используйте GPU (CUDA)")

            confirm = input("\n🚀 Начать обучение? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Обучение отменено")
                return

            # Словарь для хранения результатов сравнения
            comparison_results = {}

            # Обучение моделей
            for model_type in model_types:
                print(f"\n{'=' * 60}")
                print(f"🔧 ОБУЧЕНИЕ {model_type.upper()}...")
                print(f"{'=' * 60}")

                try:
                    if model_type == 'lstm_class':
                        result = self.trainer.train_lstm_classifier(
                            symbol=self.selected_symbol,
                            timeframe=selected_timeframe,
                            use_advanced_features=use_advanced_features,
                            verbose=True
                        )
                    elif model_type == 'xgb_class':
                        result = self.trainer.train_xgboost_classifier(
                            symbol=self.selected_symbol,
                            timeframe=selected_timeframe,
                            use_advanced_features=use_advanced_features,
                            verbose=True
                        )
                    else:
                        print(f"❌ Неизвестный тип модели: {model_type}")
                        continue

                    if result and 'model' in result:
                        model = result['model']
                        metrics = result.get('metrics', {})
                        feature_importance = result.get('feature_importance', None)

                        # Сохранение результатов для сравнения
                        comparison_results[model_type] = {
                            'metrics': metrics,
                            'feature_importance': feature_importance
                        }

                        # Сохранение модели
                        print(f"\n💾 СОХРАНЕНИЕ МОДЕЛИ...")
                        model_id = self.trainer.generate_model_id(self.selected_symbol, model_type)
                        success = self.trainer.save_model(
                            model=model,
                            model_id=model_id,
                            symbol=self.selected_symbol,
                            model_type=model_type,
                            metrics=metrics,
                            feature_importance=feature_importance,
                            verbose=True
                        )

                        if success:
                            print(f"✅ Модель {model_type} обучена и сохранена")
                            print(f"   ID модели: {model_id}")

                            # Показать метрики
                            if metrics:
                                print(f"\n📊 МЕТРИКИ МОДЕЛИ:")
                                for key, value in metrics.items():
                                    if isinstance(value, float):
                                        print(f"   {key}: {value:.4f}")
                                    else:
                                        print(f"   {key}: {value}")
                        else:
                            print(f"❌ Ошибка сохранения модели {model_type}")
                    else:
                        print(f"❌ Ошибка обучения модели {model_type}")

                except Exception as e:
                    print(f"❌ Ошибка при обучении {model_type}: {e}")
                    logger.error(f"Error training {model_type}: {e}")

            # Сравнение моделей (если обучили несколько)
            if model_choice == '4' and len(comparison_results) > 1:
                print(f"\n{'=' * 60}")
                print("🔬 СРАВНЕНИЕ МОДЕЛЕЙ")
                print(f"{'=' * 60}")

                self.trainer.compare_models(comparison_results)

            print(f"\n{'=' * 60}")
            print("🎓 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
            print(f"{'=' * 60}")

        except Exception as e:
            logger.error(f"Error training models: {e}")
            print(f"❌ Ошибка обучения: {e}")

    def run_backtest(self):
        """Запуск бэктеста для выбранной криптовалюты"""
        try:
            if not self.selected_symbol:
                print("❌ Сначала выберите криптовалюту!")
                return None

            logger.info(f"Running backtest for {self.selected_symbol}...")

            # Получаем даты для бэктеста
            start_date, end_date = state_manager.get_backtest_dates()
            days_back = (end_date - start_date).days

            print(f"\n📊 БЭКТЕСТ")
            print(f"   Криптовалюта: {self.selected_symbol}")
            print(f"   Период:       {start_date.date()} - {end_date.date()}")
            print(f"   Дней:         {days_back}")
            print("=" * 50)

            # Выбор стратегии бэктеста
            print("\n🎯 ВЫБОР СТРАТЕГИИ БЭКТЕСТА:")
            print("   1. Только лучшая модель (автовыбор)")
            print("   2. Все доступные модели")
            print("   3. Конкретная модель")

            strategy_choice = input("\nВыбор (1-3): ").strip()

            model_id = None
            if strategy_choice == '3':
                # Показать доступные модели
                models_df = self.db.get_available_models(
                    symbol=self.selected_symbol,
                    active_only=True,
                    verbose=False
                )
                if not models_df.empty:
                    print("\n📋 ДОСТУПНЫЕ МОДЕЛИ:")
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        print(f"   {i+1}. {row['model_type']} ({row['created_at'][:10]})")

                    model_idx = input(f"\nВыберите модель (1-{len(models_df)}): ").strip()
                    if model_idx.isdigit() and 1 <= int(model_idx) <= len(models_df):
                        model_id = models_df.iloc[int(model_idx)-1]['model_id']
                        print(f"✅ Выбрана модель: {model_id}")
                    else:
                        print("⚠️  Неверный выбор, будет использована лучшая модель")
                else:
                    print("⚠️  Нет доступных моделей, будет обучена новая")

            # Настройки бэктеста
            print("\n⚙️  НАСТРОЙКИ БЭКТЕСТА:")
            initial_balance = input(f"Начальный баланс (по умолчанию {config.backtest.INITIAL_BALANCE}): ").strip()
            if initial_balance:
                try:
                    initial_balance = float(initial_balance)
                except:
                    initial_balance = config.backtest.INITIAL_BALANCE
                    print(f"⚠️  Ошибка ввода, используется {initial_balance}")
            else:
                initial_balance = config.backtest.INITIAL_BALANCE

            commission = input(f"Комиссия в % (по умолчанию {config.backtest.COMMISSION*100}): ").strip()
            if commission:
                try:
                    commission = float(commission) / 100
                except:
                    commission = config.backtest.COMMISSION
                    print(f"⚠️  Ошибка ввода, используется {commission*100}%")
            else:
                commission = config.backtest.COMMISSION

            print(f"\n⚙️  ПАРАМЕТРЫ БЭКТЕСТА:")
            print(f"   Начальный баланс: ${initial_balance:,.2f}")
            print(f"   Комиссия: {commission*100:.2f}%")
            print(f"   Плечо: 1x (без маржинальной торговли)")

            confirm = input("\n🚀 Запустить бэктест? (y/n): ")
            if confirm.lower() != 'y':
                print("❌ Бэктест отменен")
                return None

            # Запуск бэктеста
            results = self.backtester.run_comprehensive_backtest(
                symbol=self.selected_symbol,
                initial_balance=initial_balance,
                commission=commission,
                model_id=model_id if model_id else None,
                verbose=True
            )

            if results and 'error' not in results:
                print("\n✅ БЭКТЕСТ ЗАВЕРШЕН УСПЕШНО!")

                # Показать основные результаты
                if 'summary' in results and 'aggregated' in results['summary']:
                    agg = results['summary']['aggregated']
                    print(f"\n📈 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")

                    # Цвет для доходности
                    total_return = agg.get('total_return', 0)
                    return_color = "\033[92m" if total_return > 0 else "\033[91m"
                    reset_color = "\033[0m"

                    print(f"   Общая доходность: {return_color}{total_return:.2f}%{reset_color}")
                    print(f"   Конечный баланс: ${agg.get('final_balance', initial_balance):,.2f}")
                    print(f"   Прибыль/убыток: ${agg.get('total_pnl', 0):,.2f}")
                    print(f"   Всего сделок: {agg.get('total_trades', 0)}")
                    print(f"   Win Rate: {agg.get('avg_win_rate', 0):.1f}%")
                    print(f"   Profit Factor: {agg.get('profit_factor', 0):.2f}")
                    print(f"   Максимальная просадка: {agg.get('max_drawdown', 0):.2f}%")

                    # Рекомендации
                    print(f"\n💡 РЕКОМЕНДАЦИИ:")
                    if total_return > 20:
                        print("   🎉 Отличные результаты! Модель показывает высокую эффективность")
                    elif total_return > 5:
                        print("   👍 Хорошие результаты, можно использовать для торговли")
                    elif total_return > -5:
                        print("   ⚠️  Результаты нейтральные, требуется доработка модели")
                    else:
                        print("   ❌ Низкая эффективность, требуется переобучение модели")

                    # Показать лучшую модель
                    if 'best_model' in results['summary']:
                        best_model = results['summary']['best_model']
                        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ:")
                        print(f"   Тип: {best_model.get('model_type', 'N/A')}")
                        print(f"   Доходность: {best_model.get('total_return', 0):.2f}%")

                return results
            else:
                error_msg = results.get('error', 'Неизвестная ошибка') if results else 'Пустые результаты'
                print(f"❌ Бэктест не удался: {error_msg}")
                return None

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            print(f"❌ Ошибка бэктеста: {e}")
            return None

    def generate_signals(self):
        """Генерация торговых сигналов для выбранной криптовалюты"""
        try:
            if not self.selected_symbol:
                print("❌ Сначала выберите криптовалюту!")
                return {}

            logger.info(f"Generating signals for {self.selected_symbol}...")

            print(f"\n📡 ГЕНЕРАЦИЯ СИГНАЛОВ")
            print(f"   Криптовалюта: {self.selected_symbol}")
            print("=" * 50)

            # Выбор модели для генерации сигналов
            print("\n🤖 ВЫБОР МОДЕЛИ ДЛЯ СИГНАЛОВ:")
            print("   1. Автоматически (лучшая доступная модель)")
            print("   2. Конкретная модель")

            model_choice = input("\nВыбор (1-2): ").strip()

            model_id = None
            if model_choice == '2':
                # Показать доступные модели
                models_df = self.db.get_available_models(
                    symbol=self.selected_symbol,
                    active_only=True,
                    verbose=False
                )
                if not models_df.empty:
                    print("\n📋 ДОСТУПНЫЕ МОДЕЛИ:")
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        print(f"   {i+1}. {row['model_type']} - создана {row['created_at'][:10]}")

                    model_idx = input(f"\nВыберите модель (1-{len(models_df)}): ").strip()
                    if model_idx.isdigit() and 1 <= int(model_idx) <= len(models_df):
                        model_id = models_df.iloc[int(model_idx)-1]['model_id']
                        print(f"✅ Выбрана модель: {model_id}")
                    else:
                        print("⚠️  Неверный выбор, будет использована лучшая модель")
                else:
                    print("⚠️  Нет доступных моделей, будет обучена новая")

            # Получение сигнала
            print(f"\n🔍 ПОЛУЧЕНИЕ АКТУАЛЬНЫХ ДАННЫХ...")

            signal = self.predictor.get_signal(
                symbol=self.selected_symbol,
                model_id=model_id,
                verbose=True
            )

            signals = {self.selected_symbol: signal}

            # Отображение сигнала
            if isinstance(signal, dict):
                signal_str = signal.get('signal', 'ERROR')
                confidence = signal.get('confidence', 0)
                price = signal.get('price', 0)
                reason = signal.get('reason', '')
                timestamp = signal.get('timestamp', datetime.now())
                model_info = signal.get('model_info', {})

                # Цвет для сигнала
                if signal_str == 'LONG':
                    signal_color = "\033[92m"  # Зеленый
                    emoji = "🟢"
                    action = "ПОКУПКА"
                elif signal_str == 'SHORT':
                    signal_color = "\033[91m"  # Красный
                    emoji = "🔴"
                    action = "ПРОДАЖА"
                else:
                    signal_color = "\033[93m"  # Желтый
                    emoji = "🟡"
                    action = "ОЖИДАНИЕ"

                reset_color = "\033[0m"

                print(f"\n{emoji} {'='*50}")
                print(f"{emoji} ТОРГОВЫЙ СИГНАЛ")
                print(f"{emoji} {'='*50}")
                print(f"{emoji} Криптовалюта: {self.selected_symbol}")
                print(f"{emoji} Время: {timestamp}")
                print(f"{emoji} Действие: {signal_color}{action}{reset_color}")
                print(f"{emoji} Цена: ${price:.2f}")
                print(f"{emoji} Уверенность: {confidence:.2%}")

                if model_info:
                    print(f"{emoji} Модель: {model_info.get('model_type', 'N/A')}")
                    print(f"{emoji} ID модели: {model_info.get('model_id', 'N/A')}")

                print(f"{emoji} Причина: {reason}")
                print(f"{emoji} {'='*50}")

                # Рекомендации
                print(f"\n💡 РЕКОМЕНДАЦИИ:")
                if confidence > 0.7:
                    print(f"   ✅ Высокая уверенность ({confidence:.0%}) - можно рассматривать для сделки")
                elif confidence > 0.5:
                    print(f"   ⚠️  Средняя уверенность ({confidence:.0%}) - требуется дополнительный анализ")
                else:
                    print(f"   ❌ Низкая уверенность ({confidence:.0%}) - рекомендуется пропустить сделку")

            else:
                print(f"\n⚠️  Сигнал: {signal}")

            logger.info(f"Signal for {self.selected_symbol}: {signal.get('signal', 'ERROR')}")
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            print(f"❌ Ошибка генерации сигналов: {e}")
            return {}

    def run_pipeline(self):
        """Запуск полного пайплайна для выбранной криптовалюты"""
        if not self.selected_symbol:
            print("❌ Сначала выберите криптовалюту!")
            return None

        logger.info("=" * 50)
        logger.info(f"Running full pipeline for {self.selected_symbol}")
        logger.info("=" * 50)

        print(f"\n🚀 ПОЛНЫЙ ПАЙПЛАЙН")
        print(f"   Криптовалюта: {self.selected_symbol}")
        print("=" * 50)

        # Показываем настройки
        train_start, train_end = state_manager.get_training_dates()
        backtest_start, backtest_end = state_manager.get_backtest_dates()

        print(f"\n📋 НАСТРОЙКИ ПАЙПЛАЙНА:")
        print(f"   Обучение: {train_start.date()} - {train_end.date()}")
        print(f"   Бэктест:  {backtest_start.date()} - {backtest_end.date()}")
        print(f"   Таймфрейм: {state_manager.get_selected_timeframe()}")

        confirm = input("\n🚀 Запустить полный пайплайн? (y/n): ")
        if confirm.lower() != 'y':
            print("❌ Пайплайн отменен")
            return None

        results = {}

        try:
            # 1. Обновление данных
            print(f"\n{'='*60}")
            print("1️⃣  ОБНОВЛЕНИЕ ДАННЫХ")
            print(f"{'='*60}")
            self.update_data()

            # 2. Обучение моделей
            print(f"\n{'='*60}")
            print("2️⃣  ОБУЧЕНИЕ МОДЕЛЕЙ")
            print(f"{'='*60}")
            self.train_specific_model()

            # 3. Бэктест
            print(f"\n{'='*60}")
            print("3️⃣  БЭКТЕСТ")
            print(f"{'='*60}")
            backtest_results = self.run_backtest()
            if backtest_results:
                results['backtest_results'] = backtest_results
            else:
                print("⚠️  Бэктест не дал результатов, продолжаем...")

            # 4. Генерация сигналов
            print(f"\n{'='*60}")
            print("4️⃣  ГЕНЕРАЦИЯ СИГНАЛОВ")
            print(f"{'='*60}")
            signals = self.generate_signals()
            if signals:
                results['signals'] = signals

            logger.info("Pipeline completed successfully")
            print(f"\n{'='*60}")
            print("✅ ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
            print(f"{'='*60}")

            # Итоговая сводка
            print(f"\n📋 ИТОГОВАЯ СВОДКА:")
            print(f"   Криптовалюта: {self.selected_symbol}")
            if 'backtest_results' in results and 'summary' in results['backtest_results']:
                agg = results['backtest_results']['summary'].get('aggregated', {})
                total_return = agg.get('total_return', 0)
                print(f"   Доходность бэктеста: {total_return:.2f}%")

            if 'signals' in results and self.selected_symbol in results['signals']:
                signal = results['signals'][self.selected_symbol]
                if isinstance(signal, dict):
                    print(f"   Текущий сигнал: {signal.get('signal', 'N/A')}")
                    print(f"   Уверенность: {signal.get('confidence', 0):.2%}")

            return results

        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            print(f"\n❌ Ошибка в пайплайне: {e}")
            return None


def display_main_menu():
    """Отображение главного меню"""
    print("\n" + "=" * 50)
    print("🤖 AI TRADING BOT v2.0")
    print("=" * 50)

    if state_manager.get_selected_symbol():
        symbol = state_manager.get_selected_symbol()
        timeframe = state_manager.get_selected_timeframe()

        print(f"📊 Текущая криптовалюта: {symbol}")
        print(f"📈 Таймфрейм: {timeframe}")

        # Показать информацию о доступных моделях
        try:
            from modules.database import Database
            db = Database()
            models_count = len(db.get_available_models(symbol=symbol, active_only=True, verbose=False))
            print(f"🤖 Активные модели: {models_count}")
        except:
            pass
    else:
        print("📊 Криптовалюта: НЕ ВЫБРАНА")

    print("=" * 50)
    print("1.  Выбрать криптовалюту")
    print("2.  Настроить периоды")
    print("3.  Обновить данные")
    print("4.  Обучить модель")
    print("5.  Запустить бэктест")
    print("6.  Сгенерировать сигналы")
    print("7.  Запустить полный пайплайн")
    print("8.  Управление моделями")
    print("9.  Режим планировщика")
    print("10. Настройки системы")
    print("0.  Выход")
    print("=" * 50)


def manage_models_menu(bot):
    """Меню управления моделями"""
    while True:
        print("\n" + "=" * 50)
        print("🤖 УПРАВЛЕНИЕ МОДЕЛЯМИ")
        print("=" * 50)
        print("1.  Просмотреть все модели")
        print("2.  Просмотреть модели по символу")
        print("3.  Просмотреть модели по типу")
        print("4.  Удалить модель")
        print("5.  Удалить все модели символа")
        print("6.  Удалить все модели типа")
        print("7.  Активировать/деактивировать модель")
        print("8.  Сравнить производительность моделей")
        print("9.  Экспорт моделей")
        print("10. Назад в главное меню")
        print("=" * 50)

        try:
            choice = input("\nВыбор (1-10): ").strip()

            if choice == '1':
                models_df = bot.db.get_available_models(active_only=False, verbose=True)
                if models_df.empty:
                    print("\n❌ Модели не найдены")
                else:
                    print(f"\n📊 НАЙДЕНО МОДЕЛЕЙ: {len(models_df)}")
                    print("-" * 100)
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        status = "✅" if row['is_active'] else "❌"
                        print(f"{i+1:3d}. {status} ID: {row['model_id']}")
                        print(f"     Символ: {row['symbol']:<10} Тип: {row['model_type']:<15}")
                        print(f"     Создана: {row['created_at']:<25} Активна: {'Да' if row['is_active'] else 'Нет'}")

                        # Показать метрики если есть
                        if 'metrics' in row and row['metrics']:
                            metrics = eval(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
                            if isinstance(metrics, dict):
                                accuracy = metrics.get('accuracy', 'N/A')
                                print(f"     Accuracy: {accuracy}")
                        print()

            elif choice == '2':
                symbol = input("Введите символ (например, BTCUSDT): ").strip().upper()
                if symbol:
                    models_df = bot.db.get_available_models(symbol=symbol, active_only=False, verbose=True)
                    if models_df.empty:
                        print(f"\n❌ Модели не найдены для {symbol}")
                    else:
                        print(f"\n📊 МОДЕЛИ ДЛЯ {symbol}: {len(models_df)}")
                        print("-" * 100)
                        for i, (_, row) in enumerate(models_df.iterrows()):
                            status = "✅" if row['is_active'] else "❌"
                            print(f"{i+1:3d}. {status} ID: {row['model_id']}")
                            print(f"     Тип: {row['model_type']:<15} Создана: {row['created_at']}")
                            print(f"     Активна: {'Да' if row['is_active'] else 'Нет'}")

                            # Показать метрики
                            if 'metrics' in row and row['metrics']:
                                metrics = eval(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
                                if isinstance(metrics, dict):
                                    print(f"     Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                                    print(f"     Precision: {metrics.get('precision', 'N/A'):.4f}")
                                    print(f"     Recall: {metrics.get('recall', 'N/A'):.4f}")
                            print()

            elif choice == '3':
                model_type = input("Введите тип модели (lstm_class, xgb_class): ").strip()
                models_df = bot.db.get_available_models(model_type=model_type, active_only=False, verbose=True)
                if models_df.empty:
                    print(f"\n❌ Модели не найдены типа '{model_type}'")
                else:
                    print(f"\n📊 МОДЕЛИ ТИПА '{model_type}': {len(models_df)}")
                    print("-" * 100)
                    for i, (_, row) in enumerate(models_df.iterrows()):
                        status = "✅" if row['is_active'] else "❌"
                        print(f"{i+1:3d}. {status} ID: {row['model_id']}")
                        print(f"     Символ: {row['symbol']:<10} Создана: {row['created_at']}")
                        print(f"     Активна: {'Да' if row['is_active'] else 'Нет'}")
                        print()

            elif choice == '4':
                model_id = input("Введите ID модели для удаления: ").strip()
                if model_id:
                    print(f"\n⚠️  УДАЛЕНИЕ МОДЕЛИ '{model_id}'")
                    confirm = input("Подтвердите (y/n): ")
                    if confirm.lower() == 'y':
                        success = bot.db.delete_model(model_id, verbose=True)
                        if success:
                            print(f"✅ Модель '{model_id}' удалена")
                        else:
                            print(f"❌ Ошибка удаления модели")

            elif choice == '5':
                symbol = input("Введите символ для удаления всех моделей: ").strip().upper()
                if symbol:
                    print(f"\n⚠️  УДАЛЕНИЕ ВСЕХ МОДЕЛЕЙ ДЛЯ {symbol}")
                    confirm = input("Подтвердите (y/n): ")
                    if confirm.lower() == 'y':
                        deleted_count = bot.db.delete_all_models(symbol=symbol, verbose=True)
                        print(f"✅ Удалено {deleted_count} моделей для {symbol}")

            elif choice == '6':
                model_type = input("Введите тип моделей для удаления: ").strip()
                if model_type:
                    print(f"\n⚠️  УДАЛЕНИЕ ВСЕХ МОДЕЛЕЙ ТИПА '{model_type}'")
                    confirm = input("Подтвердите (y/n): ")
                    if confirm.lower() == 'y':
                        deleted_count = bot.db.delete_all_models(model_type=model_type, verbose=True)
                        print(f"✅ Удалено {deleted_count} моделей типа '{model_type}'")

            elif choice == '7':
                model_id = input("Введите ID модели для активации/деактивации: ").strip()
                if model_id:
                    current_state = bot.db.get_model_state(model_id)
                    if current_state is not None:
                        new_state = not current_state
                        action = "активирована" if new_state else "деактивирована"
                        print(f"\n🔧 Изменение состояния модели '{model_id}'")
                        print(f"   Текущее состояние: {'активна' if current_state else 'неактивна'}")
                        print(f"   Новое состояние: {'активна' if new_state else 'неактивна'}")

                        confirm = input(f"\n{action.capitalize()} модель? (y/n): ")
                        if confirm.lower() == 'y':
                            success = bot.db.update_model_state(model_id, new_state)
                            if success:
                                print(f"✅ Модель {action}")
                            else:
                                print(f"❌ Ошибка изменения состояния")
                    else:
                        print(f"❌ Модель с ID '{model_id}' не найдена")

            elif choice == '8':
                symbol = input("Введите символ для сравнения моделей: ").strip().upper()
                if symbol:
                    print(f"\n🔬 СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ ДЛЯ {symbol}")
                    models_df = bot.db.get_available_models(symbol=symbol, active_only=False, verbose=False)

                    if len(models_df) < 2:
                        print(f"❌ Для сравнения нужно минимум 2 модели, найдено {len(models_df)}")
                    else:
                        print(f"\n📊 НАЙДЕНО МОДЕЛЕЙ: {len(models_df)}")
                        print("-" * 80)

                        comparison_data = []
                        for _, row in models_df.iterrows():
                            metrics = eval(row['metrics']) if row['metrics'] and isinstance(row['metrics'], str) else row['metrics']
                            if metrics and isinstance(metrics, dict):
                                comparison_data.append({
                                    'model_id': row['model_id'],
                                    'model_type': row['model_type'],
                                    'accuracy': metrics.get('accuracy', 0),
                                    'created_at': row['created_at']
                                })

                        # Сортировка по accuracy
                        comparison_data.sort(key=lambda x: x['accuracy'], reverse=True)

                        for i, data in enumerate(comparison_data):
                            rank_emoji = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else "  "))
                            print(f"{rank_emoji} {i+1:2d}. {data['model_type']:<15} Accuracy: {data['accuracy']:.4f}")
                            print(f"     ID: {data['model_id']}")
                            print(f"     Создана: {data['created_at']}")
                            print()

            elif choice == '9':
                print("\n📦 ЭКСПОРТ МОДЕЛЕЙ")
                print("1. Экспортировать конкретную модель")
                print("2. Экспортировать все модели символа")
                print("3. Назад")

                export_choice = input("\nВыбор (1-3): ").strip()

                if export_choice == '1':
                    model_id = input("Введите ID модели для экспорта: ").strip()
                    if model_id:
                        print(f"\n💾 Экспорт модели '{model_id}'...")
                        # Здесь будет вызов метода экспорта из Database
                        print("✅ Функция экспорта в разработке")

                elif export_choice == '2':
                    symbol = input("Введите символ для экспорта всех моделей: ").strip().upper()
                    if symbol:
                        print(f"\n💾 Экспорт всех моделей для {symbol}...")
                        print("✅ Функция экспорта в разработке")

            elif choice == '10':
                print("\n↩️  Возврат в главное меню...")
                break

            else:
                print("❌ Неверный выбор")

        except KeyboardInterrupt:
            print("\n\n↩️  Возврат в главное меню...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


def system_settings_menu(bot):
    """Меню настроек системы"""
    while True:
        print("\n" + "=" * 50)
        print("⚙️  НАСТРОЙКИ СИСТЕМЫ")
        print("=" * 50)
        print("1. Проверить подключение к БД")
        print("2. Проверить подключение к Binance")
        print("3. Очистить кэш данных")
        print("4. Показать статистику системы")
        print("5. Тест производительности")
        print("6. Назад в главное меню")
        print("=" * 50)

        try:
            choice = input("\nВыбор (1-6): ").strip()

            if choice == '1':
                print("\n🔍 ПРОВЕРКА ПОДКЛЮЧЕНИЯ К БАЗЕ ДАННЫХ...")
                try:
                    # Простая проверка подключения
                    test_result = bot.db.test_connection()
                    if test_result:
                        print("✅ Подключение к базе данных успешно")
                    else:
                        print("❌ Ошибка подключения к базе данных")
                except Exception as e:
                    print(f"❌ Ошибка: {e}")

            elif choice == '2':
                print("\n🔍 ПРОВЕРКА ПОДКЛЮЧЕНИЯ К BINANCE...")
                try:
                    # Проверка через data_fetcher
                    test_symbol = bot.selected_symbol or "BTCUSDT"
                    print(f"Проверка для символа: {test_symbol}")

                    # Попытка получить текущую цену
                    current_price = bot.data_fetcher.get_current_price(test_symbol)
                    if current_price:
                        print(f"✅ Подключение успешно. Текущая цена {test_symbol}: ${current_price}")
                    else:
                        print("❌ Не удалось получить данные от Binance")
                except Exception as e:
                    print(f"❌ Ошибка подключения: {e}")

            elif choice == '3':
                print("\n🧹 ОЧИСТКА КЭША ДАННЫХ")
                print("1. Очистить кэш исторических данных")
                print("2. Очистить временные файлы моделей")
                print("3. Очистить логи")
                print("4. Полная очистка")
                print("5. Назад")

                cache_choice = input("\nВыбор (1-5): ").strip()

                if cache_choice == '1':
                    confirm = input("Очистить кэш исторических данных? (y/n): ")
                    if confirm.lower() == 'y':
                        print("🧹 Очистка кэша данных...")
                        print("✅ Функция очистки кэша в разработке")

                elif cache_choice == '2':
                    confirm = input("Очистить временные файлы моделей? (y/n): ")
                    if confirm.lower() == 'y':
                        print("🧹 Очистка временных файлов...")
                        print("✅ Функция очистки в разработке")

                elif cache_choice == '3':
                    confirm = input("Очистить логи? (y/n): ")
                    if confirm.lower() == 'y':
                        print("🧹 Очистка логов...")
                        print("✅ Функция очистки логов в разработке")

                elif cache_choice == '4':
                    confirm = input("ВЫПОЛНИТЬ ПОЛНУЮ ОЧИСТКУ? (y/n): ")
                    if confirm.lower() == 'y':
                        confirm2 = input("Это удалит все временные данные. Продолжить? (y/n): ")
                        if confirm2.lower() == 'y':
                            print("🧹⚡ ПОЛНАЯ ОЧИСТКА...")
                            print("✅ Функция полной очистки в разработке")

            elif choice == '4':
                print("\n📊 СТАТИСТИКА СИСТЕМЫ")
                try:
                    # Получить статистику из базы данных
                    stats = bot.db.get_system_stats()
                    if stats:
                        print(f"   Всего моделей в базе: {stats.get('total_models', 0)}")
                        print(f"   Активных моделей: {stats.get('active_models', 0)}")
                        print(f"   Всего записей данных: {stats.get('total_data_records', 0):,}")
                        print(f"   Размер базы данных: {stats.get('db_size_mb', 0):.2f} MB")
                    else:
                        print("❌ Не удалось получить статистику")
                except Exception as e:
                    print(f"❌ Ошибка получения статистики: {e}")

            elif choice == '5':
                print("\n⚡ ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ")
                print("1. Тест загрузки данных")
                print("2. Тест обучения модели")
                print("3. Тест предсказания")
                print("4. Назад")

                perf_choice = input("\nВыбор (1-4): ").strip()

                if perf_choice == '1':
                    print("⚡ Тестирование загрузки данных...")
                    print("✅ Функция тестирования в разработке")

                elif perf_choice == '2':
                    print("⚡ Тестирование обучения...")
                    print("✅ Функция тестирования в разработке")

                elif perf_choice == '3':
                    print("⚡ Тестирование предсказания...")
                    print("✅ Функция тестирования в разработке")

            elif choice == '6':
                print("\n↩️  Возврат в главное меню...")
                break

            else:
                print("❌ Неверный выбор")

        except KeyboardInterrupt:
            print("\n\n↩️  Возврат в главное меню...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


def start_scheduler_mode(bot):
    """Запуск режима планировщика"""
    if not bot.selected_symbol:
        print("❌ Сначала выберите криптовалюту!")
        return

    print("\n" + "=" * 50)
    print("🕐 РЕЖИМ ПЛАНИРОВЩИКА")
    print("=" * 50)
    print(f"Криптовалюта: {bot.selected_symbol}")
    print(f"Таймфрейм: {state_manager.get_selected_timeframe()}")
    print(f"Обновление данных: каждые {config.data.UPDATE_INTERVAL_HOURS} часов")
    print(f"Генерация сигналов: каждый час")
    print(f"Переобучение моделей: каждые {config.model.RETRAIN_DAYS} дней")
    print("\n⚠️  Нажмите Ctrl+C для остановки")

    # Настройка интервалов
    print("\n⚙️  НАСТРОЙКА ИНТЕРВАЛОВ:")
    update_interval = input(f"Интервал обновления данных (часы, Enter для {config.data.UPDATE_INTERVAL_HOURS}): ").strip()
    if update_interval:
        try:
            update_interval = int(update_interval)
        except:
            update_interval = config.data.UPDATE_INTERVAL_HOURS
    else:
        update_interval = config.data.UPDATE_INTERVAL_HOURS

    signal_interval = input(f"Интервал сигналов (минуты, Enter для 60): ").strip()
    if signal_interval:
        try:
            signal_interval = int(signal_interval)
        except:
            signal_interval = 60
    else:
        signal_interval = 60

    print(f"\n📋 СВОДКА НАСТРОЕК:")
    print(f"   Обновление данных: каждые {update_interval} часов")
    print(f"   Генерация сигналов: каждые {signal_interval} минут")

    confirm = input("\n🚀 Запустить планировщик? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Отменено")
        return

    # Настройка расписания
    print("\n⏰ НАСТРОЙКА РАСПИСАНИЯ...")

    # Обновление данных
    schedule.every(update_interval).hours.do(
        lambda: bot.update_data()
    ).tag('data_update')

    # Генерация сигналов
    schedule.every(signal_interval).minutes.do(
        lambda: bot.generate_signals()
    ).tag('signal_generation')

    # Переобучение моделей (раз в N дней)
    schedule.every(config.model.RETRAIN_DAYS).days.do(
        lambda: bot.train_specific_model()
    ).tag('model_retraining')

    logger.info("Bot running in scheduler mode")
    print(f"\n🤖 БОТ ЗАПУЩЕН В РЕЖИМЕ ПЛАНИРОВЩИКА...")
    print(f"   Первое обновление данных через {update_interval} часов")
    print(f"   Первый сигнал через {signal_interval} минут")
    print(f"   Переобучение моделей через {config.model.RETRAIN_DAYS} дней")
    print("\n📝 Логи пишутся в файл trading_bot.log")

    # Функция для отображения статуса
    def show_scheduler_status():
        print(f"\n⏰ Статус планировщика [{datetime.now().strftime('%H:%M:%S')}]:")
        print(f"   Следующее обновление данных: {schedule.next_run('data_update')}")
        print(f"   Следующий сигнал: {schedule.next_run('signal_generation')}")
        print(f"   Следующее переобучение: {schedule.next_run('model_retraining')}")

    # Показываем статус каждые 10 минут
    schedule.every(10).minutes.do(show_scheduler_status).tag('status')

    # Бесконечный цикл
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Планировщик остановлен пользователем")
        schedule.clear()
        return


def run_interactive_mode(bot):
    """Запуск интерактивного режима"""
    # Проверяем, выбрана ли криптовалюта
    if not bot.selected_symbol:
        print("=" * 50)
        print("🤖 ДОБРО ПОЖАЛОВАТЬ В AI TRADING BOT v2.0")
        print("=" * 50)
        print("Для начала работы необходимо выбрать криптовалюту.")
        print("Рекомендуемые криптовалюты: BTCUSDT, ETHUSDT, BNBUSDT")
        bot.select_cryptocurrency()

    while True:
        display_main_menu()

        try:
            choice = input("\nВыбор (0-10): ").strip()

            if choice == '0':
                print("\n👋 Выход из программы...")
                sys.exit(0)

            elif choice == '1':
                bot.select_cryptocurrency()

            elif choice == '2':
                bot.configure_periods()

            elif choice == '3':
                bot.update_data()

            elif choice == '4':
                bot.train_specific_model()

            elif choice == '5':
                bot.run_backtest()

            elif choice == '6':
                bot.generate_signals()

            elif choice == '7':
                bot.run_pipeline()

            elif choice == '8':
                manage_models_menu(bot)

            elif choice == '9':
                start_scheduler_mode(bot)

            elif choice == '10':
                system_settings_menu(bot)

            else:
                print("❌ Неверный выбор. Пожалуйста, выберите от 0 до 10")

        except KeyboardInterrupt:
            print("\n\n👋 Выход из программы...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")


def main():
    """Основная функция"""
    print("\n" + "=" * 60)
    print("🤖 AI TRADING BOT v2.0 - АВТОМАТИЗИРОВАННАЯ ТОРГОВЛЯ")
    print("=" * 60)
    print("Версия: 2.0")
    print("Дата сборки: 2024")
    print("Автор: AI Trading Team")
    print("=" * 60)

    try:
        bot = TradingBot()
        print("✅ Бот инициализирован успешно!")

        # Проверка системных требований
        print("\n🔍 ПРОВЕРКА СИСТЕМНЫХ ТРЕБОВАНИЙ...")
        import platform
        print(f"   ОС: {platform.system()} {platform.release()}")
        print(f"   Python: {platform.python_version()}")

        # Проверка аргументов командной строки
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            print(f"\n🚀 Запуск в режиме: {mode}")

            if mode == "select":
                bot.select_cryptocurrency()
            elif mode == "update":
                bot.update_data()
            elif mode == "train":
                bot.train_specific_model()
            elif mode == "backtest":
                bot.run_backtest()
            elif mode == "signal":
                bot.generate_signals()
            elif mode == "pipeline":
                bot.run_pipeline()
            elif mode == "scheduler":
                start_scheduler_mode(bot)
            else:
                print("\n📚 ДОСТУПНЫЕ РЕЖИМЫ:")
                print("   select    - Выбор криптовалюты")
                print("   update    - Обновление данных")
                print("   train     - Обучение модели")
                print("   backtest  - Запуск бэктеста")
                print("   signal    - Генерация сигналов")
                print("   pipeline  - Полный пайплайн")
                print("   scheduler - Режим планировщика")
                print("\n💡 Или запустите без аргументов для интерактивного меню")
        else:
            # Запуск интерактивного меню
            run_interactive_mode(bot)

    except KeyboardInterrupt:
        print("\n\n👋 Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Ошибка инициализации бота: {e}")
        logger.exception("Bot initialization error")
        sys.exit(1)


if __name__ == "__main__":
    main()