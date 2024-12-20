# Что такое Model Navigator
Model Navigator — это инструмент для оптимизации и развертывания моделей глубокого обучения, разработанный для работы с NVIDIA GPU. Он автоматизирует процессы экспорта, конверсии, тестирования корректности и профилирования моделей, поддерживая такие фреймворки, как PyTorch, TensorFlow и ONNX. Model Navigator позволяет пользователям эффективно искать лучшие варианты развертывания моделей, предоставляя единый интерфейс для различных фреймворков.

# Использование Model Navigator
в папке [optimize_model](./optimize_model) есть пример использования model navigator для оптимизации модели densenet121


# Подготовка моделей для Triton NVIDIA Server
Для подготовки моделей к эксплуатации на Triton NVIDIA Server необходимо:
1. Экспортировать модель: Преобразовать модель в один из поддерживаемых форматов, таких как ONNX или TensorRT.
2. Оптимизировать модель: Использовать Model Navigator для оптимизации модели, чтобы достичь максимальной производительности.
3. Создать конфигурацию модели: Определить конфигурацию модели для Triton, включая спецификации входных и выходных тензоров.
4. Развернуть модель: Использовать API Triton для добавления модели в репозиторий и развертывания на сервере.3

# Оптимизация формата моделей и поиск наиболее производительного варианта в Triton Model Navigator
Оптимизация формата моделей и поиск наиболее производительного варианта в Triton Model Navigator включает несколько ключевых этапов, которые направлены на улучшение производительности моделей глубокого обучения и их преобразование в наиболее оптимальные форматы. Вот основные шаги этого процесса:
1. Экспорт модели: Исходная модель, созданная с использованием одного из поддерживаемых фреймворков (например, PyTorch, TensorFlow, ONNX), экспортируется в один из промежуточных форматов, таких как TorchScript, SavedModel или ONNX.1
2. Конвертация модели: Экспортированная модель затем преобразуется в целевое представление с целью достижения наилучшей производительности. Это может включать форматы TorchTensorRT, TensorFlowTensorRT, ONNX и TensorRT.1
3. Тестирование корректности: Для обеспечения корректности полученных моделей Triton Model Navigator выполняет серию тестов на корректность. Эти тесты вычисляют абсолютные и относительные значения допустимых отклонений для исходных и преобразованных моделей.1
4. Профилирование модели: Triton Model Navigator проводит профилирование производительности преобразованных моделей. Этот процесс использует Navigator Runners для выполнения инференса и измерения времени. Цель профилировщика — найти максимальную пропускную способность для каждой модели и вычислить её задержку. Эта информация затем используется для выбора лучших раннеров и предоставления данных о производительности для оптимальной конфигурации.1
5.  Верификация: После завершения профилирования Triton Model Navigator выполняет верификационные тесты для проверки метрик, предоставленных пользователем в verify_func, для всех преобразованных моделей.1
Эти шаги позволяют автоматизировать процесс оптимизации и развертывания моделей, обеспечивая их максимальную производительность на целевом оборудовании.
