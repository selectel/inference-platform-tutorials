# inference-platform-tutorials
Tutorials for Selectel Inference Platform

Инференс-платформа Selectel — это облачное решение, которое позволяет быстро и эффективно развертывать и управлять API на основе собственных моделей машинного обучения. Платформа предлагает автоматическое масштабирование ресурсов и поддержку Open Source инструментов, таких как NVIDIA Triton™ Inference Server и Istio Ingress Controller. Это позволяет избежать привязки к конкретным вендорам и обеспечивает гибкость в настройке под различные нагрузки.

### Ключевые особенности:
- **Быстрое развертывание**: Позволяет запускать AI-проекты до 3 раз быстрее, сокращая время выхода на рынок.
- **Готовые API**: Возможность получения готового endpoint с ML-моделью за несколько минут.
- **Автоматическое масштабирование**: Платформа автоматически масштабирует ресурсы в зависимости от нагрузки.
- **Высокая производительность**: Использование выделенных GPU-ресурсов и интеграция с NVIDIA Triton™ Inference Server.
- **Круглосуточная поддержка**: Доступ к технической поддержке 24/7.

### Преимущества:
- **Гибкость и независимость**: Использование Open Source решений позволяет избежать vendor lock-in.
- **Экосистема продуктов**: Возможность интеграции с другими продуктами Selectel, такими как Kubernetes, объектное хранилище и CDN.

### Описание репозитория
Этот репозиторий предназначен для обучения клиентов использованию инференс-платформы Selectel. В нем содержатся различные учебные материалы и примеры использования платформы.

- **Папка [`demo`](./demo-inferp)**: Содержит скрипты для запуска трех сценариев использования платформы:
  - Базовый деплой с авторизацией
  - Автоскейлинг
  - Канареечный деплой моделей

- **Папка [`LLM tutorials`](./LLM-tutorials)**: Включает примеры деплоя популярных LLM (Large Language Models) в нашу платформу.

- **Папка [`UI examples`](./UI-examples)**: Приведены примеры интерфейса к нашей платформе через Gradio и Telegram Bot.

- **Папка [`models format tutorial`](./models-format-tutorial)**: Содержит примеры деплоя различных форматов моделей.

Эта платформа идеально подходит для компаний, стремящихся быстро и эффективно внедрять AI-решения, минимизируя затраты на инфраструктуру и техническую поддержку.

