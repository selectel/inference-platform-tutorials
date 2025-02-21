# Быстрый старт с одной моделью

Ниже приведены шаги по использованию Model Analyzer в режиме Docker для профилирования и анализа простой модели PyTorch: add_sub.

## `Шаг 1:` Загрузка модели add_sub

---

**1. Создайте новую директорию и перейдите в неё**

```
mkdir <новая_директория> && cd <новая_директория>
```

**2. Инициализируйте git репозиторий**

```
git init && git remote add -f origin https://github.com/triton-inference-server/model_analyzer.git
```

**3. Включите частичную загрузку и скачайте директорию examples, содержащую модель add_sub**

```
git config core.sparseCheckout true && \
echo 'examples' >> .git/info/sparse-checkout && \
git pull origin main
```

## `Шаг 2:` Загрузка и запуск SDK контейнера

---

**1. Загрузите SDK контейнер:**

```
docker pull nvcr.io/nvidia/tritonserver:25.01-py3-sdk
```

**2. Запустите SDK контейнер**

```
docker run -it --gpus all \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v $(pwd)/examples/quick-start:$(pwd)/examples/quick-start \
      --net=host nvcr.io/nvidia/tritonserver:25.01-py3-sdk
```

## `Шаг 3:` Профилирование модели `add_sub`

---

Директория [examples/quick-start](../examples/quick-start) является примером
[репозитория моделей Triton](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md), который содержит простую модель libtorch, вычисляющую сумму и разность двух входных значений.

Запустите подкоманду `profile` Model Analyzer внутри контейнера:

```
model-analyzer profile \
    --model-repository <путь-к-examples-quick-start> \
    --profile-models add_sub --triton-launch-mode=docker \
    --output-model-repository-path <путь-к-выходному-репозиторию-моделей>/<выходная_директория> \
    --export-path profile_results
```

**Важно:** Необходимо указать поддиректорию `<выходная_директория>`. Нельзя указывать `--output-model-repository-path` напрямую на `<путь-к-выходному-репозиторию-моделей>`

**Важно:** Если вы уже запускали это ранее в контейнере, можете использовать опцию `--override-output-model-repository` для перезаписи предыдущих результатов.

**Важно:** Директория контрольных точек должна быть удалена между последовательными запусками команды `model-analyzer profile`.<br><br>

Это выполнит поиск по ограниченному набору настраиваемых параметров модели
`add_sub`. Процесс может занять до 60 минут. Если вы хотите более короткий
запуск (1-2 минуты) для примера, можете добавить следующие опции. Обратите внимание, что эти опции не предназначены для поиска оптимальной конфигурации:

```
--run-config-search-max-concurrency 2 \
--run-config-search-max-model-batch-size 2 \
--run-config-search-max-instance-count 2
```

- `--run-config-search-max-concurrency` устанавливает максимальное значение параллелизма, которое не будет превышено при поиске конфигурации. <br>
- `--run-config-search-max-model-batch-size` устанавливает максимальный размер батча, который не будет превышен при поиске конфигурации.
- `--run-config-search-max-instance-count`
  устанавливает максимальное количество экземпляров группы, которое не будет превышено при поиске конфигурации.<br><br>

С этими опциями model analyzer протестирует 5 конфигураций (4 новых конфигурации, а также неизмененную конфигурацию add_sub по умолчанию), и для каждой конфигурации будет запущено 2 эксперимента в Perf Analyzer (параллелизм=1 и параллелизм=2). Это значительно сокращает пространство поиска и, следовательно, время выполнения model analyzer.

Измеренные данные и итоговый отчет будут помещены в директорию
`./profile_results`. Структура директории будет следующей:

```
$HOME
  |--- model_analyzer
              |--- profile_results
              .       |--- plots
              .       |      |--- simple
              .       |      |      |--- add_sub
                      |      |              |--- gpu_mem_v_latency.png
                      |      |              |--- throughput_v_latency.png
                      |      |
                      |      |--- detailed
                      |             |--- add_sub
                      |                     |--- gpu_mem_v_latency.png
                      |                     |--- throughput_v_latency.png
                      |
                      |--- results
                      |       |--- metrics-model-inference.csv
                      |       |--- metrics-model-gpu.csv
                      |       |--- metrics-server-only.csv
                      |
                      |--- reports
                              |--- summaries
                              .        |--- add_sub
                              .                |--- result_summary.pdf
```

## `Шаг 4:` Создание подробного отчета

---

Подкоманда report Model Analyzer позволяет детально изучить производительность
варианта конфигурации модели. Например, она может показать разбивку задержек вашей
модели, чтобы помочь выявить потенциальные узкие места в производительности
модели.<br><br>
Подробные отчеты также содержат другие настраиваемые графики и
таблицу измерений для конкретной конфигурации. Вы можете сгенерировать
подробный отчет для двух конфигураций модели `add_sub`: `add_sub_config_default` и
`add_sub_config_0`, используя:

```
$ model-analyzer report --report-model-configs add_sub_config_default,add_sub_config_0 -e profile_results
```

Это создаст директории, названные в соответствии с каждой конфигурацией модели, в
`./profile_results/reports/detailed`, содержащие подробные PDF-файлы отчетов, как
показано ниже.

```
$HOME
  |--- model_analyzer
              |--- profile_results
              .       .
              .       .
                      .
                      |--- reports
                              .
                              .
                              .
                              |--- detailed
                                       |--- add_sub_config_default
                                       |        |--- detailed_report.pdf
                                       |
                                       |--- add_sub_config_0
                                                |--- detailed_report.pdf

```