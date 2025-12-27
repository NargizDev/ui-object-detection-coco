# Детекция UI элементов на скриншотах (YOLO + Faster R-CNN), в стиле автолейблинга CVAT

Этот репозиторий закрывает практическую часть под твою статью:
- берем COCO разметку из CVAT
- готовим датасет (COCO split + YOLO labels)
- обучаем 2 модели: YOLO и Faster R-CNN
- прогоняем инференс и получаем предсказания в COCO JSON
- визуализируем рамки на скриншотах
- публикуем датасет на Hugging Face

## 0) Что уже есть
В `data/raw/instances_coco.json` лежит твой COCO экспорт из CVAT (файл добавлен как пример структуры).
Картинки в репо не кладутся. Их нужно положить локально.

Ожидаемая структура входных данных:
```
data/raw/
  images/
    cv1.png
    cv2.png
    ...
  instances_coco.json
```

## 1) Установка
Python 3.10+.

Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Torch и TorchVision ставь по инструкции под свою систему (CPU или CUDA):
https://pytorch.org/get-started/locally/

## 2) Подготовка датасета (split + YOLO формат)
Команда:
```
python scripts/prepare_dataset.py ^
  --coco data/raw/instances_coco.json ^
  --images data/raw/images ^
  --out data/processed ^
  --train-ratio 0.8 ^
  --seed 42
```

Результат:
```
data/processed/
  coco/
    train.json
    val.json
  yolo/
    ui.yaml
    images/train/, images/val/
    labels/train/, labels/val/
  dataset_stats.json
```

## 3) Обучение YOLO
Минимально:
```
python training/train_yolo.py --data data/processed/yolo/ui.yaml --model yolov8n.pt --epochs 50 --imgsz 1280
```

После обучения веса обычно здесь:
`runs/detect/train/weights/best.pt`

## 4) Обучение Faster R-CNN
Минимально (CPU):
```
python training/train_fasterrcnn.py ^
  --train data/processed/coco/train.json ^
  --val data/processed/coco/val.json ^
  --images data/raw/images ^
  --epochs 10 ^
  --batch 2 ^
  --device cpu
```

Чекпойнт:
`outputs/fasterrcnn/model.pt`

## 5) Инференс и экспорт в COCO predictions
YOLO:
```
python inference/predict.py --model yolo --weights runs/detect/train/weights/best.pt --images data/raw/images --out outputs/yolo_pred.json
```

Faster R-CNN:
```
python inference/predict.py --model fasterrcnn --weights outputs/fasterrcnn/model.pt --images data/raw/images --out outputs/frcnn_pred.json
```

## 6) Визуализация рамок
```
python inference/visualize.py --images data/raw/images --pred outputs/yolo_pred.json --out-dir outputs/vis_yolo
```

## 6.1) Постобработка для иерархии и анализа изменений
Для улучшения модели для QA-тестирования UI добавлены скрипты постобработки:

- Построение иерархии элементов на основе вложенности bounding boxes:
```
python inference/postprocess.py --pred outputs/frcnn_pred.json --out outputs/frcnn_hierarchical.json
```

- Визуализация с иерархией (цвета по уровням):
```
python inference/visualize.py --images data/raw/images --pred outputs/frcnn_hierarchical.json --out-dir outputs/vis_frcnn_hier
```

- Сравнение двух состояний интерфейса (baseline vs current):
```
python inference/compare.py --baseline outputs/baseline_hierarchical.json --current outputs/current_hierarchical.json --out outputs/diff.json
```

Формат вывода теперь включает:
- Иерархию элементов (children)
- Семантическую группировку (group: interactive, text, layout, etc.)
- Интерпретируемый JSON для анализа изменений

## 6.2) Полный анализ изменений UI
Для автоматизированного анализа изменений между двумя состояниями интерфейса:
```
python inference/ui_analysis.py --baseline-images data/baseline/images --current-images data/current/images --model fasterrcnn --weights outputs/fasterrcnn/model.pt --out-dir outputs/analysis
```

Это запустит полный пайплайн: предсказание, постобработку, сравнение и визуализацию.
Сопоставление baseline/current идет по имени файла (например, `cv4.png`), поэтому в папках может быть разное число скриншотов.

## 7) Загрузка датасета на Hugging Face
Сначала логин:
```
huggingface-cli login
```

Потом:
```
python hf/push_dataset.py --dataset_id <username>/ui-screenshots-coco --images data/raw/images --coco data/raw/instances_coco.json --private 1
```

## 8) Как это приближено к CVAT
CVAT автолейблинг по сути делает то же: модель дает рамки и классы, дальше CVAT превращает это в аннотации.
Здесь мы сохраняем предсказания в COCO JSON (bbox + category + score).
Такой JSON можно:
- анализировать как результаты модели
- импортировать в пайплайн разметки/проверки
- использовать как основу для полуавтоматической разметки

Важно: текущий пилотный набор маленький. Он годится для демонстрации пайплайна, но не для стабильных метрик.

## 9) Подробная инструкция по использованию модели

### Введение

Модель предназначена для автоматизированного обнаружения и идентификации элементов пользовательского интерфейса (UI) на скриншотах веб-страниц. Основная область применения - QA-тестирование, где модель сравнивает текущее состояние интерфейса с эталонным, выявляя изменения в структуре и отображении элементов.

**Ключевые улучшения по сравнению с базовой версией:**
- **Иерархия элементов**: Элементы организованы в дерево на основе вложенности (большие элементы содержат меньшие).
- **Семантическая группировка**: Элементы разделены по типам (интерактивные, текст, иконки и т.д.).
- **Анализ изменений**: Автоматическое сравнение двух состояний интерфейса с выявлением добавленных/удаленных/измененных элементов.
- **Интерпретируемый формат**: JSON с древовидной структурой для легкого анализа.

Модель поддерживает две архитектуры: YOLOv8 (быстрая, для реального времени) и Faster R-CNN (точная, для детального анализа).

### Как работает модель

#### Общий пайплайн:
1. **Подготовка данных**: Конвертация COCO-аннотаций из CVAT в форматы для тренировки.
2. **Тренировка**: Обучение моделей на размеченных данных.
3. **Инференс**: Предсказание элементов на новых скриншотах.
4. **Постобработка**: Построение иерархии и семантической группировки.
5. **Анализ изменений**: Сравнение двух наборов предсказаний.
6. **Визуализация**: Рисование bounding boxes с учетом иерархии.

#### Категории элементов:
- 1: button (кнопка)
- 2: input (поле ввода)
- 3: dropdown (выпадающий список)
- 4: checkbox (чекбокс)
- 5: icon (иконка)
- 6: image (изображение)
- 7: text (текст)
- 8: card (карточка)
- 9: navbar (навигационная панель)

#### Семантические группы:
- `interactive`: кнопки, поля ввода, чекбоксы, выпадающие списки
- `text`: текстовые элементы
- `icon`: иконки
- `media`: изображения
- `layout`: карточки, навигационные панели
- `other`: прочие

### Как пользоваться моделью

#### 3.1. Установка и подготовка

1. **Клонируйте репозиторий и установите зависимости:**
   ```
   
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   pip install -r requirements.txt
   ```

2. **Установите PyTorch** (согласно вашей системе: CPU/CUDA).

3. **Подготовьте данные:**
   - Поместите скриншоты в `data/raw/images/`.
   - Экспортируйте аннотации из CVAT в `data/raw/instances_coco.json`.
   - Запустите подготовку датасета:
     ```
     python scripts/prepare_dataset.py --coco data/raw/instances_coco.json --images data/raw/images --out data/processed
     ```

#### 3.2. Тренировка модели

- **Faster R-CNN:**
  ```
  python training/train_fasterrcnn.py --train data/processed/coco/train.json --val data/processed/coco/val.json --images data/raw/images --epochs 10 --batch 2 --device cpu
  ```
  Модель сохранится в `outputs/fasterrcnn/model.pt`.

- **YOLOv8:**
  ```
  python training/train_yolo.py --data data/processed/yolo/ui.yaml --epochs 10 --batch 2 --device cpu
  ```
  Модель сохранится в `runs/detect/train/weights/best.pt`.

#### 3.3. Инференс (предсказание на новых изображениях)

- **Faster R-CNN:**
  ```
  python inference/predict.py --model fasterrcnn --weights outputs/fasterrcnn/model.pt --images data/raw/images --out outputs/frcnn_pred.json --conf 0.25
  ```

- **YOLO:**
  ```
  python inference/predict.py --model yolo --weights runs/detect/train/weights/best.pt --images data/raw/images --out outputs/yolo_pred.json --conf 0.25
  ```

Вывод: COCO-формат JSON с bounding boxes, категориями и score.

#### 3.4. Постобработка (построение иерархии)

```
python inference/postprocess.py --pred outputs/frcnn_pred.json --out outputs/frcnn_hierarchical.json
```

Вывод: JSON с древовидной структурой, где каждый элемент имеет `children`, `group` и другие поля.

#### 3.5. Визуализация

- **Простая визуализация:**
  ```
  python inference/visualize.py --images data/raw/images --pred outputs/frcnn_pred.json --out-dir outputs/vis_frcnn
  ```

- **Визуализация с иерархией (цвета по уровням):**
  ```
  python inference/visualize.py --images data/raw/images --pred outputs/frcnn_hierarchical.json --out-dir outputs/vis_frcnn_hier
  ```

#### 3.6. Анализ изменений

- **Сравнение двух состояний:**
  ```
  python inference/compare.py --baseline outputs/baseline_hierarchical.json --current outputs/current_hierarchical.json --out outputs/diff.json
  ```

- **Полный анализ (предсказание + постобработка + сравнение + визуализация):**
  ```
  python inference/ui_analysis.py --baseline-images path/to/baseline/images --current-images path/to/current/images --model fasterrcnn --weights outputs/fasterrcnn/model.pt --out-dir outputs/analysis
  ```

#### 3.7. Публикация датасета (опционально)

```
huggingface-cli login
python hf/push_dataset.py --dataset_id username/ui-screenshots-coco --images data/raw/images --coco data/raw/instances_coco.json --private 1
```

### Что ожидать от модели

#### Форматы вывода

- **COCO predictions (predict.py):** Плоский список объектов с `image_id`, `category_id`, `bbox` (xywh), `score`.
- **Hierarchical JSON (postprocess.py):** Дерево по `image_id`, каждый узел имеет `children` (список дочерних элементов), `group` (семантическая группа).
- **Diff JSON (compare.py):** По `image_id`: `added`, `removed`, `changed` (с рекурсивным сравнением детей).

#### Примеры

- **Иерархический элемент:**
  ```json
  {
    "image_id": 1,
    "category_id": 5,
    "bbox": [100, 50, 200, 100],
    "score": 0.95,
    "group": "icon",
    "children": [
      {
        "image_id": 1,
        "category_id": 7,
        "bbox": [110, 60, 80, 20],
        "score": 0.88,
        "group": "text",
        "children": []
      }
    ]
  }
  ```

- **Diff:**
  ```json
  {
    "1": {
      "added": [{"category_id": 1, "bbox": [10, 10, 50, 20], "group": "interactive"}],
      "removed": [],
      "changed": []
    }
  }
  ```

#### Ожидаемая точность
- На тренировочных данных: mAP 0.7-0.9 (зависит от качества аннотаций).
- В реальных условиях: Высокая точность для четких UI-элементов, возможны ложные срабатывания на сложных фонах.
- Иерархия: Автоматическая, но не идеальная (простая вложенность по bbox).

#### Ограничения
- Работает только на скриншотах (не на живых страницах).
- Требует качественных аннотаций для тренировки.
- Не распознает текст внутри элементов (только bounding boxes).
- Для анализа изменений нужны идентичные условия съемки скриншотов.
- Не обрабатывает динамические элементы (анимации, hover-состояния).

#### Советы по использованию
- Используйте `--min-score` для фильтрации слабых предсказаний.
- Для QA: Сравнивайте эталонные скриншоты с текущими после обновлений.
- Визуализируйте результаты для ручной проверки.
- Тренируйте модель на данных, близких к вашим UI (сайты, приложения).

Если возникнут вопросы или нужны доработки, дайте знать!
