import json
import re
import time
import shutil
import pandas as pd
import ollama
from pathlib import Path
from datetime import datetime


# =====================================================
# НАСТРОЙКА ПАПОК
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Результаты ПЗ5, где лежат вырезанные YOLO-объекты
PZ5_RESULT_ROOT = BASE_DIR / "results" / "pz5_yolo"

# Кадры из ПЗ2, запасной вариант
PZ2_FRAME_ROOT = BASE_DIR / "results" / "pz2_frames" / "FRAME_FOLDER"

# Результаты ПЗ7
RESULT_ROOT = BASE_DIR / "results" / "pz7_llm"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"llm_run_{RUN_TIMESTAMP}"

INPUT_IMAGES_DIR = RUN_DIR / "input_images"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
INPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def seconds_to_time(seconds):
    if seconds is None:
        return ""

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def extract_time_from_filename(filename):
    """
    Достаёт время из имени файла, если оно есть.

    Пример:
    video_frame_00005_time_00005000ms_object_000_person.jpg
    """

    match = re.search(r"time_(\d+)ms", filename)

    if not match:
        return None, ""

    milliseconds = int(match.group(1))
    seconds = milliseconds / 1000

    return round(seconds, 3), seconds_to_time(seconds)


def extract_frame_number_from_filename(filename):
    """
    Достаёт номер кадра из имени файла.
    """

    match = re.search(r"_frame_(\d+)", filename)

    if not match:
        return None

    return int(match.group(1))


def ask_int(message, default_value):
    user_input = input(message).strip()

    if user_input == "":
        return default_value

    try:
        value = int(user_input)

        if value <= 0:
            print("Значение должно быть больше 0. Используется значение по умолчанию.")
            return default_value

        return value

    except ValueError:
        print("Введено некорректное значение. Используется значение по умолчанию.")
        return default_value


# =====================================================
# ПРОВЕРКА OLLAMA
# =====================================================

def check_ollama_available():
    """
    Проверяет, запущена ли Ollama.
    """

    try:
        ollama.list()
        return True
    except Exception:
        print("\nOllama не отвечает.")
        print("Проверь, что Ollama установлена и запущена.")
        print("Можно попробовать открыть отдельный терминал и выполнить:")
        print("ollama serve")
        return False


def choose_ollama_model():
    """
    Выбор локальной vision-модели.
    По умолчанию используется Qwen2.5-VL.
    """

    print("\nВыберите LLM-модель:")
    print("1 — qwen2.5vl:latest, основной вариант")
    print("2 — llava:7b, запасной вариант")
    print("3 — ввести название модели вручную")

    choice = input("\nВведите 1, 2 или 3 (Enter = qwen2.5vl:latest): ").strip()

    if choice == "":
        return "qwen2.5vl:latest"

    if choice == "1":
        return "qwen2.5vl:latest"

    if choice == "2":
        return "llava:7b"

    if choice == "3":
        model_name = input("Введите название модели: ").strip()

        if model_name:
            return model_name

    print("Неверный выбор. Используется qwen2.5vl:latest.")
    return "qwen2.5vl:latest"


# =====================================================
# ВЫБОР ИСТОЧНИКА ИЗОБРАЖЕНИЙ
# =====================================================

def choose_source_mode():
    print("\nЧто отправлять в LLM?")
    print("1 — вырезанные объекты из ПЗ5 object_crops")
    print("2 — целые кадры из ПЗ2 FRAME_FOLDER")

    choice = input("\nВведите 1 или 2 (Enter = 1): ").strip()

    if choice == "":
        choice = "1"

    if choice == "1":
        return "pz5_object_crops"

    if choice == "2":
        return "pz2_full_frames"

    print("Неверный выбор. Используются object_crops из ПЗ5.")
    return "pz5_object_crops"


def choose_pz5_object_crops_folder():
    """
    Выбор папки object_crops из результатов ПЗ5.
    """

    if not PZ5_RESULT_ROOT.exists():
        print("Папка результатов ПЗ5 не найдена:")
        print(PZ5_RESULT_ROOT)
        return None

    yolo_runs = [
        folder for folder in sorted(PZ5_RESULT_ROOT.iterdir())
        if folder.is_dir() and folder.name.startswith("yolo_run_")
    ]

    if not yolo_runs:
        print("В results/pz5_yolo нет запусков yolo_run.")
        return None

    available_runs = []

    for run_folder in yolo_runs:
        crops_folder = run_folder / "object_crops"

        if not crops_folder.exists():
            continue

        images = [
            file for file in crops_folder.iterdir()
            if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        if images:
            available_runs.append((run_folder, crops_folder, len(images)))

    if not available_runs:
        print("В запусках ПЗ5 не найдено object_crops с изображениями.")
        return None

    print("\nНайдены папки object_crops из ПЗ5:")

    for index, item in enumerate(available_runs, start=1):
        run_folder, crops_folder, image_count = item
        print(f"{index}. {run_folder.name} — объектов: {image_count}")

    choice = input("\nВведите номер запуска ПЗ5: ").strip()

    try:
        choice_number = int(choice)

        if 1 <= choice_number <= len(available_runs):
            return available_runs[choice_number - 1][1]

        print("Неверный номер.")
        return None

    except ValueError:
        print("Введено не число.")
        return None


def choose_pz2_frames_folder():
    """
    Выбор папки с целыми кадрами из ПЗ2.
    """

    if not PZ2_FRAME_ROOT.exists():
        print("Папка кадров ПЗ2 не найдена:")
        print(PZ2_FRAME_ROOT)
        return None

    frame_folders = [
        folder for folder in sorted(PZ2_FRAME_ROOT.iterdir())
        if folder.is_dir()
    ]

    if not frame_folders:
        print("В FRAME_FOLDER нет папок с кадрами.")
        return None

    print("\nНайдены папки с кадрами ПЗ2:")

    for index, folder in enumerate(frame_folders, start=1):
        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

        print(f"{index}. {folder.name} — кадров: {len(images)}")

    choice = input("\nВведите номер папки с кадрами: ").strip()

    try:
        choice_number = int(choice)

        if 1 <= choice_number <= len(frame_folders):
            return frame_folders[choice_number - 1]

        print("Неверный номер.")
        return None

    except ValueError:
        print("Введено не число.")
        return None


def choose_images_folder():
    source_mode = choose_source_mode()

    if source_mode == "pz5_object_crops":
        images_folder = choose_pz5_object_crops_folder()
    else:
        images_folder = choose_pz2_frames_folder()

    return source_mode, images_folder


# =====================================================
# ПРОМПТ ДЛЯ LLM
# =====================================================

def build_prompt():
    """
    Промпт просит модель вернуть строго JSON.
    Это нужно, чтобы результат потом можно было использовать в общем пайплайне.
    """

    prompt = """
Ты анализируешь изображение из видеоролика для системы автоматического анализа видеоконтента.

Нужно внимательно описать, что видно на изображении, и вернуть результат строго в JSON.

Не добавляй Markdown.
Не добавляй текст до или после JSON.
Не используй ```json.

Формат ответа:

{
  "short_description_ru": "краткое описание изображения на русском языке",
  "scene_type": "человек / помещение / улица / предмет / экран / транспорт / животное / документ / другое",
  "main_objects": [
    {
      "object_name_ru": "название объекта на русском",
      "object_name_en": "название объекта на английском",
      "visual_confidence": "низкая / средняя / высокая"
    }
  ],
  "people_present": "да / нет / неясно",
  "text_or_symbols_visible": "описание видимого текста, символов, логотипов или надписей",
  "visual_context_ru": "что происходит на изображении, если это можно понять",
  "possible_risk_flags": [
    "нейтрально"
  ],
  "moderation_comment_ru": "краткий комментарий о визуальных признаках. Если опасных или подозрительных признаков нет, написать нейтрально."
}

Важно:
- Не делай юридический вывод.
- Не утверждай, что контент запрещён.
- Не придумывай то, чего нет на изображении.
- Только опиши визуальные признаки.
- Если изображение маленькое, размытое или неясное, так и напиши.
- Если признаков риска нет, в possible_risk_flags укажи только 'нейтрально'.
"""
    return prompt.strip()


def extract_json_from_response(text):
    """
    Достаёт JSON из ответа модели.
    Иногда модель добавляет лишний текст — пробуем аккуратно вытащить JSON.
    """

    if text is None:
        return None

    text = text.strip()
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start:end + 1]

        try:
            return json.loads(json_text)
        except Exception:
            return None

    return None


def analyze_image_with_ollama(model_name, image_path):
    """
    Отправляет изображение в локальную Ollama vision-модель.
    """

    prompt = build_prompt()

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(image_path)]
                }
            ],
            options={
                "temperature": 0
            }
        )

        if isinstance(response, dict):
            response_text = response.get("message", {}).get("content", "")
        else:
            response_text = response.message.content

    except Exception as error:
        return {
            "success": False,
            "raw_response": "",
            "parsed_json": None,
            "error": str(error)
        }

    parsed_json = extract_json_from_response(response_text)

    if parsed_json is None:
        return {
            "success": False,
            "raw_response": response_text,
            "parsed_json": None,
            "error": "Не удалось разобрать JSON из ответа модели"
        }

    return {
        "success": True,
        "raw_response": response_text,
        "parsed_json": parsed_json,
        "error": ""
    }


# =====================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# =====================================================

def process_images_with_llm(images_folder, source_mode, model_name, max_images):
    image_files = [
        file for file in sorted(images_folder.iterdir())
        if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("В выбранной папке нет изображений.")
        return []

    if max_images > 0:
        image_files = image_files[:max_images]

    print(f"\nБудет обработано изображений: {len(image_files)}")
    print(f"Модель: {model_name}")

    rows = []

    for index, image_path in enumerate(image_files, start=1):
        print(f"LLM-анализ {index}/{len(image_files)}: {image_path.name}")

        copied_image_path = INPUT_IMAGES_DIR / image_path.name

        try:
            shutil.copy2(image_path, copied_image_path)
        except Exception:
            copied_image_path = ""

        frame_number = extract_frame_number_from_filename(image_path.name)
        time_seconds, time_formatted = extract_time_from_filename(image_path.name)

        result = analyze_image_with_ollama(
            model_name,
            image_path
        )

        parsed = result.get("parsed_json") or {}

        main_objects = parsed.get("main_objects", [])

        if isinstance(main_objects, list):
            main_objects_text = "; ".join([
                str(item.get("object_name_ru", ""))
                for item in main_objects
                if isinstance(item, dict)
            ])
        else:
            main_objects_text = str(main_objects)

        risk_flags = parsed.get("possible_risk_flags", [])

        if isinstance(risk_flags, list):
            risk_flags_text = "; ".join([str(item) for item in risk_flags])
        else:
            risk_flags_text = str(risk_flags)

        rows.append({
            "image_file": image_path.name,
            "image_path": str(image_path),
            "copied_image_path": str(copied_image_path),
            "source_mode": source_mode,
            "model_name": model_name,
            "frame_number": frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "success": result.get("success"),
            "short_description_ru": parsed.get("short_description_ru", ""),
            "scene_type": parsed.get("scene_type", ""),
            "main_objects_text": main_objects_text,
            "people_present": parsed.get("people_present", ""),
            "text_or_symbols_visible": parsed.get("text_or_symbols_visible", ""),
            "visual_context_ru": parsed.get("visual_context_ru", ""),
            "possible_risk_flags": risk_flags_text,
            "moderation_comment_ru": parsed.get("moderation_comment_ru", ""),
            "raw_response": result.get("raw_response", ""),
            "error": result.get("error", "")
        })

        time.sleep(0.5)

    return rows


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def build_scene_summary(rows):
    if not rows:
        return []

    df = pd.DataFrame(rows)

    if "scene_type" not in df.columns:
        return []

    summary_df = (
        df.groupby("scene_type")
        .agg(
            images_count=("scene_type", "count")
        )
        .reset_index()
        .sort_values("images_count", ascending=False)
    )

    return summary_df.to_dict(orient="records")


def build_risk_summary(rows):
    if not rows:
        return []

    risk_counter = {}

    for row in rows:
        risk_text = row.get("possible_risk_flags", "")

        if not risk_text:
            continue

        flags = [flag.strip() for flag in risk_text.split(";") if flag.strip()]

        for flag in flags:
            risk_counter[flag] = risk_counter.get(flag, 0) + 1

    summary = [
        {
            "risk_flag": flag,
            "count": count
        }
        for flag, count in risk_counter.items()
    ]

    summary = sorted(summary, key=lambda item: item["count"], reverse=True)

    return summary


def save_results(images_folder, source_mode, model_name, max_images, rows):
    excel_path = RUN_DIR / "llm_image_analysis_results.xlsx"
    json_path = RUN_DIR / "llm_image_analysis_report.json"
    summary_txt_path = RUN_DIR / "llm_summary.txt"

    scene_summary = build_scene_summary(rows)
    risk_summary = build_risk_summary(rows)

    results_df = pd.DataFrame(rows)
    scene_summary_df = pd.DataFrame(scene_summary)
    risk_summary_df = pd.DataFrame(risk_summary)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="llm_results", index=False)
        scene_summary_df.to_excel(writer, sheet_name="scene_summary", index=False)
        risk_summary_df.to_excel(writer, sheet_name="risk_summary", index=False)

    json_report = {
        "report_type": "LLM_IMAGE_ANALYSIS_REPORT",
        "model_provider": "Ollama local vision model",
        "model_name": model_name,
        "source_mode": source_mode,
        "images_folder": str(images_folder),
        "max_images": max_images,
        "processed_images_count": len(rows),
        "successful_responses": sum(1 for row in rows if row.get("success")),
        "scene_summary": scene_summary,
        "risk_summary": risk_summary,
        "results": rows,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_report, file, ensure_ascii=False, indent=4)

    with open(summary_txt_path, "w", encoding="utf-8") as file:
        file.write("ПЗ7. LLM-анализ изображений\n\n")
        file.write(f"Источник изображений: {images_folder}\n")
        file.write(f"Модель: {model_name}\n")
        file.write(f"Обработано изображений: {len(rows)}\n")
        file.write(f"Успешных ответов: {sum(1 for row in rows if row.get('success'))}\n\n")

        file.write("Краткие описания:\n")

        for row in rows:
            description = row.get("short_description_ru", "")

            if description:
                file.write(f"- {row.get('image_file')}: {description}\n")

    print("\nГотово.")
    print(f"Папка результатов ПЗ7: {RUN_DIR}")
    print(f"Excel-таблица: {excel_path}")
    print(f"JSON-отчёт: {json_path}")
    print(f"Краткая сводка TXT: {summary_txt_path}")
    print(f"Копии входных изображений: {INPUT_IMAGES_DIR}")

    print("\nСтатистика:")
    print(f"Обработано изображений: {len(rows)}")
    print(f"Успешных ответов LLM: {sum(1 for row in rows if row.get('success'))}")


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ7: распознавание и описание объектов с помощью LLM")
    print("=" * 70)

    print("\nИспользуется локальная vision-LLM через Ollama.")
    print("Основная модель по умолчанию: qwen2.5vl:latest")
    print("API-ключи не требуются.")

    if not check_ollama_available():
        return

    model_name = choose_ollama_model()

    source_mode, images_folder = choose_images_folder()

    if images_folder is None:
        return

    max_images = ask_int(
        "\nСколько изображений обработать? "
        "(например 10; Enter = 10; 9999 = почти все): ",
        10
    )

    rows = process_images_with_llm(
        images_folder,
        source_mode,
        model_name,
        max_images
    )

    save_results(
        images_folder,
        source_mode,
        model_name,
        max_images,
        rows
    )


if __name__ == "__main__":
    main()