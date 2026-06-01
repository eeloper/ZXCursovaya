import base64
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

from project_config import BASE_DIR, load_run_config, RUN_CONFIG_PATH


# =====================================================
# НАСТРОЙКИ
# =====================================================

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

OLLAMA_URL = "http://localhost:11434/api/generate"

DEFAULT_MODEL = "moondream"
DEFAULT_MAX_IMAGES = 5

TAXONOMY_PATH = BASE_DIR / "risk_taxonomy.json"


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def save_run_config(config):
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def clear_folder(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists():
        shutil.rmtree(folder_path)

    folder_path.mkdir(parents=True, exist_ok=True)


def seconds_to_time(seconds):
    if seconds is None or seconds == "":
        return ""

    try:
        seconds = int(float(seconds))
    except Exception:
        return ""

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("ё", "е")

    replacements = {
        "a": "а",
        "c": "с",
        "e": "е",
        "o": "о",
        "p": "р",
        "x": "х",
        "y": "у",
        "k": "к",
        "m": "м",
        "t": "т",
        "b": "в",
        "h": "н",
        "u": "и",
        "i": "и",
    }

    for latin, cyrillic in replacements.items():
        text = text.replace(latin, cyrillic)

    text = re.sub(r"[^a-zа-я0-9 ,]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def extract_time_from_filename(filename):
    match = re.search(r"time_(\d+)ms", filename)

    if not match:
        return 0.0, "00:00:00"

    milliseconds = int(match.group(1))
    seconds = milliseconds / 1000

    return round(seconds, 3), seconds_to_time(seconds)


def extract_frame_number_from_filename(filename):
    match = re.search(r"frame_(\d+)", filename)

    if not match:
        return 0

    return int(match.group(1))


def extract_source_frame_number_from_filename(filename):
    match = re.search(r"source_(\d+)", filename)

    if not match:
        return 0

    return int(match.group(1))


def extract_yolo_class_from_filename(filename):
    """
    Пример:
    object_000013_frame_000009_source_000263_time_00009055ms_person.jpg

    Вернёт:
    person
    """

    stem = Path(filename).stem

    match = re.search(r"time_\d+ms_(.+)$", stem)

    if match:
        return match.group(1)

    parts = stem.split("_")

    if parts:
        return parts[-1]

    return ""


def get_image_files(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        return []

    files = [
        file for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(files)


def image_to_base64(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return encoded


# =====================================================
# РИСК-ТАКСОНОМИЯ
# =====================================================

def load_risk_taxonomy():
    if not TAXONOMY_PATH.exists():
        print("\nФайл risk_taxonomy.json не найден.")
        print("Проверь, что он лежит в корне проекта.")
        return {
            "taxonomy_name": "missing_taxonomy",
            "taxonomy_version": "0.0",
            "categories": {}
        }

    try:
        with open(TAXONOMY_PATH, "r", encoding="utf-8") as file:
            taxonomy = json.load(file)

        return taxonomy

    except Exception as error:
        print("\nОшибка чтения risk_taxonomy.json:")
        print(error)

        return {
            "taxonomy_name": "broken_taxonomy",
            "taxonomy_version": "0.0",
            "categories": {}
        }


def flatten_terms(term_block):
    terms = []

    if isinstance(term_block, list):
        return term_block

    if isinstance(term_block, dict):
        for values in term_block.values():
            if isinstance(values, list):
                terms.extend(values)

    return terms


def build_keyword_dict(taxonomy):
    keyword_dict = {}

    categories = taxonomy.get("categories", {})

    for category_name, category_data in categories.items():
        terms = []

        terms.extend(flatten_terms(category_data.get("strong_terms", {})))
        terms.extend(flatten_terms(category_data.get("weak_terms", {})))
        terms.extend(category_data.get("model_labels", []))

        clean_terms = []

        for term in terms:
            term = clean_text(term)

            if term and term not in clean_terms:
                clean_terms.append(term)

        keyword_dict[category_name] = clean_terms

    return keyword_dict


def detect_risk_flags_from_taxonomy(text, yolo_class, keyword_dict):
    text_norm = normalize_text(text)
    yolo_class_norm = normalize_text(yolo_class)

    combined = f"{text_norm} {yolo_class_norm}"

    detected_categories = []

    for category_name, keywords in keyword_dict.items():
        for keyword in keywords:
            keyword_norm = normalize_text(keyword)

            if keyword_norm and keyword_norm in combined:
                detected_categories.append(category_name)
                break

    detected_categories = list(sorted(set(detected_categories)))

    if not detected_categories:
        return "none"

    return ",".join(detected_categories)


# =====================================================
# ВЫБОР МОДЕЛИ И РЕЖИМА
# =====================================================

def choose_llm_model():
    print("\nВыберите режим LLM-анализа:")
    print("1 — быстрый режим: moondream")
    print("2 — стандартный режим: llava:7b")
    print("3 — качественный режим: qwen2.5vl")
    print("4 — ввести модель вручную")
    print("Enter — быстрый режим: moondream")

    choice = input("\nВведите номер: ").strip()

    if choice == "" or choice == "1":
        return "moondream", "fast"

    if choice == "2":
        return "llava:7b", "balanced"

    if choice == "3":
        return "qwen2.5vl", "quality"

    if choice == "4":
        model_name = input("Введите название модели Ollama: ").strip()

        if model_name:
            return model_name, "custom"

    print("Неверный выбор. Используется moondream.")
    return DEFAULT_MODEL, "fast"


def choose_max_images(total_images):
    print("\nСколько изображений отправить в LLM?")
    print("Для быстрой демонстрации лучше 3–5.")
    print("Enter — 5")

    user_input = input("\nКоличество изображений: ").strip()

    if user_input == "":
        return min(DEFAULT_MAX_IMAGES, total_images)

    try:
        value = int(user_input)

        if value <= 0:
            print("Количество должно быть больше 0. Используется 5.")
            return min(DEFAULT_MAX_IMAGES, total_images)

        return min(value, total_images)

    except ValueError:
        print("Некорректное значение. Используется 5.")
        return min(DEFAULT_MAX_IMAGES, total_images)


# =====================================================
# OLLAMA
# =====================================================

def check_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def build_prompt(yolo_class, taxonomy):
    categories = taxonomy.get("categories", {})
    category_names = list(categories.keys())

    category_text = ", ".join(category_names) if category_names else "weapon, violence"

    prompt = f"""
You are an image moderation assistant.

The image is an object crop extracted from a video by YOLO.
YOLO class hint: "{yolo_class}"

The project uses a risk taxonomy with these categories:
{category_text}

Analyze the image and return ONLY valid JSON.
Do not write explanations outside JSON.
Do not use markdown.
Do not use ```json.

Use this exact JSON schema:

{{
  "short_description_ru": "brief description in Russian",
  "scene_type": "object/person/weapon/violence/unknown",
  "main_objects_text": "main visible objects, comma-separated",
  "people_present": "yes/no/unknown",
  "text_or_symbols_visible": "visible text or symbols, or none",
  "visual_context_ru": "short visual context in Russian",
  "possible_risk_flags": "weapon/violence/weapon,violence/none",
  "moderation_comment_ru": "brief moderation comment in Russian"
}}

Rules:
- If the image contains a risk marker related to weapons, use possible_risk_flags = "weapon".
- If the image contains a risk marker related to violence, use possible_risk_flags = "violence".
- If both are present, use possible_risk_flags = "weapon,violence".
- If there are no visible risk signs, use possible_risk_flags = "none".
- If you are uncertain, write "unknown" in unclear fields, but still return valid JSON.
"""
    return prompt.strip()


def call_ollama_vision(model_name, image_path, yolo_class, taxonomy):
    image_b64 = image_to_base64(image_path)

    payload = {
        "model": model_name,
        "prompt": build_prompt(yolo_class, taxonomy),
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama вернула ошибку {response.status_code}: {response.text}"
        )

    data = response.json()

    return data.get("response", "")


# =====================================================
# ПАРСИНГ И НОРМАЛИЗАЦИЯ ОТВЕТА LLM
# =====================================================

def extract_json_from_text(text):
    if text is None:
        return {}

    text = str(text).strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        json_text = match.group(0)

        try:
            return json.loads(json_text)
        except Exception:
            return {}

    return {}


def guess_scene_type(yolo_class, risk_flags):
    yolo_class = clean_text(yolo_class).lower()

    if "weapon" in risk_flags:
        return "weapon"

    if "violence" in risk_flags:
        return "violence"

    if yolo_class in ["person", "man", "woman", "boy", "girl"]:
        return "person"

    if yolo_class:
        return "object"

    return "unknown"


def build_fallback_result(raw_response, yolo_class, keyword_dict):
    raw_response = clean_text(raw_response)
    yolo_class = clean_text(yolo_class)

    risk_flags = detect_risk_flags_from_taxonomy(
        text=raw_response,
        yolo_class=yolo_class,
        keyword_dict=keyword_dict
    )

    scene_type = guess_scene_type(yolo_class, risk_flags)

    if raw_response:
        description = raw_response
    else:
        description = f"Обнаружен объект класса {yolo_class}" if yolo_class else "Описание изображения не получено"

    people_present = "yes" if yolo_class.lower() == "person" else "unknown"

    return {
        "short_description_ru": description,
        "scene_type": scene_type,
        "main_objects_text": yolo_class if yolo_class else "unknown",
        "people_present": people_present,
        "text_or_symbols_visible": "none",
        "visual_context_ru": description,
        "possible_risk_flags": risk_flags,
        "moderation_comment_ru": "Требуется ручная проверка, если объект относится к риск-маркерам."
    }


def normalize_possible_risk_flags(value, raw_response, yolo_class, keyword_dict):
    value = clean_text(value).lower()
    value = value.replace(" ", "")

    allowed = [
        "weapon",
        "violence",
        "weapon,violence",
        "violence,weapon",
        "none"
    ]

    if value in allowed:
        if value == "violence,weapon":
            return "weapon,violence"

        return value

    return detect_risk_flags_from_taxonomy(
        text=raw_response,
        yolo_class=yolo_class,
        keyword_dict=keyword_dict
    )


def normalize_llm_result(raw_response, yolo_class, keyword_dict):
    parsed = extract_json_from_text(raw_response)

    if not parsed:
        return build_fallback_result(raw_response, yolo_class, keyword_dict)

    result = {
        "short_description_ru": clean_text(parsed.get("short_description_ru", "")),
        "scene_type": clean_text(parsed.get("scene_type", "")),
        "main_objects_text": clean_text(parsed.get("main_objects_text", "")),
        "people_present": clean_text(parsed.get("people_present", "")),
        "text_or_symbols_visible": clean_text(parsed.get("text_or_symbols_visible", "")),
        "visual_context_ru": clean_text(parsed.get("visual_context_ru", "")),
        "possible_risk_flags": normalize_possible_risk_flags(
            parsed.get("possible_risk_flags", ""),
            raw_response,
            yolo_class,
            keyword_dict
        ),
        "moderation_comment_ru": clean_text(parsed.get("moderation_comment_ru", ""))
    }

    fallback = build_fallback_result(raw_response, yolo_class, keyword_dict)

    for key, value in result.items():
        if value == "":
            result[key] = fallback[key]

    return result


# =====================================================
# ОТБОР ИЗОБРАЖЕНИЙ
# =====================================================

def select_images_for_llm(object_crops_dir, max_images):
    image_files = get_image_files(object_crops_dir)

    if not image_files:
        return []

    if max_images >= len(image_files):
        return image_files

    if max_images == 1:
        return [image_files[0]]

    step = max(1, len(image_files) // max_images)
    selected = image_files[::step][:max_images]

    return selected


# =====================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# =====================================================

def analyze_images_with_llm(image_files, model_name, taxonomy, keyword_dict):
    rows = []

    for index, image_path in enumerate(image_files, start=1):
        print(f"LLM-анализ {index}/{len(image_files)}: {image_path.name}")

        time_seconds, time_formatted = extract_time_from_filename(image_path.name)
        frame_number = extract_frame_number_from_filename(image_path.name)
        source_frame_number = extract_source_frame_number_from_filename(image_path.name)
        yolo_class = extract_yolo_class_from_filename(image_path.name)

        try:
            raw_response = call_ollama_vision(
                model_name=model_name,
                image_path=image_path,
                yolo_class=yolo_class,
                taxonomy=taxonomy
            )

            normalized = normalize_llm_result(
                raw_response=raw_response,
                yolo_class=yolo_class,
                keyword_dict=keyword_dict
            )

            success = True
            error = ""

        except Exception as error_text:
            raw_response = ""

            normalized = build_fallback_result(
                raw_response="",
                yolo_class=yolo_class,
                keyword_dict=keyword_dict
            )

            success = False
            error = str(error_text)

        rows.append({
            "image_file": image_path.name,
            "image_path": str(image_path),
            "frame_number": frame_number,
            "source_frame_number": source_frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "yolo_class_hint": yolo_class,
            "success": success,
            "error": error,
            "model_name": model_name,
            "raw_response": raw_response,
            "short_description_ru": normalized["short_description_ru"],
            "scene_type": normalized["scene_type"],
            "main_objects_text": normalized["main_objects_text"],
            "people_present": normalized["people_present"],
            "text_or_symbols_visible": normalized["text_or_symbols_visible"],
            "visual_context_ru": normalized["visual_context_ru"],
            "possible_risk_flags": normalized["possible_risk_flags"],
            "moderation_comment_ru": normalized["moderation_comment_ru"]
        })

    return rows


# =====================================================
# СВОДКИ
# =====================================================

def build_scene_summary(rows):
    scene_counter = {}

    for row in rows:
        scene_type = row.get("scene_type", "")

        if not scene_type:
            scene_type = "unknown"

        scene_counter[scene_type] = scene_counter.get(scene_type, 0) + 1

    summary_rows = []

    for scene_type, count in sorted(scene_counter.items()):
        summary_rows.append({
            "scene_type": scene_type,
            "images_count": count
        })

    return summary_rows


def build_risk_summary(rows):
    risk_counter = {}

    for row in rows:
        risk_flags = row.get("possible_risk_flags", "")

        if not risk_flags:
            risk_flags = "unknown"

        risk_counter[risk_flags] = risk_counter.get(risk_flags, 0) + 1

    summary_rows = []

    for risk_flags, count in sorted(risk_counter.items()):
        summary_rows.append({
            "possible_risk_flags": risk_flags,
            "images_count": count
        })

    return summary_rows


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def save_llm_results(
    llm_dir,
    rows,
    scene_summary_rows,
    risk_summary_rows,
    model_name,
    llm_mode,
    taxonomy
):
    llm_dir = Path(llm_dir)
    llm_dir.mkdir(parents=True, exist_ok=True)

    excel_path = llm_dir / "llm_image_analysis_results.xlsx"
    json_path = llm_dir / "llm_image_analysis_report.json"

    results_df = pd.DataFrame(rows)
    scene_summary_df = pd.DataFrame(scene_summary_rows)
    risk_summary_df = pd.DataFrame(risk_summary_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="llm_results", index=False)
        scene_summary_df.to_excel(writer, sheet_name="scene_summary", index=False)
        risk_summary_df.to_excel(writer, sheet_name="risk_summary", index=False)

    report = {
        "report_type": "LLM_IMAGE_ANALYSIS",
        "model_name": model_name,
        "llm_mode": llm_mode,
        "taxonomy_name": taxonomy.get("taxonomy_name", ""),
        "taxonomy_version": taxonomy.get("taxonomy_version", ""),
        "processed_images_count": len(rows),
        "success_count": sum(1 for row in rows if row.get("success") is True),
        "error_count": sum(1 for row in rows if row.get("success") is False),
        "results": rows,
        "scene_summary": scene_summary_rows,
        "risk_summary": risk_summary_rows,
        "excel_path": str(excel_path),
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return {
        "excel_path": excel_path,
        "json_path": json_path
    }


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ7: LLM-анализ изображений через Ollama")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выполни ПЗ5.")
        print(error)
        return

    taxonomy = load_risk_taxonomy()
    keyword_dict = build_keyword_dict(taxonomy)

    print("\nТаксономия риска:")
    print(f"taxonomy_name: {taxonomy.get('taxonomy_name')}")
    print(f"taxonomy_version: {taxonomy.get('taxonomy_version')}")
    print(f"categories: {list(keyword_dict.keys())}")

    object_crops_dir = Path(config["object_crops_dir"])
    llm_dir = Path(config["llm_dir"])

    if not object_crops_dir.exists():
        print("\nПапка object_crops не найдена:")
        print(object_crops_dir)
        return

    image_files_all = get_image_files(object_crops_dir)

    if not image_files_all:
        print("\nВ object_crops нет изображений для LLM:")
        print(object_crops_dir)
        return

    print("\nИзображения для LLM берутся из папки:")
    print(object_crops_dir)

    print("\nРезультаты LLM будут сохранены в папку:")
    print(llm_dir)

    if not check_ollama_available():
        print("\nOllama не отвечает по адресу:")
        print("http://localhost:11434")
        print("\nПроверь, что Ollama запущена.")
        print("Также проверь, что модель установлена, например:")
        print("ollama pull moondream")
        return

    model_name, llm_mode = choose_llm_model()
    max_images = choose_max_images(len(image_files_all))

    selected_images = select_images_for_llm(
        object_crops_dir=object_crops_dir,
        max_images=max_images
    )

    print("\nБудет обработано изображений:")
    print(len(selected_images))

    clear_folder(llm_dir)

    try:
        rows = analyze_images_with_llm(
            image_files=selected_images,
            model_name=model_name,
            taxonomy=taxonomy,
            keyword_dict=keyword_dict
        )

        scene_summary_rows = build_scene_summary(rows)
        risk_summary_rows = build_risk_summary(rows)

        saved_paths = save_llm_results(
            llm_dir=llm_dir,
            rows=rows,
            scene_summary_rows=scene_summary_rows,
            risk_summary_rows=risk_summary_rows,
            model_name=model_name,
            llm_mode=llm_mode,
            taxonomy=taxonomy
        )

    except Exception as error:
        print("\nОшибка при выполнении ПЗ7:")
        print(error)

        config["pz7_status"] = "error"
        config["pz7_error"] = str(error)
        config["pz7_finished_at"] = datetime.now().isoformat()
        save_run_config(config)

        return

    config["llm_dir"] = str(llm_dir)
    config["llm_model"] = model_name
    config["llm_mode"] = llm_mode
    config["llm_taxonomy_name"] = taxonomy.get("taxonomy_name", "")
    config["llm_taxonomy_version"] = taxonomy.get("taxonomy_version", "")
    config["llm_processed_images_count"] = len(rows)
    config["llm_results_path"] = str(saved_paths["excel_path"])
    config["llm_report_path"] = str(saved_paths["json_path"])
    config["pz7_status"] = "success"
    config["pz7_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\nПЗ7 завершено успешно.")
    print(f"Модель: {model_name}")
    print(f"Режим: {llm_mode}")
    print(f"Обработано изображений: {len(rows)}")
    print(f"Excel: {saved_paths['excel_path']}")
    print(f"JSON: {saved_paths['json_path']}")
    print("\nrun_config.json обновлён.")


if __name__ == "__main__":
    main()