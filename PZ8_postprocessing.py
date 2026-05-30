import json
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import pandas as pd


# =====================================================
# НАСТРОЙКА ПАПОК
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

PZ3_ROOT = BASE_DIR / "results" / "pz3_ocr"
PZ4_ROOT = BASE_DIR / "results" / "pz4_whisper"
PZ5_ROOT = BASE_DIR / "results" / "pz5_yolo"
PZ6_ROOT = BASE_DIR / "results" / "pz6_resnet"
PZ7_ROOT = BASE_DIR / "results" / "pz7_llm"

RESULT_ROOT = BASE_DIR / "results" / "pz8_postprocessing"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"postprocess_run_{RUN_TIMESTAMP}"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# СЛОВАРИ ДЛЯ АНАЛИЗА РИСКОВ
# =====================================================

WEAPON_KEYWORDS = [
    "оружие", "пистолет", "нож", "автомат", "винтовка", "ружье", "ружьё",
    "граната", "патрон", "боеприпас", "стрельба", "выстрел", "пуля",
    "gun", "pistol", "knife", "rifle", "weapon", "firearm", "grenade",
    "bullet", "ammo", "ammunition", "shooting", "revolver", "shotgun",
    "machine gun", "assault rifle"
]

VIOLENCE_KEYWORDS = [
    "насилие", "драка", "нападение", "кровь", "убийство", "угроза",
    "избиение", "удар", "ранение", "стрелять", "убить", "атака",
    "violence", "fight", "attack", "blood", "murder", "threat",
    "kill", "killing", "wound", "injury", "hit", "assault"
]


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

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
    text = text.replace("|", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_text(text):
    """
    Нормализация нужна только для сравнения и поиска ключевых слов.
    Исходный текст в результатах не меняется.
    """

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

    text = re.sub(r"[^a-zа-я0-9 ]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def text_similarity(text_1, text_2):
    norm_1 = normalize_text(text_1)
    norm_2 = normalize_text(text_2)

    if not norm_1 or not norm_2:
        return 0

    shorter = min(norm_1, norm_2, key=len)
    longer = max(norm_1, norm_2, key=len)

    if len(shorter) >= 8 and shorter in longer:
        return 0.92

    return SequenceMatcher(None, norm_1, norm_2).ratio()


def deduplicate_texts(texts, threshold=0.82):
    unique_texts = []

    for text in texts:
        text = clean_text(text)

        if not text:
            continue

        already_exists = False

        for existing in unique_texts:
            if text_similarity(existing, text) >= threshold:
                already_exists = True
                break

        if not already_exists:
            unique_texts.append(text)

    return unique_texts


def contains_any_keyword(text, keywords):
    text_norm = normalize_text(text)

    if not text_norm:
        return []

    found = []

    for keyword in keywords:
        keyword_norm = normalize_text(keyword)

        if keyword_norm and keyword_norm in text_norm:
            found.append(keyword)

    return found


def read_json(json_path):
    if json_path is None or not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def read_excel_safe(excel_path, sheet_name=0):
    """
    Безопасно читает Excel.

    Важно:
    если sheet_name=None, pandas возвращает dict всех листов.
    Поэтому по умолчанию читаем первый лист.
    """

    if excel_path is None or not excel_path.exists():
        return pd.DataFrame()

    try:
        data = pd.read_excel(excel_path, sheet_name=sheet_name)

        if isinstance(data, dict):
            if len(data) == 0:
                return pd.DataFrame()

            first_sheet_name = list(data.keys())[0]
            return data[first_sheet_name]

        return data

    except Exception:
        return pd.DataFrame()


def dataframe_to_records(df):
    """
    Переводит DataFrame в список словарей.
    Защищает от случая, когда вместо DataFrame пришёл dict.
    """

    if df is None:
        return []

    if isinstance(df, dict):
        if len(df) == 0:
            return []

        first_sheet_name = list(df.keys())[0]
        df = df[first_sheet_name]

    if not hasattr(df, "empty"):
        return []

    if df.empty:
        return []

    df = df.fillna("")
    return df.to_dict(orient="records")


def get_basename_from_path(path_value):
    if path_value is None or path_value == "":
        return ""

    try:
        return Path(str(path_value)).name
    except Exception:
        return ""


def find_first_file(folder, pattern):
    if folder is None or not folder.exists():
        return None

    files = list(folder.glob(pattern))

    if not files:
        return None

    return sorted(files)[0]


# =====================================================
# ВЫБОР ЗАПУСКОВ ПЗ3–ПЗ7
# =====================================================

def choose_run_folder(root_folder, prefix, title, allow_skip=True):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if not root_folder.exists():
        print("Папка не найдена:")
        print(root_folder)

        if allow_skip:
            print("Этот блок будет пропущен.")
            return None

        return None

    run_folders = [
        folder for folder in sorted(root_folder.iterdir())
        if folder.is_dir() and folder.name.startswith(prefix)
    ]

    if not run_folders:
        print("Запуски не найдены:")
        print(root_folder)

        if allow_skip:
            print("Этот блок будет пропущен.")
            return None

        return None

    print("Найдены запуски:")

    for index, folder in enumerate(run_folders, start=1):
        print(f"{index}. {folder.name}")

    print("\nEnter — выбрать последний запуск")

    if allow_skip:
        print("0 — пропустить этот блок")

    choice = input("\nВведите номер запуска: ").strip()

    if choice == "":
        return run_folders[-1]

    if allow_skip and choice == "0":
        return None

    try:
        choice_number = int(choice)

        if 1 <= choice_number <= len(run_folders):
            return run_folders[choice_number - 1]

        print("Неверный номер. Выбран последний запуск.")
        return run_folders[-1]

    except ValueError:
        print("Введено не число. Выбран последний запуск.")
        return run_folders[-1]


def choose_all_runs():
    pz3_run = choose_run_folder(
        PZ3_ROOT,
        "ocr_run_",
        "Выбор результатов ПЗ3 OCR"
    )

    pz4_run = choose_run_folder(
        PZ4_ROOT,
        "whisper_run_",
        "Выбор результатов ПЗ4 Whisper"
    )

    pz5_run = choose_run_folder(
        PZ5_ROOT,
        "yolo_run_",
        "Выбор результатов ПЗ5 YOLO"
    )

    pz6_run = choose_run_folder(
        PZ6_ROOT,
        "resnet_run_",
        "Выбор результатов ПЗ6 ResNet"
    )

    pz7_run = choose_run_folder(
        PZ7_ROOT,
        "llm_run_",
        "Выбор результатов ПЗ7 LLM"
    )

    return {
        "pz3_run": pz3_run,
        "pz4_run": pz4_run,
        "pz5_run": pz5_run,
        "pz6_run": pz6_run,
        "pz7_run": pz7_run
    }


# =====================================================
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЗ3
# =====================================================

def load_pz3_results(pz3_run):
    if pz3_run is None:
        return {
            "ocr_unique_texts": [],
            "ocr_raw_rows": [],
            "ocr_segments": [],
            "ocr_json": {}
        }

    unique_texts_path = pz3_run / "unique_texts.txt"
    raw_excel_path = pz3_run / "raw_ocr_results.xlsx"
    segments_excel_path = pz3_run / "deduplicated_segments.xlsx"
    json_path = pz3_run / "ocr_results.json"

    ocr_unique_texts = []

    if unique_texts_path.exists():
        with open(unique_texts_path, "r", encoding="utf-8") as file:
            ocr_unique_texts = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

    raw_df = read_excel_safe(raw_excel_path)
    segments_df = read_excel_safe(segments_excel_path)

    return {
        "ocr_unique_texts": deduplicate_texts(ocr_unique_texts),
        "ocr_raw_rows": dataframe_to_records(raw_df),
        "ocr_segments": dataframe_to_records(segments_df),
        "ocr_json": read_json(json_path)
    }


# =====================================================
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЗ4
# =====================================================

def load_pz4_results(pz4_run):
    if pz4_run is None:
        return {
            "audio_full_text": "",
            "audio_segments": [],
            "audio_json": {}
        }

    transcripts_dir = pz4_run / "transcripts"

    transcript_txt = find_first_file(transcripts_dir, "*_transcript.txt")
    segments_excel = find_first_file(transcripts_dir, "*_whisper_segments.xlsx")
    json_path = find_first_file(transcripts_dir, "*_whisper_report.json")

    audio_full_text = ""

    if transcript_txt is not None and transcript_txt.exists():
        with open(transcript_txt, "r", encoding="utf-8") as file:
            audio_full_text = file.read().strip()

    segments_df = read_excel_safe(segments_excel)

    return {
        "audio_full_text": audio_full_text,
        "audio_segments": dataframe_to_records(segments_df),
        "audio_json": read_json(json_path)
    }


# =====================================================
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЗ5
# =====================================================

def load_pz5_results(pz5_run):
    if pz5_run is None:
        return {
            "yolo_objects": [],
            "yolo_frames": [],
            "yolo_class_summary": [],
            "yolo_json": {}
        }

    excel_path = pz5_run / "yolo_detection_results.xlsx"
    json_path = pz5_run / "yolo_detection_report.json"

    objects_df = read_excel_safe(excel_path, sheet_name="objects")
    frames_df = read_excel_safe(excel_path, sheet_name="frames")
    class_summary_df = read_excel_safe(excel_path, sheet_name="class_summary")

    return {
        "yolo_objects": dataframe_to_records(objects_df),
        "yolo_frames": dataframe_to_records(frames_df),
        "yolo_class_summary": dataframe_to_records(class_summary_df),
        "yolo_json": read_json(json_path)
    }


# =====================================================
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЗ6
# =====================================================

def load_pz6_results(pz6_run):
    if pz6_run is None:
        return {
            "resnet_top1": [],
            "resnet_topk": [],
            "resnet_class_summary": [],
            "resnet_json": {}
        }

    excel_path = pz6_run / "resnet_classification_results.xlsx"
    json_path = pz6_run / "resnet_classification_report.json"

    top1_df = read_excel_safe(excel_path, sheet_name="top1_results")
    topk_df = read_excel_safe(excel_path, sheet_name="topk_results")
    summary_df = read_excel_safe(excel_path, sheet_name="class_summary")

    return {
        "resnet_top1": dataframe_to_records(top1_df),
        "resnet_topk": dataframe_to_records(topk_df),
        "resnet_class_summary": dataframe_to_records(summary_df),
        "resnet_json": read_json(json_path)
    }


# =====================================================
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЗ7
# =====================================================

def load_pz7_results(pz7_run):
    if pz7_run is None:
        return {
            "llm_results": [],
            "llm_scene_summary": [],
            "llm_risk_summary": [],
            "llm_json": {}
        }

    excel_path = pz7_run / "llm_image_analysis_results.xlsx"
    json_path = pz7_run / "llm_image_analysis_report.json"

    results_df = read_excel_safe(excel_path, sheet_name="llm_results")
    scene_summary_df = read_excel_safe(excel_path, sheet_name="scene_summary")
    risk_summary_df = read_excel_safe(excel_path, sheet_name="risk_summary")

    return {
        "llm_results": dataframe_to_records(results_df),
        "llm_scene_summary": dataframe_to_records(scene_summary_df),
        "llm_risk_summary": dataframe_to_records(risk_summary_df),
        "llm_json": read_json(json_path)
    }


# =====================================================
# ПОСТОБРАБОТКА ТЕКСТА
# =====================================================

def build_combined_texts(pz3_data, pz4_data):
    combined_texts = []

    for text in pz3_data.get("ocr_unique_texts", []):
        combined_texts.append({
            "source": "ocr_screen_text",
            "time_formatted": "",
            "text": clean_text(text)
        })

    audio_segments = pz4_data.get("audio_segments", [])

    if audio_segments:
        for segment in audio_segments:
            text = segment.get("text", "")

            if text:
                combined_texts.append({
                    "source": "whisper_audio_text",
                    "time_formatted": segment.get("time_interval", ""),
                    "text": clean_text(text)
                })
    else:
        full_text = pz4_data.get("audio_full_text", "")

        if full_text:
            combined_texts.append({
                "source": "whisper_audio_text",
                "time_formatted": "",
                "text": clean_text(full_text)
            })

    unique_text_values = deduplicate_texts(
        [item["text"] for item in combined_texts],
        threshold=0.82
    )

    unique_rows = []

    for index, text in enumerate(unique_text_values, start=1):
        source = "mixed"
        time_formatted = ""

        for item in combined_texts:
            if text_similarity(item["text"], text) >= 0.90:
                source = item["source"]
                time_formatted = item.get("time_formatted", "")
                break

        unique_rows.append({
            "text_id": index,
            "source": source,
            "time_formatted": time_formatted,
            "text": text
        })

    return combined_texts, unique_rows


# =====================================================
# ГРУППИРОВКА YOLO-ОБЪЕКТОВ
# =====================================================

def build_object_group(class_name, group_rows):
    times = []

    for row in group_rows:
        try:
            if row.get("time_seconds", "") != "":
                times.append(float(row.get("time_seconds")))
        except Exception:
            pass

    if times:
        start_time = min(times)
        end_time = max(times)
    else:
        start_time = ""
        end_time = ""

    confidences = []

    for row in group_rows:
        try:
            confidences.append(float(row.get("confidence", 0)))
        except Exception:
            pass

    if confidences:
        max_confidence = round(max(confidences), 4)
        mean_confidence = round(sum(confidences) / len(confidences), 4)
    else:
        max_confidence = 0
        mean_confidence = 0

    example_frame = group_rows[0].get("frame_file", "")
    example_crop = group_rows[0].get("crop_path", "")

    return {
        "class_name": class_name,
        "start_time_seconds": start_time,
        "end_time_seconds": end_time,
        "start_time": seconds_to_time(start_time),
        "end_time": seconds_to_time(end_time),
        "time_interval": f"{seconds_to_time(start_time)} - {seconds_to_time(end_time)}",
        "detections_count": len(group_rows),
        "max_confidence": max_confidence,
        "mean_confidence": mean_confidence,
        "example_frame": example_frame,
        "example_crop": example_crop
    }


def group_yolo_objects(yolo_objects, max_gap_seconds=3.0):
    """
    Склеивает повторяющиеся YOLO-срабатывания по классу и времени.
    """

    if not yolo_objects:
        return []

    grouped_rows = []

    df = pd.DataFrame(yolo_objects).fillna("")

    if "class_name" not in df.columns:
        return []

    if "time_seconds" not in df.columns:
        df["time_seconds"] = ""

    for class_name, class_df in df.groupby("class_name"):
        rows = class_df.to_dict(orient="records")

        rows = sorted(
            rows,
            key=lambda row: (
                float(row.get("time_seconds", 0)) if row.get("time_seconds", "") != "" else 999999,
                int(row.get("frame_number", 0)) if row.get("frame_number", "") != "" else 999999
            )
        )

        current_group = []

        for row in rows:
            if not current_group:
                current_group.append(row)
                continue

            previous = current_group[-1]

            try:
                current_time = float(row.get("time_seconds", 0))
                previous_time = float(previous.get("time_seconds", 0))
                time_gap = current_time - previous_time
            except Exception:
                time_gap = 0

            if time_gap <= max_gap_seconds:
                current_group.append(row)
            else:
                grouped_rows.append(build_object_group(class_name, current_group))
                current_group = [row]

        if current_group:
            grouped_rows.append(build_object_group(class_name, current_group))

    for index, row in enumerate(grouped_rows, start=1):
        row["object_group_id"] = index

    return grouped_rows


# =====================================================
# ОБЪЕДИНЕНИЕ YOLO + RESNET + LLM
# =====================================================

def build_enriched_objects(yolo_objects, resnet_top1, llm_results):
    """
    Соединяет YOLO, ResNet и LLM по имени crop-изображения.
    """

    resnet_by_image = {}

    for row in resnet_top1:
        image_file = str(row.get("image_file", ""))

        if image_file:
            resnet_by_image[image_file] = row

    llm_by_image = {}

    for row in llm_results:
        image_file = str(row.get("image_file", ""))

        if image_file:
            llm_by_image[image_file] = row

    enriched = []

    for index, yolo_row in enumerate(yolo_objects, start=1):
        crop_file = get_basename_from_path(yolo_row.get("crop_path", ""))

        resnet_row = resnet_by_image.get(crop_file, {})
        llm_row = llm_by_image.get(crop_file, {})

        enriched.append({
            "object_id": index,
            "frame_file": yolo_row.get("frame_file", ""),
            "frame_number": yolo_row.get("frame_number", ""),
            "time_seconds": yolo_row.get("time_seconds", ""),
            "time_formatted": yolo_row.get("time_formatted", ""),
            "yolo_class": yolo_row.get("class_name", ""),
            "yolo_confidence": yolo_row.get("confidence", ""),
            "crop_file": crop_file,
            "crop_path": yolo_row.get("crop_path", ""),
            "resnet_top1_class": resnet_row.get("top1_class_name", ""),
            "resnet_top1_confidence": resnet_row.get("top1_confidence", ""),
            "llm_description": llm_row.get("short_description_ru", ""),
            "llm_scene_type": llm_row.get("scene_type", ""),
            "llm_main_objects": llm_row.get("main_objects_text", ""),
            "llm_people_present": llm_row.get("people_present", ""),
            "llm_risk_flags": llm_row.get("possible_risk_flags", ""),
            "llm_comment": llm_row.get("moderation_comment_ru", "")
        })

    return enriched


# =====================================================
# АНАЛИЗ РИСКОВ: ОРУЖИЕ И НАСИЛИЕ
# =====================================================

def add_risk_source(risk_sources, source, risk_type, evidence, time_formatted="", confidence=""):
    evidence = clean_text(evidence)

    if not evidence:
        return

    risk_sources.append({
        "source": source,
        "risk_type": risk_type,
        "evidence": evidence,
        "time_formatted": time_formatted,
        "confidence": confidence
    })


def analyze_text_risks(unique_texts):
    risk_sources = []

    for row in unique_texts:
        text = row.get("text", "")
        source = row.get("source", "text")
        time_formatted = row.get("time_formatted", "")

        weapon_hits = contains_any_keyword(text, WEAPON_KEYWORDS)
        violence_hits = contains_any_keyword(text, VIOLENCE_KEYWORDS)

        if weapon_hits:
            add_risk_source(
                risk_sources,
                source,
                "weapon",
                f"Найдены ключевые слова оружия: {', '.join(weapon_hits)}. Фрагмент: {text}",
                time_formatted
            )

        if violence_hits:
            add_risk_source(
                risk_sources,
                source,
                "violence",
                f"Найдены ключевые слова насилия: {', '.join(violence_hits)}. Фрагмент: {text}",
                time_formatted
            )

    return risk_sources


def analyze_yolo_risks(yolo_objects):
    risk_sources = []

    for row in yolo_objects:
        class_name = row.get("class_name", "")
        time_formatted = row.get("time_formatted", "")
        confidence = row.get("confidence", "")

        weapon_hits = contains_any_keyword(class_name, WEAPON_KEYWORDS)
        violence_hits = contains_any_keyword(class_name, VIOLENCE_KEYWORDS)

        if weapon_hits:
            add_risk_source(
                risk_sources,
                "yolo",
                "weapon",
                f"YOLO обнаружила объект класса: {class_name}",
                time_formatted,
                confidence
            )

        if violence_hits:
            add_risk_source(
                risk_sources,
                "yolo",
                "violence",
                f"YOLO обнаружила объект класса: {class_name}",
                time_formatted,
                confidence
            )

    return risk_sources


def analyze_resnet_risks(resnet_top1):
    risk_sources = []

    for row in resnet_top1:
        class_name = row.get("top1_class_name", "")
        time_formatted = row.get("time_formatted", "")
        confidence = row.get("top1_confidence", "")

        weapon_hits = contains_any_keyword(class_name, WEAPON_KEYWORDS)
        violence_hits = contains_any_keyword(class_name, VIOLENCE_KEYWORDS)

        if weapon_hits:
            add_risk_source(
                risk_sources,
                "resnet",
                "weapon",
                f"ResNet классифицировала изображение как: {class_name}",
                time_formatted,
                confidence
            )

        if violence_hits:
            add_risk_source(
                risk_sources,
                "resnet",
                "violence",
                f"ResNet классифицировала изображение как: {class_name}",
                time_formatted,
                confidence
            )

    return risk_sources


def analyze_llm_risks(llm_results):
    risk_sources = []

    for row in llm_results:
        time_formatted = row.get("time_formatted", "")

        fields = [
            row.get("short_description_ru", ""),
            row.get("main_objects_text", ""),
            row.get("text_or_symbols_visible", ""),
            row.get("visual_context_ru", ""),
            row.get("possible_risk_flags", ""),
            row.get("moderation_comment_ru", "")
        ]

        full_text = " ".join([
            clean_text(field)
            for field in fields
            if field
        ])

        weapon_hits = contains_any_keyword(full_text, WEAPON_KEYWORDS)
        violence_hits = contains_any_keyword(full_text, VIOLENCE_KEYWORDS)

        if weapon_hits:
            add_risk_source(
                risk_sources,
                "llm",
                "weapon",
                f"LLM-описание содержит признаки оружия: {', '.join(weapon_hits)}. Описание: {full_text}",
                time_formatted
            )

        if violence_hits:
            add_risk_source(
                risk_sources,
                "llm",
                "violence",
                f"LLM-описание содержит признаки насилия: {', '.join(violence_hits)}. Описание: {full_text}",
                time_formatted
            )

    return risk_sources


def deduplicate_risk_sources(risk_sources):
    unique_sources = []

    for source in risk_sources:
        evidence = source.get("evidence", "")

        duplicate = False

        for existing in unique_sources:
            if (
                source.get("source") == existing.get("source")
                and source.get("risk_type") == existing.get("risk_type")
                and text_similarity(evidence, existing.get("evidence", "")) >= 0.88
            ):
                duplicate = True
                break

        if not duplicate:
            unique_sources.append(source)

    return unique_sources


def calculate_risk_level(risk_sources):
    if not risk_sources:
        return "none"

    visual_sources = [
        item for item in risk_sources
        if item.get("source") in ["yolo", "resnet", "llm"]
    ]

    text_sources = [
        item for item in risk_sources
        if item.get("source") in ["ocr_screen_text", "whisper_audio_text", "text"]
    ]

    weapon_sources = [
        item for item in risk_sources
        if item.get("risk_type") == "weapon"
    ]

    violence_sources = [
        item for item in risk_sources
        if item.get("risk_type") == "violence"
    ]

    if len(weapon_sources) >= 2 and len(visual_sources) >= 1:
        return "high"

    if len(violence_sources) >= 2 and len(visual_sources) >= 1:
        return "high"

    if len(visual_sources) >= 1:
        return "medium"

    if len(text_sources) >= 1:
        return "medium"

    return "low"


def build_risk_analysis(unique_texts, pz5_data, pz6_data, pz7_data):
    risk_sources = []

    risk_sources.extend(analyze_text_risks(unique_texts))
    risk_sources.extend(analyze_yolo_risks(pz5_data.get("yolo_objects", [])))
    risk_sources.extend(analyze_resnet_risks(pz6_data.get("resnet_top1", [])))
    risk_sources.extend(analyze_llm_risks(pz7_data.get("llm_results", [])))

    risk_sources = deduplicate_risk_sources(risk_sources)

    weapon_detected = any(
        item.get("risk_type") == "weapon"
        for item in risk_sources
    )

    violence_detected = any(
        item.get("risk_type") == "violence"
        for item in risk_sources
    )

    risk_level = calculate_risk_level(risk_sources)

    if risk_level == "none":
        final_comment = "Признаки оружия или насильственного контента не обнаружены."
    elif risk_level == "low":
        final_comment = "Обнаружены слабые косвенные признаки. Рекомендуется ручная проверка."
    elif risk_level == "medium":
        final_comment = "Обнаружены признаки потенциально опасного контента. Требуется ручная проверка."
    else:
        final_comment = "Обнаружены выраженные признаки оружия или насильственного контента. Требуется приоритетная ручная проверка."

    return {
        "weapon_detected": weapon_detected,
        "violence_detected": violence_detected,
        "risk_level": risk_level,
        "risk_sources_count": len(risk_sources),
        "risk_sources": risk_sources,
        "final_comment": final_comment
    }


# =====================================================
# ИТОГОВАЯ СВОДКА
# =====================================================

def build_final_summary(
    pz3_data,
    pz4_data,
    pz5_data,
    pz6_data,
    pz7_data,
    unique_texts,
    grouped_objects,
    enriched_objects,
    risk_analysis
):
    summary = {
        "ocr_unique_texts_count": len(pz3_data.get("ocr_unique_texts", [])),
        "audio_segments_count": len(pz4_data.get("audio_segments", [])),
        "combined_unique_texts_count": len(unique_texts),
        "yolo_detections_count": len(pz5_data.get("yolo_objects", [])),
        "grouped_objects_count": len(grouped_objects),
        "resnet_images_count": len(pz6_data.get("resnet_top1", [])),
        "llm_descriptions_count": len(pz7_data.get("llm_results", [])),
        "enriched_objects_count": len(enriched_objects),
        "weapon_detected": risk_analysis.get("weapon_detected"),
        "violence_detected": risk_analysis.get("violence_detected"),
        "risk_level": risk_analysis.get("risk_level"),
        "risk_sources_count": risk_analysis.get("risk_sources_count")
    }

    return summary


def build_summary_table(summary):
    labels = {
        "ocr_unique_texts_count": "Уникальные OCR-фразы",
        "audio_segments_count": "Сегменты аудиорасшифровки Whisper",
        "combined_unique_texts_count": "Объединённые уникальные текстовые фрагменты",
        "yolo_detections_count": "Всего YOLO-детекций",
        "grouped_objects_count": "Сгруппированные объекты YOLO",
        "resnet_images_count": "Изображения, классифицированные ResNet",
        "llm_descriptions_count": "Изображения, описанные LLM",
        "enriched_objects_count": "Объединённые записи YOLO + ResNet + LLM",
        "weapon_detected": "Обнаружены признаки оружия",
        "violence_detected": "Обнаружены признаки насилия",
        "risk_level": "Итоговый уровень риска",
        "risk_sources_count": "Количество найденных риск-признаков"
    }

    rows = []

    for key, value in summary.items():
        rows.append({
            "metric": key,
            "description": labels.get(key, key),
            "value": value
        })

    return rows


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def save_txt_summary(summary, unique_texts, grouped_objects, risk_analysis):
    txt_path = RUN_DIR / "final_text_summary.txt"

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("ПЗ8. Итоговая постобработка результатов анализа видео\n\n")

        file.write("Сводка:\n")
        file.write(f"- Уникальные OCR-фразы: {summary['ocr_unique_texts_count']}\n")
        file.write(f"- Сегменты аудиорасшифровки: {summary['audio_segments_count']}\n")
        file.write(f"- Объединённые уникальные текстовые фрагменты: {summary['combined_unique_texts_count']}\n")
        file.write(f"- YOLO-детекции: {summary['yolo_detections_count']}\n")
        file.write(f"- Сгруппированные объекты: {summary['grouped_objects_count']}\n")
        file.write(f"- ResNet-классификаций: {summary['resnet_images_count']}\n")
        file.write(f"- LLM-описаний: {summary['llm_descriptions_count']}\n\n")

        file.write("Анализ риска:\n")
        file.write(f"- Признаки оружия: {risk_analysis.get('weapon_detected')}\n")
        file.write(f"- Признаки насилия: {risk_analysis.get('violence_detected')}\n")
        file.write(f"- Уровень риска: {risk_analysis.get('risk_level')}\n")
        file.write(f"- Комментарий: {risk_analysis.get('final_comment')}\n\n")

        file.write("Источники риск-признаков:\n")

        if risk_analysis.get("risk_sources"):
            for row in risk_analysis.get("risk_sources", []):
                file.write(
                    f"- [{row.get('source')}] {row.get('risk_type')} | "
                    f"{row.get('time_formatted')} | {row.get('evidence')}\n"
                )
        else:
            file.write("- Риск-признаки не обнаружены.\n")

        file.write("\nОбъединённые уникальные тексты:\n")

        for row in unique_texts:
            file.write(f"- [{row.get('source')}] {row.get('text')}\n")

        file.write("\nСгруппированные объекты:\n")

        for row in grouped_objects:
            file.write(
                f"- {row.get('class_name')} | "
                f"{row.get('time_interval')} | "
                f"срабатываний: {row.get('detections_count')}\n"
            )

    return txt_path


def save_deduplicated_texts(unique_texts):
    txt_path = RUN_DIR / "deduplicated_texts.txt"

    with open(txt_path, "w", encoding="utf-8") as file:
        for row in unique_texts:
            file.write(row.get("text", "") + "\n")

    return txt_path


def save_excel(
    summary_rows,
    unique_texts,
    pz4_data,
    pz5_data,
    grouped_objects,
    pz6_data,
    pz7_data,
    enriched_objects,
    risk_analysis
):
    excel_path = RUN_DIR / "final_analysis_tables.xlsx"

    summary_df = pd.DataFrame(summary_rows)
    texts_df = pd.DataFrame(unique_texts)
    audio_df = pd.DataFrame(pz4_data.get("audio_segments", []))
    yolo_df = pd.DataFrame(pz5_data.get("yolo_objects", []))
    grouped_df = pd.DataFrame(grouped_objects)
    resnet_df = pd.DataFrame(pz6_data.get("resnet_top1", []))
    llm_df = pd.DataFrame(pz7_data.get("llm_results", []))
    enriched_df = pd.DataFrame(enriched_objects)

    risk_summary_df = pd.DataFrame([{
        "weapon_detected": risk_analysis.get("weapon_detected"),
        "violence_detected": risk_analysis.get("violence_detected"),
        "risk_level": risk_analysis.get("risk_level"),
        "risk_sources_count": risk_analysis.get("risk_sources_count"),
        "final_comment": risk_analysis.get("final_comment")
    }])

    risk_sources_df = pd.DataFrame(risk_analysis.get("risk_sources", []))

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        risk_summary_df.to_excel(writer, sheet_name="risk_summary", index=False)
        risk_sources_df.to_excel(writer, sheet_name="risk_sources", index=False)
        texts_df.to_excel(writer, sheet_name="texts", index=False)
        audio_df.to_excel(writer, sheet_name="audio_segments", index=False)
        yolo_df.to_excel(writer, sheet_name="yolo_objects", index=False)
        grouped_df.to_excel(writer, sheet_name="grouped_objects", index=False)
        resnet_df.to_excel(writer, sheet_name="resnet_results", index=False)
        llm_df.to_excel(writer, sheet_name="llm_results", index=False)
        enriched_df.to_excel(writer, sheet_name="enriched_objects", index=False)

    return excel_path


def save_json_report(
    selected_runs,
    pz3_data,
    pz4_data,
    pz5_data,
    pz6_data,
    pz7_data,
    combined_texts,
    unique_texts,
    grouped_objects,
    enriched_objects,
    risk_analysis,
    summary
):
    json_path = RUN_DIR / "final_analysis_report.json"

    report = {
        "report_type": "FINAL_VIDEO_ANALYSIS_REPORT",
        "task": "Detection of visual, textual and audio signs of weapons and potentially violent content",
        "selected_runs": {
            "pz3_run": str(selected_runs.get("pz3_run")) if selected_runs.get("pz3_run") else "",
            "pz4_run": str(selected_runs.get("pz4_run")) if selected_runs.get("pz4_run") else "",
            "pz5_run": str(selected_runs.get("pz5_run")) if selected_runs.get("pz5_run") else "",
            "pz6_run": str(selected_runs.get("pz6_run")) if selected_runs.get("pz6_run") else "",
            "pz7_run": str(selected_runs.get("pz7_run")) if selected_runs.get("pz7_run") else ""
        },
        "risk_analysis": risk_analysis,
        "text_analysis": {
            "ocr_unique_texts": pz3_data.get("ocr_unique_texts", []),
            "audio_full_text": pz4_data.get("audio_full_text", ""),
            "audio_segments": pz4_data.get("audio_segments", []),
            "combined_texts": combined_texts,
            "combined_unique_texts": unique_texts
        },
        "object_analysis": {
            "yolo_objects": pz5_data.get("yolo_objects", []),
            "grouped_objects": grouped_objects,
            "resnet_top1": pz6_data.get("resnet_top1", []),
            "llm_results": pz7_data.get("llm_results", []),
            "enriched_objects": enriched_objects
        },
        "final_summary": summary,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4, default=str)

    return json_path


def save_all_results(
    selected_runs,
    pz3_data,
    pz4_data,
    pz5_data,
    pz6_data,
    pz7_data,
    combined_texts,
    unique_texts,
    grouped_objects,
    enriched_objects,
    risk_analysis,
    summary
):
    summary_rows = build_summary_table(summary)

    excel_path = save_excel(
        summary_rows,
        unique_texts,
        pz4_data,
        pz5_data,
        grouped_objects,
        pz6_data,
        pz7_data,
        enriched_objects,
        risk_analysis
    )

    json_path = save_json_report(
        selected_runs,
        pz3_data,
        pz4_data,
        pz5_data,
        pz6_data,
        pz7_data,
        combined_texts,
        unique_texts,
        grouped_objects,
        enriched_objects,
        risk_analysis,
        summary
    )

    txt_summary_path = save_txt_summary(
        summary,
        unique_texts,
        grouped_objects,
        risk_analysis
    )

    deduplicated_texts_path = save_deduplicated_texts(unique_texts)

    print("\n" + "=" * 70)
    print("ПЗ8 ГОТОВО")
    print("=" * 70)

    print(f"Папка результатов ПЗ8: {RUN_DIR}")
    print(f"Итоговый JSON: {json_path}")
    print(f"Итоговый Excel: {excel_path}")
    print(f"Текстовая сводка: {txt_summary_path}")
    print(f"Очищенные тексты: {deduplicated_texts_path}")

    print("\nАнализ риска:")
    print(f"Признаки оружия: {risk_analysis.get('weapon_detected')}")
    print(f"Признаки насилия: {risk_analysis.get('violence_detected')}")
    print(f"Уровень риска: {risk_analysis.get('risk_level')}")
    print(f"Комментарий: {risk_analysis.get('final_comment')}")

    print("\nКраткая статистика:")

    for key, value in summary.items():
        print(f"{key}: {value}")


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ8: постобработка и итоговый анализ риска")
    print("=" * 70)

    print("\nСкрипт соберёт результаты ПЗ3–ПЗ7 в единый JSON, Excel и TXT.")
    print("Дополнительно будет выполнен анализ признаков оружия и насилия.")
    print("Можно нажимать Enter, чтобы выбирать последние запуски.")

    selected_runs = choose_all_runs()

    print("\nЗагружаем данные...")

    pz3_data = load_pz3_results(selected_runs.get("pz3_run"))
    pz4_data = load_pz4_results(selected_runs.get("pz4_run"))
    pz5_data = load_pz5_results(selected_runs.get("pz5_run"))
    pz6_data = load_pz6_results(selected_runs.get("pz6_run"))
    pz7_data = load_pz7_results(selected_runs.get("pz7_run"))

    print("Выполняем постобработку текста...")
    combined_texts, unique_texts = build_combined_texts(pz3_data, pz4_data)

    print("Выполняем группировку YOLO-объектов...")
    grouped_objects = group_yolo_objects(
        pz5_data.get("yolo_objects", []),
        max_gap_seconds=3.0
    )

    print("Объединяем YOLO, ResNet и LLM...")
    enriched_objects = build_enriched_objects(
        pz5_data.get("yolo_objects", []),
        pz6_data.get("resnet_top1", []),
        pz7_data.get("llm_results", [])
    )

    print("Выполняем анализ признаков оружия и насилия...")
    risk_analysis = build_risk_analysis(
        unique_texts,
        pz5_data,
        pz6_data,
        pz7_data
    )

    summary = build_final_summary(
        pz3_data,
        pz4_data,
        pz5_data,
        pz6_data,
        pz7_data,
        unique_texts,
        grouped_objects,
        enriched_objects,
        risk_analysis
    )

    save_all_results(
        selected_runs,
        pz3_data,
        pz4_data,
        pz5_data,
        pz6_data,
        pz7_data,
        combined_texts,
        unique_texts,
        grouped_objects,
        enriched_objects,
        risk_analysis,
        summary
    )


if __name__ == "__main__":
    main()