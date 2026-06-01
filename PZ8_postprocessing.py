import json
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import cv2
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
TAXONOMY_PATH = BASE_DIR / "risk_taxonomy.json"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"postprocess_run_{RUN_TIMESTAMP}"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# НАСТРОЙКИ
# =====================================================

DEFAULT_FPS = 30.0
MIN_VIDEO_CONFIDENCE = 0.30


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

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


def contains_any_keyword(text, keywords):
    text_norm = normalize_text(text)

    if not text_norm:
        return []

    found = []

    for keyword in keywords:
        keyword_norm = normalize_text(keyword)

        if keyword_norm and keyword_norm in text_norm:
            found.append(keyword)

    return list(sorted(set(found)))


def seconds_to_time(seconds):
    if seconds is None or seconds == "":
        return "00:00:00"

    try:
        seconds = int(float(seconds))
    except Exception:
        return "00:00:00"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def safe_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def frame_from_seconds(seconds, fps):
    seconds = safe_float(seconds, 0.0)
    fps = safe_float(fps, DEFAULT_FPS)

    if fps <= 0:
        fps = DEFAULT_FPS

    return int(round(seconds * fps))


def read_json(json_path):
    if json_path is None or not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def read_excel_safe(excel_path, sheet_name=0):
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


def find_first_file(folder, pattern):
    if folder is None or not folder.exists():
        return None

    files = list(folder.glob(pattern))

    if not files:
        return None

    return sorted(files)[0]


# =====================================================
# ЗАГРУЗКА ТАКСОНОМИИ РИСКОВ
# =====================================================

def load_risk_taxonomy():
    """
    Загружает риск-таксономию из risk_taxonomy.json.

    Внутри taxonomy есть категории:
    - weapon
    - violence

    Для каждой категории берутся:
    - strong_terms
    - weak_terms
    - model_labels
    """

    if not TAXONOMY_PATH.exists():
        print("Файл risk_taxonomy.json не найден.")
        print("Проверь, что он лежит в корне проекта рядом с PZ8_postprocessing.py.")
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
        print("Не удалось прочитать risk_taxonomy.json.")
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
    """
    Преобразует таксономию в словарь вида:
    {
        "weapon": [...],
        "violence": [...]
    }
    """

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


# =====================================================
# АВТООПРЕДЕЛЕНИЕ FPS И КОЛИЧЕСТВА КАДРОВ
# =====================================================

def get_video_metadata(video_path):
    metadata = {
        "fps": DEFAULT_FPS,
        "frameCount": 0,
        "video_duration_seconds": 0.0,
        "video_duration_formatted": "00:00:00",
        "metadata_source": "fallback"
    }

    if not video_path:
        return metadata

    video_path = str(video_path)

    if not Path(video_path).exists():
        return metadata

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        return metadata

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    fps = safe_float(fps, DEFAULT_FPS)
    frame_count = safe_int(frame_count, 0)

    if fps <= 0:
        fps = DEFAULT_FPS

    duration_seconds = 0.0

    if frame_count > 0 and fps > 0:
        duration_seconds = round(frame_count / fps, 3)

    metadata = {
        "fps": round(fps, 3),
        "frameCount": frame_count,
        "video_duration_seconds": duration_seconds,
        "video_duration_formatted": seconds_to_time(duration_seconds),
        "metadata_source": "opencv"
    }

    return metadata


# =====================================================
# ВЫБОР ЗАПУСКОВ
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
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЗ3–ПЗ7
# =====================================================

def load_pz3_results(pz3_run):
    if pz3_run is None:
        return {
            "ocr_unique_texts": [],
            "ocr_segments": []
        }

    unique_texts_path = pz3_run / "unique_texts.txt"
    segments_excel_path = pz3_run / "deduplicated_segments.xlsx"

    ocr_unique_texts = []

    if unique_texts_path.exists():
        with open(unique_texts_path, "r", encoding="utf-8") as file:
            ocr_unique_texts = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

    segments_df = read_excel_safe(segments_excel_path)

    return {
        "ocr_unique_texts": ocr_unique_texts,
        "ocr_segments": dataframe_to_records(segments_df)
    }


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


def load_pz5_results(pz5_run):
    if pz5_run is None:
        return {
            "yolo_objects": [],
            "yolo_frames": [],
            "yolo_json": {}
        }

    excel_path = pz5_run / "yolo_detection_results.xlsx"
    json_path = pz5_run / "yolo_detection_report.json"

    objects_df = read_excel_safe(excel_path, sheet_name="objects")
    frames_df = read_excel_safe(excel_path, sheet_name="frames")

    return {
        "yolo_objects": dataframe_to_records(objects_df),
        "yolo_frames": dataframe_to_records(frames_df),
        "yolo_json": read_json(json_path)
    }


def load_pz6_results(pz6_run):
    if pz6_run is None:
        return {
            "resnet_top1": []
        }

    excel_path = pz6_run / "resnet_classification_results.xlsx"
    top1_df = read_excel_safe(excel_path, sheet_name="top1_results")

    return {
        "resnet_top1": dataframe_to_records(top1_df)
    }


def load_pz7_results(pz7_run):
    if pz7_run is None:
        return {
            "llm_results": []
        }

    excel_path = pz7_run / "llm_image_analysis_results.xlsx"
    results_df = read_excel_safe(excel_path, sheet_name="llm_results")

    return {
        "llm_results": dataframe_to_records(results_df)
    }


# =====================================================
# СОЗДАНИЕ DETECTIONS В ФОРМАТЕ ПРЕПОДАВАТЕЛЯ
# =====================================================

def make_detection(
    subclass,
    detection_type,
    start_seconds,
    end_seconds,
    fps,
    confidence=0.9
):
    start_seconds = safe_float(start_seconds, 0.0)
    end_seconds = safe_float(end_seconds, start_seconds)

    if end_seconds < start_seconds:
        end_seconds = start_seconds

    start_frame = frame_from_seconds(start_seconds, fps)
    end_frame = frame_from_seconds(end_seconds, fps)

    start_time = seconds_to_time(start_seconds)
    end_time = seconds_to_time(end_seconds)

    return {
        "startFrame": start_frame,
        "endFrame": end_frame,
        "start_time": start_time,
        "end_time": end_time,
        "time_interval": f"{start_time} - {end_time}",
        "subclass": subclass,
        "confidence": round(float(confidence), 4),
        "type": detection_type
    }


def add_keyword_detections_from_text(
    detections,
    text,
    start_seconds,
    end_seconds,
    fps,
    detection_type,
    base_confidence,
    keyword_dict
):
    for subclass, keywords in keyword_dict.items():
        hits = contains_any_keyword(text, keywords)

        if hits:
            detections.append(
                make_detection(
                    subclass=subclass,
                    detection_type=detection_type,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    fps=fps,
                    confidence=base_confidence
                )
            )


def build_text_detections(pz3_data, pz4_data, fps, keyword_dict):
    detections = []

    for row in pz3_data.get("ocr_segments", []):
        text = row.get("final_text", "")

        if not text:
            text = row.get("recognized_text", "")

        start_seconds = row.get("start_time_seconds", "")
        end_seconds = row.get("end_time_seconds", "")

        add_keyword_detections_from_text(
            detections=detections,
            text=text,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            fps=fps,
            detection_type="text",
            base_confidence=0.85,
            keyword_dict=keyword_dict
        )

    if not pz3_data.get("ocr_segments"):
        for text in pz3_data.get("ocr_unique_texts", []):
            add_keyword_detections_from_text(
                detections=detections,
                text=text,
                start_seconds=0,
                end_seconds=0,
                fps=fps,
                detection_type="text",
                base_confidence=0.75,
                keyword_dict=keyword_dict
            )

    for row in pz4_data.get("audio_segments", []):
        text = row.get("text", "")

        start_seconds = row.get("start_seconds", "")
        end_seconds = row.get("end_seconds", "")

        add_keyword_detections_from_text(
            detections=detections,
            text=text,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            fps=fps,
            detection_type="audio",
            base_confidence=0.85,
            keyword_dict=keyword_dict
        )

    return detections


def build_yolo_detections(pz5_data, fps, keyword_dict):
    detections = []

    for row in pz5_data.get("yolo_objects", []):
        class_name = row.get("class_name", "")
        confidence = safe_float(row.get("confidence", 0.8), 0.8)
        time_seconds = row.get("time_seconds", "")

        for subclass, keywords in keyword_dict.items():
            hits = contains_any_keyword(class_name, keywords)

            if hits:
                detections.append(
                    make_detection(
                        subclass=subclass,
                        detection_type="video",
                        start_seconds=time_seconds,
                        end_seconds=safe_float(time_seconds, 0.0) + 1.0,
                        fps=fps,
                        confidence=confidence
                    )
                )

    return detections


def build_resnet_detections(pz6_data, fps, keyword_dict):
    detections = []

    for row in pz6_data.get("resnet_top1", []):
        class_name = row.get("top1_class_name", "")
        confidence = safe_float(row.get("top1_confidence", 0.75), 0.75)
        time_seconds = row.get("time_seconds", "")

        for subclass, keywords in keyword_dict.items():
            hits = contains_any_keyword(class_name, keywords)

            if hits:
                detections.append(
                    make_detection(
                        subclass=subclass,
                        detection_type="video",
                        start_seconds=time_seconds,
                        end_seconds=safe_float(time_seconds, 0.0) + 1.0,
                        fps=fps,
                        confidence=confidence
                    )
                )

    return detections


def build_llm_detections(pz7_data, fps, keyword_dict):
    detections = []

    for row in pz7_data.get("llm_results", []):
        time_seconds = row.get("time_seconds", "")

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

        add_keyword_detections_from_text(
            detections=detections,
            text=full_text,
            start_seconds=time_seconds,
            end_seconds=safe_float(time_seconds, 0.0) + 1.0,
            fps=fps,
            detection_type="video",
            base_confidence=0.9,
            keyword_dict=keyword_dict
        )

    return detections


# =====================================================
# ФИЛЬТРАЦИЯ И СКЛЕЙКА DETECTIONS
# =====================================================

def filter_low_confidence_detections(detections):
    filtered = []

    for det in detections:
        det_type = det.get("type", "")
        confidence = safe_float(det.get("confidence", 0.0), 0.0)

        if det_type == "video" and confidence < MIN_VIDEO_CONFIDENCE:
            continue

        filtered.append(det)

    return filtered


def merge_time_based_detections(detections, fps, max_gap_seconds=3.0):
    if not detections:
        return []

    prepared = sorted(
        detections,
        key=lambda row: (
            row.get("subclass", ""),
            row.get("type", ""),
            row.get("startFrame", 0)
        )
    )

    grouped = []
    current = None

    max_gap_frames = int(max_gap_seconds * fps)

    for det in prepared:
        if current is None:
            current = det.copy()
            continue

        same_class = det.get("subclass") == current.get("subclass")
        same_type = det.get("type") == current.get("type")

        gap_frames = safe_int(det.get("startFrame", 0), 0) - safe_int(current.get("endFrame", 0), 0)

        if same_class and same_type and gap_frames <= max_gap_frames:
            current["endFrame"] = max(
                safe_int(current.get("endFrame", 0), 0),
                safe_int(det.get("endFrame", 0), 0)
            )

            current["confidence"] = round(
                max(
                    safe_float(current.get("confidence", 0.0), 0.0),
                    safe_float(det.get("confidence", 0.0), 0.0)
                ),
                4
            )
        else:
            grouped.append(current)
            current = det.copy()

    if current is not None:
        grouped.append(current)

    return grouped


def normalize_detection_times(detections, fps):
    normalized = []

    for det in detections:
        start_frame = safe_int(det.get("startFrame", 0), 0)
        end_frame = safe_int(det.get("endFrame", 0), 0)

        start_seconds = start_frame / fps
        end_seconds = end_frame / fps

        start_time = seconds_to_time(start_seconds)
        end_time = seconds_to_time(end_seconds)

        normalized.append({
            "startFrame": start_frame,
            "endFrame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "time_interval": f"{start_time} - {end_time}",
            "subclass": det.get("subclass", ""),
            "confidence": round(safe_float(det.get("confidence", 0.0), 0.0), 4),
            "type": det.get("type", "")
        })

    return normalized


def build_all_detections(
    pz3_data,
    pz4_data,
    pz5_data,
    pz6_data,
    pz7_data,
    fps,
    keyword_dict
):
    detections = []

    detections.extend(build_text_detections(pz3_data, pz4_data, fps, keyword_dict))
    detections.extend(build_yolo_detections(pz5_data, fps, keyword_dict))
    detections.extend(build_resnet_detections(pz6_data, fps, keyword_dict))
    detections.extend(build_llm_detections(pz7_data, fps, keyword_dict))

    detections = filter_low_confidence_detections(detections)

    detections = merge_time_based_detections(
        detections,
        fps=fps,
        max_gap_seconds=3.0
    )

    detections = normalize_detection_times(detections, fps)

    detections = sorted(
        detections,
        key=lambda row: (
            row.get("startFrame", 0),
            row.get("subclass", ""),
            row.get("type", "")
        )
    )

    return detections


# =====================================================
# SOURCE INFO
# =====================================================

def get_video_path_from_data(pz4_data, pz5_data):
    audio_json = pz4_data.get("audio_json", {})

    if audio_json.get("video_path"):
        return audio_json.get("video_path")

    yolo_json = pz5_data.get("yolo_json", {})

    if yolo_json.get("video_path"):
        return yolo_json.get("video_path")

    return ""


def calculate_frame_count_fallback(pz5_data, detections):
    yolo_frames = pz5_data.get("yolo_frames", [])

    frame_numbers = []

    for row in yolo_frames:
        frame_number = row.get("frame_number", "")

        if frame_number != "":
            frame_numbers.append(safe_int(frame_number, 0))

    if frame_numbers:
        return max(frame_numbers) + 1

    detection_frames = [
        safe_int(row.get("endFrame", 0), 0)
        for row in detections
    ]

    if detection_frames:
        return max(detection_frames) + 1

    return 0


def build_source_info(pz4_data, pz5_data, detections, metadata):
    video_path = get_video_path_from_data(pz4_data, pz5_data)

    fps = safe_float(metadata.get("fps", DEFAULT_FPS), DEFAULT_FPS)
    frame_count = safe_int(metadata.get("frameCount", 0), 0)

    if frame_count <= 0:
        frame_count = calculate_frame_count_fallback(pz5_data, detections)

    if fps <= 0:
        fps = DEFAULT_FPS

    if frame_count > 0:
        duration_seconds = round(frame_count / fps, 3)
    else:
        duration_seconds = 0.0

    source_info = {
        "frameCount": frame_count,
        "fps": fps,
        "video_path": video_path,
        "video_duration_seconds": duration_seconds,
        "video_duration_formatted": seconds_to_time(duration_seconds),
        "analysis_timestamp": datetime.now().isoformat()
    }

    return source_info


# =====================================================
# СОХРАНЕНИЕ ФИНАЛЬНОГО JSON
# =====================================================

def save_final_json(source_info, detections):
    json_path = RUN_DIR / "final_analysis_report.json"

    report = {
        "report_type": "TIME_BASED_REPORT",
        "source_info": source_info,
        "detections": detections
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return json_path


def save_excel_preview(source_info, detections):
    excel_path = RUN_DIR / "final_analysis_preview.xlsx"

    source_df = pd.DataFrame([source_info])
    detections_df = pd.DataFrame(detections)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="source_info", index=False)
        detections_df.to_excel(writer, sheet_name="detections", index=False)

    return excel_path


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ8: финальный JSON в формате TIME_BASED_REPORT")
    print("=" * 70)

    print("\nЭтот скрипт формирует один главный JSON:")
    print("final_analysis_report.json")
    print("\nФормат JSON:")
    print("report_type + source_info + detections")
    print(f"\nФильтр video-срабатываний: confidence >= {MIN_VIDEO_CONFIDENCE}")

    taxonomy = load_risk_taxonomy()
    keyword_dict = build_keyword_dict(taxonomy)

    print("\nТаксономия риска:")
    print(f"taxonomy_name: {taxonomy.get('taxonomy_name')}")
    print(f"taxonomy_version: {taxonomy.get('taxonomy_version')}")
    print(f"categories: {list(keyword_dict.keys())}")

    selected_runs = choose_all_runs()

    print("\nЗагружаем результаты ПЗ3–ПЗ7...")

    pz3_data = load_pz3_results(selected_runs.get("pz3_run"))
    pz4_data = load_pz4_results(selected_runs.get("pz4_run"))
    pz5_data = load_pz5_results(selected_runs.get("pz5_run"))
    pz6_data = load_pz6_results(selected_runs.get("pz6_run"))
    pz7_data = load_pz7_results(selected_runs.get("pz7_run"))

    video_path = get_video_path_from_data(pz4_data, pz5_data)
    metadata = get_video_metadata(video_path)

    fps = safe_float(metadata.get("fps", DEFAULT_FPS), DEFAULT_FPS)

    print("\nМетаданные видео:")
    print(f"video_path: {video_path}")
    print(f"fps: {fps}")
    print(f"frameCount: {metadata.get('frameCount')}")
    print(f"metadata_source: {metadata.get('metadata_source')}")

    print("\nФормируем detections...")

    detections = build_all_detections(
        pz3_data=pz3_data,
        pz4_data=pz4_data,
        pz5_data=pz5_data,
        pz6_data=pz6_data,
        pz7_data=pz7_data,
        fps=fps,
        keyword_dict=keyword_dict
    )

    source_info = build_source_info(
        pz4_data=pz4_data,
        pz5_data=pz5_data,
        detections=detections,
        metadata=metadata
    )

    json_path = save_final_json(
        source_info=source_info,
        detections=detections
    )

    excel_path = save_excel_preview(
        source_info=source_info,
        detections=detections
    )

    print("\n" + "=" * 70)
    print("ПЗ8 ГОТОВО")
    print("=" * 70)

    print(f"Папка результата: {RUN_DIR}")
    print(f"Главный JSON: {json_path}")
    print(f"Excel для проверки: {excel_path}")

    print("\nКраткая статистика:")
    print(f"frameCount: {source_info.get('frameCount')}")
    print(f"fps: {source_info.get('fps')}")
    print(f"video_duration_formatted: {source_info.get('video_duration_formatted')}")
    print(f"detections: {len(detections)}")

    if detections:
        print("\nПервые detections:")
        for item in detections[:5]:
            print(item)
    else:
        print("\nРиск-срабатывания не найдены.")
        print("Если на видео точно есть оружие, значит предыдущие модули не передали признаки в текст/YOLO/ResNet/LLM.")


if __name__ == "__main__":
    main()