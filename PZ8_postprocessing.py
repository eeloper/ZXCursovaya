import json
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import pandas as pd

from project_config import BASE_DIR, load_run_config, RUN_CONFIG_PATH


# =====================================================
# НАСТРОЙКИ
# =====================================================

DEFAULT_FPS = 30.0
MIN_VIDEO_CONFIDENCE = 0.30

TAXONOMY_PATH = BASE_DIR / "risk_taxonomy.json"


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def save_run_config(config):
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
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

    text = re.sub(r"[^a-zа-я0-9 ,]", " ", text, flags=re.IGNORECASE)
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
    json_path = Path(json_path)

    if not json_path.exists():
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def read_excel_safe(excel_path, sheet_name=0):
    excel_path = Path(excel_path)

    if not excel_path.exists():
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


def detect_categories(text, keyword_dict):
    detected = []

    for subclass, keywords in keyword_dict.items():
        hits = contains_any_keyword(text, keywords)

        if hits:
            detected.append(subclass)

    return list(sorted(set(detected)))


# =====================================================
# МЕТАДАННЫЕ ВИДЕО
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

    video_path = Path(video_path)

    if not video_path.exists():
        return metadata

    cap = cv2.VideoCapture(str(video_path))

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
# ЗАГРУЗКА РЕЗУЛЬТАТОВ ИЗ CURRENT_RUN
# =====================================================

def load_pz3_results(config):
    ocr_dir = Path(config["ocr_dir"])

    unique_texts_path = ocr_dir / "unique_texts.txt"
    segments_excel_path = ocr_dir / "deduplicated_segments.xlsx"
    raw_excel_path = ocr_dir / "raw_ocr_results.xlsx"
    json_path = ocr_dir / "ocr_results.json"

    unique_texts = []

    if unique_texts_path.exists():
        with open(unique_texts_path, "r", encoding="utf-8") as file:
            unique_texts = [
                line.strip()
                for line in file.readlines()
                if line.strip()
            ]

    segments_df = read_excel_safe(segments_excel_path)
    raw_df = read_excel_safe(raw_excel_path)

    return {
        "ocr_dir": str(ocr_dir),
        "ocr_unique_texts": unique_texts,
        "ocr_segments": dataframe_to_records(segments_df),
        "ocr_raw_rows": dataframe_to_records(raw_df),
        "ocr_json": read_json(json_path)
    }


def load_pz4_results(config):
    whisper_dir = Path(config["whisper_dir"])

    transcript_path = whisper_dir / "input_video_transcript.txt"
    segments_excel_path = whisper_dir / "input_video_whisper_segments.xlsx"
    json_path = whisper_dir / "input_video_whisper_report.json"

    full_text = ""

    if transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as file:
            full_text = file.read().strip()

    segments_df = read_excel_safe(segments_excel_path)

    return {
        "whisper_dir": str(whisper_dir),
        "audio_full_text": full_text,
        "audio_segments": dataframe_to_records(segments_df),
        "audio_json": read_json(json_path)
    }


def load_pz5_results(config):
    yolo_dir = Path(config["yolo_dir"])

    excel_path = yolo_dir / "yolo_detection_results.xlsx"
    json_path = yolo_dir / "yolo_detection_report.json"

    objects_df = read_excel_safe(excel_path, sheet_name="objects")
    frames_df = read_excel_safe(excel_path, sheet_name="frames")
    class_summary_df = read_excel_safe(excel_path, sheet_name="class_summary")

    return {
        "yolo_dir": str(yolo_dir),
        "yolo_objects": dataframe_to_records(objects_df),
        "yolo_frames": dataframe_to_records(frames_df),
        "yolo_class_summary": dataframe_to_records(class_summary_df),
        "yolo_json": read_json(json_path)
    }


def load_pz6_results(config):
    resnet_dir = Path(config["resnet_dir"])

    excel_path = resnet_dir / "resnet_classification_results.xlsx"
    json_path = resnet_dir / "resnet_classification_report.json"

    top1_df = read_excel_safe(excel_path, sheet_name="top1_results")
    topk_df = read_excel_safe(excel_path, sheet_name="topk_results")
    class_summary_df = read_excel_safe(excel_path, sheet_name="class_summary")

    return {
        "resnet_dir": str(resnet_dir),
        "resnet_top1": dataframe_to_records(top1_df),
        "resnet_topk": dataframe_to_records(topk_df),
        "resnet_class_summary": dataframe_to_records(class_summary_df),
        "resnet_json": read_json(json_path)
    }


def load_pz7_results(config):
    llm_dir = Path(config["llm_dir"])

    excel_path = llm_dir / "llm_image_analysis_results.xlsx"
    json_path = llm_dir / "llm_image_analysis_report.json"

    llm_json = read_json(json_path)

    results_from_json = llm_json.get("results", [])

    if results_from_json:
        return {
            "llm_dir": str(llm_dir),
            "llm_results": results_from_json,
            "llm_json": llm_json
        }

    results_df = read_excel_safe(excel_path, sheet_name="llm_results")

    return {
        "llm_dir": str(llm_dir),
        "llm_results": dataframe_to_records(results_df),
        "llm_json": llm_json
    }


# =====================================================
# DETECTIONS В ФОРМАТЕ ПРЕПОДАВАТЕЛЯ
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
    categories = detect_categories(text, keyword_dict)

    for subclass in categories:
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


def build_ocr_detections(pz3_data, fps, keyword_dict):
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

    return detections


def build_audio_detections(pz4_data, fps, keyword_dict):
    detections = []

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

        categories = detect_categories(class_name, keyword_dict)

        for subclass in categories:
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

        categories = detect_categories(class_name, keyword_dict)

        for subclass in categories:
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


def normalize_risk_flags(value):
    value = clean_text(value).lower()
    value = value.replace(" ", "")

    if value in ["", "none", "no", "unknown"]:
        return []

    parts = [
        part.strip()
        for part in value.split(",")
        if part.strip()
    ]

    return list(sorted(set(parts)))


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
            row.get("moderation_comment_ru", ""),
            row.get("yolo_class_hint", "")
        ]

        full_text = " ".join([
            clean_text(field)
            for field in fields
            if field
        ])

        subclasses = normalize_risk_flags(row.get("possible_risk_flags", ""))

        if not subclasses:
            subclasses = detect_categories(full_text, keyword_dict)

        for subclass in subclasses:
            if subclass == "none":
                continue

            detections.append(
                make_detection(
                    subclass=subclass,
                    detection_type="video",
                    start_seconds=time_seconds,
                    end_seconds=safe_float(time_seconds, 0.0) + 1.0,
                    fps=fps,
                    confidence=0.9
                )
            )

    return detections


# =====================================================
# ФИЛЬТРАЦИЯ И СКЛЕЙКА
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

        gap_frames = (
            safe_int(det.get("startFrame", 0), 0)
            - safe_int(current.get("endFrame", 0), 0)
        )

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


def build_all_detections(pz3_data, pz4_data, pz5_data, pz6_data, pz7_data, fps, keyword_dict):
    detections = []

    detections.extend(build_ocr_detections(pz3_data, fps, keyword_dict))
    detections.extend(build_audio_detections(pz4_data, fps, keyword_dict))
    detections.extend(build_yolo_detections(pz5_data, fps, keyword_dict))
    detections.extend(build_resnet_detections(pz6_data, fps, keyword_dict))
    detections.extend(build_llm_detections(pz7_data, fps, keyword_dict))

    detections = filter_low_confidence_detections(detections)

    detections = merge_time_based_detections(
        detections=detections,
        fps=fps,
        max_gap_seconds=3.0
    )

    detections = normalize_detection_times(
        detections=detections,
        fps=fps
    )

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
# SUMMARY И RECOMMENDATION
# =====================================================

def build_summary(detections):
    detections_count = len(detections)

    subclass_counter = {}
    type_counter = {}

    max_confidence = 0.0

    for det in detections:
        subclass = det.get("subclass", "unknown")
        detection_type = det.get("type", "unknown")
        confidence = safe_float(det.get("confidence", 0.0), 0.0)

        subclass_counter[subclass] = subclass_counter.get(subclass, 0) + 1
        type_counter[detection_type] = type_counter.get(detection_type, 0) + 1

        if confidence > max_confidence:
            max_confidence = confidence

    summary = {
        "detections_count": detections_count,
        "weapon_detections_count": subclass_counter.get("weapon", 0),
        "violence_detections_count": subclass_counter.get("violence", 0),
        "video_detections_count": type_counter.get("video", 0),
        "audio_detections_count": type_counter.get("audio", 0),
        "text_detections_count": type_counter.get("text", 0),
        "max_confidence": round(max_confidence, 4),
        "detections_by_subclass": subclass_counter,
        "detections_by_type": type_counter
    }

    return summary


def build_recommendation(summary, detections):
    detections_count = summary.get("detections_count", 0)
    weapon_count = summary.get("weapon_detections_count", 0)
    violence_count = summary.get("violence_detections_count", 0)
    max_confidence = summary.get("max_confidence", 0.0)

    has_high_confidence_weapon = any(
        det.get("subclass") == "weapon"
        and safe_float(det.get("confidence", 0.0), 0.0) >= 0.80
        for det in detections
    )

    has_high_confidence_violence = any(
        det.get("subclass") == "violence"
        and safe_float(det.get("confidence", 0.0), 0.0) >= 0.80
        for det in detections
    )

    if detections_count == 0:
        return {
            "is_dangerous": False,
            "risk_level": "low",
            "decision": "no_risk_detected",
            "comment": "По результатам автоматической проверки признаки потенциально опасного контента не обнаружены."
        }

    if (
        detections_count >= 3
        or weapon_count >= 2
        or violence_count >= 2
        or has_high_confidence_weapon
        or has_high_confidence_violence
        or max_confidence >= 0.90
    ):
        return {
            "is_dangerous": True,
            "risk_level": "high",
            "decision": "manual_review_required",
            "comment": "В видеоролике обнаружены признаки потенциально опасного контента. Рекомендуется ручная проверка."
        }

    return {
        "is_dangerous": True,
        "risk_level": "medium",
        "decision": "manual_review_recommended",
        "comment": "В видеоролике обнаружены отдельные риск-срабатывания. Рекомендуется дополнительная ручная проверка."
    }


# =====================================================
# SOURCE_INFO И СОХРАНЕНИЕ
# =====================================================

def build_source_info(config, metadata, taxonomy):
    fps = safe_float(metadata.get("fps", DEFAULT_FPS), DEFAULT_FPS)
    frame_count = safe_int(metadata.get("frameCount", 0), 0)

    if frame_count <= 0:
        frame_count = safe_int(config.get("frameCount", 0), 0)

    if fps <= 0:
        fps = DEFAULT_FPS

    if frame_count > 0:
        duration_seconds = round(frame_count / fps, 3)
    else:
        duration_seconds = safe_float(config.get("video_duration_seconds", 0.0), 0.0)

    source_info = {
        "frameCount": frame_count,
        "fps": fps,
        "video_path": config.get("video_path", ""),
        "video_duration_seconds": duration_seconds,
        "video_duration_formatted": seconds_to_time(duration_seconds),
        "analysis_timestamp": datetime.now().isoformat(),
        "taxonomy_name": taxonomy.get("taxonomy_name", ""),
        "taxonomy_version": taxonomy.get("taxonomy_version", "")
    }

    return source_info


def save_final_json(final_dir, source_info, summary, recommendation, detections):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    json_path = final_dir / "final_analysis_report.json"

    report = {
        "report_type": "TIME_BASED_REPORT",
        "source_info": source_info,
        "summary": summary,
        "recommendation": recommendation,
        "detections": detections
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return json_path


def save_excel_preview(final_dir, source_info, summary, recommendation, detections):
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    excel_path = final_dir / "final_analysis_preview.xlsx"

    source_df = pd.DataFrame([source_info])
    summary_df = pd.DataFrame([summary])
    recommendation_df = pd.DataFrame([recommendation])
    detections_df = pd.DataFrame(detections)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="source_info", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        recommendation_df.to_excel(writer, sheet_name="recommendation", index=False)
        detections_df.to_excel(writer, sheet_name="detections", index=False)

    return excel_path


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ8: постобработка и финальный JSON")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выполни предыдущие этапы.")
        print(error)
        return

    taxonomy = load_risk_taxonomy()
    keyword_dict = build_keyword_dict(taxonomy)

    print("\nТаксономия риска:")
    print(f"taxonomy_name: {taxonomy.get('taxonomy_name')}")
    print(f"taxonomy_version: {taxonomy.get('taxonomy_version')}")
    print(f"categories: {list(keyword_dict.keys())}")

    video_path = config.get("video_path", "")
    metadata = get_video_metadata(video_path)
    fps = safe_float(metadata.get("fps", DEFAULT_FPS), DEFAULT_FPS)

    print("\nМетаданные видео:")
    print(f"video_path: {video_path}")
    print(f"fps: {fps}")
    print(f"frameCount: {metadata.get('frameCount')}")
    print(f"metadata_source: {metadata.get('metadata_source')}")

    print("\nЗагружаем результаты из current_run...")

    pz3_data = load_pz3_results(config)
    pz4_data = load_pz4_results(config)
    pz5_data = load_pz5_results(config)
    pz6_data = load_pz6_results(config)
    pz7_data = load_pz7_results(config)

    print("\nНайдено данных:")
    print(f"OCR segments: {len(pz3_data.get('ocr_segments', []))}")
    print(f"Whisper segments: {len(pz4_data.get('audio_segments', []))}")
    print(f"YOLO objects: {len(pz5_data.get('yolo_objects', []))}")
    print(f"ResNet top1 rows: {len(pz6_data.get('resnet_top1', []))}")
    print(f"LLM results: {len(pz7_data.get('llm_results', []))}")

    print("\nФормируем итоговые detections...")

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
        config=config,
        metadata=metadata,
        taxonomy=taxonomy
    )

    summary = build_summary(detections)
    recommendation = build_recommendation(summary, detections)

    final_dir = Path(config["final_dir"])

    json_path = save_final_json(
        final_dir=final_dir,
        source_info=source_info,
        summary=summary,
        recommendation=recommendation,
        detections=detections
    )

    excel_path = save_excel_preview(
        final_dir=final_dir,
        source_info=source_info,
        summary=summary,
        recommendation=recommendation,
        detections=detections
    )

    config["final_json_path"] = str(json_path)
    config["final_preview_excel_path"] = str(excel_path)
    config["final_detections_count"] = summary.get("detections_count", 0)
    config["final_risk_level"] = recommendation.get("risk_level", "")
    config["final_is_dangerous"] = recommendation.get("is_dangerous", False)
    config["final_decision"] = recommendation.get("decision", "")
    config["pz8_status"] = "success"
    config["pz8_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\n" + "=" * 70)
    print("ПЗ8 ГОТОВО")
    print("=" * 70)

    print(f"Главный JSON: {json_path}")
    print(f"Excel для проверки: {excel_path}")

    print("\nКраткая статистика:")
    print(f"frameCount: {source_info.get('frameCount')}")
    print(f"fps: {source_info.get('fps')}")
    print(f"video_duration_formatted: {source_info.get('video_duration_formatted')}")
    print(f"detections_count: {summary.get('detections_count')}")
    print(f"weapon_detections_count: {summary.get('weapon_detections_count')}")
    print(f"violence_detections_count: {summary.get('violence_detections_count')}")
    print(f"risk_level: {recommendation.get('risk_level')}")
    print(f"decision: {recommendation.get('decision')}")
    print(f"is_dangerous: {recommendation.get('is_dangerous')}")

    if detections:
        print("\nПервые detections:")
        for item in detections[:5]:
            print(item)
    else:
        print("\nРиск-срабатывания не найдены.")


if __name__ == "__main__":
    main()