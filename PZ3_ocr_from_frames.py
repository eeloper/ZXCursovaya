import json
import re
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher

import cv2
import pandas as pd
import easyocr

from project_config import load_run_config, RUN_CONFIG_PATH


# =====================================================
# НАСТРОЙКИ
# =====================================================

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

MIN_TEXT_CONFIDENCE = 0.20
TEXT_SIMILARITY_THRESHOLD = 0.78


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def save_run_config(config):
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def clear_output_folder(folder_path):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()


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


def extract_time_from_filename(filename):
    """
    Извлекает время из имени кадра.

    Пример:
    frame_000001_source_000029_time_00000990ms.jpg
    """

    match = re.search(r"time_(\d+)ms", filename)

    if not match:
        return None, ""

    milliseconds = int(match.group(1))
    seconds = milliseconds / 1000

    return round(seconds, 3), seconds_to_time(seconds)


def extract_frame_number_from_filename(filename):
    match = re.search(r"frame_(\d+)", filename)

    if not match:
        return None

    return int(match.group(1))


def extract_source_frame_number_from_filename(filename):
    match = re.search(r"source_(\d+)", filename)

    if not match:
        return None

    return int(match.group(1))


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
        "i": "и"
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

    if len(shorter) >= 6 and shorter in longer:
        return 0.92

    return SequenceMatcher(None, norm_1, norm_2).ratio()


def get_frame_files(frames_dir):
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        return []

    frame_files = [
        file for file in frames_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(frame_files)


# =====================================================
# OCR
# =====================================================

def create_easyocr_reader():
    print("\nЗагружается EasyOCR...")
    print("Используются языки: ru, en")

    reader = easyocr.Reader(
        ["ru", "en"],
        gpu=False
    )

    return reader


def preprocess_frame_for_ocr(image_path):
    """
    Лёгкая предобработка кадра перед OCR.
    Не меняет исходный файл, только готовит изображение в памяти.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Увеличиваем изображение, чтобы OCR лучше видел мелкие титры.
    scale = 1.5
    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # Усиление контраста.
    equalized = cv2.equalizeHist(resized)

    return equalized


def recognize_text_on_frame(reader, frame_path):
    prepared_image = preprocess_frame_for_ocr(frame_path)

    if prepared_image is None:
        return "", 0.0, []

    try:
        results = reader.readtext(
            prepared_image,
            detail=1,
            paragraph=False
        )
    except Exception as error:
        return "", 0.0, [{
            "error": str(error)
        }]

    text_parts = []
    confidences = []
    details = []

    for item in results:
        bbox, text, confidence = item

        text = clean_text(text)
        confidence = float(confidence)

        if not text:
            continue

        if confidence < MIN_TEXT_CONFIDENCE:
            continue

        text_parts.append(text)
        confidences.append(confidence)

        details.append({
            "text": text,
            "confidence": round(confidence, 4)
        })

    full_text = clean_text(" ".join(text_parts))

    if confidences:
        mean_confidence = sum(confidences) / len(confidences)
    else:
        mean_confidence = 0.0

    return full_text, round(mean_confidence, 4), details


def process_frames(frames_dir):
    frame_files = get_frame_files(frames_dir)

    if not frame_files:
        print("\nВ папке кадров нет изображений:")
        print(frames_dir)
        return []

    print(f"\nНайдено кадров для OCR: {len(frame_files)}")

    reader = create_easyocr_reader()

    rows = []

    for index, frame_path in enumerate(frame_files, start=1):
        print(f"OCR {index}/{len(frame_files)}: {frame_path.name}")

        time_seconds, time_formatted = extract_time_from_filename(frame_path.name)
        frame_number = extract_frame_number_from_filename(frame_path.name)
        source_frame_number = extract_source_frame_number_from_filename(frame_path.name)

        recognized_text, confidence, details = recognize_text_on_frame(
            reader,
            frame_path
        )

        rows.append({
            "frame_file": frame_path.name,
            "frame_path": str(frame_path),
            "frame_number": frame_number,
            "source_frame_number": source_frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "recognized_text": recognized_text,
            "confidence": confidence,
            "ocr_details_json": json.dumps(details, ensure_ascii=False)
        })

    return rows


# =====================================================
# ДЕДУБЛИКАЦИЯ OCR
# =====================================================

def build_deduplicated_segments(raw_rows):
    """
    Склеивает похожие OCR-тексты с соседних кадров в сегменты.
    """

    segments = []

    current_segment = None

    for row in raw_rows:
        text = clean_text(row.get("recognized_text", ""))

        if not text:
            continue

        time_seconds = row.get("time_seconds", "")
        frame_number = row.get("frame_number", "")
        source_frame_number = row.get("source_frame_number", "")
        confidence = row.get("confidence", 0.0)

        if current_segment is None:
            current_segment = {
                "start_frame": frame_number,
                "end_frame": frame_number,
                "start_source_frame": source_frame_number,
                "end_source_frame": source_frame_number,
                "start_time_seconds": time_seconds,
                "end_time_seconds": time_seconds,
                "start_time": seconds_to_time(time_seconds),
                "end_time": seconds_to_time(time_seconds),
                "final_text": text,
                "texts": [text],
                "max_confidence": confidence,
                "frames_count": 1
            }
            continue

        similarity = text_similarity(current_segment["final_text"], text)

        if similarity >= TEXT_SIMILARITY_THRESHOLD:
            current_segment["end_frame"] = frame_number
            current_segment["end_source_frame"] = source_frame_number
            current_segment["end_time_seconds"] = time_seconds
            current_segment["end_time"] = seconds_to_time(time_seconds)
            current_segment["texts"].append(text)
            current_segment["frames_count"] += 1

            if confidence > current_segment["max_confidence"]:
                current_segment["max_confidence"] = confidence
                current_segment["final_text"] = text

        else:
            segments.append(current_segment)

            current_segment = {
                "start_frame": frame_number,
                "end_frame": frame_number,
                "start_source_frame": source_frame_number,
                "end_source_frame": source_frame_number,
                "start_time_seconds": time_seconds,
                "end_time_seconds": time_seconds,
                "start_time": seconds_to_time(time_seconds),
                "end_time": seconds_to_time(time_seconds),
                "final_text": text,
                "texts": [text],
                "max_confidence": confidence,
                "frames_count": 1
            }

    if current_segment is not None:
        segments.append(current_segment)

    final_segments = []

    for index, segment in enumerate(segments, start=1):
        final_segments.append({
            "segment_id": index,
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "start_source_frame": segment["start_source_frame"],
            "end_source_frame": segment["end_source_frame"],
            "start_time_seconds": segment["start_time_seconds"],
            "end_time_seconds": segment["end_time_seconds"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "time_interval": f"{segment['start_time']} - {segment['end_time']}",
            "final_text": segment["final_text"],
            "max_confidence": round(float(segment["max_confidence"]), 4),
            "frames_count": segment["frames_count"]
        })

    return final_segments


def build_unique_texts(segments):
    unique_texts = []

    for segment in segments:
        text = clean_text(segment.get("final_text", ""))

        if not text:
            continue

        already_exists = False

        for existing_text in unique_texts:
            if text_similarity(existing_text, text) >= TEXT_SIMILARITY_THRESHOLD:
                already_exists = True
                break

        if not already_exists:
            unique_texts.append(text)

    return unique_texts


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def save_results(ocr_dir, raw_rows, segments, unique_texts):
    ocr_dir = Path(ocr_dir)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    raw_excel_path = ocr_dir / "raw_ocr_results.xlsx"
    segments_excel_path = ocr_dir / "deduplicated_segments.xlsx"
    unique_texts_path = ocr_dir / "unique_texts.txt"
    json_path = ocr_dir / "ocr_results.json"

    raw_df = pd.DataFrame(raw_rows)
    segments_df = pd.DataFrame(segments)

    raw_df.to_excel(raw_excel_path, index=False)
    segments_df.to_excel(segments_excel_path, index=False)

    with open(unique_texts_path, "w", encoding="utf-8") as file:
        for text in unique_texts:
            file.write(text + "\n")

    report = {
        "report_type": "OCR_RESULTS",
        "processed_frames_count": len(raw_rows),
        "segments_count": len(segments),
        "unique_texts_count": len(unique_texts),
        "raw_ocr_results_path": str(raw_excel_path),
        "deduplicated_segments_path": str(segments_excel_path),
        "unique_texts_path": str(unique_texts_path),
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return {
        "raw_excel_path": raw_excel_path,
        "segments_excel_path": segments_excel_path,
        "unique_texts_path": unique_texts_path,
        "json_path": json_path
    }


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ3: OCR текста с кадров")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выполни ПЗ2.")
        print(error)
        return

    frames_dir = Path(config["frames_dir"])
    ocr_dir = Path(config["ocr_dir"])

    if not frames_dir.exists():
        print("\nПапка кадров не найдена:")
        print(frames_dir)
        return

    frame_files = get_frame_files(frames_dir)

    if not frame_files:
        print("\nВ папке кадров нет изображений:")
        print(frames_dir)
        return

    print("\nКадры берутся из папки:")
    print(frames_dir)

    print("\nРезультаты OCR будут сохранены в папку:")
    print(ocr_dir)

    clear_output_folder(ocr_dir)

    try:
        raw_rows = process_frames(frames_dir)
        segments = build_deduplicated_segments(raw_rows)
        unique_texts = build_unique_texts(segments)

        saved_paths = save_results(
            ocr_dir=ocr_dir,
            raw_rows=raw_rows,
            segments=segments,
            unique_texts=unique_texts
        )

    except Exception as error:
        print("\nОшибка при выполнении OCR:")
        print(error)

        config["pz3_status"] = "error"
        config["pz3_error"] = str(error)
        config["pz3_finished_at"] = datetime.now().isoformat()
        save_run_config(config)

        return

    config["ocr_dir"] = str(ocr_dir)
    config["ocr_raw_results_path"] = str(saved_paths["raw_excel_path"])
    config["ocr_segments_path"] = str(saved_paths["segments_excel_path"])
    config["ocr_unique_texts_path"] = str(saved_paths["unique_texts_path"])
    config["ocr_json_path"] = str(saved_paths["json_path"])
    config["ocr_processed_frames_count"] = len(raw_rows)
    config["ocr_segments_count"] = len(segments)
    config["ocr_unique_texts_count"] = len(unique_texts)
    config["pz3_status"] = "success"
    config["pz3_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\nПЗ3 завершено успешно.")
    print(f"Обработано кадров: {len(raw_rows)}")
    print(f"Сегментов текста: {len(segments)}")
    print(f"Уникальных текстов: {len(unique_texts)}")
    print(f"Папка OCR: {ocr_dir}")
    print("\nrun_config.json обновлён.")


if __name__ == "__main__":
    main()