import cv2
import json
import re
import pandas as pd
import easyocr
import numpy as np
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher


# =====================================================
# НАСТРОЙКА ПАПОК
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

FRAME_ROOT = BASE_DIR / "results" / "pz2_frames" / "FRAME_FOLDER"
RESULT_ROOT = BASE_DIR / "results" / "pz3_ocr"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"ocr_run_{RUN_TIMESTAMP}"

RECOGNIZED_FRAMES_DIR = RUN_DIR / "recognized_frames"
OCR_AREAS_DIR = RUN_DIR / "ocr_areas"
PROCESSED_AREAS_DIR = RUN_DIR / "processed_areas"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
RECOGNIZED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
OCR_AREAS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_AREAS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# =====================================================
# ЧТЕНИЕ И СОХРАНЕНИЕ
# =====================================================

def read_image_correctly(image_path):
    image_array = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def save_image_correctly(save_path, image):
    extension = save_path.suffix
    success, encoded_image = cv2.imencode(extension, image)

    if success:
        encoded_image.tofile(str(save_path))
        return True

    return False


# =====================================================
# ВРЕМЯ И НОМЕР КАДРА
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
    match = re.search(r"time_(\d+)ms", filename)

    if not match:
        return None, ""

    milliseconds = int(match.group(1))
    seconds = milliseconds / 1000

    return round(seconds, 3), seconds_to_time(seconds)


def extract_frame_number_from_filename(filename):
    match = re.search(r"_frame_(\d+)", filename)

    if not match:
        return None

    return int(match.group(1))


# =====================================================
# ОЧИСТКА И СРАВНЕНИЕ ТЕКСТА
# =====================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("|", " ")
    text = text.replace("_", " ")
    text = text.replace("~", " ")

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def normalize_for_compare(text):
    """
    Нормализация нужна только для сравнения дублей.
    Сам текст в результатах не портим.
    """

    if not text:
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
    norm_1 = normalize_for_compare(text_1)
    norm_2 = normalize_for_compare(text_2)

    if not norm_1 or not norm_2:
        return 0

    shorter = min(norm_1, norm_2, key=len)
    longer = max(norm_1, norm_2, key=len)

    if len(shorter) >= 5 and shorter in longer:
        return 0.92

    return SequenceMatcher(None, norm_1, norm_2).ratio()


def chunks_are_similar(chunk_1, chunk_2, threshold=0.70):
    """
    Проверяет, похожи ли два фрагмента из нескольких слов.
    """

    if len(chunk_1) != len(chunk_2):
        return False

    scores = []

    for word_1, word_2 in zip(chunk_1, chunk_2):
        norm_1 = normalize_for_compare(word_1)
        norm_2 = normalize_for_compare(word_2)

        if not norm_1 or not norm_2:
            scores.append(0)
        else:
            scores.append(SequenceMatcher(None, norm_1, norm_2).ratio())

    if not scores:
        return False

    return sum(scores) / len(scores) >= threshold


def deduplicate_repeated_fragments(text):
    """
    Удаляет повторы внутри одной строки OCR.

    Было:
    aemop сценарця aemop сценарuя Александра Аверьянова Александра Аверьянова

    Станет:
    aemop сценарця Александра Аверьянова
    """

    text = clean_text(text)

    if not text:
        return ""

    words = text.split()

    if len(words) <= 1:
        return text

    changed = True

    while changed:
        changed = False
        result = []
        i = 0

        while i < len(words):
            duplicate_found = False

            max_chunk_size = min(8, (len(words) - i) // 2)

            for chunk_size in range(max_chunk_size, 0, -1):
                first_chunk = words[i:i + chunk_size]
                second_chunk = words[i + chunk_size:i + chunk_size * 2]

                if chunks_are_similar(first_chunk, second_chunk):
                    result.extend(first_chunk)
                    i += chunk_size * 2
                    duplicate_found = True
                    changed = True
                    break

            if not duplicate_found:
                result.append(words[i])
                i += 1

        words = result

    return clean_text(" ".join(words))


# =====================================================
# ВЫБОР ПАПКИ С КАДРАМИ
# =====================================================

def choose_frames_folder():
    if not FRAME_ROOT.exists():
        print("Папка с кадрами не найдена:")
        print(FRAME_ROOT)
        return None

    frame_folders = [
        folder for folder in sorted(FRAME_ROOT.iterdir())
        if folder.is_dir()
    ]

    if not frame_folders:
        print("В FRAME_FOLDER нет папок с кадрами.")
        print(FRAME_ROOT)
        return None

    print("\nНайдены папки с кадрами:")

    for index, folder in enumerate(frame_folders, start=1):
        count = len([
            file for file in folder.iterdir()
            if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])

        print(f"{index}. {folder.name} — кадров: {count}")

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


def choose_area_mode():
    print("\nГде искать текст?")
    print("1 — весь кадр")
    print("2 — нижняя часть кадра")
    print("3 — верхняя часть кадра")

    choice = input("\nВведите 1, 2 или 3 (Enter = 1): ").strip()

    if choice == "":
        choice = "1"

    if choice == "1":
        return "full"

    if choice == "2":
        return "bottom"

    if choice == "3":
        return "top"

    print("Неверный выбор. Используется весь кадр.")
    return "full"


def get_ocr_area(frame, area_mode):
    height, width = frame.shape[:2]

    if area_mode == "full":
        return frame, (0, 0, width, height)

    if area_mode == "bottom":
        y1 = int(height * 0.50)
        y2 = height
        return frame[y1:y2, 0:width], (0, y1, width, y2)

    if area_mode == "top":
        y1 = 0
        y2 = int(height * 0.50)
        return frame[y1:y2, 0:width], (0, y1, width, y2)

    return frame, (0, 0, width, height)


# =====================================================
# ПРЕДОБРАБОТКА
# =====================================================

def preprocess_area(area):
    """
    Делаем вторую версию области для OCR.
    OCR будет запускаться и по оригиналу, и по обработанной области.
    """

    gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

    scale = 1.5

    gray_big = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    contrast = clahe.apply(gray_big)

    return contrast, scale


def scale_box_to_original(box, scale):
    new_box = []

    for point in box:
        x = point[0] / scale
        y = point[1] / scale
        new_box.append([x, y])

    return new_box


# =====================================================
# OCR
# =====================================================

def run_easyocr(reader, image, source_name, scale=1.0):
    try:
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False
        )
    except Exception:
        return []

    entries = []

    for box, text, confidence in results:
        text = clean_text(text)

        if text == "":
            continue

        if len(text) == 1 and not re.search(r"[A-Za-zА-Яа-яЁё0-9]", text):
            continue

        if scale != 1.0:
            box = scale_box_to_original(box, scale)

        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        entries.append({
            "text": text,
            "confidence": float(confidence),
            "box": box,
            "x": min(xs),
            "y": min(ys),
            "source": source_name
        })

    return entries


def merge_ocr_entries(entries):
    """
    Собирает OCR-блоки одного кадра в строку
    и удаляет повторы внутри строки.
    """

    if not entries:
        return "", 0

    entries_sorted = sorted(entries, key=lambda item: (item["y"], item["x"]))

    parts = []
    confidences = []

    for entry in entries_sorted:
        text = clean_text(entry["text"])

        if not text:
            continue

        duplicate = False

        for existing in parts:
            if text_similarity(existing, text) >= 0.80:
                duplicate = True
                break

        if not duplicate:
            parts.append(text)
            confidences.append(entry["confidence"])

    full_text = clean_text(" ".join(parts))
    full_text = deduplicate_repeated_fragments(full_text)

    if confidences:
        avg_confidence = round(sum(confidences) / len(confidences), 3)
    else:
        avg_confidence = 0

    return full_text, avg_confidence


def draw_ocr_on_frame(frame, bbox, entries, final_text):
    result = frame.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3
    )

    for entry in entries:
        box = np.array(entry["box"], dtype=np.int32)

        box[:, 0] += x1
        box[:, 1] += y1

        cv2.polylines(
            result,
            [box],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

    if final_text:
        cv2.putText(
            result,
            final_text[:80],
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return result


# =====================================================
# ОБРАБОТКА КАДРОВ
# =====================================================

def process_frames(frames_folder, area_mode):
    image_files = [
        file for file in sorted(frames_folder.iterdir())
        if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("В выбранной папке нет изображений.")
        return []

    print(f"\nНайдено кадров: {len(image_files)}")
    print("Загружаем EasyOCR...")
    print("Первый запуск может занять несколько минут.")

    reader = easyocr.Reader(["ru", "en"], gpu=False)

    raw_rows = []

    for index, frame_path in enumerate(image_files, start=1):
        print(f"Обработка кадра {index}/{len(image_files)}: {frame_path.name}")

        frame = read_image_correctly(frame_path)

        if frame is None:
            continue

        frame_number = extract_frame_number_from_filename(frame_path.name)
        time_seconds, time_formatted = extract_time_from_filename(frame_path.name)

        area, bbox = get_ocr_area(frame, area_mode)

        area_path = OCR_AREAS_DIR / f"{frame_path.stem}_area.png"
        save_image_correctly(area_path, area)

        processed_area, scale = preprocess_area(area)

        processed_area_path = PROCESSED_AREAS_DIR / f"{frame_path.stem}_processed.png"
        save_image_correctly(processed_area_path, processed_area)

        original_entries = run_easyocr(
            reader,
            area,
            source_name="original",
            scale=1.0
        )

        processed_entries = run_easyocr(
            reader,
            processed_area,
            source_name="processed",
            scale=scale
        )

        all_entries = original_entries + processed_entries

        recognized_text, avg_confidence = merge_ocr_entries(all_entries)

        if recognized_text:
            annotated = draw_ocr_on_frame(
                frame,
                bbox,
                all_entries,
                recognized_text
            )

            annotated_path = RECOGNIZED_FRAMES_DIR / f"{frame_path.stem}_ocr.png"
            save_image_correctly(annotated_path, annotated)
        else:
            annotated_path = ""

        raw_rows.append({
            "frame_file": frame_path.name,
            "frame_path": str(frame_path),
            "frame_number": frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "area_mode": area_mode,
            "ocr_area_path": str(area_path),
            "processed_area_path": str(processed_area_path),
            "recognized_text": recognized_text,
            "normalized_text": normalize_for_compare(recognized_text),
            "average_confidence": avg_confidence,
            "annotated_frame_path": str(annotated_path)
        })

    return raw_rows


# =====================================================
# ДЕДУБЛИКАЦИЯ МЕЖДУ КАДРАМИ
# =====================================================

def build_unique_texts(raw_rows):
    unique_texts = []

    for row in raw_rows:
        text = deduplicate_repeated_fragments(row.get("recognized_text", ""))

        if not text:
            continue

        already_exists = False

        for existing in unique_texts:
            if text_similarity(existing, text) >= 0.78:
                already_exists = True
                break

        if not already_exists:
            unique_texts.append(text)

    return unique_texts


def build_deduplicated_segments(raw_rows):
    rows_with_text = [
        row for row in raw_rows
        if row.get("recognized_text")
    ]

    if not rows_with_text:
        return []

    rows_with_text = sorted(
        rows_with_text,
        key=lambda row: (
            row["time_seconds"] if row["time_seconds"] is not None else 999999,
            row["frame_number"] if row["frame_number"] is not None else 999999
        )
    )

    groups = []
    current_group = [rows_with_text[0]]

    max_gap_seconds = 5.0

    for row in rows_with_text[1:]:
        previous = current_group[-1]

        current_time = row.get("time_seconds")
        previous_time = previous.get("time_seconds")

        if current_time is not None and previous_time is not None:
            time_gap = current_time - previous_time
        else:
            time_gap = 0

        similarity = text_similarity(
            previous.get("recognized_text", ""),
            row.get("recognized_text", "")
        )

        if time_gap <= max_gap_seconds and similarity >= 0.60:
            current_group.append(row)
        else:
            groups.append(current_group)
            current_group = [row]

    groups.append(current_group)

    segments = []

    for index, group in enumerate(groups, start=1):
        best_row = max(
            group,
            key=lambda row: len(normalize_for_compare(row.get("recognized_text", "")))
        )

        start_time = group[0].get("time_seconds")
        end_time = group[-1].get("time_seconds")

        variants = []

        for row in group:
            text = deduplicate_repeated_fragments(row.get("recognized_text", ""))

            if text and text not in variants:
                variants.append(text)

        final_text = deduplicate_repeated_fragments(best_row.get("recognized_text", ""))

        segments.append({
            "segment_id": index,
            "start_time_seconds": start_time,
            "end_time_seconds": end_time,
            "start_time": seconds_to_time(start_time),
            "end_time": seconds_to_time(end_time),
            "time_interval": f"{seconds_to_time(start_time)} - {seconds_to_time(end_time)}",
            "start_frame": group[0].get("frame_number"),
            "end_frame": group[-1].get("frame_number"),
            "final_text": final_text,
            "frames_in_segment": len(group),
            "variants_count": len(variants),
            "text_variants": " | ".join(variants[:5]),
            "example_frame": best_row.get("frame_file", ""),
            "annotated_frame_path": best_row.get("annotated_frame_path", "")
        })

    return segments


# =====================================================
# СОХРАНЕНИЕ
# =====================================================

def save_results(frames_folder, area_mode, raw_rows, unique_texts, segments):
    raw_excel_path = RUN_DIR / "raw_ocr_results.xlsx"
    unique_excel_path = RUN_DIR / "unique_texts.xlsx"
    segments_excel_path = RUN_DIR / "deduplicated_segments.xlsx"
    unique_txt_path = RUN_DIR / "unique_texts.txt"
    json_path = RUN_DIR / "ocr_results.json"

    pd.DataFrame(raw_rows).to_excel(raw_excel_path, index=False)
    pd.DataFrame({"unique_text": unique_texts}).to_excel(unique_excel_path, index=False)
    pd.DataFrame(segments).to_excel(segments_excel_path, index=False)

    with open(unique_txt_path, "w", encoding="utf-8") as file:
        for text in unique_texts:
            file.write(text + "\n")

    report = {
        "report_type": "FRAME_OCR_REPORT",
        "frames_folder": str(frames_folder),
        "area_mode": area_mode,
        "frames_count": len(raw_rows),
        "frames_with_text": sum(1 for row in raw_rows if row.get("recognized_text")),
        "unique_texts_count": len(unique_texts),
        "segments_count": len(segments),
        "unique_texts": unique_texts,
        "segments": segments,
        "raw_frames": raw_rows,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print("\nГотово.")
    print(f"Папка результатов: {RUN_DIR}")
    print(f"Сырой OCR по всем кадрам: {raw_excel_path}")
    print(f"Уникальные тексты Excel: {unique_excel_path}")
    print(f"Уникальные тексты TXT: {unique_txt_path}")
    print(f"Сегменты без дублей: {segments_excel_path}")
    print(f"JSON-отчёт: {json_path}")

    print("\nСтатистика:")
    print(f"Всего кадров обработано: {len(raw_rows)}")
    print(f"Кадров с текстом: {sum(1 for row in raw_rows if row.get('recognized_text'))}")
    print(f"Уникальных текстов: {len(unique_texts)}")
    print(f"Сегментов после дедубликации: {len(segments)}")


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ3: распознавание текста из потока изображений")
    print("=" * 70)

    frames_folder = choose_frames_folder()

    if frames_folder is None:
        return

    area_mode = choose_area_mode()

    raw_rows = process_frames(frames_folder, area_mode)

    unique_texts = build_unique_texts(raw_rows)
    segments = build_deduplicated_segments(raw_rows)

    save_results(
        frames_folder,
        area_mode,
        raw_rows,
        unique_texts,
        segments
    )


if __name__ == "__main__":
    main()