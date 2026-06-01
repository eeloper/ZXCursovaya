import json
import re
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import torch

from project_config import load_run_config, RUN_CONFIG_PATH


# =====================================================
# НАСТРОЙКИ
# =====================================================

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

DEFAULT_CONFIDENCE_THRESHOLD = 0.25

BASE_DIR = Path(__file__).resolve().parent
YOLO_REPO_DIR = BASE_DIR / "models" / "yolov5"
YOLO_WEIGHTS_DIR = BASE_DIR / "models" / "yolov5_weights"


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


def get_frame_files(frames_dir):
    frames_dir = Path(frames_dir)

    if not frames_dir.exists():
        return []

    files = [
        file for file in frames_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(files)


# =====================================================
# ВЫБОР И ЗАГРУЗКА МОДЕЛИ YOLOv5
# =====================================================

def choose_yolo_model():
    print("\nВыберите модель YOLOv5:")
    print("1 — yolov5s, самая быстрая")
    print("2 — yolov5m, средний вариант")
    print("3 — yolov5l, более точная, но медленнее")
    print("Enter — yolov5l")

    choice = input("\nВведите номер модели: ").strip()

    if choice == "1":
        return "yolov5s"

    if choice == "2":
        return "yolov5m"

    if choice == "3" or choice == "":
        return "yolov5l"

    print("Неверный выбор. Используется yolov5l.")
    return "yolov5l"


def choose_confidence_threshold():
    print("\nВведите минимальную уверенность YOLO.")
    print("Например: 0.25, 0.30, 0.50")
    print("Enter — 0.25")

    user_input = input("\nconfidence threshold: ").strip()

    if user_input == "":
        return DEFAULT_CONFIDENCE_THRESHOLD

    try:
        value = float(user_input.replace(",", "."))

        if value < 0 or value > 1:
            print("Значение должно быть от 0 до 1. Используется 0.25.")
            return DEFAULT_CONFIDENCE_THRESHOLD

        return value

    except ValueError:
        print("Некорректное значение. Используется 0.25.")
        return DEFAULT_CONFIDENCE_THRESHOLD


def get_weights_path(model_name):
    weights_path = YOLO_WEIGHTS_DIR / f"{model_name}.pt"

    if weights_path.exists():
        return weights_path

    return None


def load_yolo_model(model_name, confidence_threshold):
    """
    Загружает YOLOv5.

    Приоритет:
    1. локальный репозиторий models/yolov5 + локальные веса;
    2. если локальных весов нет, PyTorch Hub сам попробует скачать модель.
    """

    print("\nЗагружается YOLOv5...")
    print(f"Модель: {model_name}")
    print(f"confidence threshold: {confidence_threshold}")

    weights_path = get_weights_path(model_name)

    if YOLO_REPO_DIR.exists() and weights_path is not None:
        print("\nИспользуется локальный репозиторий YOLOv5 и локальные веса:")
        print(YOLO_REPO_DIR)
        print(weights_path)

        model = torch.hub.load(
            str(YOLO_REPO_DIR),
            "custom",
            path=str(weights_path),
            source="local"
        )
    else:
        print("\nЛокальные веса не найдены. Пробуем загрузить модель через torch.hub.")
        print("Если интернета нет, скачивание может не сработать.")

        model = torch.hub.load(
            "ultralytics/yolov5",
            model_name,
            pretrained=True
        )

    model.conf = confidence_threshold

    return model


# =====================================================
# ДЕТЕКЦИЯ И СОХРАНЕНИЕ
# =====================================================

def crop_object(image, x1, y1, x2, y2):
    height, width = image.shape[:2]

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def draw_detection(image, x1, y1, x2, y2, label, confidence):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    text = f"{label} {confidence:.2f}"

    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def process_frames_with_yolo(model, frames_dir, yolo_dir, object_crops_dir):
    frames_dir = Path(frames_dir)
    yolo_dir = Path(yolo_dir)
    object_crops_dir = Path(object_crops_dir)

    annotated_frames_dir = yolo_dir / "annotated_frames"

    clear_folder(yolo_dir)
    object_crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_frames_dir.mkdir(parents=True, exist_ok=True)

    frame_files = get_frame_files(frames_dir)

    if not frame_files:
        print("\nКадры не найдены:")
        print(frames_dir)
        return [], [], []

    print(f"\nНайдено кадров для YOLO: {len(frame_files)}")

    object_rows = []
    frame_rows = []
    class_counter = {}

    crop_global_index = 0

    for frame_index, frame_path in enumerate(frame_files, start=1):
        print(f"YOLO {frame_index}/{len(frame_files)}: {frame_path.name}")

        image = cv2.imread(str(frame_path))

        if image is None:
            continue

        original_image = image.copy()

        time_seconds, time_formatted = extract_time_from_filename(frame_path.name)
        frame_number = extract_frame_number_from_filename(frame_path.name)
        source_frame_number = extract_source_frame_number_from_filename(frame_path.name)

        results = model(str(frame_path))
        detections_df = results.pandas().xyxy[0]

        detections_count = 0

        for _, detection in detections_df.iterrows():
            x1 = float(detection["xmin"])
            y1 = float(detection["ymin"])
            x2 = float(detection["xmax"])
            y2 = float(detection["ymax"])
            confidence = float(detection["confidence"])
            class_id = int(detection["class"])
            class_name = str(detection["name"])

            crop = crop_object(
                original_image,
                x1,
                y1,
                x2,
                y2
            )

            crop_path = ""

            if crop is not None:
                crop_global_index += 1

                crop_filename = (
                    f"object_{crop_global_index:06d}"
                    f"_frame_{frame_number:06d}"
                    f"_source_{source_frame_number:06d}"
                    f"_time_{int(time_seconds * 1000):08d}ms"
                    f"_{class_name}.jpg"
                )

                crop_path = object_crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop)

            draw_detection(
                image,
                x1,
                y1,
                x2,
                y2,
                class_name,
                confidence
            )

            object_rows.append({
                "object_id": crop_global_index,
                "frame_file": frame_path.name,
                "frame_path": str(frame_path),
                "frame_number": frame_number,
                "source_frame_number": source_frame_number,
                "time_seconds": time_seconds,
                "time_formatted": time_formatted,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "crop_path": str(crop_path) if crop_path else ""
            })

            class_counter[class_name] = class_counter.get(class_name, 0) + 1
            detections_count += 1

        annotated_frame_path = annotated_frames_dir / frame_path.name
        cv2.imwrite(str(annotated_frame_path), image)

        frame_rows.append({
            "frame_file": frame_path.name,
            "frame_path": str(frame_path),
            "annotated_frame_path": str(annotated_frame_path),
            "frame_number": frame_number,
            "source_frame_number": source_frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "detections_count": detections_count
        })

    class_summary_rows = []

    for class_name, count in sorted(class_counter.items()):
        class_summary_rows.append({
            "class_name": class_name,
            "detections_count": count
        })

    return object_rows, frame_rows, class_summary_rows


def save_yolo_results(yolo_dir, object_rows, frame_rows, class_summary_rows, model_name, confidence_threshold):
    yolo_dir = Path(yolo_dir)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    excel_path = yolo_dir / "yolo_detection_results.xlsx"
    json_path = yolo_dir / "yolo_detection_report.json"

    objects_df = pd.DataFrame(object_rows)
    frames_df = pd.DataFrame(frame_rows)
    class_summary_df = pd.DataFrame(class_summary_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        objects_df.to_excel(writer, sheet_name="objects", index=False)
        frames_df.to_excel(writer, sheet_name="frames", index=False)
        class_summary_df.to_excel(writer, sheet_name="class_summary", index=False)

    report = {
        "report_type": "YOLO_OBJECT_DETECTION",
        "model_name": model_name,
        "confidence_threshold": confidence_threshold,
        "objects_count": len(object_rows),
        "processed_frames_count": len(frame_rows),
        "classes_count": len(class_summary_rows),
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
    print("ПЗ5: детектирование объектов YOLOv5")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выполни ПЗ2.")
        print(error)
        return

    frames_dir = Path(config["frames_dir"])
    yolo_dir = Path(config["yolo_dir"])
    object_crops_dir = Path(config["object_crops_dir"])

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

    print("\nРезультаты YOLO будут сохранены в папку:")
    print(yolo_dir)

    model_name = choose_yolo_model()
    confidence_threshold = choose_confidence_threshold()

    try:
        model = load_yolo_model(
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )

        object_rows, frame_rows, class_summary_rows = process_frames_with_yolo(
            model=model,
            frames_dir=frames_dir,
            yolo_dir=yolo_dir,
            object_crops_dir=object_crops_dir
        )

        saved_paths = save_yolo_results(
            yolo_dir=yolo_dir,
            object_rows=object_rows,
            frame_rows=frame_rows,
            class_summary_rows=class_summary_rows,
            model_name=model_name,
            confidence_threshold=confidence_threshold
        )

    except Exception as error:
        print("\nОшибка при выполнении ПЗ5:")
        print(error)

        config["pz5_status"] = "error"
        config["pz5_error"] = str(error)
        config["pz5_finished_at"] = datetime.now().isoformat()
        save_run_config(config)

        return

    config["yolo_dir"] = str(yolo_dir)
    config["object_crops_dir"] = str(object_crops_dir)
    config["yolo_model"] = model_name
    config["yolo_confidence_threshold"] = confidence_threshold
    config["yolo_results_path"] = str(saved_paths["excel_path"])
    config["yolo_report_path"] = str(saved_paths["json_path"])
    config["yolo_objects_count"] = len(object_rows)
    config["yolo_processed_frames_count"] = len(frame_rows)
    config["yolo_classes_count"] = len(class_summary_rows)
    config["pz5_status"] = "success"
    config["pz5_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\nПЗ5 завершено успешно.")
    print(f"Обработано кадров: {len(frame_rows)}")
    print(f"Найдено объектов: {len(object_rows)}")
    print(f"Количество классов: {len(class_summary_rows)}")
    print(f"Excel: {saved_paths['excel_path']}")
    print(f"JSON: {saved_paths['json_path']}")
    print(f"Вырезанные объекты: {object_crops_dir}")
    print("\nrun_config.json обновлён.")


if __name__ == "__main__":
    main()