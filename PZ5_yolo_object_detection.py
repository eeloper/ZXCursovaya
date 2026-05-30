import cv2
import json
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# =====================================================
# НАСТРОЙКА ПАПОК
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Кадры после ПЗ2
FRAME_ROOT = BASE_DIR / "results" / "pz2_frames" / "FRAME_FOLDER"

# Локальная папка с YOLOv5
YOLOV5_REPO_DIR = BASE_DIR / "models" / "yolov5" / "yolov5-7.0"

# Локальная папка с весами YOLOv5
YOLOV5_WEIGHTS_DIR = BASE_DIR / "models" / "yolov5_weights"

# Результаты ПЗ5
RESULT_ROOT = BASE_DIR / "results" / "pz5_yolo"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"yolo_run_{RUN_TIMESTAMP}"

ANNOTATED_FRAMES_DIR = RUN_DIR / "annotated_frames"
OBJECT_CROPS_DIR = RUN_DIR / "object_crops"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
OBJECT_CROPS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# =====================================================
# ЧТЕНИЕ И СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ
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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
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


def ask_float(message, default_value):
    user_input = input(message).strip()

    if user_input == "":
        return default_value

    try:
        return float(user_input.replace(",", "."))
    except ValueError:
        print("Введено некорректное значение. Используется значение по умолчанию.")
        return default_value


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


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_-]", "_", text)
    return text


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
        image_count = len([
            file for file in folder.iterdir()
            if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])

        print(f"{index}. {folder.name} — кадров: {image_count}")

    choice = input("\nВведите номер папки с кадрами: ").strip()

    try:
        choice_number = int(choice)

        if 1 <= choice_number <= len(frame_folders):
            return frame_folders[choice_number - 1]

        print("Неверный номер папки.")
        return None

    except ValueError:
        print("Введено не число.")
        return None


# =====================================================
# ВЫБОР МОДЕЛИ YOLO
# =====================================================

def choose_yolo_model():
    print("\nВыберите модель YOLOv5:")
    print("1 — yolov5s, быстрая и лёгкая")
    print("2 — yolov5m, точнее, но медленнее")
    print("3 — yolov5l, ещё точнее, но может работать долго")

    choice = input("\nВведите 1, 2 или 3 (Enter = yolov5s): ").strip()

    if choice == "":
        return "yolov5s"

    if choice == "1":
        return "yolov5s"

    if choice == "2":
        return "yolov5m"

    if choice == "3":
        return "yolov5l"

    print("Неверный выбор. Используется yolov5s.")
    return "yolov5s"


def load_yolov5_model(model_name, confidence_threshold):
    """
    Загружает YOLOv5 локально, без скачивания с GitHub.
    Используется YOLOv5 v7.0, не YOLOv8.

    Исправление:
    новые версии PyTorch по умолчанию загружают веса в режиме weights_only=True,
    а старый формат YOLOv5 требует weights_only=False.
    """

    print("\nЗагружаем модель YOLOv5 локально...")
    print(f"Модель: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    weights_path = YOLOV5_WEIGHTS_DIR / f"{model_name}.pt"

    if not YOLOV5_REPO_DIR.exists():
        print("\nПапка YOLOv5 не найдена:")
        print(YOLOV5_REPO_DIR)
        print("\nПроверь, что YOLOv5 v7.0 распакован сюда:")
        print("models/yolov5/yolov5-7.0")
        return None, None

    if not (YOLOV5_REPO_DIR / "hubconf.py").exists():
        print("\nВ папке YOLOv5 не найден hubconf.py:")
        print(YOLOV5_REPO_DIR / "hubconf.py")
        print("\nСкорее всего, архив YOLOv5 распакован не в ту папку.")
        return None, None

    if not weights_path.exists():
        print("\nФайл весов YOLOv5 не найден:")
        print(weights_path)
        print(f"\nСкачай файл {model_name}.pt и положи его сюда:")
        print(YOLOV5_WEIGHTS_DIR)
        return None, None

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = patched_torch_load

        model = torch.hub.load(
            str(YOLOV5_REPO_DIR),
            "custom",
            path=str(weights_path),
            source="local",
            trust_repo=True
        )

    except Exception as error:
        print("\nНе удалось загрузить YOLOv5 локально.")
        print("Текст ошибки:")
        print(error)
        print("\nЕсли ошибка связана с зависимостями, выполни:")
        print(f'python -m pip install -r "{YOLOV5_REPO_DIR / "requirements.txt"}"')
        return None, None

    finally:
        torch.load = original_torch_load

    model.to(device)
    model.conf = confidence_threshold
    model.iou = 0.45

    print("YOLOv5 успешно загружена локально.")

    return model, device


# =====================================================
# ОБРАБОТКА КАДРОВ YOLO
# =====================================================

def draw_detection(image, class_name, confidence, x1, y1, x2, y2):
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return image


def process_frames_with_yolo(frames_folder, model, frame_step):
    image_files = [
        file for file in sorted(frames_folder.iterdir())
        if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("В выбранной папке нет изображений.")
        return [], []

    selected_files = image_files[::frame_step]

    print(f"\nВсего кадров в папке: {len(image_files)}")
    print(f"Будет обработано кадров: {len(selected_files)}")
    print(f"Шаг обработки: каждый {frame_step}-й кадр")

    detection_rows = []
    frame_summary_rows = []

    for index, frame_path in enumerate(selected_files, start=1):
        print(f"Обработка кадра {index}/{len(selected_files)}: {frame_path.name}")

        frame = read_image_correctly(frame_path)

        if frame is None:
            print("Не удалось прочитать кадр.")
            continue

        frame_number = extract_frame_number_from_filename(frame_path.name)
        time_seconds, time_formatted = extract_time_from_filename(frame_path.name)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = model(frame_rgb, size=640)
        except Exception as error:
            print("Ошибка при обработке кадра YOLO:")
            print(error)
            continue

        detections_df = results.pandas().xyxy[0]

        annotated_frame = frame.copy()
        detections_count = len(detections_df)

        annotated_path = ""
        classes_on_frame = []

        if detections_count > 0:
            for detection_index, row in detections_df.iterrows():
                x1 = int(max(0, round(float(row["xmin"]))))
                y1 = int(max(0, round(float(row["ymin"]))))
                x2 = int(min(frame.shape[1], round(float(row["xmax"]))))
                y2 = int(min(frame.shape[0], round(float(row["ymax"]))))

                confidence = round(float(row["confidence"]), 4)
                class_id = int(row["class"])
                class_name = str(row["name"])

                classes_on_frame.append(class_name)

                annotated_frame = draw_detection(
                    annotated_frame,
                    class_name,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2
                )

                crop_path = ""

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]

                    crop_filename = (
                        f"{frame_path.stem}_object_{detection_index:03d}"
                        f"_{safe_filename(class_name)}.jpg"
                    )

                    crop_path_obj = OBJECT_CROPS_DIR / crop_filename
                    save_image_correctly(crop_path_obj, crop)
                    crop_path = str(crop_path_obj)

                detection_rows.append({
                    "frame_file": frame_path.name,
                    "frame_path": str(frame_path),
                    "frame_number": frame_number,
                    "time_seconds": time_seconds,
                    "time_formatted": time_formatted,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "crop_path": crop_path
                })

            annotated_path_obj = ANNOTATED_FRAMES_DIR / f"{frame_path.stem}_yolo.jpg"
            save_image_correctly(annotated_path_obj, annotated_frame)
            annotated_path = str(annotated_path_obj)

        unique_classes = sorted(list(set(classes_on_frame)))

        frame_summary_rows.append({
            "frame_file": frame_path.name,
            "frame_path": str(frame_path),
            "frame_number": frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "detections_count": detections_count,
            "classes_on_frame": ", ".join(unique_classes),
            "annotated_frame_path": annotated_path
        })

    return detection_rows, frame_summary_rows


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def build_class_summary(detection_rows):
    if not detection_rows:
        return []

    df = pd.DataFrame(detection_rows)

    summary_df = (
        df.groupby("class_name")
        .agg(
            detections_count=("class_name", "count"),
            max_confidence=("confidence", "max"),
            mean_confidence=("confidence", "mean")
        )
        .reset_index()
        .sort_values("detections_count", ascending=False)
    )

    summary_df["mean_confidence"] = summary_df["mean_confidence"].round(4)

    return summary_df.to_dict(orient="records")


def save_results(frames_folder, model_name, confidence_threshold, frame_step, detection_rows, frame_summary_rows):
    class_summary_rows = build_class_summary(detection_rows)

    excel_path = RUN_DIR / "yolo_detection_results.xlsx"
    json_path = RUN_DIR / "yolo_detection_report.json"

    detections_df = pd.DataFrame(detection_rows)
    frames_df = pd.DataFrame(frame_summary_rows)
    classes_df = pd.DataFrame(class_summary_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        detections_df.to_excel(writer, sheet_name="objects", index=False)
        frames_df.to_excel(writer, sheet_name="frames", index=False)
        classes_df.to_excel(writer, sheet_name="class_summary", index=False)

    frames_with_detections = sum(
        1 for row in frame_summary_rows
        if row.get("detections_count", 0) > 0
    )

    json_report = {
        "report_type": "YOLOV5_OBJECT_DETECTION_REPORT",
        "model_type": "YOLOv5",
        "model_version": "v7.0",
        "model_name": model_name,
        "loading_mode": "local",
        "frames_folder": str(frames_folder),
        "confidence_threshold": confidence_threshold,
        "frame_step": frame_step,
        "processed_frames_count": len(frame_summary_rows),
        "frames_with_detections": frames_with_detections,
        "detections_count": len(detection_rows),
        "class_summary": class_summary_rows,
        "detections": detection_rows,
        "frames": frame_summary_rows,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_report, file, ensure_ascii=False, indent=4, default=str)

    print("\nГотово.")
    print(f"Папка результатов ПЗ5: {RUN_DIR}")
    print(f"Excel-таблица: {excel_path}")
    print(f"JSON-отчёт: {json_path}")
    print(f"Кадры с рамками: {ANNOTATED_FRAMES_DIR}")
    print(f"Вырезанные объекты: {OBJECT_CROPS_DIR}")

    print("\nСтатистика:")
    print(f"Обработано кадров: {len(frame_summary_rows)}")
    print(f"Кадров с объектами: {frames_with_detections}")
    print(f"Всего найдено объектов: {len(detection_rows)}")


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ5: распознавание объектов на кадрах с помощью YOLOv5")
    print("=" * 70)

    print("\nВажно: используется YOLOv5 v7.0, загруженная локально.")
    print("YOLOv8 в этом задании не используется.")

    frames_folder = choose_frames_folder()

    if frames_folder is None:
        return

    model_name = choose_yolo_model()

    confidence_threshold = ask_float(
        "\nВведите порог уверенности YOLO "
        "(например 0.25, 0.35, 0.5; Enter = 0.35): ",
        0.35
    )

    frame_step = ask_int(
        "\nВведите шаг обработки кадров "
        "(1 — каждый кадр, 2 — каждый второй, 5 — каждый пятый; Enter = 1): ",
        1
    )

    model, device = load_yolov5_model(
        model_name,
        confidence_threshold
    )

    if model is None:
        return

    detection_rows, frame_summary_rows = process_frames_with_yolo(
        frames_folder,
        model,
        frame_step
    )

    save_results(
        frames_folder,
        model_name,
        confidence_threshold,
        frame_step,
        detection_rows,
        frame_summary_rows
    )


if __name__ == "__main__":
    main()