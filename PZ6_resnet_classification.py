import json
import re
import shutil
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from torchvision.models import (
    resnet18,
    resnet34,
    ResNet18_Weights,
    ResNet34_Weights
)


# =====================================================
# НАСТРОЙКА ПАПОК
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Результаты ПЗ5, где лежат object_crops
PZ5_RESULT_ROOT = BASE_DIR / "results" / "pz5_yolo"

# Кадры из ПЗ2 — запасной вариант, если нужно классифицировать весь кадр
PZ2_FRAME_ROOT = BASE_DIR / "results" / "pz2_frames" / "FRAME_FOLDER"

# Результаты ПЗ6
RESULT_ROOT = BASE_DIR / "results" / "pz6_resnet"

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"resnet_run_{RUN_TIMESTAMP}"

ANNOTATED_IMAGES_DIR = RUN_DIR / "annotated_images"
COPIED_INPUTS_DIR = RUN_DIR / "input_images"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
COPIED_INPUTS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def safe_filename(text):
    text = str(text)
    text = re.sub(r"[^A-Za-zА-Яа-яЁё0-9_-]", "_", text)
    return text


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
    Достаёт время из имени кадра/объекта, если оно есть.

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
# ВЫБОР ИСТОЧНИКА ИЗОБРАЖЕНИЙ
# =====================================================

def choose_source_mode():
    print("\nЧто классифицировать через ResNet?")
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
    Выбор папки object_crops из запусков ПЗ5.
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
    Запасной режим: классификация целых кадров.
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
        folder = choose_pz5_object_crops_folder()
    else:
        folder = choose_pz2_frames_folder()

    return source_mode, folder


# =====================================================
# ВЫБОР RESNET
# =====================================================

def choose_resnet_model():
    print("\nВыберите модель ResNet:")
    print("1 — ResNet18, быстрее")
    print("2 — ResNet34, лучше качество, не ResNet50")

    choice = input("\nВведите 1 или 2 (Enter = ResNet34): ").strip()

    if choice == "":
        return "resnet34"

    if choice == "1":
        return "resnet18"

    if choice == "2":
        return "resnet34"

    print("Неверный выбор. Используется ResNet34.")
    return "resnet34"


def load_resnet_model(model_name):
    """
    Загружает предобученную ResNet.
    Важно: ResNet50 не используется.
    """

    print("\nЗагружаем модель ResNet...")
    print(f"Модель: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    try:
        if model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        else:
            weights = ResNet34_Weights.DEFAULT
            model = resnet34(weights=weights)

    except Exception as error:
        print("\nНе удалось загрузить предобученные веса ResNet.")
        print("Скорее всего, PyTorch пытается скачать веса, но нет доступа к интернету.")
        print("Попробуй временно отключить прокси/VPN и запустить снова.")
        print("\nТекст ошибки:")
        print(error)
        return None, None, None

    model.to(device)
    model.eval()

    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    print("ResNet успешно загружена.")

    return model, preprocess, categories


# =====================================================
# АННОТАЦИЯ ИЗОБРАЖЕНИЙ
# =====================================================

def draw_prediction_on_image(image_path, top1_class, top1_confidence, save_path):
    """
    Сохраняет копию изображения с подписью класса ResNet.
    """

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return ""

    draw = ImageDraw.Draw(image)

    text = f"{top1_class} | {top1_confidence:.3f}"

    # Простая подпись без зависимости от внешних шрифтов
    rectangle_height = 32
    draw.rectangle(
        [(0, 0), (image.width, rectangle_height)],
        fill=(255, 255, 255)
    )

    draw.text(
        (8, 8),
        text,
        fill=(0, 0, 0)
    )

    image.save(save_path)

    return str(save_path)


# =====================================================
# КЛАССИФИКАЦИЯ
# =====================================================

def classify_single_image(image_path, model, preprocess, categories, top_k, device):
    """
    Классифицирует одно изображение с помощью ResNet.
    """

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        return None, []

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, top_k)

    top_results = []

    for rank, index in enumerate(top_indices, start=1):
        class_id = int(index.item())
        class_name = categories[class_id]
        confidence = float(top_probabilities[rank - 1].item())

        top_results.append({
            "rank": rank,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 6)
        })

    top1 = top_results[0] if top_results else None

    return top1, top_results


def process_images_with_resnet(images_folder, model, preprocess, categories, top_k, source_mode, model_name):
    """
    Классифицирует изображения из выбранной папки.
    """

    image_files = [
        file for file in sorted(images_folder.iterdir())
        if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    if not image_files:
        print("В выбранной папке нет изображений.")
        return [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nНайдено изображений: {len(image_files)}")
    print(f"Top-K: {top_k}")

    top1_rows = []
    topk_rows = []

    for image_index, image_path in enumerate(image_files, start=1):
        print(f"Классификация {image_index}/{len(image_files)}: {image_path.name}")

        frame_number = extract_frame_number_from_filename(image_path.name)
        time_seconds, time_formatted = extract_time_from_filename(image_path.name)

        top1, top_results = classify_single_image(
            image_path,
            model,
            preprocess,
            categories,
            top_k,
            device
        )

        if top1 is None:
            continue

        copied_input_path = COPIED_INPUTS_DIR / image_path.name

        try:
            shutil.copy2(image_path, copied_input_path)
        except Exception:
            copied_input_path = ""

        annotated_filename = f"{image_path.stem}_resnet.jpg"
        annotated_path = ANNOTATED_IMAGES_DIR / annotated_filename

        annotated_path_str = draw_prediction_on_image(
            image_path,
            top1["class_name"],
            top1["confidence"],
            annotated_path
        )

        top1_rows.append({
            "image_file": image_path.name,
            "image_path": str(image_path),
            "source_mode": source_mode,
            "model_name": model_name,
            "frame_number": frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "top1_class_id": top1["class_id"],
            "top1_class_name": top1["class_name"],
            "top1_confidence": top1["confidence"],
            "annotated_image_path": annotated_path_str
        })

        for result in top_results:
            topk_rows.append({
                "image_file": image_path.name,
                "image_path": str(image_path),
                "source_mode": source_mode,
                "model_name": model_name,
                "frame_number": frame_number,
                "time_seconds": time_seconds,
                "time_formatted": time_formatted,
                "rank": result["rank"],
                "class_id": result["class_id"],
                "class_name": result["class_name"],
                "confidence": result["confidence"]
            })

    return top1_rows, topk_rows


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def build_class_summary(top1_rows):
    if not top1_rows:
        return []

    df = pd.DataFrame(top1_rows)

    summary_df = (
        df.groupby("top1_class_name")
        .agg(
            images_count=("top1_class_name", "count"),
            max_confidence=("top1_confidence", "max"),
            mean_confidence=("top1_confidence", "mean")
        )
        .reset_index()
        .sort_values("images_count", ascending=False)
    )

    summary_df["mean_confidence"] = summary_df["mean_confidence"].round(6)

    return summary_df.to_dict(orient="records")


def save_results(images_folder, source_mode, model_name, top_k, top1_rows, topk_rows):
    """
    Сохраняет результаты ResNet в Excel и JSON.
    """

    class_summary_rows = build_class_summary(top1_rows)

    excel_path = RUN_DIR / "resnet_classification_results.xlsx"
    json_path = RUN_DIR / "resnet_classification_report.json"

    top1_df = pd.DataFrame(top1_rows)
    topk_df = pd.DataFrame(topk_rows)
    summary_df = pd.DataFrame(class_summary_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        top1_df.to_excel(writer, sheet_name="top1_results", index=False)
        topk_df.to_excel(writer, sheet_name="topk_results", index=False)
        summary_df.to_excel(writer, sheet_name="class_summary", index=False)

    json_report = {
        "report_type": "RESNET_CLASSIFICATION_REPORT",
        "model_type": "ResNet",
        "model_name": model_name,
        "note": "ResNet50 is not used",
        "source_mode": source_mode,
        "images_folder": str(images_folder),
        "top_k": top_k,
        "processed_images_count": len(top1_rows),
        "class_summary": class_summary_rows,
        "top1_results": top1_rows,
        "topk_results": topk_rows,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_report, file, ensure_ascii=False, indent=4)

    print("\nГотово.")
    print(f"Папка результатов ПЗ6: {RUN_DIR}")
    print(f"Excel-таблица: {excel_path}")
    print(f"JSON-отчёт: {json_path}")
    print(f"Изображения с подписями: {ANNOTATED_IMAGES_DIR}")

    print("\nСтатистика:")
    print(f"Обработано изображений: {len(top1_rows)}")
    print(f"Уникальных top1-классов: {len(class_summary_rows)}")


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ6: классификация объектов с помощью ResNet")
    print("=" * 70)

    print("\nВажно: ResNet50 в данном задании не используется.")
    print("Основная рекомендуемая модель — ResNet34.")

    source_mode, images_folder = choose_images_folder()

    if images_folder is None:
        return

    model_name = choose_resnet_model()

    top_k = ask_int(
        "\nВведите количество вариантов классификации Top-K "
        "(например 3 или 5; Enter = 5): ",
        5
    )

    model, preprocess, categories = load_resnet_model(model_name)

    if model is None:
        return

    top1_rows, topk_rows = process_images_with_resnet(
        images_folder,
        model,
        preprocess,
        categories,
        top_k,
        source_mode,
        model_name
    )

    save_results(
        images_folder,
        source_mode,
        model_name,
        top_k,
        top1_rows,
        topk_rows
    )


if __name__ == "__main__":
    main()