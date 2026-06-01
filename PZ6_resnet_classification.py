import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from project_config import load_run_config, RUN_CONFIG_PATH


# =====================================================
# НАСТРОЙКИ
# =====================================================

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

DEFAULT_TOP_K = 5


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


def get_image_files(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        return []

    files = [
        file for file in folder_path.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    return sorted(files)


# =====================================================
# IMAGENET LABELS
# =====================================================

def load_imagenet_labels():
    """
    Загружает список классов ImageNet.

    Сначала пробуем взять labels из torchvision weights.
    Если не получилось, используем короткий fallback,
    чтобы программа не падала.
    """

    try:
        weights = models.ResNet34_Weights.DEFAULT
        categories = weights.meta["categories"]
        return categories
    except Exception:
        return [f"class_{index}" for index in range(1000)]


# =====================================================
# МОДЕЛЬ RESNET
# =====================================================

def choose_top_k():
    print("\nВведите количество top-k классов.")
    print("Например:")
    print("3 — показать 3 наиболее вероятных класса")
    print("5 — показать 5 наиболее вероятных классов")
    print("Enter — 5")

    user_input = input("\nTop-k: ").strip()

    if user_input == "":
        return DEFAULT_TOP_K

    try:
        value = int(user_input)

        if value <= 0:
            print("Top-k должен быть больше 0. Используется 5.")
            return DEFAULT_TOP_K

        if value > 10:
            print("Слишком большое значение. Используется 10.")
            return 10

        return value

    except ValueError:
        print("Некорректное значение. Используется 5.")
        return DEFAULT_TOP_K


def load_resnet34_model():
    """
    Загружает ResNet34 с предобученными весами ImageNet.
    """

    print("\nЗагружается ResNet34...")

    weights = models.ResNet34_Weights.DEFAULT
    model = models.resnet34(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Устройство: {device}")

    return model, preprocess, device


def classify_image(model, preprocess, device, image_path, labels, top_k):
    image_path = Path(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "top1": None,
            "topk": []
        }

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_probabilities, top_indices = torch.topk(probabilities, top_k)

    topk_results = []

    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities, top_indices),
        start=1
    ):
        class_index = int(class_index.item())
        confidence = float(probability.item())

        class_name = labels[class_index] if class_index < len(labels) else f"class_{class_index}"

        topk_results.append({
            "rank": rank,
            "class_id": class_index,
            "class_name": class_name,
            "confidence": round(confidence, 4)
        })

    top1 = topk_results[0] if topk_results else None

    return {
        "success": True,
        "error": "",
        "top1": top1,
        "topk": topk_results
    }


# =====================================================
# ОБРАБОТКА OBJECT CROPS
# =====================================================

def process_object_crops(object_crops_dir, top_k):
    object_crops_dir = Path(object_crops_dir)

    image_files = get_image_files(object_crops_dir)

    if not image_files:
        print("\nВ папке object_crops нет изображений:")
        print(object_crops_dir)
        return [], [], []

    print(f"\nНайдено изображений для ResNet: {len(image_files)}")

    labels = load_imagenet_labels()
    model, preprocess, device = load_resnet34_model()

    top1_rows = []
    topk_rows = []
    class_counter = Counter()

    for image_index, image_path in enumerate(image_files, start=1):
        print(f"ResNet {image_index}/{len(image_files)}: {image_path.name}")

        time_seconds, time_formatted = extract_time_from_filename(image_path.name)
        frame_number = extract_frame_number_from_filename(image_path.name)
        source_frame_number = extract_source_frame_number_from_filename(image_path.name)

        result = classify_image(
            model=model,
            preprocess=preprocess,
            device=device,
            image_path=image_path,
            labels=labels,
            top_k=top_k
        )

        if not result["success"]:
            top1_rows.append({
                "image_file": image_path.name,
                "image_path": str(image_path),
                "frame_number": frame_number,
                "source_frame_number": source_frame_number,
                "time_seconds": time_seconds,
                "time_formatted": time_formatted,
                "success": False,
                "error": result["error"],
                "top1_class_id": "",
                "top1_class_name": "",
                "top1_confidence": ""
            })
            continue

        top1 = result["top1"]

        top1_rows.append({
            "image_file": image_path.name,
            "image_path": str(image_path),
            "frame_number": frame_number,
            "source_frame_number": source_frame_number,
            "time_seconds": time_seconds,
            "time_formatted": time_formatted,
            "success": True,
            "error": "",
            "top1_class_id": top1["class_id"],
            "top1_class_name": top1["class_name"],
            "top1_confidence": top1["confidence"]
        })

        class_counter[top1["class_name"]] += 1

        for item in result["topk"]:
            topk_rows.append({
                "image_file": image_path.name,
                "image_path": str(image_path),
                "frame_number": frame_number,
                "source_frame_number": source_frame_number,
                "time_seconds": time_seconds,
                "time_formatted": time_formatted,
                "rank": item["rank"],
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "confidence": item["confidence"]
            })

    class_summary_rows = []

    for class_name, count in class_counter.most_common():
        class_summary_rows.append({
            "class_name": class_name,
            "images_count": count
        })

    return top1_rows, topk_rows, class_summary_rows


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def save_resnet_results(resnet_dir, top1_rows, topk_rows, class_summary_rows, top_k):
    resnet_dir = Path(resnet_dir)
    resnet_dir.mkdir(parents=True, exist_ok=True)

    excel_path = resnet_dir / "resnet_classification_results.xlsx"
    json_path = resnet_dir / "resnet_classification_report.json"

    top1_df = pd.DataFrame(top1_rows)
    topk_df = pd.DataFrame(topk_rows)
    class_summary_df = pd.DataFrame(class_summary_rows)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        top1_df.to_excel(writer, sheet_name="top1_results", index=False)
        topk_df.to_excel(writer, sheet_name="topk_results", index=False)
        class_summary_df.to_excel(writer, sheet_name="class_summary", index=False)

    report = {
        "report_type": "RESNET_IMAGE_CLASSIFICATION",
        "model_name": "resnet34",
        "top_k": top_k,
        "processed_images_count": len(top1_rows),
        "topk_rows_count": len(topk_rows),
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
    print("ПЗ6: классификация объектов ResNet34")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выполни ПЗ5.")
        print(error)
        return

    object_crops_dir = Path(config["object_crops_dir"])
    resnet_dir = Path(config["resnet_dir"])

    if not object_crops_dir.exists():
        print("\nПапка object_crops не найдена:")
        print(object_crops_dir)
        return

    image_files = get_image_files(object_crops_dir)

    if not image_files:
        print("\nВ object_crops нет изображений для классификации:")
        print(object_crops_dir)
        return

    print("\nИзображения объектов берутся из папки:")
    print(object_crops_dir)

    print("\nРезультаты ResNet будут сохранены в папку:")
    print(resnet_dir)

    top_k = choose_top_k()

    clear_folder(resnet_dir)

    try:
        top1_rows, topk_rows, class_summary_rows = process_object_crops(
            object_crops_dir=object_crops_dir,
            top_k=top_k
        )

        saved_paths = save_resnet_results(
            resnet_dir=resnet_dir,
            top1_rows=top1_rows,
            topk_rows=topk_rows,
            class_summary_rows=class_summary_rows,
            top_k=top_k
        )

    except Exception as error:
        print("\nОшибка при выполнении ПЗ6:")
        print(error)

        config["pz6_status"] = "error"
        config["pz6_error"] = str(error)
        config["pz6_finished_at"] = datetime.now().isoformat()
        save_run_config(config)

        return

    config["resnet_dir"] = str(resnet_dir)
    config["resnet_model"] = "resnet34"
    config["resnet_top_k"] = top_k
    config["resnet_results_path"] = str(saved_paths["excel_path"])
    config["resnet_report_path"] = str(saved_paths["json_path"])
    config["resnet_processed_images_count"] = len(top1_rows)
    config["resnet_classes_count"] = len(class_summary_rows)
    config["pz6_status"] = "success"
    config["pz6_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\nПЗ6 завершено успешно.")
    print(f"Классифицировано изображений: {len(top1_rows)}")
    print(f"Количество классов top1: {len(class_summary_rows)}")
    print(f"Excel: {saved_paths['excel_path']}")
    print(f"JSON: {saved_paths['json_path']}")
    print("\nrun_config.json обновлён.")


if __name__ == "__main__":
    main()