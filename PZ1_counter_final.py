import cv2
import numpy as np
import pandas as pd
import easyocr
import re
from pathlib import Path


# =====================================================
# НАСТРОЙКА ПАПОК ПРОЕКТА
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "data" / "input_images"

RESULT_DIR = BASE_DIR / "results" / "pz1_opencv"
CHANNELS_DIR = RESULT_DIR / "channels"
COLLAGES_DIR = RESULT_DIR / "collages"
OCR_DIR = RESULT_DIR / "counter_number_ocr"
ROI_DIR = OCR_DIR / "number_roi"
ANNOTATED_DIR = OCR_DIR / "annotated"
PROCESSED_DIR = OCR_DIR / "processed_roi"

EXCEL_PATH = OCR_DIR / "counter_number_results.xlsx"
DEFECT_REPORT_PATH = OCR_DIR / "defect_analysis.txt"

for folder in [
    RESULT_DIR,
    CHANNELS_DIR,
    COLLAGES_DIR,
    OCR_DIR,
    ROI_DIR,
    ANNOTATED_DIR,
    PROCESSED_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


# =====================================================
# ЧТЕНИЕ И СОХРАНЕНИЕ ИЗОБРАЖЕНИЙ
# =====================================================

def read_image_correctly(image_path):
    """
    Надёжное чтение изображения.
    Такой способ нормально работает с русскими буквами в пути Windows.
    """
    image_array = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def save_image_correctly(save_path, image):
    """
    Надёжное сохранение изображения.
    """
    extension = save_path.suffix
    success, encoded_image = cv2.imencode(extension, image)

    if success:
        encoded_image.tofile(str(save_path))
    else:
        print(f"Не удалось сохранить файл: {save_path}")


# =====================================================
# РАЗЛОЖЕНИЕ НА КАНАЛЫ
# =====================================================

def split_channels(image, image_name):
    """
    Сохраняет Ч/Б, R, G, B каналы.
    """

    image_channel_dir = CHANNELS_DIR / image_name
    image_channel_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blue, green, red = cv2.split(image)

    save_image_correctly(image_channel_dir / "gray.png", gray)
    save_image_correctly(image_channel_dir / "red.png", red)
    save_image_correctly(image_channel_dir / "green.png", green)
    save_image_correctly(image_channel_dir / "blue.png", blue)

    return {
        "gray": gray,
        "red": red,
        "green": green,
        "blue": blue
    }


def make_colored_channel(channel, color_name):
    """
    Делает наглядное цветное изображение одного канала.
    """

    zeros = np.zeros_like(channel)

    if color_name == "red":
        return cv2.merge([zeros, zeros, channel])
    if color_name == "green":
        return cv2.merge([zeros, channel, zeros])
    if color_name == "blue":
        return cv2.merge([channel, zeros, zeros])

    return cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)


def add_label(image, label):
    """
    Добавляет подпись на изображение.
    """

    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 35), (255, 255, 255), -1)
    cv2.putText(
        result,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )
    return result


def create_collage(image, channels, image_name):
    """
    Создаёт коллаж:
    оригинал + R + G + B + grayscale.
    """

    target_width = 350
    target_height = 260

    original = cv2.resize(image, (target_width, target_height))

    red_img = make_colored_channel(channels["red"], "red")
    green_img = make_colored_channel(channels["green"], "green")
    blue_img = make_colored_channel(channels["blue"], "blue")
    gray_img = cv2.cvtColor(channels["gray"], cv2.COLOR_GRAY2BGR)

    red_img = cv2.resize(red_img, (target_width, target_height))
    green_img = cv2.resize(green_img, (target_width, target_height))
    blue_img = cv2.resize(blue_img, (target_width, target_height))
    gray_img = cv2.resize(gray_img, (target_width, target_height))

    original = add_label(original, "Original")
    red_img = add_label(red_img, "R channel")
    green_img = add_label(green_img, "G channel")
    blue_img = add_label(blue_img, "B channel")
    gray_img = add_label(gray_img, "Grayscale")

    empty = np.ones_like(original) * 255

    row1 = np.hstack([original, red_img, green_img])
    row2 = np.hstack([blue_img, gray_img, empty])

    collage = np.vstack([row1, row2])

    collage_path = COLLAGES_DIR / f"{image_name}_collage.png"
    save_image_correctly(collage_path, collage)

    return collage_path


# =====================================================
# АНАЛИЗ ДЕФЕКТОВ ИЗОБРАЖЕНИЯ
# =====================================================

def analyze_image_quality(image, channels):
    """
    Анализирует качество изображения:
    яркость, контраст, размытость, шум, цветовой перекос.
    """

    gray = channels["gray"]
    red = channels["red"]
    green = channels["green"]
    blue = channels["blue"]

    defects = []

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float(np.std(gray - blur))

    mean_red = float(np.mean(red))
    mean_green = float(np.mean(green))
    mean_blue = float(np.mean(blue))

    color_diff = max(mean_red, mean_green, mean_blue) - min(mean_red, mean_green, mean_blue)

    if brightness < 80:
        defects.append("низкая яркость")

    if brightness > 200:
        defects.append("пересветка")

    if contrast < 30:
        defects.append("низкий контраст")

    if sharpness < 50:
        defects.append("размытость")

    if noise > 15:
        defects.append("шум")

    if color_diff > 40:
        defects.append("цветовой перекос")

    if not defects:
        defects.append("выраженных дефектов не обнаружено")

    if "цветовой перекос" in defects or "пересветка" in defects:
        recommendation = "RGB-разложение + выбор канала"
    else:
        recommendation = "Ч/Б + повышение контраста"

    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "noise": round(noise, 2),
        "mean_red": round(mean_red, 2),
        "mean_green": round(mean_green, 2),
        "mean_blue": round(mean_blue, 2),
        "color_diff": round(color_diff, 2),
        "defects": defects,
        "recommendation": recommendation
    }


# =====================================================
# ПОИСК ОБЛАСТИ С КРУПНЫМИ ЦИФРАМИ
# =====================================================

def find_big_number_roi(image):
    """
    Ищет область с крупным номером счётчика.

    Логика:
    1. Берём верхнюю часть изображения, потому что номер обычно сверху.
    2. Усиливаем контраст.
    3. Делаем бинаризацию.
    4. Морфологией объединяем крупные цифры в одну область.
    5. Выбираем самый вероятный прямоугольник с номером.
    """

    original_height, original_width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Берём верхнюю часть изображения.
    # На твоём фото крупный номер расположен именно сверху.
    top_limit = int(original_height * 0.35)
    top_gray = gray[0:top_limit, :]

    # Усиливаем локальный контраст
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(top_gray)

    # Немного убираем шум
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    # Делаем тёмные цифры белыми на чёрном фоне
    _, binary_inv = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Склеиваем символы по горизонтали, чтобы 442974 стало одной областью
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
    connected = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Дополнительно немного расширяем область
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    connected = cv2.dilate(connected, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:
            continue

        aspect = w / h
        area = w * h

        # Отсеиваем мелкие подписи и случайный шум
        if w < original_width * 0.18:
            continue

        if h < top_limit * 0.12:
            continue

        if aspect < 1.8 or aspect > 15:
            continue

        # Чем крупнее и выше область, тем вероятнее, что это главный номер
        score = area + h * 2000 - y * 500

        candidates.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "aspect": aspect,
            "score": score
        })

    # Если нашли подходящие области — берём лучшую
    if candidates:
        best = max(candidates, key=lambda item: item["score"])

        pad_x = int(best["w"] * 0.08)
        pad_y = int(best["h"] * 0.25)

        x1 = max(0, best["x"] - pad_x)
        y1 = max(0, best["y"] - pad_y)
        x2 = min(original_width, best["x"] + best["w"] + pad_x)
        y2 = min(top_limit, best["y"] + best["h"] + pad_y)

        roi = image[y1:y2, x1:x2]

        return roi, (x1, y1, x2, y2), "auto_contour"

    # Если автоматический поиск не сработал,
    # используем запасной вариант: верхняя левая часть изображения.
    fallback_x1 = 0
    fallback_y1 = 0
    fallback_x2 = int(original_width * 0.75)
    fallback_y2 = int(original_height * 0.22)

    roi = image[fallback_y1:fallback_y2, fallback_x1:fallback_x2]

    return roi, (fallback_x1, fallback_y1, fallback_x2, fallback_y2), "fallback_top_area"


# =====================================================
# ПРЕДОБРАБОТКА ROI ДЛЯ OCR
# =====================================================

def prepare_roi_variants(roi):
    """
    Создаёт несколько вариантов обработки вырезанной области.
    OCR будет пробовать их все.
    """

    variants = []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Увеличиваем область, чтобы OCR легче прочитал цифры
    gray_big = cv2.resize(
        gray,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray_big)

    # Резкость
    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(contrast, -1, sharp_kernel)

    # Otsu
    _, otsu = cv2.threshold(
        sharp,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu_inv = cv2.bitwise_not(otsu)

    # Adaptive
    adaptive = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    adaptive_inv = cv2.bitwise_not(adaptive)

    variants.append(("gray_big", gray_big))
    variants.append(("contrast", contrast))
    variants.append(("sharp", sharp))
    variants.append(("otsu", otsu))
    variants.append(("otsu_inv", otsu_inv))
    variants.append(("adaptive", adaptive))
    variants.append(("adaptive_inv", adaptive_inv))

    return variants


# =====================================================
# OCR И ВЫБОР ЛУЧШЕГО НОМЕРА
# =====================================================

def clean_digits(text):
    """
    Оставляет только цифры.
    """
    return re.sub(r"\D", "", text)


def normalize_candidate(candidate):
    """
    Приводит найденный номер к нормальному виду.

    Нам нужен крупный серийный номер счётчика.
    Обычно на таких фото он 5-6 цифр.
    Если OCR дал длинную строку, пробуем взять наиболее вероятные 6 цифр.
    """

    if not candidate:
        return ""

    candidate = clean_digits(candidate)

    if 5 <= len(candidate) <= 6:
        return candidate

    if len(candidate) > 6:
        return candidate[:6]

    return candidate


def recognize_digits_from_roi(reader, roi, image_name):
    """
    Распознаёт цифры только из вырезанной области с крупным номером.
    """

    variants = prepare_roi_variants(roi)

    candidates = []

    for variant_name, variant_image in variants:
        processed_path = PROCESSED_DIR / f"{image_name}_{variant_name}.png"
        save_image_correctly(processed_path, variant_image)

        try:
            results = reader.readtext(
                variant_image,
                detail=1,
                paragraph=False,
                allowlist="0123456789"
            )
        except Exception:
            results = []

        # Вариант 1: берём отдельные OCR-блоки
        for box, text, confidence in results:
            digits = clean_digits(text)

            if len(digits) >= 4:
                number = normalize_candidate(digits)

                length_bonus = 20 if len(number) == 6 else 0
                score = float(confidence) * 100 + length_bonus

                candidates.append({
                    "number": number,
                    "confidence": float(confidence),
                    "variant": variant_name,
                    "source": "single_block",
                    "score": score
                })

        # Вариант 2: если OCR разбил цифры на части, склеиваем их слева направо
        parts = []

        for box, text, confidence in results:
            digits = clean_digits(text)

            if not digits:
                continue

            xs = [point[0] for point in box]
            x_left = min(xs)

            parts.append({
                "x": x_left,
                "digits": digits,
                "confidence": float(confidence)
            })

        if parts:
            parts_sorted = sorted(parts, key=lambda item: item["x"])
            joined = "".join([item["digits"] for item in parts_sorted])
            joined = normalize_candidate(joined)

            if len(joined) >= 4:
                avg_confidence = float(np.mean([item["confidence"] for item in parts_sorted]))

                length_bonus = 25 if len(joined) == 6 else 0
                score = avg_confidence * 100 + length_bonus

                candidates.append({
                    "number": joined,
                    "confidence": avg_confidence,
                    "variant": variant_name,
                    "source": "joined_blocks",
                    "score": score
                })

    if not candidates:
        return {
            "number": "не распознан",
            "confidence": 0,
            "variant": "-",
            "source": "-"
        }

    # Сначала предпочитаем 6-значные номера
    six_digit_candidates = [
        candidate for candidate in candidates
        if len(candidate["number"]) == 6
    ]

    if six_digit_candidates:
        best = max(six_digit_candidates, key=lambda item: item["score"])
    else:
        best = max(candidates, key=lambda item: item["score"])

    return {
        "number": best["number"],
        "confidence": round(best["confidence"], 3),
        "variant": best["variant"],
        "source": best["source"]
    }


def draw_result_on_image(image, bbox, number):
    """
    Рисует рамку вокруг найденной области с номером.
    """

    result = image.copy()

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        4
    )

    cv2.putText(
        result,
        str(number),
        (x1, max(y1 - 15, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        4
    )

    return result


# =====================================================
# ЗАКЛЮЧЕНИЕ ПО ДЕФЕКТАМ
# =====================================================

def build_defect_report(results):
    """
    Формирует текстовое заключение по дефектам.
    """

    lines = []

    lines.append("ЗАКЛЮЧЕНИЕ ПО ТИПАМ ДЕФЕКТОВ ИЗОБРАЖЕНИЙ")
    lines.append("=" * 70)
    lines.append("")
    lines.append("В рамках практического задания выполнялась обработка фотографий счётчиков.")
    lines.append("Для каждого изображения были сформированы Ч/Б-представление и RGB-каналы,")
    lines.append("а также выполнено распознавание крупного серийного номера счётчика.")
    lines.append("")

    lines.append("1. Когда помогает перевод изображения в Ч/Б")
    lines.append("-" * 70)
    lines.append("Перевод в Ч/Б помогает, когда основная информация определяется не цветом,")
    lines.append("а яркостным контрастом: тёмные цифры на светлом фоне или светлые цифры на тёмном фоне.")
    lines.append("Такой подход полезен при низком контрасте, слабой яркости, шуме и небольшом размытии.")
    lines.append("После перевода в Ч/Б можно применять CLAHE, Otsu threshold и adaptive threshold.")
    lines.append("")

    lines.append("2. Когда требуется разложение на RGB")
    lines.append("-" * 70)
    lines.append("RGB-разложение требуется, когда изображение имеет цветовой перекос, блики,")
    lines.append("цветные артефакты или когда отдельный цветовой канал содержит более контрастные цифры.")
    lines.append("Например, при синей/жёлтой подсветке один канал может давать более читаемое изображение,")
    lines.append("чем обычное Ч/Б-представление.")
    lines.append("")

    lines.append("3. Результаты по изображениям")
    lines.append("-" * 70)

    for item in results:
        defects = item["Дефекты"]
        number = item["Распознанный номер"]
        recommendation = item["Рекомендация по обработке"]

        lines.append(f"{item['Имя файла']}:")
        lines.append(f"  Распознанный номер: {number}")
        lines.append(f"  Дефекты: {defects}")
        lines.append(f"  Рекомендация: {recommendation}")
        lines.append("")

    return "\n".join(lines)


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ1: обработка фотографий счётчиков")
    print("OpenCV + контрастная предобработка + EasyOCR")
    print("=" * 70)

    image_files = [
        file for file in sorted(INPUT_DIR.iterdir())
        if file.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print("Изображения не найдены.")
        print(f"Положи фотографии счётчиков в папку: {INPUT_DIR}")
        return

    print(f"Найдено изображений: {len(image_files)}")
    print("Загружаем EasyOCR. Первый запуск может занять несколько минут...")

    reader = easyocr.Reader(["en"], gpu=False)

    table_rows = []

    for image_path in image_files:
        print(f"\nОбработка файла: {image_path.name}")

        image = read_image_correctly(image_path)

        if image is None:
            print("Не удалось прочитать изображение.")
            continue

        image_name = image_path.stem

        # 1. Разложение на каналы
        channels = split_channels(image, image_name)

        # 2. Коллаж
        collage_path = create_collage(image, channels, image_name)

        # 3. Анализ качества изображения
        quality = analyze_image_quality(image, channels)

        # 4. Поиск области с крупным номером
        roi, bbox, roi_method = find_big_number_roi(image)

        roi_path = ROI_DIR / f"{image_name}_number_roi.png"
        save_image_correctly(roi_path, roi)

        # 5. OCR только по области с крупными цифрами
        ocr_result = recognize_digits_from_roi(reader, roi, image_name)

        number = ocr_result["number"]

        # 6. Картинка с рамкой
        annotated = draw_result_on_image(image, bbox, number)

        annotated_path = ANNOTATED_DIR / f"{image_name}_annotated.png"
        save_image_correctly(annotated_path, annotated)

        print(f"Найденный номер: {number}")
        print(f"Уверенность: {ocr_result['confidence']}")
        print(f"Метод поиска области: {roi_method}")
        print(f"Вариант OCR: {ocr_result['variant']}")

        defects_text = "; ".join(quality["defects"])

        table_rows.append({
            "Имя файла": image_path.name,
            "Путь к исходному изображению": str(image_path),
            "Коллаж каналов": str(collage_path),
            "Вырезанная область номера": str(roi_path),
            "Изображение с найденной областью": str(annotated_path),
            "Распознанный номер": number,
            "Уверенность OCR": ocr_result["confidence"],
            "Метод поиска области": roi_method,
            "Вариант OCR": ocr_result["variant"],
            "Источник OCR": ocr_result["source"],
            "Дефекты": defects_text,
            "Рекомендация по обработке": quality["recommendation"],
            "Яркость": quality["brightness"],
            "Контраст": quality["contrast"],
            "Резкость": quality["sharpness"],
            "Шум": quality["noise"],
            "Mean R": quality["mean_red"],
            "Mean G": quality["mean_green"],
            "Mean B": quality["mean_blue"],
        })

    # 7. Excel
    df = pd.DataFrame(table_rows)
    df.to_excel(EXCEL_PATH, index=False)

    # 8. Текстовое заключение
    report_text = build_defect_report(table_rows)
    DEFECT_REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("\nГотово.")
    print(f"Excel-таблица: {EXCEL_PATH}")
    print(f"Заключение по дефектам: {DEFECT_REPORT_PATH}")
    print(f"Области с номерами: {ROI_DIR}")
    print(f"Картинки с рамками: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()