import cv2
import json
import pandas as pd
import requests
import shutil
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse


# =====================================================
# НАСТРОЙКА ПАПОК ПРОЕКТА
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

RESULT_DIR = BASE_DIR / "results" / "pz2_frames"

VIDEO_FOLDER = RESULT_DIR / "VIDEO_FOLDER"
FRAME_FOLDER = RESULT_DIR / "FRAME_FOLDER"
SAMPLE_FRAMES_DIR = RESULT_DIR / "sample_frames"

VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)
FRAME_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def seconds_to_time(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def ask_float(message, default_value):
    user_input = input(message).strip()

    if user_input == "":
        return default_value

    try:
        return float(user_input.replace(",", "."))
    except ValueError:
        print("Введено некорректное значение. Используется значение по умолчанию.")
        return default_value


def ask_text(message, default_value=""):
    user_input = input(message).strip()

    if user_input == "":
        return default_value

    return user_input


def is_video_file(file_path):
    return file_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def clear_folder_contents(folder_path):
    """
    Очищает только служебные папки.
    Основная папка FRAME_FOLDER не очищается, чтобы кадры оставались для ПЗ3.
    """

    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        return

    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def make_safe_video_name(extension=".mp4"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"downloaded_video_{timestamp}{extension}"


def save_frame_correctly(save_path, frame):
    success, encoded_image = cv2.imencode(".jpg", frame)

    if success:
        encoded_image.tofile(str(save_path))
        return True

    return False


def check_video_can_be_opened(video_path):
    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        video.release()
        return False

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    video.release()

    return frame_count > 0 and fps > 0


# =====================================================
# ВЫБОР ВИДЕО С КОМПЬЮТЕРА
# =====================================================

def find_local_videos():
    videos = [
        file for file in sorted(VIDEO_FOLDER.iterdir())
        if is_video_file(file)
    ]

    return videos


def choose_local_video():
    videos = find_local_videos()

    if not videos:
        print("\nВ папке не найдено видео:")
        print(VIDEO_FOLDER)
        print("\nПоложи видеофайл в эту папку и запусти программу снова.")
        return []

    print("\nНайдены видеофайлы:")

    for index, video in enumerate(videos, start=1):
        print(f"{index}. {video.name}")

    print("0. Обработать все видео из папки")

    choice = input("\nВведите номер видео или 0 для обработки всех видео: ").strip()

    if choice == "0":
        return videos

    try:
        choice_number = int(choice)

        if 1 <= choice_number <= len(videos):
            return [videos[choice_number - 1]]

        print("Неверный номер. Будут обработаны все видео.")
        return videos

    except ValueError:
        print("Введено не число. Будут обработаны все видео.")
        return videos


# =====================================================
# СКАЧИВАНИЕ ВИДЕО ПО ССЫЛКЕ
# =====================================================

def get_ytdlp_format(quality):
    quality = str(quality).lower().strip()

    if quality == "best":
        return "best[ext=mp4]/best"

    if quality == "worst":
        return "worst[ext=mp4]/worst"

    try:
        quality_int = int(quality)
        return f"best[height<={quality_int}][ext=mp4]/best[height<={quality_int}]/best"
    except ValueError:
        return "best[ext=mp4]/best"


def download_with_ytdlp(video_url, quality):
    """
    Скачивает видео через yt-dlp.
    Для Rutube добавлены увеличенный timeout, повторы и заголовки браузера.
    """

    try:
        import yt_dlp
    except ImportError:
        print("Библиотека yt-dlp не установлена.")
        print("Установи её командой: pip install -U yt-dlp")
        return None

    print("\nСкачивание видео через yt-dlp...")
    print(f"Ссылка: {video_url}")
    print(f"Качество: {quality}")

    before_files = set(VIDEO_FOLDER.glob("*"))

    ydl_options = {
        "outtmpl": str(VIDEO_FOLDER / "downloaded_video_%(id)s.%(ext)s"),
        "format": get_ytdlp_format(quality),
        "noplaylist": True,
        "restrictfilenames": True,
        "merge_output_format": "mp4",

        # Важно для Rutube: увеличиваем время ожидания
        "socket_timeout": 120,

        # Повторы при ошибках соединения
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 10,

        # Иногда помогает при проблемах с сетью
        "force_ipv4": True,

        # Заголовок обычного браузера
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_options) as ydl:
            ydl.download([video_url])
    except Exception as error:
        print("yt-dlp не смог скачать видео.")
        print(f"Ошибка: {error}")
        return None

    after_files = set(VIDEO_FOLDER.glob("*"))
    new_files = list(after_files - before_files)

    video_files = [
        file for file in new_files
        if is_video_file(file)
    ]

    if video_files:
        downloaded_video = sorted(video_files, key=lambda file: file.stat().st_mtime)[-1]
    else:
        all_videos = find_local_videos()

        if not all_videos:
            print("Файл скачан, но видеофайл не найден.")
            return None

        downloaded_video = sorted(all_videos, key=lambda file: file.stat().st_mtime)[-1]

    safe_extension = downloaded_video.suffix.lower()

    if safe_extension not in SUPPORTED_VIDEO_EXTENSIONS:
        safe_extension = ".mp4"

    safe_video_path = VIDEO_FOLDER / make_safe_video_name(safe_extension)

    try:
        if downloaded_video != safe_video_path:
            downloaded_video.rename(safe_video_path)
            downloaded_video = safe_video_path
    except Exception:
        pass

    print(f"Видео скачано: {downloaded_video}")

    if not check_video_can_be_opened(downloaded_video):
        print("Внимание: скачанное видео не удалось открыть через OpenCV.")
        print("Файл может иметь неподдерживаемый кодек или быть повреждён.")
        return None

    return downloaded_video


def download_direct_file(video_url):
    """
    Пробует скачать видео как прямой файл.
    Для Rutube обычно не подходит, потому что Rutube-ссылка ведёт на страницу,
    а не на прямой mp4-файл.
    """

    print("\nПробуем скачать ссылку как прямой видеофайл...")

    parsed_url = urlparse(video_url)
    file_name = Path(parsed_url.path).name

    if not file_name:
        file_name = "downloaded_video.mp4"

    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        extension = ".mp4"

    output_path = VIDEO_FOLDER / make_safe_video_name(extension)

    try:
        with requests.get(video_url, stream=True, timeout=60) as response:
            response.raise_for_status()

            with open(output_path, "wb") as file:
                shutil.copyfileobj(response.raw, file)

        print(f"Файл скачан: {output_path}")

        if not check_video_can_be_opened(output_path):
            print("Скачанный файл не удалось открыть как видео.")
            return None

        return output_path

    except Exception as error:
        print("Не удалось скачать файл напрямую.")
        print(f"Ошибка: {error}")
        return None


def is_yandex_disk_link(url):
    return "yadi.sk" in url or "disk.yandex" in url


def get_yandex_resource_info(public_key):
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources"

    response = requests.get(
        api_url,
        params={
            "public_key": public_key,
            "limit": 100
        },
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def get_yandex_download_link(public_key, path=None):
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

    params = {
        "public_key": public_key
    }

    if path is not None:
        params["path"] = path

    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()

    return response.json()["href"]


def download_yandex_file(public_url, file_name, item_path=None):
    extension = Path(file_name).suffix.lower()

    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        extension = ".mp4"

    output_path = VIDEO_FOLDER / make_safe_video_name(extension)

    try:
        download_link = get_yandex_download_link(public_url, item_path)

        with requests.get(download_link, stream=True, timeout=60) as response:
            response.raise_for_status()

            with open(output_path, "wb") as file:
                shutil.copyfileobj(response.raw, file)

        print(f"Видео скачано: {output_path}")

        if not check_video_can_be_opened(output_path):
            print("Скачанный файл не удалось открыть как видео.")
            return None

        return output_path

    except Exception as error:
        print(f"Не удалось скачать файл {file_name}")
        print(f"Ошибка: {error}")
        return None


def download_yandex_public_resource(public_url):
    print("\nОбработка ссылки Яндекс.Диска...")

    try:
        resource_info = get_yandex_resource_info(public_url)
    except Exception as error:
        print("Не удалось получить информацию о ресурсе Яндекс.Диска.")
        print(f"Ошибка: {error}")
        return []

    downloaded_videos = []
    resource_type = resource_info.get("type")

    if resource_type == "file":
        file_name = resource_info.get("name", "yandex_video.mp4")
        file_path = Path(file_name)

        if not is_video_file(file_path):
            print("Файл на Яндекс.Диске не является поддерживаемым видео.")
            return []

        video = download_yandex_file(public_url, file_name)

        if video is not None:
            downloaded_videos.append(video)

    elif resource_type == "dir":
        items = resource_info.get("_embedded", {}).get("items", [])

        video_items = [
            item for item in items
            if Path(item.get("name", "")).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ]

        if not video_items:
            print("В публичной папке Яндекс.Диска не найдено видео.")
            return []

        print("\nВ папке Яндекс.Диска найдены видео:")

        for index, item in enumerate(video_items, start=1):
            print(f"{index}. {item.get('name')}")

        print("0. Скачать все видео из папки")

        choice = input("\nВведите номер видео или 0 для скачивания всех: ").strip()

        if choice == "0":
            selected_items = video_items
        else:
            try:
                choice_number = int(choice)

                if 1 <= choice_number <= len(video_items):
                    selected_items = [video_items[choice_number - 1]]
                else:
                    print("Неверный номер. Будут скачаны все видео.")
                    selected_items = video_items

            except ValueError:
                print("Введено не число. Будут скачаны все видео.")
                selected_items = video_items

        for item in selected_items:
            file_name = item.get("name")
            item_path = item.get("path")

            video = download_yandex_file(public_url, file_name, item_path)

            if video is not None:
                downloaded_videos.append(video)

    else:
        print("Неизвестный тип ресурса Яндекс.Диска.")

    return downloaded_videos


def download_video_by_url():
    video_url = ask_text("\nВведите ссылку на видео: ")

    if not video_url:
        print("Ссылка не введена.")
        return []

    if is_yandex_disk_link(video_url):
        return download_yandex_public_resource(video_url)

    quality = ask_text(
        "Введите качество скачивания: best / worst / 1080 / 720 / 480 "
        "(Enter = 720): ",
        "720"
    )

    downloaded_video = download_with_ytdlp(video_url, quality)

    if downloaded_video is not None:
        return [downloaded_video]

    direct_video = download_direct_file(video_url)

    if direct_video is not None:
        return [direct_video]

    print("Видео по ссылке скачать не удалось.")
    print("Для Rutube это может быть временная проблема соединения или блокировка ответа сайта.")
    print("В таком случае можно скачать видео вручную и положить его в VIDEO_FOLDER.")
    return []


# =====================================================
# НАРЕЗКА ВИДЕО НА КАДРЫ
# =====================================================

def extract_frames_from_video(video_path, extract_fps):
    print("\n" + "=" * 70)
    print(f"Обработка видео: {video_path.name}")
    print("=" * 70)

    video_name = video_path.stem

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fps_label = str(extract_fps).replace(".", "_")

    output_folder_name = f"{video_name}_fps_{fps_label}_{run_timestamp}"
    output_frames_dir = FRAME_FOLDER / output_folder_name
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        print("Не удалось открыть видео через OpenCV.")
        print(f"Путь к видео: {video_path}")
        return None, []

    source_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or frame_count <= 0:
        print("Не удалось определить параметры видео.")
        video.release()
        return None, []

    duration_seconds = frame_count / source_fps
    duration_formatted = seconds_to_time(duration_seconds)

    if extract_fps <= 0:
        print("Частота нарезки должна быть больше 0. Используется 1 кадр/сек.")
        extract_fps = 1.0

    if extract_fps > source_fps:
        print("Заданная частота выше FPS исходного видео.")
        print("Будут сохранены все доступные кадры.")
        extract_fps = source_fps

    extraction_interval_seconds = 1 / extract_fps

    print(f"FPS исходного видео: {round(source_fps, 2)}")
    print(f"Всего кадров в видео: {frame_count}")
    print(f"Длительность видео: {duration_formatted}")
    print(f"Частота нарезки: {extract_fps} кадр/сек")
    print(f"Интервал сохранения: каждые {round(extraction_interval_seconds, 3)} сек")
    print(f"Папка текущего запуска: {output_frames_dir}")

    saved_frames_info = []

    current_frame_number = 0
    saved_frame_number = 0
    next_save_time = 0.0

    while True:
        success, frame = video.read()

        if not success:
            break

        current_time_seconds = current_frame_number / source_fps

        if current_time_seconds + 0.0001 >= next_save_time:
            time_ms = int(current_time_seconds * 1000)

            frame_filename = (
                f"{video_name}_frame_{saved_frame_number:05d}"
                f"_time_{time_ms:08d}ms.jpg"
            )

            frame_path = output_frames_dir / frame_filename
            save_frame_correctly(frame_path, frame)

            saved_frames_info.append({
                "video_name": video_path.name,
                "frame_file": frame_filename,
                "frame_path": str(frame_path),
                "frames_folder": str(output_frames_dir),
                "original_frame_number": current_frame_number,
                "saved_frame_number": saved_frame_number,
                "time_seconds": round(current_time_seconds, 3),
                "time_formatted": seconds_to_time(current_time_seconds)
            })

            if saved_frame_number < 10:
                sample_path = SAMPLE_FRAMES_DIR / frame_filename
                save_frame_correctly(sample_path, frame)

            saved_frame_number += 1
            next_save_time += extraction_interval_seconds

        current_frame_number += 1

    video.release()

    video_info = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "source_fps": round(source_fps, 3),
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "duration_formatted": duration_formatted,
        "extract_fps": extract_fps,
        "saved_frames": saved_frame_number,
        "frames_output_dir": str(output_frames_dir),
        "analysis_timestamp": datetime.now().isoformat()
    }

    print(f"Сохранено кадров: {saved_frame_number}")
    print(f"Папка с кадрами: {output_frames_dir}")

    return video_info, saved_frames_info


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ2: нарезка видео на изображения")
    print("=" * 70)

    print("\nРабочие папки:")
    print(f"Папка с исходными видео: {VIDEO_FOLDER}")
    print(f"Папка для сохранения кадров: {FRAME_FOLDER}")

    print("\nСтарые папки с кадрами удаляться не будут.")
    print("Это нужно, чтобы кадры из ПЗ2 оставались доступными для ПЗ3.")

    clear_folder_contents(SAMPLE_FRAMES_DIR)
    print("Папка sample_frames очищена.")

    print("\nВыберите источник видео:")
    print("1 — выбрать видео из папки на компьютере")
    print("2 — скачать видео по ссылке")

    source_choice = input("\nВведите 1 или 2: ").strip()

    if source_choice == "1":
        videos_to_process = choose_local_video()
        source_mode = "local_folder"

    elif source_choice == "2":
        videos_to_process = download_video_by_url()
        source_mode = "url_download"

    else:
        print("Неверный выбор.")
        return

    if not videos_to_process:
        print("Видео для обработки не найдено.")
        return

    extract_fps = ask_float(
        "\nВведите частоту нарезки кадров в кадрах/сек "
        "(например: 1 — один кадр в секунду; "
        "2 — два кадра в секунду; "
        "0.5 — один кадр каждые 2 секунды; "
        "Enter = 1): ",
        1.0
    )

    all_video_info = []
    all_frames_info = []

    for video_path in videos_to_process:
        video_info, frames_info = extract_frames_from_video(video_path, extract_fps)

        if video_info is not None:
            all_video_info.append(video_info)
            all_frames_info.extend(frames_info)

    frames_excel_path = RESULT_DIR / "frames_table.xlsx"
    videos_excel_path = RESULT_DIR / "videos_table.xlsx"

    frames_df = pd.DataFrame(all_frames_info)
    videos_df = pd.DataFrame(all_video_info)

    try:
        frames_df.to_excel(frames_excel_path, index=False)
        videos_df.to_excel(videos_excel_path, index=False)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        frames_excel_path = RESULT_DIR / f"frames_table_{timestamp}.xlsx"
        videos_excel_path = RESULT_DIR / f"videos_table_{timestamp}.xlsx"

        frames_df.to_excel(frames_excel_path, index=False)
        videos_df.to_excel(videos_excel_path, index=False)

        print("\nОсновной Excel-файл был открыт или заблокирован.")
        print("Результаты сохранены в новые файлы:")
        print(frames_excel_path)
        print(videos_excel_path)

    json_report = {
        "report_type": "VIDEO_FRAME_EXTRACTION_REPORT",
        "source_mode": source_mode,
        "extract_fps": extract_fps,
        "videos_count": len(all_video_info),
        "total_saved_frames": len(all_frames_info),
        "videos": all_video_info,
        "frames": all_frames_info,
        "analysis_timestamp": datetime.now().isoformat()
    }

    json_path = RESULT_DIR / "video_frame_extraction_report.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_report, file, ensure_ascii=False, indent=4)

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print(f"Обработано видео: {len(all_video_info)}")
    print(f"Сохранено кадров: {len(all_frames_info)}")
    print(f"Кадры сохранены в папку: {FRAME_FOLDER}")
    print(f"Excel по кадрам: {frames_excel_path}")
    print(f"Excel по видео: {videos_excel_path}")
    print(f"JSON-отчёт: {json_path}")
    print(f"Примеры кадров: {SAMPLE_FRAMES_DIR}")


if __name__ == "__main__":
    main()