import subprocess
import sys
from pathlib import Path

from project_config import (
    BASE_DIR,
    CURRENT_RUN_DIR,
    RUN_CONFIG_PATH,
    load_run_config,
    save_run_config,
    prepare_new_run_from_local_video,
    prepare_empty_run_for_download
)


PIPELINE_STEPS = [
    {
        "title": "ПЗ2: нарезка видео на кадры",
        "script": "PZ2_video_to_frames.py"
    },
    {
        "title": "ПЗ3: OCR текста с кадров",
        "script": "PZ3_ocr_from_frames.py"
    },
    {
        "title": "ПЗ4: извлечение аудио и распознавание речи",
        "script": "PZ4_audio_whisper.py"
    },
    {
        "title": "ПЗ5: детектирование объектов YOLOv5",
        "script": "PZ5_yolo_object_detection.py"
    },
    {
        "title": "ПЗ6: классификация объектов ResNet34",
        "script": "PZ6_resnet_classification.py"
    },
    {
        "title": "ПЗ7: LLM-анализ изображений",
        "script": "PZ7_llm_ollama_analysis.py"
    },
    {
        "title": "ПЗ8: постобработка и финальный JSON",
        "script": "PZ8_postprocessing.py"
    }
]


def print_header():
    print("=" * 80)
    print("СИСТЕМА АНАЛИЗА ВИДЕОКОНТЕНТА")
    print("=" * 80)
    print()
    print("Главный сценарий проекта:")
    print("1. Выбор видео с компьютера или скачивание по ссылке")
    print("2. Создание единой рабочей папки results/current_run")
    print("3. Нарезка видео на кадры")
    print("4. OCR, Whisper, YOLOv5, ResNet34, LLM")
    print("5. Формирование финального JSON в формате TIME_BASED_REPORT")
    print()


def choose_local_video_path():
    print("\nВыбор локального видео.")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )

        root.destroy()

        if file_path:
            return file_path

    except Exception:
        pass

    print("Окно выбора файла не открылось.")
    print("Вставь полный путь к видео вручную.")
    print(r"Пример: C:\Users\Asus\Desktop\video.mp4")

    return input("Путь к видео: ").strip().strip('"')


def choose_video_quality():
    print("\nВыберите качество скачиваемого видео:")
    print("1 — лучшее доступное качество")
    print("2 — до 720p")
    print("3 — до 480p")
    print("4 — до 360p")
    print("Enter — лучшее доступное качество")

    choice = input("Введите номер: ").strip()

    if choice == "2":
        return "720"
    if choice == "3":
        return "480"
    if choice == "4":
        return "360"

    return "best"


def build_ytdlp_format(quality):
    if quality == "720":
        return "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
    if quality == "480":
        return "bestvideo[height<=480]+bestaudio/best[height<=480]/best"
    if quality == "360":
        return "bestvideo[height<=360]+bestaudio/best[height<=360]/best"

    return "bestvideo+bestaudio/best"


def download_video_with_ytdlp(url, output_path, quality):
    try:
        import yt_dlp
    except ImportError:
        print("\nyt-dlp не установлен.")
        print("Установи его командой:")
        print("python -m pip install yt-dlp")
        return False

    output_path = Path(output_path)

    ydl_opts = {
        "format": build_ytdlp_format(quality),
        "outtmpl": str(output_path),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if output_path.exists():
            return True

        print("\nФайл после скачивания не найден:")
        print(output_path)
        return False

    except Exception as error:
        print("\nОшибка при скачивании видео:")
        print(error)
        print("\nЕсли это Rutube и снова ошибка, проверь прокси/VPN, как раньше.")
        return False


def prepare_video_source():
    print("=" * 80)
    print("ВЫБОР ИСТОЧНИКА ВИДЕО")
    print("=" * 80)
    print()
    print("1 — выбрать видео с компьютера")
    print("2 — скачать видео по ссылке / Rutube")
    print("0 — выйти")
    print()

    choice = input("Введите номер: ").strip()

    if choice == "0":
        return None

    if choice == "1":
        video_path = choose_local_video_path()

        if not video_path:
            print("Видео не выбрано.")
            return None

        try:
            config = prepare_new_run_from_local_video(video_path)
            print("\nВидео скопировано в current_run:")
            print(config["video_path"])
            return config

        except Exception as error:
            print("\nНе удалось подготовить локальное видео:")
            print(error)
            return None

    if choice == "2":
        url = input("\nВставь ссылку на видео: ").strip()

        if not url:
            print("Ссылка не введена.")
            return None

        quality = choose_video_quality()

        try:
            config = prepare_empty_run_for_download(url, quality)
        except Exception as error:
            print("\nНе удалось создать папку current_run:")
            print(error)
            return None

        output_video_path = config["video_path"]

        print("\nНачинается скачивание видео...")
        print(f"Ссылка: {url}")
        print(f"Качество: {quality}")
        print(f"Файл будет сохранён: {output_video_path}")

        success = download_video_with_ytdlp(
            url=url,
            output_path=output_video_path,
            quality=quality
        )

        if not success:
            print("\nСкачивание не выполнено.")
            return None

        config["download_status"] = "success"
        save_run_config(config)

        print("\nВидео скачано и сохранено в current_run:")
        print(output_video_path)

        return config

    print("Неизвестный вариант.")
    return None


def check_script(script_name):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        print(f"Файл не найден: {script_path}")
        return False

    return True


def run_script(script_name):
    script_path = BASE_DIR / script_name

    print()
    print("-" * 80)
    print(f"Запуск: {script_name}")
    print("-" * 80)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR)
        )

        if result.returncode == 0:
            print(f"Этап завершён успешно: {script_name}")
            return True

        print(f"Этап завершился с ошибкой: {script_name}")
        print(f"Код ошибки: {result.returncode}")
        return False

    except KeyboardInterrupt:
        print("Выполнение остановлено пользователем.")
        return False

    except Exception as error:
        print(f"Ошибка при запуске {script_name}:")
        print(error)
        return False


def run_full_pipeline():
    completed = []
    failed = []

    for step in PIPELINE_STEPS:
        title = step["title"]
        script = step["script"]

        print()
        print("=" * 80)
        print(title)
        print("=" * 80)

        if not check_script(script):
            failed.append(title)
            continue

        success = run_script(script)

        if success:
            completed.append(title)
        else:
            failed.append(title)

            choice = input("Продолжить следующие этапы? Enter — да, 0 — остановить: ").strip()

            if choice == "0":
                break

    return completed, failed


def show_result_paths():
    try:
        config = load_run_config()
    except Exception as error:
        print("Не удалось загрузить текущий запуск.")
        print(error)
        return

    print()
    print("=" * 80)
    print("ПУТИ ТЕКУЩЕГО ЗАПУСКА")
    print("=" * 80)
    print(f"Папка current_run: {CURRENT_RUN_DIR}")
    print(f"Источник: {config.get('source_type')}")
    print(f"Исходное значение: {config.get('source_value')}")
    print(f"Видео: {config.get('video_path')}")
    print(f"Кадры: {config.get('frames_dir')}")
    print(f"YOLO crops: {config.get('object_crops_dir')}")
    print(f"Финальная папка: {config.get('final_dir')}")
    print(f"Финальный JSON: {config.get('final_json_path')}")


def main():
    print_header()

    print("Режим работы:")
    print("1 — новый полный анализ видео")
    print("2 — показать пути текущего запуска")
    print("0 — выйти")
    print()

    choice = input("Введите номер: ").strip()

    if choice == "0":
        print("Выход.")
        return

    if choice == "2":
        show_result_paths()
        return

    if choice != "1":
        print("Неизвестный режим.")
        return

    config = prepare_video_source()

    if config is None:
        print("\nЗапуск не подготовлен.")
        return

    print()
    print("Текущий запуск подготовлен.")
    print(f"Папка запуска: {config.get('current_run_dir')}")
    print(f"Видео: {config.get('video_path')}")

    completed, failed = run_full_pipeline()

    print()
    print("=" * 80)
    print("ПАЙПЛАЙН ЗАВЕРШЁН")
    print("=" * 80)
    print(f"Успешно выполнено этапов: {len(completed)}")
    print(f"Этапов с ошибками: {len(failed)}")

    show_result_paths()


if __name__ == "__main__":
    main()