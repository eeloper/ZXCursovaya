import json
import subprocess
import pandas as pd
import whisper
from pathlib import Path
from datetime import datetime


# =====================================================
# НАСТРОЙКА ПАПОК
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

# Видео берём из папки ПЗ2
VIDEO_FOLDER = BASE_DIR / "results" / "pz2_frames" / "VIDEO_FOLDER"

# Общая папка результатов ПЗ4
RESULT_ROOT = BASE_DIR / "results" / "pz4_whisper"

# Каждый запуск сохраняем в отдельную папку
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULT_ROOT / f"whisper_run_{RUN_TIMESTAMP}"

AUDIO_DIR = RUN_DIR / "audio"
TRANSCRIPT_DIR = RUN_DIR / "transcripts"

RESULT_ROOT.mkdir(parents=True, exist_ok=True)
RUN_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================

def seconds_to_time(seconds):
    """
    Переводит секунды в формат ЧЧ:ММ:СС.
    """

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def find_video_files():
    """
    Ищет видеофайлы в папке VIDEO_FOLDER.
    """

    if not VIDEO_FOLDER.exists():
        return []

    videos = [
        file for file in sorted(VIDEO_FOLDER.iterdir())
        if file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]

    return videos


def choose_video():
    """
    Даёт пользователю выбрать видео для обработки.
    """

    videos = find_video_files()

    if not videos:
        print("В папке VIDEO_FOLDER нет видеофайлов.")
        print(VIDEO_FOLDER)
        return None

    print("\nНайдены видеофайлы:")

    for index, video in enumerate(videos, start=1):
        print(f"{index}. {video.name}")

    choice = input("\nВведите номер видео для обработки: ").strip()

    try:
        choice_number = int(choice)

        if 1 <= choice_number <= len(videos):
            return videos[choice_number - 1]

        print("Неверный номер видео.")
        return None

    except ValueError:
        print("Введено не число.")
        return None


def choose_whisper_model():
    """
    Позволяет выбрать модель Whisper.
    """

    print("\nВыберите модель Whisper:")
    print("1 — tiny   быстро, но ниже качество")
    print("2 — base   средний вариант")
    print("3 — small  лучше качество, но медленнее")
    print("4 — medium ещё лучше, но может долго работать")

    choice = input("\nВведите 1, 2, 3 или 4 (Enter = base): ").strip()

    if choice == "":
        return "base"

    if choice == "1":
        return "tiny"

    if choice == "2":
        return "base"

    if choice == "3":
        return "small"

    if choice == "4":
        return "medium"

    print("Неверный выбор. Используется модель base.")
    return "base"


def choose_language():
    """
    Позволяет выбрать язык распознавания.
    """

    print("\nВыберите язык распознавания:")
    print("1 — русский")
    print("2 — английский")
    print("3 — автоопределение")

    choice = input("\nВведите 1, 2 или 3 (Enter = русский): ").strip()

    if choice == "":
        return "ru"

    if choice == "1":
        return "ru"

    if choice == "2":
        return "en"

    if choice == "3":
        return None

    print("Неверный выбор. Используется русский язык.")
    return "ru"


# =====================================================
# ИЗВЛЕЧЕНИЕ АУДИО
# =====================================================

def extract_audio_from_video(video_path):
    """
    Извлекает аудиодорожку из видео и сохраняет её в WAV.
    """

    audio_path = AUDIO_DIR / f"{video_path.stem}_audio.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path)
    ]

    print("\nИзвлекаем аудио из видео...")
    print(f"Видео: {video_path}")
    print(f"Аудио будет сохранено: {audio_path}")

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        print("FFmpeg не найден.")
        print("Установи FFmpeg и перезапусти PyCharm.")
        return None
    except subprocess.CalledProcessError:
        print("Не удалось извлечь аудио из видео.")
        return None

    print("Аудио успешно извлечено.")
    return audio_path


# =====================================================
# WHISPER
# =====================================================

def transcribe_audio(audio_path, model_name, language):
    """
    Распознаёт речь из аудиофайла с помощью Whisper.
    """

    print("\nЗагружаем модель Whisper...")
    print(f"Модель: {model_name}")

    model = whisper.load_model(model_name)

    print("Начинаем распознавание аудио...")

    if language is None:
        result = model.transcribe(str(audio_path))
    else:
        result = model.transcribe(
            str(audio_path),
            language=language
        )

    return result


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def save_results(video_path, audio_path, whisper_result, model_name, language):
    """
    Сохраняет результаты распознавания в TXT, Excel и JSON.
    """

    full_text = whisper_result.get("text", "").strip()
    segments = whisper_result.get("segments", [])

    txt_path = TRANSCRIPT_DIR / f"{video_path.stem}_transcript.txt"

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(full_text)

    rows = []

    for segment in segments:
        start = float(segment.get("start", 0))
        end = float(segment.get("end", 0))
        text = segment.get("text", "").strip()

        rows.append({
            "video_name": video_path.name,
            "audio_file": str(audio_path),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "start_time": seconds_to_time(start),
            "end_time": seconds_to_time(end),
            "time_interval": f"{seconds_to_time(start)} - {seconds_to_time(end)}",
            "text": text
        })

    excel_path = TRANSCRIPT_DIR / f"{video_path.stem}_whisper_segments.xlsx"

    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)

    json_path = TRANSCRIPT_DIR / f"{video_path.stem}_whisper_report.json"

    json_report = {
        "report_type": "AUDIO_TRANSCRIPTION_REPORT",
        "video_name": video_path.name,
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "model_name": model_name,
        "language": language if language is not None else "auto",
        "full_text": full_text,
        "segments_count": len(segments),
        "segments": rows,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_report, file, ensure_ascii=False, indent=4)

    print("\nРезультаты сохранены.")
    print(f"Папка текущего запуска: {RUN_DIR}")
    print(f"Аудиофайл: {audio_path}")
    print(f"TXT-расшифровка: {txt_path}")
    print(f"Excel по сегментам: {excel_path}")
    print(f"JSON-отчёт: {json_path}")


# =====================================================
# ОСНОВНАЯ ПРОГРАММА
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ4: извлечение аудио и распознавание речи через Whisper")
    print("=" * 70)

    print("\nКаждый запуск сохраняется в новую папку:")
    print(RUN_DIR)

    video_path = choose_video()

    if video_path is None:
        return

    model_name = choose_whisper_model()
    language = choose_language()

    audio_path = extract_audio_from_video(video_path)

    if audio_path is None:
        return

    whisper_result = transcribe_audio(
        audio_path,
        model_name,
        language
    )

    save_results(
        video_path,
        audio_path,
        whisper_result,
        model_name,
        language
    )

    print("\nГотово.")


if __name__ == "__main__":
    main()