import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import whisper

from project_config import load_run_config, RUN_CONFIG_PATH


# =====================================================
# НАСТРОЙКИ
# =====================================================

DEFAULT_WHISPER_MODEL = "base"


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


def choose_whisper_model():
    print("\nВыберите модель Whisper:")
    print("1 — tiny, самая быстрая")
    print("2 — base, стандартный вариант")
    print("3 — small, точнее, но медленнее")
    print("4 — medium, ещё точнее, но заметно медленнее")
    print("Enter — base")

    choice = input("\nВведите номер модели: ").strip()

    if choice == "1":
        return "tiny"

    if choice == "2" or choice == "":
        return "base"

    if choice == "3":
        return "small"

    if choice == "4":
        return "medium"

    print("Неверный выбор. Используется base.")
    return DEFAULT_WHISPER_MODEL


# =====================================================
# ИЗВЛЕЧЕНИЕ АУДИО
# =====================================================

def check_ffmpeg_available():
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        return result.returncode == 0

    except Exception:
        return False


def extract_audio_from_video(video_path, audio_path):
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not check_ffmpeg_available():
        raise RuntimeError(
            "FFmpeg не найден. Установи FFmpeg и добавь его в PATH."
        )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Ошибка при извлечении аудио через FFmpeg:\n"
            + result.stderr
        )

    if not audio_path.exists():
        raise RuntimeError("Аудиофайл не был создан.")

    return audio_path


# =====================================================
# WHISPER
# =====================================================

def transcribe_audio(audio_path, model_name):
    print("\nЗагружается модель Whisper...")
    print(f"Модель: {model_name}")

    model = whisper.load_model(model_name)

    print("\nНачинается распознавание аудио...")

    result = model.transcribe(
        str(audio_path),
        language="ru",
        verbose=False
    )

    return result


def build_segments(result):
    segments = []

    raw_segments = result.get("segments", [])

    for index, segment in enumerate(raw_segments, start=1):
        start_seconds = float(segment.get("start", 0))
        end_seconds = float(segment.get("end", 0))
        text = str(segment.get("text", "")).strip()

        segments.append({
            "segment_id": index,
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "start_time": seconds_to_time(start_seconds),
            "end_time": seconds_to_time(end_seconds),
            "time_interval": f"{seconds_to_time(start_seconds)} - {seconds_to_time(end_seconds)}",
            "text": text
        })

    return segments


# =====================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =====================================================

def save_whisper_results(whisper_dir, audio_path, model_name, result, segments):
    whisper_dir = Path(whisper_dir)
    whisper_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = whisper_dir / "input_video_transcript.txt"
    segments_excel_path = whisper_dir / "input_video_whisper_segments.xlsx"
    report_json_path = whisper_dir / "input_video_whisper_report.json"

    full_text = str(result.get("text", "")).strip()

    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(full_text)

    segments_df = pd.DataFrame(segments)
    segments_df.to_excel(segments_excel_path, index=False)

    report = {
        "report_type": "WHISPER_AUDIO_RECOGNITION",
        "model_name": model_name,
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path),
        "segments_excel_path": str(segments_excel_path),
        "segments_count": len(segments),
        "full_text": full_text,
        "analysis_timestamp": datetime.now().isoformat()
    }

    with open(report_json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    return {
        "transcript_path": transcript_path,
        "segments_excel_path": segments_excel_path,
        "report_json_path": report_json_path,
        "full_text": full_text
    }


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ4: извлечение аудио и распознавание речи Whisper")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выбери видео.")
        print(error)
        return

    video_path = Path(config["video_path"])
    audio_dir = Path(config["audio_dir"])
    whisper_dir = Path(config["whisper_dir"])

    if not video_path.exists():
        print("\nВидео не найдено:")
        print(video_path)
        return

    print("\nИспользуется видео текущего запуска:")
    print(video_path)

    print("\nАудио будет сохранено в папку:")
    print(audio_dir)

    print("\nРезультаты Whisper будут сохранены в папку:")
    print(whisper_dir)

    model_name = choose_whisper_model()

    clear_folder(audio_dir)
    clear_folder(whisper_dir)

    audio_path = audio_dir / "input_video_audio.wav"

    try:
        extracted_audio_path = extract_audio_from_video(
            video_path=video_path,
            audio_path=audio_path
        )

        result = transcribe_audio(
            audio_path=extracted_audio_path,
            model_name=model_name
        )

        segments = build_segments(result)

        saved_paths = save_whisper_results(
            whisper_dir=whisper_dir,
            audio_path=extracted_audio_path,
            model_name=model_name,
            result=result,
            segments=segments
        )

    except Exception as error:
        print("\nОшибка при выполнении ПЗ4:")
        print(error)

        config["pz4_status"] = "error"
        config["pz4_error"] = str(error)
        config["pz4_finished_at"] = datetime.now().isoformat()
        save_run_config(config)

        return

    config["audio_dir"] = str(audio_dir)
    config["audio_path"] = str(extracted_audio_path)
    config["whisper_dir"] = str(whisper_dir)
    config["whisper_model"] = model_name
    config["whisper_transcript_path"] = str(saved_paths["transcript_path"])
    config["whisper_segments_path"] = str(saved_paths["segments_excel_path"])
    config["whisper_report_path"] = str(saved_paths["report_json_path"])
    config["whisper_segments_count"] = len(segments)
    config["whisper_full_text"] = saved_paths["full_text"]
    config["pz4_status"] = "success"
    config["pz4_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\nПЗ4 завершено успешно.")
    print(f"Аудиофайл: {extracted_audio_path}")
    print(f"Сегментов речи: {len(segments)}")
    print(f"Текстовая расшифровка: {saved_paths['transcript_path']}")
    print(f"Excel с сегментами: {saved_paths['segments_excel_path']}")
    print("\nrun_config.json обновлён.")


if __name__ == "__main__":
    main()