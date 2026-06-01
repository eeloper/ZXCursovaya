import json
import shutil
from pathlib import Path
from datetime import datetime

import cv2

from project_config import load_run_config, RUN_CONFIG_PATH


# =====================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================

def seconds_to_time(seconds):
    seconds = int(float(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    sec = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def clear_folder(folder_path):
    folder_path = Path(folder_path)

    if folder_path.exists():
        shutil.rmtree(folder_path)

    folder_path.mkdir(parents=True, exist_ok=True)


def save_run_config(config):
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def ask_extraction_fps():
    print("\nВведите частоту сохранения кадров.")
    print("Например:")
    print("1  — сохранить 1 кадр в секунду")
    print("2  — сохранить 2 кадра в секунду")
    print("5  — сохранить 5 кадров в секунду")
    print("Enter — значение по умолчанию: 1 кадр в секунду")

    user_input = input("\nЧастота кадров для сохранения: ").strip()

    if user_input == "":
        return 1.0

    try:
        value = float(user_input.replace(",", "."))

        if value <= 0:
            print("Частота должна быть больше 0. Используется 1 кадр в секунду.")
            return 1.0

        return value

    except ValueError:
        print("Введено некорректное значение. Используется 1 кадр в секунду.")
        return 1.0


# =====================================================
# ПОЛУЧЕНИЕ МЕТАДАННЫХ ВИДЕО
# =====================================================

def get_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 0:
        fps = 30.0

    duration_seconds = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": round(float(fps), 3),
        "frame_count": frame_count,
        "duration_seconds": round(duration_seconds, 3),
        "duration_formatted": seconds_to_time(duration_seconds)
    }


# =====================================================
# НАРЕЗКА ВИДЕО НА КАДРЫ
# =====================================================

def extract_frames(video_path, frames_dir, extraction_fps):
    video_path = Path(video_path)
    frames_dir = Path(frames_dir)

    metadata = get_video_metadata(video_path)

    source_fps = metadata["fps"]
    source_frame_count = metadata["frame_count"]

    if extraction_fps > source_fps:
        print("\nЗапрошенная частота выше FPS исходного видео.")
        print(f"FPS видео: {source_fps}")
        print(f"Будет использована частота: {source_fps}")
        extraction_fps = source_fps

    frame_step = max(1, int(round(source_fps / extraction_fps)))

    print("\nМетаданные видео:")
    print(f"FPS исходного видео: {source_fps}")
    print(f"Количество кадров в видео: {source_frame_count}")
    print(f"Длительность: {metadata['duration_formatted']}")
    print(f"Будет сохраняться примерно {extraction_fps} кадр(ов) в секунду")
    print(f"Шаг сохранения: каждый {frame_step}-й кадр")

    clear_folder(frames_dir)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    saved_frames_count = 0
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_index % frame_step == 0:
            time_seconds = frame_index / source_fps
            time_ms = int(time_seconds * 1000)

            frame_filename = (
                f"frame_{saved_frames_count:06d}"
                f"_source_{frame_index:06d}"
                f"_time_{time_ms:08d}ms.jpg"
            )

            frame_path = frames_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)

            saved_frames_count += 1

        frame_index += 1

    cap.release()

    return {
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "video_duration_seconds": metadata["duration_seconds"],
        "video_duration_formatted": metadata["duration_formatted"],
        "extraction_fps": extraction_fps,
        "frame_step": frame_step,
        "saved_frames_count": saved_frames_count
    }


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("ПЗ2: нарезка видео на кадры")
    print("=" * 70)

    try:
        config = load_run_config()
    except Exception as error:
        print("\nНе удалось загрузить run_config.json.")
        print("Сначала запусти main.py и выбери видео.")
        print(error)
        return

    video_path = Path(config["video_path"])
    frames_dir = Path(config["frames_dir"])

    if not video_path.exists():
        print("\nВидео не найдено:")
        print(video_path)
        return

    print("\nИспользуется видео текущего запуска:")
    print(video_path)

    print("\nКадры будут сохранены в папку:")
    print(frames_dir)

    extraction_fps = ask_extraction_fps()

    try:
        result = extract_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            extraction_fps=extraction_fps
        )

    except Exception as error:
        print("\nОшибка при нарезке видео на кадры:")
        print(error)

        config["pz2_status"] = "error"
        config["pz2_error"] = str(error)
        config["pz2_finished_at"] = datetime.now().isoformat()
        save_run_config(config)

        return

    config["source_fps"] = result["source_fps"]
    config["fps"] = result["source_fps"]
    config["source_frame_count"] = result["source_frame_count"]
    config["frameCount"] = result["source_frame_count"]
    config["video_duration_seconds"] = result["video_duration_seconds"]
    config["video_duration_formatted"] = result["video_duration_formatted"]
    config["extraction_fps"] = result["extraction_fps"]
    config["frame_step"] = result["frame_step"]
    config["saved_frames_count"] = result["saved_frames_count"]
    config["frames_dir"] = str(frames_dir)
    config["pz2_status"] = "success"
    config["pz2_finished_at"] = datetime.now().isoformat()

    save_run_config(config)

    print("\nПЗ2 завершено успешно.")
    print(f"Сохранено кадров: {result['saved_frames_count']}")
    print(f"Папка с кадрами: {frames_dir}")
    print("\nrun_config.json обновлён.")


if __name__ == "__main__":
    main()