import json
import shutil
from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = BASE_DIR / "results"
CURRENT_RUN_DIR = RESULTS_DIR / "current_run"
RUN_CONFIG_PATH = CURRENT_RUN_DIR / "run_config.json"


def reset_current_run():
    if CURRENT_RUN_DIR.exists():
        shutil.rmtree(CURRENT_RUN_DIR)

    create_current_run_dirs()


def create_current_run_dirs():
    folders = [
        CURRENT_RUN_DIR,
        CURRENT_RUN_DIR / "input",
        CURRENT_RUN_DIR / "frames",
        CURRENT_RUN_DIR / "audio",
        CURRENT_RUN_DIR / "ocr",
        CURRENT_RUN_DIR / "whisper",
        CURRENT_RUN_DIR / "yolo",
        CURRENT_RUN_DIR / "yolo" / "object_crops",
        CURRENT_RUN_DIR / "resnet",
        CURRENT_RUN_DIR / "llm",
        CURRENT_RUN_DIR / "postprocessing",
        CURRENT_RUN_DIR / "final"
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def save_run_config(config):
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


def load_run_config():
    if not RUN_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Файл run_config.json не найден. Сначала запусти main.py и выбери видео."
        )

    with open(RUN_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_base_config(video_path, source_type, source_value="", quality=""):
    config = {
        "run_id": datetime.now().strftime("run_%Y%m%d_%H%M%S"),
        "source_type": source_type,
        "source_value": source_value,
        "quality": quality,
        "video_path": str(video_path),
        "current_run_dir": str(CURRENT_RUN_DIR),
        "input_dir": str(CURRENT_RUN_DIR / "input"),
        "frames_dir": str(CURRENT_RUN_DIR / "frames"),
        "audio_dir": str(CURRENT_RUN_DIR / "audio"),
        "ocr_dir": str(CURRENT_RUN_DIR / "ocr"),
        "whisper_dir": str(CURRENT_RUN_DIR / "whisper"),
        "yolo_dir": str(CURRENT_RUN_DIR / "yolo"),
        "object_crops_dir": str(CURRENT_RUN_DIR / "yolo" / "object_crops"),
        "resnet_dir": str(CURRENT_RUN_DIR / "resnet"),
        "llm_dir": str(CURRENT_RUN_DIR / "llm"),
        "postprocessing_dir": str(CURRENT_RUN_DIR / "postprocessing"),
        "final_dir": str(CURRENT_RUN_DIR / "final"),
        "final_json_path": str(CURRENT_RUN_DIR / "final" / "final_analysis_report.json"),
        "created_at": datetime.now().isoformat()
    }

    return config


def prepare_new_run_from_local_video(video_path):
    reset_current_run()

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    target_video_path = CURRENT_RUN_DIR / "input" / "input_video.mp4"
    shutil.copy2(video_path, target_video_path)

    config = build_base_config(
        video_path=target_video_path,
        source_type="local_file",
        source_value=str(video_path),
        quality=""
    )

    save_run_config(config)

    return config


def prepare_empty_run_for_download(url, quality):
    reset_current_run()

    target_video_path = CURRENT_RUN_DIR / "input" / "input_video.mp4"

    config = build_base_config(
        video_path=target_video_path,
        source_type="url",
        source_value=url,
        quality=quality
    )

    save_run_config(config)

    return config