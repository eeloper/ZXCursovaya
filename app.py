import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from project_config import (
    BASE_DIR,
    CURRENT_RUN_DIR,
    load_run_config,
    save_run_config,
    prepare_new_run_from_local_video,
    prepare_empty_run_for_download
)


PIPELINE_STEPS = [
    ("ПЗ2: нарезка видео на кадры", "PZ2_video_to_frames.py"),
    ("ПЗ3: OCR текста с кадров", "PZ3_ocr_from_frames.py"),
    ("ПЗ4: Whisper-анализ аудио", "PZ4_audio_whisper.py"),
    ("ПЗ5: YOLOv5-детекция объектов", "PZ5_yolo_object_detection.py"),
    ("ПЗ6: ResNet34-классификация", "PZ6_resnet_classification.py"),
    ("ПЗ7: LLM-анализ изображений", "PZ7_llm_ollama_analysis.py"),
    ("ПЗ8: финальный JSON", "PZ8_postprocessing.py"),
]


def build_ytdlp_format(quality):
    if quality == "720":
        return "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
    if quality == "480":
        return "bestvideo[height<=480]+bestaudio/best[height<=480]/best"
    if quality == "360":
        return "bestvideo[height<=360]+bestaudio/best[height<=360]/best"
    return "bestvideo+bestaudio/best"


def download_video_with_ytdlp(url, output_path, quality, log_callback):
    try:
        import yt_dlp
    except ImportError:
        log_callback("yt-dlp не установлен. Выполни команду:\npython -m pip install yt-dlp\n")
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
        log_callback(f"Скачивание видео...\nСсылка: {url}\nКачество: {quality}\n")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if output_path.exists():
            log_callback(f"Видео скачано:\n{output_path}\n")
            return True

        log_callback(f"Файл после скачивания не найден:\n{output_path}\n")
        return False

    except Exception as error:
        log_callback(f"Ошибка скачивания видео:\n{error}\n")
        return False


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ видеоконтента")
        self.root.geometry("950x680")

        self.is_running = False

        self.source_mode = tk.StringVar(value="local")
        self.local_video_path = tk.StringVar()
        self.video_url = tk.StringVar()
        self.download_quality = tk.StringVar(value="best")

        self.extraction_fps = tk.StringVar(value="1")
        self.whisper_model = tk.StringVar(value="base")
        self.yolo_model = tk.StringVar(value="yolov5l")
        self.yolo_confidence = tk.StringVar(value="0.25")
        self.resnet_top_k = tk.StringVar(value="5")
        self.llm_model = tk.StringVar(value="moondream")
        self.llm_images_count = tk.StringVar(value="5")

        self.create_ui()

    def create_ui(self):
        title = tk.Label(
            self.root,
            text="Система анализа видеоконтента",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=8)

        source_frame = tk.LabelFrame(self.root, text="1. Видео", padx=10, pady=10)
        source_frame.pack(fill="x", padx=12, pady=5)

        tk.Radiobutton(
            source_frame,
            text="Файл с компьютера",
            variable=self.source_mode,
            value="local",
            command=self.update_source_state
        ).grid(row=0, column=0, sticky="w", pady=3)

        tk.Radiobutton(
            source_frame,
            text="Ссылка / Rutube",
            variable=self.source_mode,
            value="url",
            command=self.update_source_state
        ).grid(row=1, column=0, sticky="w", pady=3)

        self.local_entry = tk.Entry(
            source_frame,
            textvariable=self.local_video_path,
            width=75
        )
        self.local_entry.grid(row=0, column=1, padx=8, sticky="we")

        self.choose_button = tk.Button(
            source_frame,
            text="Выбрать",
            command=self.choose_file
        )
        self.choose_button.grid(row=0, column=2, padx=5)

        self.url_frame = tk.Frame(source_frame)
        self.url_frame.grid(row=1, column=1, padx=8, sticky="we")

        self.url_entry = tk.Entry(
            self.url_frame,
            textvariable=self.video_url,
            width=65
        )
        self.url_entry.pack(side="left", fill="x", expand=True)

        # Вставка ссылки через Ctrl+V и через правую кнопку мыши
        self.url_entry.bind("<Control-v>", self.paste_url_event)
        self.url_entry.bind("<Control-V>", self.paste_url_event)
        self.url_entry.bind("<Button-3>", self.show_url_context_menu)

        self.paste_url_button = tk.Button(
            self.url_frame,
            text="Вставить ссылку",
            command=self.paste_url_from_clipboard
        )
        self.paste_url_button.pack(side="left", padx=5)

        self.quality_box = ttk.Combobox(
            source_frame,
            textvariable=self.download_quality,
            values=["best", "720", "480", "360"],
            width=10,
            state="readonly"
        )
        self.quality_box.grid(row=1, column=2, padx=5)

        source_frame.columnconfigure(1, weight=1)

        settings_frame = tk.LabelFrame(self.root, text="2. Настройки анализа", padx=10, pady=10)
        settings_frame.pack(fill="x", padx=12, pady=5)

        self.add_setting(settings_frame, 0, "Кадров в секунду для ПЗ2:", self.extraction_fps, ["1", "2", "3", "5"])
        self.add_setting(settings_frame, 1, "Whisper:", self.whisper_model, ["tiny", "base", "small", "medium"])
        self.add_setting(settings_frame, 2, "YOLOv5:", self.yolo_model, ["yolov5s", "yolov5m", "yolov5l"])
        self.add_setting(settings_frame, 3, "YOLO confidence:", self.yolo_confidence, ["0.25", "0.30", "0.50"])
        self.add_setting(settings_frame, 4, "ResNet top-k:", self.resnet_top_k, ["3", "5", "7", "10"])
        self.add_setting(settings_frame, 5, "LLM модель:", self.llm_model, ["moondream", "llava:7b", "qwen2.5vl"])
        self.add_setting(settings_frame, 6, "Картинок для LLM:", self.llm_images_count, ["3", "5", "10"])

        buttons_frame = tk.Frame(self.root)
        buttons_frame.pack(fill="x", padx=12, pady=8)

        tk.Button(
            buttons_frame,
            text="1. Подготовить запуск",
            width=25,
            command=self.prepare_thread
        ).grid(row=0, column=0, padx=5, pady=3)

        tk.Button(
            buttons_frame,
            text="2. Запустить полный анализ",
            width=25,
            command=self.run_all_thread
        ).grid(row=0, column=1, padx=5, pady=3)

        tk.Button(
            buttons_frame,
            text="Пересобрать финальный JSON",
            width=25,
            command=self.run_pz8_thread
        ).grid(row=0, column=2, padx=5, pady=3)

        tk.Button(
            buttons_frame,
            text="Открыть результат",
            width=20,
            command=self.open_result_folder
        ).grid(row=0, column=3, padx=5, pady=3)

        log_frame = tk.LabelFrame(self.root, text="Лог выполнения", padx=8, pady=8)
        log_frame.pack(fill="both", expand=True, padx=12, pady=5)

        self.log_area = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.log_area.pack(fill="both", expand=True)

        self.status = tk.Label(
            self.root,
            text="Готово",
            anchor="w"
        )
        self.status.pack(fill="x", padx=12, pady=4)

        self.update_source_state()

    def add_setting(self, parent, row, label_text, variable, values):
        tk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=3)

        box = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            width=18
        )
        box.grid(row=row, column=1, sticky="w", pady=3)

    def update_source_state(self):
        if self.source_mode.get() == "local":
            self.local_entry.config(state="normal")
            self.choose_button.config(state="normal")
            self.url_entry.config(state="disabled")
            self.paste_url_button.config(state="disabled")
            self.quality_box.config(state="disabled")
        else:
            self.local_entry.config(state="disabled")
            self.choose_button.config(state="disabled")
            self.url_entry.config(state="normal")
            self.paste_url_button.config(state="normal")
            self.quality_box.config(state="readonly")
            self.url_entry.focus_set()

    def choose_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.local_video_path.set(file_path)

    def paste_url_from_clipboard(self):
        try:
            text = self.root.clipboard_get().strip()

            if not text:
                messagebox.showwarning("Буфер обмена", "Буфер обмена пуст.")
                return

            self.source_mode.set("url")
            self.video_url.set(text)
            self.update_source_state()
            self.url_entry.focus_set()
            self.url_entry.icursor(tk.END)
            self.log(f"Ссылка вставлена:\n{text}\n")

        except Exception as error:
            messagebox.showerror("Ошибка", f"Не удалось вставить ссылку:\n{error}")

    def paste_url_event(self, event=None):
        try:
            text = self.root.clipboard_get().strip()

            if text:
                self.source_mode.set("url")
                self.video_url.set(text)
                self.update_source_state()
                self.url_entry.focus_set()
                self.url_entry.icursor(tk.END)
                self.log(f"Ссылка вставлена через Ctrl+V:\n{text}\n")

            return "break"

        except Exception as error:
            messagebox.showerror("Ошибка", f"Не удалось вставить ссылку:\n{error}")
            return "break"

    def show_url_context_menu(self, event):
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Вставить", command=self.paste_url_from_clipboard)
        menu.tk_popup(event.x_root, event.y_root)

    def log(self, text):
        self.log_area.insert(tk.END, text)
        self.log_area.see(tk.END)
        self.root.update_idletasks()

    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def prepare_thread(self):
        threading.Thread(target=self.prepare_run, daemon=True).start()

    def prepare_run(self):
        if self.is_running:
            messagebox.showwarning("Выполнение", "Сейчас уже идёт выполнение.")
            return

        self.is_running = True
        self.set_status("Подготовка запуска...")

        try:
            if self.source_mode.get() == "local":
                video_path = self.local_video_path.get().strip().strip('"')

                if not video_path:
                    messagebox.showerror("Ошибка", "Выберите видеофайл.")
                    return

                self.log("Подготовка запуска с локальным видео...\n")
                config = prepare_new_run_from_local_video(video_path)
                self.log(f"Видео скопировано:\n{config['video_path']}\n")

            else:
                url = self.video_url.get().strip()

                if not url:
                    messagebox.showerror("Ошибка", "Введите ссылку на видео.")
                    return

                quality = self.download_quality.get()

                self.log("Подготовка запуска со скачиванием видео...\n")
                config = prepare_empty_run_for_download(url, quality)

                success = download_video_with_ytdlp(
                    url=url,
                    output_path=config["video_path"],
                    quality=quality,
                    log_callback=self.log
                )

                if not success:
                    messagebox.showerror("Ошибка", "Не удалось скачать видео.")
                    return

                config["download_status"] = "success"
                save_run_config(config)

            self.log(f"Папка запуска:\n{CURRENT_RUN_DIR}\n")
            self.set_status("Запуск подготовлен")
            messagebox.showinfo("Готово", "Запуск подготовлен. Теперь можно запускать полный анализ.")

        except Exception as error:
            self.log(f"Ошибка подготовки:\n{error}\n")
            messagebox.showerror("Ошибка", str(error))

        finally:
            self.is_running = False

    def get_script_input(self, script_name):
        if script_name == "PZ2_video_to_frames.py":
            return f"{self.extraction_fps.get()}\n"

        if script_name == "PZ4_audio_whisper.py":
            model = self.whisper_model.get()

            model_map = {
                "tiny": "1",
                "base": "2",
                "small": "3",
                "medium": "4"
            }

            return f"{model_map.get(model, '2')}\n"

        if script_name == "PZ5_yolo_object_detection.py":
            yolo = self.yolo_model.get()

            yolo_map = {
                "yolov5s": "1",
                "yolov5m": "2",
                "yolov5l": "3"
            }

            return f"{yolo_map.get(yolo, '3')}\n{self.yolo_confidence.get()}\n"

        if script_name == "PZ6_resnet_classification.py":
            return f"{self.resnet_top_k.get()}\n"

        if script_name == "PZ7_llm_ollama_analysis.py":
            llm = self.llm_model.get()

            llm_map = {
                "moondream": "1",
                "llava:7b": "2",
                "qwen2.5vl": "3"
            }

            return f"{llm_map.get(llm, '1')}\n{self.llm_images_count.get()}\n"

        return "\n"

    def run_script(self, script_name):
        script_path = BASE_DIR / script_name

        if not script_path.exists():
            self.log(f"Файл не найден: {script_path}\n")
            return False

        self.log("\n" + "=" * 80 + "\n")
        self.log(f"Запуск: {script_name}\n")
        self.log("=" * 80 + "\n")

        self.set_status(f"Выполняется: {script_name}")

        input_text = self.get_script_input(script_name)

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            if process.stdin:
                process.stdin.write(input_text)
                process.stdin.flush()
                process.stdin.close()

            if process.stdout:
                for line in process.stdout:
                    self.log(line)
                    self.root.update_idletasks()

            process.wait()

            if process.returncode == 0:
                self.log(f"\nЭтап завершён успешно: {script_name}\n")
                self.set_status(f"Завершено: {script_name}")
                return True

            self.log(f"\nЭтап завершился с ошибкой: {script_name}\n")
            self.log(f"Код ошибки: {process.returncode}\n")
            self.set_status(f"Ошибка: {script_name}")
            return False

        except Exception as error:
            self.log(f"Ошибка запуска {script_name}:\n{error}\n")
            self.set_status(f"Ошибка запуска: {script_name}")
            return False

    def run_all_thread(self):
        threading.Thread(target=self.run_all, daemon=True).start()

    def run_all(self):
        if self.is_running:
            messagebox.showwarning("Выполнение", "Сейчас уже идёт выполнение.")
            return

        self.is_running = True
        self.set_status("Выполняется полный анализ...")

        try:
            try:
                load_run_config()
            except Exception:
                messagebox.showerror("Ошибка", "Сначала нажмите «Подготовить запуск».")
                return

            completed = 0
            failed = 0

            for title, script in PIPELINE_STEPS:
                self.set_status(f"Выполняется: {title}")
                self.log("\n\n" + "=" * 80 + "\n")
                self.log(f"{title}\n")
                self.log("=" * 80 + "\n")

                success = self.run_script(script)

                if success:
                    completed += 1
                else:
                    failed += 1
                    break

            self.log("\n" + "=" * 80 + "\n")
            self.log("ПАЙПЛАЙН ЗАВЕРШЁН\n")
            self.log(f"Успешно выполнено этапов: {completed}\n")
            self.log(f"Этапов с ошибками: {failed}\n")

            self.show_result_paths()
            self.set_status("Анализ завершён")

        finally:
            self.is_running = False

    def run_pz8_thread(self):
        threading.Thread(target=self.run_pz8, daemon=True).start()

    def run_pz8(self):
        if self.is_running:
            messagebox.showwarning("Выполнение", "Сейчас уже идёт выполнение.")
            return

        self.is_running = True
        self.set_status("Пересборка финального JSON...")

        try:
            self.run_script("PZ8_postprocessing.py")
            self.show_result_paths()
            self.set_status("Финальный JSON пересобран")
        finally:
            self.is_running = False

    def show_result_paths(self):
        try:
            config = load_run_config()
        except Exception as error:
            self.log(f"Не удалось загрузить run_config.json:\n{error}\n")
            return

        self.log("\nРезультаты:\n")
        self.log(f"current_run: {CURRENT_RUN_DIR}\n")
        self.log(f"final JSON: {config.get('final_json_path')}\n")
        self.log(f"final Excel: {config.get('final_preview_excel_path')}\n")
        self.log(f"detections_count: {config.get('final_detections_count')}\n")
        self.log(f"risk_level: {config.get('final_risk_level')}\n")
        self.log(f"is_dangerous: {config.get('final_is_dangerous')}\n")
        self.log(f"decision: {config.get('final_decision')}\n")

    def open_result_folder(self):
        if not CURRENT_RUN_DIR.exists():
            messagebox.showerror("Ошибка", "Папка current_run ещё не создана.")
            return

        subprocess.Popen(f'explorer "{CURRENT_RUN_DIR}"')


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()