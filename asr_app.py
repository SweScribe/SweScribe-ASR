import sys
import math
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import onnx_asr
from scipy.signal import resample_poly

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
MODEL_NAME = "KlangAI/pianissimo-sv-onnx"
MODEL_SAMPLE_RATE = 16000
LIVE_CHUNK_SECONDS = 4
LIVE_OVERLAP_SECONDS = 0.75
FILE_SEGMENT_SECONDS = 20  # long files are transcribed in pieces of this length


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_model_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert to mono float32 at 16 kHz."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != MODEL_SAMPLE_RATE:
        g = math.gcd(MODEL_SAMPLE_RATE, int(sample_rate))
        audio = resample_poly(
            audio, MODEL_SAMPLE_RATE // g, int(sample_rate) // g
        ).astype(np.float32)
    return audio


def recognize(model, audio_16k: np.ndarray) -> str:
    return str(model.recognize(audio_16k, sample_rate=MODEL_SAMPLE_RATE)).strip()


# ----------------------------------------------------------------------------
# Worker threads
# ----------------------------------------------------------------------------
class ModelLoader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            model = onnx_asr.load_model(MODEL_NAME, quantization="int8")
            self.loaded.emit(model)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class LiveWorker(QThread):
    segment = Signal(float, float, str)
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, model, device_index, parent=None):
        super().__init__(parent)
        self.model = model
        self.device_index = device_index
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            info = sd.query_devices(self.device_index, "input")
            input_rate = int(info["default_samplerate"])
            audio_q: queue.Queue = queue.Queue()

            def callback(indata, frames, time_info, status):
                audio_q.put(indata.copy())

            chunk_size = int(input_rate * LIVE_CHUNK_SECONDS)
            overlap_size = int(input_rate * LIVE_OVERLAP_SECONDS)
            buffer = np.empty(0, dtype=np.float32)
            consumed = 0  # samples (at input rate) already moved past

            with sd.InputStream(
                device=self.device_index,
                samplerate=input_rate,
                channels=1,
                dtype="float32",
                callback=callback,
                blocksize=0,
            ):
                while not self._stop.is_set():
                    try:
                        data = audio_q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    self.level.emit(float(np.sqrt(np.mean(data**2))))
                    buffer = np.concatenate((buffer, data))

                    if len(buffer) < chunk_size:
                        continue

                    chunk = buffer[:chunk_size]
                    start = consumed / input_rate
                    end = start + LIVE_CHUNK_SECONDS
                    step = chunk_size - overlap_size
                    buffer = buffer[step:]
                    consumed += step

                    text = recognize(self.model, to_model_rate(chunk, input_rate))
                    if text:
                        self.segment.emit(start, end, text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class FileWorker(QThread):
    segment = Signal(float, float, str)
    progress = Signal(int)
    failed = Signal(str)

    def __init__(self, model, path, parent=None):
        super().__init__(parent)
        self.model = model
        self.path = path
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            audio, sr = sf.read(self.path, dtype="float32")
            audio = to_model_rate(audio, sr)
            seg_len = FILE_SEGMENT_SECONDS * MODEL_SAMPLE_RATE
            total = len(audio)
            pos = 0
            while pos < total and not self._stop.is_set():
                piece = audio[pos : pos + seg_len]
                start = pos / MODEL_SAMPLE_RATE
                end = (pos + len(piece)) / MODEL_SAMPLE_RATE
                text = recognize(self.model, piece)
                if text:
                    self.segment.emit(start, end, text)
                pos += seg_len
                self.progress.emit(min(100, int(pos * 100 / total)))
            self.progress.emit(100)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
STYLE = """
* { font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif; font-size: 13px; }
QMainWindow, QWidget#root { background: #0f1117; }
QLabel { color: #c9ceda; }
QLabel#title { font-size: 22px; font-weight: 600; color: #ffffff; }
QLabel#subtitle { color: #7b8294; }
QLabel#pill { background: #1a1e2a; border-radius: 11px; padding: 4px 12px; color: #9aa3b8; }
QLabel#pill[state="ready"] { background: #12301f; color: #5fe08f; }
QLabel#pill[state="busy"] { background: #33290f; color: #f0c05a; }
QLabel#pill[state="live"] { background: #35171c; color: #ff6b7d; }
QLabel#pill[state="error"] { background: #35171c; color: #ff6b7d; }
QFrame#card { background: #161a24; border: 1px solid #232838; border-radius: 14px; }
QPushButton {
    background: #1d2231; color: #dfe3ee; border: 1px solid #2a3047;
    border-radius: 9px; padding: 9px 18px;
}
QPushButton:hover { background: #252b3f; }
QPushButton:disabled { color: #566077; background: #171b27; border-color: #1f2434; }
QPushButton#primary { background: #5b6cff; border: none; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #7280ff; }
QPushButton#primary:disabled { background: #2a3050; color: #6b7390; }
QPushButton#danger { background: #e5484d; border: none; color: white; font-weight: 600; }
QPushButton#danger:hover { background: #ee6065; }
QPushButton#tab { background: transparent; border: none; border-radius: 8px; padding: 8px 22px; color: #8a92a8; }
QPushButton#tab:checked { background: #5b6cff; color: white; font-weight: 600; }
QPushButton#tab:disabled { color: #4a5268; }
QFrame#tabbar { background: #161a24; border: 1px solid #232838; border-radius: 12px; }
QComboBox {
    background: #1d2231; color: #dfe3ee; border: 1px solid #2a3047;
    border-radius: 9px; padding: 8px 12px;
}
QComboBox QAbstractItemView { background: #1d2231; color: #dfe3ee; selection-background-color: #5b6cff; border: 1px solid #2a3047; }
QProgressBar { background: #1d2231; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #5b6cff; border-radius: 4px; }
QProgressBar#level::chunk { background: #3ddc84; }
QTextEdit {
    background: #0f1117; color: #e6e9f2; border: 1px solid #232838;
    border-radius: 10px; padding: 12px; font-size: 14px;
    selection-background-color: #5b6cff;
}
QLabel#drop {
    border: 2px dashed #2f3650; border-radius: 12px; padding: 22px;
    color: #8a92a8; background: #121622;
}
QLabel#drop[hover="true"] { border-color: #5b6cff; color: #c9ceff; }
"""


class DropLabel(QLabel):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__("Drop an audio file here\nWAV · FLAC · OGG")
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)

    def _hover(self, on: bool):
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover(True)

    def dragLeaveEvent(self, e):
        self._hover(False)

    def dropEvent(self, e):
        self._hover(False)
        urls = e.mimeData().urls()
        if urls:
            self.fileDropped.emit(urls[0].toLocalFile())


# ----------------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pianissimo Transcriber")
        self.resize(860, 760)

        self.model = None
        self.segments: list[Segment] = []
        self.live_worker: LiveWorker | None = None
        self.file_worker: FileWorker | None = None
        self.file_path: str | None = None
        self.source_label = ""

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(16)

        # Header
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        t = QLabel("Pianissimo Transcriber")
        t.setObjectName("title")
        s = QLabel("Swedish speech-to-text · live or from file")
        s.setObjectName("subtitle")
        titles.addWidget(t)
        titles.addWidget(s)
        header.addLayout(titles)
        header.addStretch()
        self.pill = QLabel("")
        self.pill.setObjectName("pill")
        header.addWidget(self.pill, alignment=Qt.AlignTop)
        outer.addLayout(header)

        # Mode toggle
        tabbar = QFrame()
        tabbar.setObjectName("tabbar")
        tb = QHBoxLayout(tabbar)
        tb.setContentsMargins(5, 5, 5, 5)
        tb.setSpacing(4)
        self.btn_live = QPushButton("🎙  Live")
        self.btn_file = QPushButton("📁  File")
        self.mode_group = QButtonGroup(self)
        for i, b in enumerate((self.btn_live, self.btn_file)):
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.mode_group.addButton(b, i)
            tb.addWidget(b)
        self.btn_live.setChecked(True)
        tb.addStretch()
        outer.addWidget(tabbar, alignment=Qt.AlignLeft)

        # Control card with stacked pages
        card = QFrame()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 18, 20, 18)
        self.stack = QStackedWidget()
        cl.addWidget(self.stack)
        self.stack.addWidget(self._build_live_page())
        self.stack.addWidget(self._build_file_page())
        outer.addWidget(card)
        self.mode_group.idClicked.connect(self.stack.setCurrentIndex)

        # Transcript
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Transcription will appear here…")
        outer.addWidget(self.text, stretch=1)

        # Footer
        footer = QHBoxLayout()
        self.count_label = QLabel("0 segments")
        self.count_label.setObjectName("subtitle")
        footer.addWidget(self.count_label)
        footer.addStretch()
        self.btn_clear = QPushButton("Clear")
        self.btn_save = QPushButton("Save as .txt")
        self.btn_save.setObjectName("primary")
        for b in (self.btn_clear, self.btn_save):
            b.setCursor(Qt.PointingHandCursor)
            footer.addWidget(b)
        outer.addLayout(footer)
        self.btn_clear.clicked.connect(self.clear_transcript)
        self.btn_save.clicked.connect(self.save_transcript)

        self.setStyleSheet(STYLE)
        self.refresh_devices()
        self.update_controls()

        # Load model in the background
        self.set_status("Loading model…", "busy")
        self.loader = ModelLoader()
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.failed.connect(self.on_model_failed)
        self.loader.start()

    # ---- page builders ----------------------------------------------------
    def _build_live_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        row = QHBoxLayout()
        row.addWidget(QLabel("Microphone"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(320)
        row.addWidget(self.device_combo, stretch=1)
        self.btn_start = QPushButton("Start listening")
        self.btn_start.setObjectName("primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.toggle_live)
        row.addWidget(self.btn_start)
        lay.addLayout(row)

        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("level")
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        lay.addWidget(self.level_bar)

        hint = QLabel(
            f"Audio is transcribed in {LIVE_CHUNK_SECONDS}-second chunks, "
            "so text appears a few seconds after you speak."
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return page

    def _build_file_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        self.drop = DropLabel()
        self.drop.fileDropped.connect(self.set_file)
        lay.addWidget(self.drop)

        row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.clicked.connect(self.browse_file)
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("subtitle")
        self.btn_transcribe = QPushButton("Transcribe file")
        self.btn_transcribe.setObjectName("primary")
        self.btn_transcribe.setCursor(Qt.PointingHandCursor)
        self.btn_transcribe.clicked.connect(self.toggle_file)
        row.addWidget(self.btn_browse)
        row.addWidget(self.file_label, stretch=1)
        row.addWidget(self.btn_transcribe)
        lay.addLayout(row)

        self.file_progress = QProgressBar()
        self.file_progress.setRange(0, 100)
        self.file_progress.setTextVisible(False)
        lay.addWidget(self.file_progress)
        return page

    # ---- status / state ---------------------------------------------------
    def set_status(self, text: str, state: str):
        self.pill.setText(text)
        self.pill.setProperty("state", state)
        self.pill.style().unpolish(self.pill)
        self.pill.style().polish(self.pill)

    def is_busy(self) -> bool:
        return bool(
            (self.live_worker and self.live_worker.isRunning())
            or (self.file_worker and self.file_worker.isRunning())
        )

    def update_controls(self):
        ready = self.model is not None
        live_on = bool(self.live_worker and self.live_worker.isRunning())
        file_on = bool(self.file_worker and self.file_worker.isRunning())

        self.btn_live.setEnabled(not file_on and not live_on)
        self.btn_file.setEnabled(not file_on and not live_on)

        self.btn_start.setEnabled(ready and not file_on)
        self.device_combo.setEnabled(not live_on)
        self.btn_start.setText("Stop" if live_on else "Start listening")
        self.btn_start.setObjectName("danger" if live_on else "primary")

        self.btn_transcribe.setEnabled(ready and not live_on and (self.file_path is not None or file_on))
        self.btn_transcribe.setText("Cancel" if file_on else "Transcribe file")
        self.btn_transcribe.setObjectName("danger" if file_on else "primary")
        self.btn_browse.setEnabled(not file_on)
        self.drop.setAcceptDrops(not file_on)

        for b in (self.btn_start, self.btn_transcribe):
            b.style().unpolish(b)
            b.style().polish(b)

        self.btn_save.setEnabled(bool(self.segments))
        self.btn_clear.setEnabled(bool(self.segments) and not self.is_busy())

    # ---- model ------------------------------------------------------------
    def on_model_loaded(self, model):
        self.model = model
        self.set_status("Ready", "ready")
        self.update_controls()

    def on_model_failed(self, msg: str):
        self.set_status("Model failed to load", "error")
        QMessageBox.critical(self, "Model error", f"Could not load the model:\n\n{msg}")

    # ---- devices ----------------------------------------------------------
    def refresh_devices(self):
        self.device_combo.clear()
        try:
            default_in = sd.default.device[0]
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    self.device_combo.addItem(dev["name"], i)
                    if i == default_in:
                        self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception as exc:  # noqa: BLE001
            self.device_combo.addItem(f"No input devices ({exc})", None)

    # ---- live -------------------------------------------------------------
    def toggle_live(self):
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.stop()
            self.btn_start.setEnabled(False)
            self.set_status("Finishing…", "busy")
            return
        device = self.device_combo.currentData()
        if device is None:
            QMessageBox.warning(self, "No microphone", "No input device available.")
            return
        self.source_label = f"Live microphone ({self.device_combo.currentText()})"
        self.live_worker = LiveWorker(self.model, device)
        self.live_worker.segment.connect(self.add_segment)
        self.live_worker.level.connect(self.on_level)
        self.live_worker.failed.connect(self.on_worker_failed)
        self.live_worker.finished.connect(self.on_worker_finished)
        self.live_worker.start()
        self.set_status("● Listening", "live")
        self.update_controls()

    def on_level(self, rms: float):
        self.level_bar.setValue(min(100, int(rms * 400)))

    # ---- file -------------------------------------------------------------
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an audio file",
            "",
            "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff);;All files (*)",
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str):
        self.file_path = path
        self.file_label.setText(Path(path).name)
        self.drop.setText(f"✔  {Path(path).name}")
        self.file_progress.setValue(0)
        self.update_controls()

    def toggle_file(self):
        if self.file_worker and self.file_worker.isRunning():
            self.file_worker.stop()
            self.btn_transcribe.setEnabled(False)
            self.set_status("Cancelling…", "busy")
            return
        if not self.file_path:
            return
        self.source_label = f"File: {Path(self.file_path).name}"
        self.file_progress.setValue(0)
        self.file_worker = FileWorker(self.model, self.file_path)
        self.file_worker.segment.connect(self.add_segment)
        self.file_worker.progress.connect(self.file_progress.setValue)
        self.file_worker.failed.connect(self.on_worker_failed)
        self.file_worker.finished.connect(self.on_worker_finished)
        self.file_worker.start()
        self.set_status("Transcribing…", "busy")
        self.update_controls()

    # ---- shared worker callbacks -------------------------------------------
    def on_worker_failed(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    def on_worker_finished(self):
        self.level_bar.setValue(0)
        self.set_status("Ready", "ready")
        self.update_controls()

    # ---- transcript -------------------------------------------------------
    def add_segment(self, start: float, end: float, text: str):
        # Live mode: continue timestamps after existing text only within the same session
        self.segments.append(Segment(start, end, text))
        self.text.append(
            f'<span style="color:#7c8cff">[{fmt_time(start)}]</span> {escape(text)}'
        )
        n = len(self.segments)
        self.count_label.setText(f"{n} segment{'s' if n != 1 else ''}")
        self.update_controls()

    def clear_transcript(self):
        self.segments.clear()
        self.text.clear()
        self.count_label.setText("0 segments")
        self.update_controls()

    def save_transcript(self):
        if not self.segments:
            return
        default = f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save transcript", default, "Text files (*.txt)"
        )
        if not path:
            return
        lines = [
            "Transcript",
            f"Source: {self.source_label or 'unknown'}",
            f"Saved: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "-" * 40,
        ]
        for seg in self.segments:
            lines.append(f"[{fmt_time(seg.start)} - {fmt_time(seg.end)}] {seg.text}")
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.set_status("Saved ✓", "ready")

    # ---- shutdown ---------------------------------------------------------
    def closeEvent(self, event):
        for w in (self.live_worker, self.file_worker):
            if w and w.isRunning():
                w.stop()
                w.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
