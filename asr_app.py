"""Pianissimo transcriber: live microphone or audio-file speech-to-text, with an ONNX or NeMo engine.

Run with:  uv run asr_app.py
"""
import importlib.util
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

from PySide6.QtCore import QRectF, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ONNX_MODEL_NAME = "KlangAI/pianissimo-sv-onnx"
NEMO_MODEL_NAME = "KlangAI/pianissimo-sv"
MODEL_SAMPLE_RATE = 16000
LIVE_CHUNK_SECONDS = 4
LIVE_OVERLAP_SECONDS = 0.75
FILE_SEGMENT_SECONDS = 20  # long files are transcribed in pieces of this length

ENGINE_NAMES = {"onnx": "ONNX", "nemo": "NeMo"}
ENGINE_KEYS = list(ENGINE_NAMES)

# Mic sensitivity (noise gate). The slider runs 0..100; higher = picks up quieter sounds.
DEFAULT_SENSITIVITY = 60
GATE_LOUDEST_DB = -20.0  # gate level at sensitivity 0 (only loud sounds get through)
GATE_QUIETEST_DB = -60.0  # gate level at sensitivity 100, which switches the gate off
METER_FLOOR_DB = -60.0  # the level meter shows METER_FLOOR_DB (empty) .. 0 dBFS (full)
GATE_FRAME_SECONDS = 0.05
GATE_MIN_SPEECH_SECONDS = 0.3  # a chunk needs this much sound above the gate to be transcribed


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class EngineStatus:
    level: str  # "ok" | "warn" (works, with a caveat) | "error" (cannot run)
    summary: str
    details: str = ""
    fix: str = ""  # "" | "install" (torch + NeMo) | "reinstall_torch" (swap CPU build for GPU build)


def repolish(widget):
    """Re-apply the stylesheet after changing a widget's objectName or a dynamic property."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_model_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert to mono float32 at 16 kHz."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sample_rate != MODEL_SAMPLE_RATE:
        g = math.gcd(MODEL_SAMPLE_RATE, int(sample_rate))
        audio = resample_poly(audio, MODEL_SAMPLE_RATE // g, int(sample_rate) // g).astype(np.float32)
    return audio


def gate_db(sensitivity: float) -> float:
    """Noise-gate level in dBFS for a 0..100 sensitivity: sounds quieter than this are ignored."""
    return GATE_LOUDEST_DB + (GATE_QUIETEST_DB - GATE_LOUDEST_DB) * sensitivity / 100


def db_to_meter(db: float) -> float:
    """Position on the level meter, 0..1."""
    return min(1.0, max(0.0, (db - METER_FLOOR_DB) / -METER_FLOOR_DB))


def rms_to_meter(rms: float) -> float:
    return db_to_meter(20 * math.log10(max(rms, 1e-6)))


def has_speech(audio: np.ndarray, rate: int, threshold_db: float) -> bool:
    """True if enough of `audio` is louder than the noise gate to be worth transcribing.
    Noise-only chunks are skipped, because speech models invent words for background noise."""
    if threshold_db <= GATE_QUIETEST_DB:  # slider at maximum: gate is off
        return True
    frame = int(rate * GATE_FRAME_SECONDS)
    n = len(audio) // frame
    if n == 0:
        return False
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    loud_frames = np.count_nonzero(rms >= 10 ** (threshold_db / 20))
    return loud_frames * GATE_FRAME_SECONDS >= GATE_MIN_SPEECH_SECONDS


def remove_overlap(previous: str, current: str) -> str:
    """Drop words at the start of `current` that repeat the end of `previous`
    (live chunks overlap, so boundary words get transcribed twice)."""
    prev = previous.lower().split()
    words = current.split()
    for n in range(min(len(prev), len(words), 12), 0, -1):  # longest match first
        if prev[-n:] == [w.lower() for w in words[:n]]:
            return " ".join(words[n:])
    return current.strip()


def hyp_text(result) -> str:
    """NeMo returns different shapes depending on model type and version."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        result = result[0]
    return str(getattr(result, "text", result)).strip()


class StoppableThread(QThread):
    """QThread with a stop flag that run() loops are expected to poll."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()


# ----------------------------------------------------------------------------
# Engines: availability checks
# ----------------------------------------------------------------------------
def check_onnx() -> EngineStatus:
    for module, package in (("onnx_asr", "onnx-asr"), ("onnxruntime", "onnxruntime")):
        if importlib.util.find_spec(module) is None:
            return EngineStatus("error", f"The {package} package is not installed.", "Fix: run `uv sync`.")
    try:
        import onnx_asr  # noqa: F401
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        return EngineStatus(
            "error",
            f"ONNX failed to import: {type(exc).__name__}: {exc}",
            "Try reinstalling: uv pip install --reinstall onnx-asr onnxruntime",
        )
    return EngineStatus("ok", "Ready · int8 model via ONNX Runtime.", "ONNX Runtime providers: " + ", ".join(providers))


def check_nemo() -> EngineStatus:
    install_hint = "Click the install button to set it up automatically (PyTorch is matched to your GPU driver)."
    if importlib.util.find_spec("torch") is None:
        return EngineStatus("error", "PyTorch is not installed.", install_hint, fix="install")
    if importlib.util.find_spec("nemo") is None:
        return EngineStatus("error", "NVIDIA NeMo is not installed.", install_hint, fix="install")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return EngineStatus(
            "error",
            f"PyTorch failed to import: {type(exc).__name__}: {exc}",
            "A broken PyTorch/CUDA install is the usual cause. Reinstall PyTorch from pytorch.org.",
        )
    try:
        from nemo.collections.asr.models import ASRModel  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return EngineStatus(
            "error",
            f"NeMo is installed but failed to import: {type(exc).__name__}: {exc}",
            "This is usually a version conflict between torch, numpy and nemo_toolkit.",
        )

    if torch.cuda.is_available():
        return EngineStatus(
            "ok",
            f"Ready · GPU: {torch.cuda.get_device_name(0)}",
            f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}",
        )

    # No usable GPU: work out why, then fall back to CPU.
    cuda_build = torch.version.cuda
    has_driver = shutil.which("nvidia-smi") is not None
    fix = ""
    if cuda_build is None and has_driver:
        reason = (
            "An NVIDIA driver was found, but your PyTorch is a CPU-only build. "
            "Click the button to switch to the GPU build that matches your driver."
        )
        fix = "reinstall_torch"
    elif cuda_build is None:
        reason = (
            "PyTorch is a CPU-only build and no NVIDIA driver was found "
            "(nvidia-smi is missing), so this machine may have no NVIDIA GPU."
        )
    elif not has_driver:
        reason = (
            "No NVIDIA driver was found (nvidia-smi is missing). Install the NVIDIA driver, "
            "or this machine has no NVIDIA GPU."
        )
    else:
        reason = (
            f"PyTorch is built for CUDA {cuda_build} and nvidia-smi exists, but CUDA is not usable. "
            "The NVIDIA driver is probably too old or not loaded; update it and restart."
        )
    return EngineStatus("warn", "Works, but on CPU only (slower).", reason, fix=fix)


class EngineChecker(QThread):
    result = Signal(str, object)

    def run(self):
        for key, check in (("onnx", check_onnx), ("nemo", check_nemo)):
            try:
                status = check()
            except Exception as exc:  # noqa: BLE001
                status = EngineStatus("error", f"Check failed: {type(exc).__name__}: {exc}")
            self.result.emit(key, status)


# ----------------------------------------------------------------------------
# Engines: one-click install of PyTorch + NeMo (via uv)
# ----------------------------------------------------------------------------
def find_uv() -> str | None:
    # `uv run` exports the path of the uv binary in $UV; otherwise look on PATH.
    return os.environ.get("UV") or shutil.which("uv")


def install_commands(fix: str, uv: str) -> list[list[str]]:
    """uv commands that install into the environment this app is running in.

    --torch-backend=auto makes uv look at the NVIDIA driver and pick the matching
    PyTorch build (CUDA version, or CPU if there is no usable GPU). It only exists
    in the `uv pip` interface, which is why we don't use `uv sync` for this.
    """
    base = [uv, "pip", "install", "--python", sys.executable, "--torch-backend=auto"]
    if fix == "reinstall_torch":
        family = [n for n in ("torch", "torchaudio", "torchvision") if n == "torch" or importlib.util.find_spec(n)]
        reinstall = [arg for n in family for arg in ("--reinstall-package", n)]
        return [base + reinstall + family]
    # torch first, so NeMo's own dependency on it is already satisfied by the right build
    return [base + ["torch"], base + ["nemo_toolkit[asr]"]]


def run_commands(cmds, on_line, should_stop) -> int:
    """Run commands in order, streaming output lines. 0 = ok, -1 = cancelled, else exit code."""
    for cmd in cmds:
        on_line("$ " + " ".join(cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            on_line(line.rstrip())
            if should_stop():
                proc.terminate()
                proc.wait()
                return -1
        code = proc.wait()
        if code != 0:
            return code
    return 0


class InstallThread(StoppableThread):
    line = Signal(str)
    done = Signal(int)

    def __init__(self, cmds, parent=None):
        super().__init__(parent)
        self.cmds = cmds

    def run(self):
        try:
            code = run_commands(self.cmds, self.line.emit, self._stop.is_set)
        except Exception as exc:  # noqa: BLE001
            self.line.emit(f"Error: {exc}")
            code = 1
        self.done.emit(code)


class InstallDialog(QDialog):
    def __init__(self, parent, cmds):
        super().__init__(parent)
        self.setWindowTitle("Installing packages")
        self.resize(720, 420)
        self.success = False

        lay = QVBoxLayout(self)
        self.status = QLabel(
            "Installing… this can take several minutes and download a few GB. "
            "You can keep this window open while it works."
        )
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)  # busy indicator
        lay.addWidget(self.bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: monospace; font-size: 12px;")
        lay.addWidget(self.log, stretch=1)
        self.btn = QPushButton("Cancel")
        self.btn.clicked.connect(self.on_button)
        lay.addWidget(self.btn, alignment=Qt.AlignRight)

        self.worker = InstallThread(cmds, self)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def on_button(self):
        if self.worker.isRunning():
            self.btn.setEnabled(False)
            self.status.setText("Cancelling…")
            self.worker.stop()
        else:
            self.accept()

    def on_done(self, code: int):
        self.bar.setRange(0, 1)
        self.bar.setValue(1 if code == 0 else 0)
        self.btn.setEnabled(True)
        self.btn.setText("Close")
        self.success = code == 0
        if code == 0:
            self.status.setText("Done. The app needs a restart to start using the new packages.")
        elif code == -1:
            self.status.setText("Cancelled. Nothing is guaranteed to be fully installed; you can try again.")
        else:
            self.status.setText(f"Install failed (exit code {code}). See the log below for details.")

    def reject(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        super().reject()


# ----------------------------------------------------------------------------
# Engines: recognizers
# ----------------------------------------------------------------------------
class OnnxRecognizer:
    label = "ONNX (int8)"

    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_16k: np.ndarray) -> str:
        return str(self.model.recognize(audio_16k, sample_rate=MODEL_SAMPLE_RATE)).strip()


class NemoRecognizer:
    def __init__(self, model, torch_module, device):
        self.model = model
        self.torch = torch_module
        self.label = f"NeMo ({'GPU' if device.type == 'cuda' else 'CPU'})"

    def transcribe(self, audio_16k: np.ndarray) -> str:
        if len(audio_16k) == 0:
            return ""
        audio = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)
        with self.torch.inference_mode():
            result = self.model.transcribe(audio=[audio], batch_size=1, return_hypotheses=True)
        return hyp_text(result)


def load_onnx() -> OnnxRecognizer:
    import onnx_asr

    return OnnxRecognizer(onnx_asr.load_model(ONNX_MODEL_NAME, quantization="int8"))


def load_nemo() -> NemoRecognizer:
    import torch
    from nemo.collections.asr.models import ASRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASRModel.from_pretrained(model_name=NEMO_MODEL_NAME, map_location=device).eval()
    return NemoRecognizer(model, torch, device)


# ----------------------------------------------------------------------------
# Worker threads
# ----------------------------------------------------------------------------
class ModelLoader(QThread):
    loaded = Signal(str, object)
    failed = Signal(str, str)

    def __init__(self, key: str, parent=None):
        super().__init__(parent)
        self.key = key

    def run(self):
        try:
            recognizer = load_nemo() if self.key == "nemo" else load_onnx()
            self.loaded.emit(self.key, recognizer)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self.key, f"{type(exc).__name__}: {exc}")


class LiveWorker(StoppableThread):
    segment = Signal(float, float, str)  # start, end, text
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, recognizer, device_index, sensitivity=DEFAULT_SENSITIVITY, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.device_index = device_index
        self.sensitivity = sensitivity  # 0..100; the UI may change it while we run

    def run(self):
        try:
            rate = int(sd.query_devices(self.device_index, "input")["default_samplerate"])
            audio_q: queue.Queue = queue.Queue()

            def callback(indata, frames, time_info, status):
                audio_q.put(indata[:, 0].copy())

            chunk = int(rate * LIVE_CHUNK_SECONDS)
            step = chunk - int(rate * LIVE_OVERLAP_SECONDS)
            parts: list[np.ndarray] = []  # incoming blocks, joined only once a full chunk is available
            have = 0
            consumed = 0  # samples (at the input rate) already moved past
            previous_text = ""

            with sd.InputStream(
                device=self.device_index,
                samplerate=rate,
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
                    self.level.emit(float(np.sqrt(np.mean(data**2))))
                    parts.append(data)
                    have += len(data)
                    if have < chunk:
                        continue

                    buffer = np.concatenate(parts)
                    start = consumed / rate
                    consumed += step
                    parts, have = [buffer[step:]], len(buffer) - step  # keep the overlap for the next chunk

                    audio = buffer[:chunk]
                    if not has_speech(audio, rate, gate_db(self.sensitivity)):
                        previous_text = ""  # a gap: don't de-duplicate against text from before it
                        continue

                    text = self.recognizer.transcribe(to_model_rate(audio, rate))
                    if not text:
                        continue
                    new_text = remove_overlap(previous_text, text)
                    previous_text = text
                    if new_text:
                        self.segment.emit(start, start + LIVE_CHUNK_SECONDS, new_text)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class FileWorker(StoppableThread):
    segment = Signal(float, float, str)
    progress = Signal(int)
    failed = Signal(str)

    def __init__(self, recognizer, path, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.path = path

    def run(self):
        try:
            audio, sr = sf.read(self.path, dtype="float32")
            audio = to_model_rate(audio, sr)
            size = FILE_SEGMENT_SECONDS * MODEL_SAMPLE_RATE
            total = len(audio)
            for pos in range(0, total, size):
                if self._stop.is_set():
                    return
                piece = audio[pos : pos + size]
                end = pos + len(piece)
                text = self.recognizer.transcribe(piece)
                if text:
                    self.segment.emit(pos / MODEL_SAMPLE_RATE, end / MODEL_SAMPLE_RATE, text)
                self.progress.emit(end * 100 // total)
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
QLabel#engineinfo { color: #9aa3b8; font-size: 12px; }
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
QPushButton#tab:checked:disabled { background: #2a3050; color: #8a92a8; }
QFrame#tabbar { background: #161a24; border: 1px solid #232838; border-radius: 12px; }
QComboBox {
    background: #1d2231; color: #dfe3ee; border: 1px solid #2a3047;
    border-radius: 9px; padding: 8px 12px;
}
QComboBox QAbstractItemView { background: #1d2231; color: #dfe3ee; selection-background-color: #5b6cff; border: 1px solid #2a3047; }
QProgressBar { background: #1d2231; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #5b6cff; border-radius: 4px; }
QSlider::groove:horizontal { height: 6px; background: #1d2231; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #5b6cff; border-radius: 3px; }
QSlider::handle:horizontal { background: #dfe3ee; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: #ffffff; }
QTextEdit, QPlainTextEdit {
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

LEVEL_ICON = {"ok": ("✔", "#5fe08f"), "warn": ("⚠", "#f0c05a"), "error": ("✖", "#ff6b7d")}


class LevelMeter(QWidget):
    """Microphone level bar with an orange marker at the noise-gate level.
    The bar is grey while the sound is below the marker (ignored) and green above it."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(16)
        self.level = 0.0  # 0..1
        self.gate = 0.0  # 0..1

    def set_level(self, level: float):
        # rise instantly, fall smoothly, so the bar is readable instead of flickering
        self.level = level if level > self.level else self.level * 0.85 + level * 0.15
        self.update()

    def set_gate(self, gate: float):
        self.gate = gate
        self.update()

    def reset(self):
        self.level = 0.0
        self.update()

    def paintEvent(self, _event):
        w, h = self.width(), self.height()
        bar_h = 8
        top = (h - bar_h) / 2
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#1d2231"))
        p.drawRoundedRect(QRectF(0, top, w, bar_h), 4, 4)
        if self.level > 0.01:
            p.setBrush(QColor("#3ddc84" if self.level >= self.gate else "#566077"))
            p.drawRoundedRect(QRectF(0, top, max(float(bar_h), w * self.level), bar_h), 4, 4)
        p.setBrush(QColor("#ffb86b"))
        p.drawRoundedRect(QRectF(w * self.gate - 1.5, 0, 3, h), 1.5, 1.5)


class DropLabel(QLabel):
    fileDropped = Signal(str)

    def __init__(self):
        super().__init__("Drop an audio file here\nWAV · FLAC · OGG")
        self.setObjectName("drop")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)

    def _hover(self, on: bool):
        self.setProperty("hover", on)
        repolish(self)

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

        self.recognizers: dict[str, object] = {}  # loaded engines, kept so switching back is instant
        self.statuses: dict[str, EngineStatus] = {}
        self.engine_key = "onnx"
        self.nemo_fix = ""
        self.loading = False
        self.loader: ModelLoader | None = None
        self.segments: list[Segment] = []
        self.engines_used: list[str] = []
        self.live_worker: LiveWorker | None = None
        self.file_worker: FileWorker | None = None
        self.file_path: str | None = None
        self.source_label = ""
        self.settings = QSettings("SweScribe", "Transcriber")  # remembers the sensitivity between runs

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(14)

        # Header
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        title = QLabel("Pianissimo Transcriber")
        title.setObjectName("title")
        subtitle = QLabel("Swedish speech-to-text · live or from file")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch()
        self.pill = QLabel("")
        self.pill.setObjectName("pill")
        header.addWidget(self.pill, alignment=Qt.AlignTop)
        outer.addLayout(header)

        # Mode toggle (left) + engine toggle (right)
        toggles = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        mode_bar, (self.btn_live, self.btn_file) = self._make_tab_bar(["🎙  Live", "📁  File"], self.mode_group)
        self.btn_live.setChecked(True)
        toggles.addWidget(mode_bar)
        toggles.addStretch()
        engine_caption = QLabel("Engine")
        engine_caption.setObjectName("subtitle")
        toggles.addWidget(engine_caption)
        self.engine_group = QButtonGroup(self)
        engine_bar, engine_btns = self._make_tab_bar([ENGINE_NAMES[k] for k in ENGINE_KEYS], self.engine_group)
        self.engine_buttons = dict(zip(ENGINE_KEYS, engine_btns))
        self.engine_buttons[self.engine_key].setChecked(True)
        toggles.addWidget(engine_bar)
        outer.addLayout(toggles)

        # Engine availability details + install button
        info_row = QHBoxLayout()
        self.engine_info = QLabel()
        self.engine_info.setObjectName("engineinfo")
        self.engine_info.setTextFormat(Qt.RichText)
        self.engine_info.setWordWrap(True)
        info_row.addWidget(self.engine_info, stretch=1)
        self.btn_fix = QPushButton("")
        self.btn_fix.setCursor(Qt.PointingHandCursor)
        self.btn_fix.setVisible(False)
        self.btn_fix.clicked.connect(self.start_install)
        info_row.addWidget(self.btn_fix, alignment=Qt.AlignTop)
        outer.addLayout(info_row)

        # Control card with stacked pages
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        self.stack = QStackedWidget()
        card_layout.addWidget(self.stack)
        self.stack.addWidget(self._build_live_page())
        self.stack.addWidget(self._build_file_page())
        outer.addWidget(card)
        self.mode_group.idClicked.connect(self.stack.setCurrentIndex)
        self.engine_group.idClicked.connect(self.on_engine_clicked)

        # Transcript
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Transcription will appear here…")
        outer.addWidget(self.text, stretch=1)

        # Footer
        footer = QHBoxLayout()
        self.count_label = QLabel()
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
        self.refresh_engine_info()
        self.update_controls()

        # Check which engines can run (imports can be slow, so do it off the UI thread)
        self.set_status("Checking engines…", "busy")
        self.checker = EngineChecker()
        self.checker.result.connect(self.on_engine_checked)
        self.checker.start()

    # ---- widget builders --------------------------------------------------
    @staticmethod
    def _make_tab_bar(labels, group):
        bar = QFrame()
        bar.setObjectName("tabbar")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(4)
        buttons = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            group.addButton(b, i)
            lay.addWidget(b)
            buttons.append(b)
        return bar, buttons

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

        self.meter = LevelMeter()
        lay.addWidget(self.meter)

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Sensitivity"))
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setToolTip("Lower it to ignore more background noise, raise it to pick up quieter speech.")
        try:
            saved = int(self.settings.value("sensitivity", DEFAULT_SENSITIVITY))
        except (TypeError, ValueError):
            saved = DEFAULT_SENSITIVITY
        self.sensitivity.setValue(max(0, min(100, saved)))
        self.sensitivity.valueChanged.connect(self.on_sensitivity_changed)
        sens_row.addWidget(self.sensitivity, stretch=1)
        self.sens_label = QLabel()
        self.sens_label.setObjectName("subtitle")
        self.sens_label.setMinimumWidth(48)
        self.sens_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sens_row.addWidget(self.sens_label)
        lay.addLayout(sens_row)
        self.on_sensitivity_changed(self.sensitivity.value())  # sync marker and label with the saved value

        hint = QLabel(
            f"Audio is transcribed in {LIVE_CHUNK_SECONDS}-second chunks, so text appears a few seconds after "
            "you speak. Sound that stays left of the orange marker is ignored: if background noise is being "
            "transcribed, move the slider left until the noise turns grey."
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
    @property
    def recognizer(self):
        return self.recognizers.get(self.engine_key)

    @property
    def live_on(self) -> bool:
        return bool(self.live_worker and self.live_worker.isRunning())

    @property
    def file_on(self) -> bool:
        return bool(self.file_worker and self.file_worker.isRunning())

    def set_status(self, text: str, state: str):
        self.pill.setText(text)
        self.pill.setProperty("state", state)
        repolish(self.pill)

    def ready_text(self) -> str:
        rec = self.recognizer
        return f"Ready · {rec.label}" if rec else "Ready"

    @staticmethod
    def _set_action(button, active: bool, idle_text: str, active_text: str):
        """Primary action button that turns into a red Stop/Cancel button while its job runs."""
        button.setText(active_text if active else idle_text)
        button.setObjectName("danger" if active else "primary")
        repolish(button)

    def update_controls(self):
        live_on, file_on = self.live_on, self.file_on
        busy = live_on or file_on
        ready = self.recognizer is not None and not self.loading

        self.btn_live.setEnabled(not busy)
        self.btn_file.setEnabled(not busy)

        # Engine toggle: only engines that passed the check, and not mid-session / mid-load
        for key, btn in self.engine_buttons.items():
            st = self.statuses.get(key)
            btn.setEnabled(st is not None and st.level != "error" and not busy and not self.loading)
            mark = {"warn": " ⚠", "error": " ✖"}.get(st.level, "") if st else ""
            btn.setText(ENGINE_NAMES[key] + mark)
            btn.setToolTip(f"{st.summary} {st.details}".strip() if st else "Checking…")
        self.btn_fix.setEnabled(not busy and not self.loading)

        self.btn_start.setEnabled(ready and not file_on)
        self.device_combo.setEnabled(not live_on)
        self._set_action(self.btn_start, live_on, "Start listening", "Stop")

        self.btn_transcribe.setEnabled(ready and not live_on and (self.file_path is not None or file_on))
        self._set_action(self.btn_transcribe, file_on, "Transcribe file", "Cancel")
        self.btn_browse.setEnabled(not file_on)
        self.drop.setAcceptDrops(not file_on)

        self.update_footer()

    def update_footer(self):
        n = len(self.segments)
        self.count_label.setText(f"{n} segment{'s' if n != 1 else ''}")
        self.btn_save.setEnabled(n > 0)
        self.btn_clear.setEnabled(n > 0 and not (self.live_on or self.file_on))

    # ---- engines ----------------------------------------------------------
    def refresh_engine_info(self):
        lines = []
        for key in ENGINE_KEYS:
            name = ENGINE_NAMES[key]
            st = self.statuses.get(key)
            if st is None:
                lines.append(f'<b>{name}</b> <span style="color:#7b8294">checking…</span>')
                continue
            icon, color = LEVEL_ICON[st.level]
            extra = f" {escape(st.details)}" if st.level != "ok" and st.details else ""
            lines.append(f'<b>{name}</b> <span style="color:{color}">{icon}</span> {escape(st.summary)}{extra}')
        self.engine_info.setText("<br>".join(lines))

        nemo = self.statuses.get("nemo")
        self.nemo_fix = nemo.fix if nemo else ""
        self.btn_fix.setVisible(bool(self.nemo_fix))
        self.btn_fix.setText("Install GPU PyTorch" if self.nemo_fix == "reinstall_torch" else "Install NeMo support")

    def start_install(self):
        uv = find_uv()
        if not uv:
            QMessageBox.warning(
                self,
                "uv not found",
                "The installer needs uv, but it wasn't found. Start the app with "
                "`uv run asr_app.py`, or install uv from https://docs.astral.sh/uv/.",
            )
            return
        dlg = InstallDialog(self, install_commands(self.nemo_fix, uv))
        dlg.exec()
        if dlg.success:
            reply = QMessageBox.question(
                self, "Restart needed", "The packages were installed. Restart the app now to start using them?"
            )
            if reply == QMessageBox.Yes:
                os.execv(sys.executable, [sys.executable, *sys.argv])

    def on_engine_checked(self, key: str, status: EngineStatus):
        self.statuses[key] = status
        self.refresh_engine_info()
        if key == self.engine_key:
            self.activate_engine()
        else:
            self.update_controls()

    def on_engine_clicked(self, idx: int):
        self.engine_key = ENGINE_KEYS[idx]
        self.activate_engine()

    def activate_engine(self):
        """Make the selected engine ready: use the cached model, load it, or explain why not."""
        key = self.engine_key
        st = self.statuses.get(key)
        if self.loading:
            pass
        elif key in self.recognizers:
            self.set_status(self.ready_text(), "ready")
        elif st is None:
            self.set_status("Checking engines…", "busy")
        elif st.level == "error":
            self.set_status(f"{ENGINE_NAMES[key]} unavailable", "error")
        else:
            self.start_loading(key)
        self.update_controls()

    def start_loading(self, key: str):
        self.loading = True
        self.set_status(f"Loading {ENGINE_NAMES[key]} model…", "busy")
        self.loader = ModelLoader(key)
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.failed.connect(self.on_model_failed)
        self.loader.start()

    def on_model_loaded(self, key: str, recognizer):
        self.loading = False
        self.recognizers[key] = recognizer
        self.set_status(self.ready_text(), "ready")
        self.update_controls()

    def on_model_failed(self, key: str, msg: str):
        self.loading = False
        self.set_status(f"{ENGINE_NAMES[key]} failed to load", "error")
        self.update_controls()
        QMessageBox.critical(
            self,
            "Model error",
            f"Could not load the {ENGINE_NAMES[key]} model:\n\n{msg}\n\nClick the engine button again to retry.",
        )

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

    # ---- running a transcription ------------------------------------------
    def _launch(self, worker, source: str, status: str, state: str):
        """Wire up and start a live/file worker (it must already be stored in self.live_worker/file_worker)."""
        label = self.recognizer.label
        if label not in self.engines_used:
            self.engines_used.append(label)
        self.source_label = source
        worker.segment.connect(self.add_segment)
        worker.failed.connect(self.on_worker_failed)
        worker.finished.connect(self.on_worker_finished)
        worker.start()
        self.set_status(status, state)
        self.update_controls()

    def toggle_live(self):
        if self.live_on:
            self.live_worker.stop()
            self.btn_start.setEnabled(False)
            self.set_status("Finishing…", "busy")
            return
        device = self.device_combo.currentData()
        if device is None:
            QMessageBox.warning(self, "No microphone", "No input device available.")
            return
        self.live_worker = LiveWorker(self.recognizer, device, self.sensitivity.value())
        self.live_worker.level.connect(self.on_level)
        self._launch(self.live_worker, f"Live microphone ({self.device_combo.currentText()})", "● Listening", "live")

    def on_level(self, rms: float):
        self.meter.set_level(rms_to_meter(rms))

    def on_sensitivity_changed(self, value: int):
        db = gate_db(value)
        self.meter.set_gate(db_to_meter(db))
        self.sens_label.setText("Gate off" if db <= GATE_QUIETEST_DB else f"{db:.0f} dB")
        self.settings.setValue("sensitivity", value)
        if self.live_worker is not None:
            self.live_worker.sensitivity = value  # takes effect on the next chunk, even while listening

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose an audio file", "", "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff);;All files (*)"
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str):
        self.file_path = path
        name = Path(path).name
        self.file_label.setText(name)
        self.drop.setText(f"✔  {name}")
        self.file_progress.setValue(0)
        self.update_controls()

    def toggle_file(self):
        if self.file_on:
            self.file_worker.stop()
            self.btn_transcribe.setEnabled(False)
            self.set_status("Cancelling…", "busy")
            return
        if not self.file_path:
            return
        self.file_progress.setValue(0)
        self.file_worker = FileWorker(self.recognizer, self.file_path)
        self.file_worker.progress.connect(self.file_progress.setValue)
        self._launch(self.file_worker, f"File: {Path(self.file_path).name}", "Transcribing…", "busy")

    def on_worker_failed(self, msg: str):
        QMessageBox.critical(self, "Error", msg)

    def on_worker_finished(self):
        self.meter.reset()
        self.set_status(self.ready_text(), "ready")
        self.update_controls()

    # ---- transcript -------------------------------------------------------
    def add_segment(self, start: float, end: float, text: str):
        self.segments.append(Segment(start, end, text))
        self.text.append(f'<span style="color:#7c8cff">[{fmt_time(start)}]</span> {escape(text)}')
        self.update_footer()

    def clear_transcript(self):
        self.segments.clear()
        self.engines_used.clear()
        self.text.clear()
        self.update_footer()

    def save_transcript(self):
        if not self.segments:
            return
        default = f"transcript_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path, _ = QFileDialog.getSaveFileName(self, "Save transcript", default, "Text files (*.txt)")
        if not path:
            return
        lines = [
            "Transcript",
            f"Source: {self.source_label or 'unknown'}",
            f"Engine: {', '.join(self.engines_used) or 'unknown'}",
            f"Saved: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "-" * 40,
            *(f"[{fmt_time(s.start)} - {fmt_time(s.end)}] {s.text}" for s in self.segments),
        ]
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.set_status("Saved ✓", "ready")

    # ---- shutdown ---------------------------------------------------------
    def closeEvent(self, event):
        for t in (self.live_worker, self.file_worker, self.checker, self.loader):
            if t and t.isRunning():
                if hasattr(t, "stop"):
                    t.stop()
                if not t.wait(3000):  # an import / model load / inference can't be interrupted politely
                    t.terminate()
                    t.wait(1000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
