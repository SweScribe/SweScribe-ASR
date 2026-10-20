"""Pianissimo transcriber: live microphone or audio-file speech-to-text, with an ONNX or NeMo engine.

Run with:  uv run asr_app.py
"""
import importlib.util
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from string import Template

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

from PySide6.QtCore import QEasingCurve, QRectF, QSettings, Qt, QThread, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
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
# Theme
# ----------------------------------------------------------------------------
COLOR_KEYS = ("bg", "surface", "accent", "text")
COLOR_LABELS = {"bg": "Background", "surface": "Panels", "accent": "Accent", "text": "Text"}
THEME_MODES = ("dark", "light")
THEME_DEFAULTS = {
    "dark": {"bg": "#0f1117", "surface": "#161a24", "accent": "#5b6cff", "text": "#dfe3ee"},
    "light": {"bg": "#f3f5fa", "surface": "#ffffff", "accent": "#4f5df0", "text": "#1d2233"},
}
HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def rgb_to_hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c))) for c in rgb)


def mix(a: str, b: str, t: float) -> str:
    """Blend two #rrggbb colors: t=0 gives a, t=1 gives b."""
    return rgb_to_hex([x + (y - x) * t for x, y in zip(hex_to_rgb(a), hex_to_rgb(b))])


def luminance(color: str) -> float:
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in hex_to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def build_palette(base: dict) -> dict:
    """Derive every UI color from the four the user can edit (background, panels, accent, text)."""
    bg, surface, accent, text = (base[k] for k in COLOR_KEYS)
    dark = luminance(bg) < 0.2
    pal = dict(base)
    pal.update(
        muted=mix(text, bg, 0.45),
        faint=mix(text, bg, 0.65),
        border=mix(surface, text, 0.12),
        control=mix(surface, text, 0.07),
        control_hover=mix(surface, text, 0.13),
        accent_hover=mix(accent, "#ffffff" if dark else "#000000", 0.15),
        accent_dim=mix(surface, accent, 0.3),
        on_accent="#ffffff" if luminance(accent) < 0.4 else "#111111",
        ok="#3ddc84" if dark else "#1c9a58",
        warn="#f0c05a" if dark else "#b7791f",
        err="#ff6b7d" if dark else "#d6293e",
        gate="#ffb86b" if dark else "#e07a10",
        danger="#e5484d",
        danger_hover="#ee6065",
    )
    for name in ("ok", "warn", "err"):
        pal[name + "_bg"] = mix(surface, pal[name], 0.18)
    return pal


def theme_colors(mode: str, custom: dict) -> dict:
    return {**THEME_DEFAULTS[mode], **custom.get(mode, {})}


def load_theme(raw) -> tuple[str, dict]:
    """Read the saved theme; anything missing or malformed falls back to the defaults."""
    mode, custom = "dark", {m: {} for m in THEME_MODES}
    try:
        data = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        data = {}
    if isinstance(data, dict):
        if data.get("mode") in THEME_MODES:
            mode = data["mode"]
        saved = data.get("custom")
        if isinstance(saved, dict):
            for m in THEME_MODES:
                colors = saved.get(m)
                if isinstance(colors, dict):
                    custom[m] = {
                        k: v.lower()
                        for k, v in colors.items()
                        if k in COLOR_KEYS and isinstance(v, str) and HEX_RE.fullmatch(v)
                    }
    return mode, custom


def dump_theme(mode: str, custom: dict) -> str:
    return json.dumps({"mode": mode, "custom": custom})


# Colors currently in use; custom-painted widgets read this when they paint.
THEME_COLORS: dict = build_palette(THEME_DEFAULTS["dark"])

STYLE = Template("""
* { font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif; font-size: 13px; }
QMainWindow, QWidget#root { background: $bg; }
QToolTip { background: $surface; color: $text; border: 1px solid $border; padding: 4px; }
QLabel { color: $text; }
QLabel#title { font-size: 22px; font-weight: 600; }
QLabel#subtitle { color: $muted; }
QLabel#engineinfo { color: $muted; font-size: 12px; }
QLabel#cardtitle { font-size: 15px; font-weight: 600; }
QLabel#pill { background: $control; border-radius: 11px; padding: 4px 12px; color: $muted; }
QLabel#pill[state="ready"] { background: $ok_bg; color: $ok; }
QLabel#pill[state="busy"] { background: $warn_bg; color: $warn; }
QLabel#pill[state="live"] { background: $err_bg; color: $err; }
QLabel#pill[state="error"] { background: $err_bg; color: $err; }
QFrame#card { background: $surface; border: 1px solid $border; border-radius: 14px; }
QFrame#engine { background: $bg; border: 1px solid $border; border-radius: 12px; }
QFrame#engine[active="true"] { border: 1px solid $accent; }
QPushButton {
    background: $control; color: $text; border: 1px solid $border;
    border-radius: 9px; padding: 9px 18px;
}
QPushButton:hover { background: $control_hover; }
QPushButton:disabled { color: $faint; }
QPushButton#primary { background: $accent; border: none; color: $on_accent; font-weight: 600; }
QPushButton#primary:hover { background: $accent_hover; }
QPushButton#primary:disabled { background: $accent_dim; color: $faint; }
QPushButton#danger { background: $danger; border: none; color: white; font-weight: 600; }
QPushButton#danger:hover { background: $danger_hover; }
QPushButton#segbtn { background: transparent; border: none; border-radius: 8px; padding: 8px 22px; color: $muted; }
QPushButton#segbtn:hover { background: transparent; color: $text; }
QPushButton#segbtn:checked { color: $on_accent; font-weight: 600; }
QPushButton#segbtn:disabled { color: $faint; }
QComboBox {
    background: $control; color: $text; border: 1px solid $border;
    border-radius: 9px; padding: 8px 12px;
}
QComboBox QAbstractItemView { background: $surface; color: $text; selection-background-color: $accent; selection-color: $on_accent; border: 1px solid $border; }
QProgressBar { background: $control; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent; }
QProgressBar::chunk { background: $accent; border-radius: 4px; }
QSlider::groove:horizontal { height: 6px; background: $control; border-radius: 3px; }
QSlider::sub-page:horizontal { background: $accent; border-radius: 3px; }
QSlider::handle:horizontal { background: $text; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
QSlider::handle:horizontal:hover { background: $accent_hover; }
QTextEdit, QPlainTextEdit {
    background: $bg; color: $text; border: 1px solid $border;
    border-radius: 10px; padding: 12px; font-size: 14px;
    selection-background-color: $accent; selection-color: $on_accent;
}
QLabel#drop {
    border: 2px dashed $border; border-radius: 12px; padding: 22px;
    color: $muted; background: $bg;
}
QLabel#drop[hover="true"] { border-color: $accent; color: $text; }
""")

LEVEL_ICON = {"ok": ("✔", "ok"), "warn": ("⚠", "warn"), "error": ("✖", "err")}  # (icon, palette color)

ENGINE_DESCRIPTIONS = {
    "onnx": (
        "Lightweight and quick to start. A small quantized (int8) model that runs well on any CPU, with no GPU "
        "or extra downloads needed. Best for everyday use and live transcription."
    ),
    "nemo": (
        "The full NVIDIA NeMo version of the model. Can be more accurate and is fastest on an NVIDIA GPU, but "
        "needs PyTorch and NeMo (several GB). Best for long recordings and when accuracy matters most."
    ),
}


# ----------------------------------------------------------------------------
# Custom widgets
# ----------------------------------------------------------------------------
class SegmentedControl(QWidget):
    """Pill-shaped toggle whose highlight slides to the selected option."""

    changed = Signal(int)  # emitted when the user clicks an option

    def __init__(self, labels):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(4)
        self.group = QButtonGroup(self)
        self.buttons: list[QPushButton] = []
        for i, text in enumerate(labels):
            b = QPushButton(text)
            b.setObjectName("segbtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.buttons.append(b)
        self.buttons[0].setChecked(True)
        self.index = 0
        self._from = self._to = QRectF()
        self._t = 1.0
        self.anim = QVariantAnimation(self)
        self.anim.setDuration(240)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.valueChanged.connect(self._step)
        self.group.idClicked.connect(self._clicked)

    # -- public API
    def set_current(self, index: int, animate: bool = True):
        self.buttons[index].setChecked(True)
        self.index = index
        self._move(animate)

    def set_enabled_at(self, index: int, enabled: bool):
        if self.buttons[index].isEnabled() != enabled:
            self.buttons[index].setEnabled(enabled)
            self.update()

    def set_text(self, index: int, text: str):
        if self.buttons[index].text() != text:
            self.buttons[index].setText(text)
            self.layout().activate()
            self._move(False)

    def set_tooltip(self, index: int, text: str):
        self.buttons[index].setToolTip(text)

    # -- animation
    def _clicked(self, index: int):
        self.index = index
        self._move(True)
        self.changed.emit(index)

    def _step(self, value):
        self._t = float(value)
        self.update()

    def _shown(self) -> QRectF:
        a, b, t = self._from, self._to, self._t
        return QRectF(
            a.x() + (b.x() - a.x()) * t,
            a.y() + (b.y() - a.y()) * t,
            a.width() + (b.width() - a.width()) * t,
            a.height() + (b.height() - a.height()) * t,
        )

    def _move(self, animate: bool):
        target = QRectF(self.buttons[self.index].geometry())
        self.anim.stop()
        if animate and self.isVisible() and not self._to.isNull():
            self._from = self._shown()
            self._to = target
            self._t = 0.0
            self.anim.start()
        else:
            self._from = self._to = target
            self._t = 1.0
            self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.layout().activate()
        self._move(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._move(False)

    def paintEvent(self, _event):
        c = THEME_COLORS
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(c["border"]), 1))
        p.setBrush(QColor(c["surface"]))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        if not self._to.isNull():
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c["accent"] if self.buttons[self.index].isEnabled() else c["accent_dim"]))
            p.drawRoundedRect(self._shown(), 8, 8)


class LevelMeter(QWidget):
    """Microphone level bar with a marker at the noise-gate level.
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
        c = THEME_COLORS
        w, h = self.width(), self.height()
        bar_h = 8
        top = (h - bar_h) / 2
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["control"]))
        p.drawRoundedRect(QRectF(0, top, w, bar_h), 4, 4)
        if self.level > 0.01:
            p.setBrush(QColor(c["ok"] if self.level >= self.gate else c["faint"]))
            p.drawRoundedRect(QRectF(0, top, max(float(bar_h), w * self.level), bar_h), 4, 4)
        p.setBrush(QColor(c["gate"]))
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


class EngineCard(QFrame):
    """Name, description and availability of one engine; the active engine gets an accent border."""

    def __init__(self, name: str, description: str, with_action: bool = False):
        super().__init__()
        self.setObjectName("engine")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)
        top = QHBoxLayout()
        title = QLabel(name)
        title.setObjectName("cardtitle")
        top.addWidget(title)
        top.addStretch()
        self.button: QPushButton | None = None
        if with_action:
            self.button = QPushButton("")
            self.button.setCursor(Qt.PointingHandCursor)
            self.button.setVisible(False)
            top.addWidget(self.button)
        lay.addLayout(top)
        desc = QLabel(description)
        desc.setWordWrap(True)
        lay.addWidget(desc)
        self.status = QLabel()
        self.status.setObjectName("engineinfo")
        self.status.setTextFormat(Qt.RichText)
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

    def set_active(self, active: bool):
        self.setProperty("active", active)
        repolish(self)


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
        self.settings = QSettings("SweScribe", "Transcriber")  # remembers sensitivity and theme between runs
        self.theme_mode, self.theme_custom = load_theme(self.settings.value("theme", ""))
        self.meters: list[LevelMeter] = []
        self.segs: list[SegmentedControl] = []

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

        # Main navigation: Live / File / Settings
        self.main_seg = self._segmented(["🎙  Live", "📁  File", "⚙  Settings"])
        outer.addWidget(self.main_seg, alignment=Qt.AlignLeft)

        self.main_stack = QStackedWidget()
        self.main_stack.addWidget(self._build_transcribe_page())
        self.main_stack.addWidget(self._build_settings_page())
        outer.addWidget(self.main_stack, stretch=1)
        self.main_seg.changed.connect(self.on_main_tab)

        self.refresh_devices()
        self.sensitivity.valueChanged.connect(self.on_sensitivity_changed)
        self.device_combo.currentIndexChanged.connect(self.update_mic_caption)
        self.on_sensitivity_changed(self.sensitivity.value())  # sync markers and labels with the saved value
        self.mode_seg.set_current(THEME_MODES.index(self.theme_mode), animate=False)
        self.apply_theme()
        self.update_controls()

        # Check which engines can run (imports can be slow, so do it off the UI thread)
        self.set_status("Checking engines…", "busy")
        self.checker = EngineChecker()
        self.checker.result.connect(self.on_engine_checked)
        self.checker.start()

    # ---- widget builders --------------------------------------------------
    def _segmented(self, labels) -> SegmentedControl:
        seg = SegmentedControl(labels)
        self.segs.append(seg)
        return seg

    @staticmethod
    def _page(spacing: int = 12):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(spacing)
        return page, lay

    @staticmethod
    def _card(inner: QWidget) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.addWidget(inner)
        return card

    @staticmethod
    def _caption(text: str, width: int = 96) -> QLabel:
        label = QLabel(text)
        label.setMinimumWidth(width)
        return label

    def _build_transcribe_page(self) -> QWidget:
        page, lay = self._page(14)
        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_live_page())
        self.mode_stack.addWidget(self._build_file_page())
        lay.addWidget(self._card(self.mode_stack))

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Transcription will appear here…")
        lay.addWidget(self.text, stretch=1)

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
        lay.addLayout(footer)
        self.btn_clear.clicked.connect(self.clear_transcript)
        self.btn_save.clicked.connect(self.save_transcript)
        return page

    def _build_live_page(self) -> QWidget:
        page, lay = self._page()
        row = QHBoxLayout()
        self.mic_caption = QLabel()
        self.mic_caption.setWordWrap(True)
        row.addWidget(self.mic_caption, stretch=1)
        self.btn_start = QPushButton("Start listening")
        self.btn_start.setObjectName("primary")
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.toggle_live)
        row.addWidget(self.btn_start)
        lay.addLayout(row)

        meter = LevelMeter()
        self.meters.append(meter)
        lay.addWidget(meter)

        hint = QLabel(
            f"Audio is transcribed in {LIVE_CHUNK_SECONDS}-second chunks, so text appears a few seconds after "
            "you speak. Choose the microphone and filter out background noise under Settings → Microphone."
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return page

    def _build_file_page(self) -> QWidget:
        page, lay = self._page()
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

    def _build_settings_page(self) -> QWidget:
        page, lay = self._page(14)
        self.sub_seg = self._segmented(["Microphone", "General", "Customization"])
        lay.addWidget(self.sub_seg, alignment=Qt.AlignLeft)
        self.sub_stack = QStackedWidget()
        self.sub_stack.addWidget(self._build_mic_page())
        self.sub_stack.addWidget(self._build_general_page())
        self.sub_stack.addWidget(self._build_custom_page())
        self.sub_seg.changed.connect(self.sub_stack.setCurrentIndex)
        lay.addWidget(self._card(self.sub_stack))
        lay.addStretch()
        return page

    def _build_mic_page(self) -> QWidget:
        page, lay = self._page()
        row = QHBoxLayout()
        row.addWidget(self._caption("Microphone"))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(320)
        row.addWidget(self.device_combo, stretch=1)
        lay.addLayout(row)

        sens_row = QHBoxLayout()
        sens_row.addWidget(self._caption("Sensitivity"))
        self.sensitivity = QSlider(Qt.Horizontal)
        self.sensitivity.setRange(0, 100)
        self.sensitivity.setToolTip("Lower it to ignore more background noise, raise it to pick up quieter speech.")
        try:
            saved = int(self.settings.value("sensitivity", DEFAULT_SENSITIVITY))
        except (TypeError, ValueError):
            saved = DEFAULT_SENSITIVITY
        self.sensitivity.setValue(max(0, min(100, saved)))
        sens_row.addWidget(self.sensitivity, stretch=1)
        self.sens_label = QLabel()
        self.sens_label.setObjectName("subtitle")
        self.sens_label.setMinimumWidth(56)
        self.sens_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sens_row.addWidget(self.sens_label)
        lay.addLayout(sens_row)

        meter = LevelMeter()
        self.meters.append(meter)
        lay.addWidget(meter)

        hint = QLabel(
            "Sound that stays left of the orange marker is ignored. If background noise is being transcribed, "
            "move the slider left until the noise turns grey. The level shows while you are listening."
        )
        hint.setObjectName("subtitle")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return page

    def _build_general_page(self) -> QWidget:
        page, lay = self._page()
        row = QHBoxLayout()
        row.addWidget(self._caption("Engine", 60))
        self.engine_seg = self._segmented([ENGINE_NAMES[k] for k in ENGINE_KEYS])
        row.addWidget(self.engine_seg)
        row.addStretch()
        lay.addLayout(row)

        self.cards = {
            key: EngineCard(ENGINE_NAMES[key], ENGINE_DESCRIPTIONS[key], with_action=(key == "nemo"))
            for key in ENGINE_KEYS
        }
        self.btn_fix = self.cards["nemo"].button
        self.btn_fix.clicked.connect(self.start_install)
        for card in self.cards.values():
            lay.addWidget(card)
        self.engine_seg.set_current(ENGINE_KEYS.index(self.engine_key), animate=False)
        self.engine_seg.changed.connect(self.on_engine_clicked)
        return page

    def _build_custom_page(self) -> QWidget:
        page, lay = self._page(14)
        row = QHBoxLayout()
        row.addWidget(self._caption("Theme", 60))
        self.mode_seg = self._segmented(["🌙  Dark", "☀  Light"])
        row.addWidget(self.mode_seg)
        row.addStretch()
        lay.addLayout(row)
        self.mode_seg.changed.connect(self.on_theme_mode)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.swatches: dict[str, QPushButton] = {}
        self.hex_labels: dict[str, QLabel] = {}
        for r, key in enumerate(COLOR_KEYS):
            swatch = QPushButton("")
            swatch.setFixedSize(56, 28)
            swatch.setCursor(Qt.PointingHandCursor)
            swatch.setToolTip("Click to choose a color")
            swatch.clicked.connect(lambda _checked=False, k=key: self.pick_color(k))
            hex_label = QLabel()
            hex_label.setObjectName("subtitle")
            self.swatches[key] = swatch
            self.hex_labels[key] = hex_label
            grid.addWidget(QLabel(COLOR_LABELS[key]), r, 0)
            grid.addWidget(swatch, r, 1)
            grid.addWidget(hex_label, r, 2)
        grid.setColumnStretch(3, 1)
        lay.addLayout(grid)

        bottom = QHBoxLayout()
        self.btn_reset_colors = QPushButton("Reset colors")
        self.btn_reset_colors.setCursor(Qt.PointingHandCursor)
        self.btn_reset_colors.clicked.connect(self.reset_colors)
        bottom.addWidget(self.btn_reset_colors)
        note = QLabel("Colors are remembered separately for dark and light mode.")
        note.setObjectName("subtitle")
        bottom.addWidget(note, stretch=1)
        lay.addLayout(bottom)
        return page

    # ---- theme ------------------------------------------------------------
    def apply_theme(self):
        colors = theme_colors(self.theme_mode, self.theme_custom)
        pal = build_palette(colors)
        THEME_COLORS.clear()
        THEME_COLORS.update(pal)
        self.setStyleSheet(STYLE.substitute(pal))
        for key, swatch in self.swatches.items():
            swatch.setStyleSheet(f"background: {colors[key]}; border: 1px solid {pal['border']}; border-radius: 8px; padding: 0;")
            self.hex_labels[key].setText(colors[key].upper())
        for widget in (*self.segs, *self.meters):
            widget.update()
        self.refresh_engine_info()
        self.render_transcript()

    def save_theme(self):
        self.settings.setValue("theme", dump_theme(self.theme_mode, self.theme_custom))

    def on_theme_mode(self, idx: int):
        self.theme_mode = THEME_MODES[idx]
        self.save_theme()
        self.apply_theme()

    def pick_color(self, key: str):
        current = theme_colors(self.theme_mode, self.theme_custom)[key]
        chosen = QColorDialog.getColor(QColor(current), self, f"Choose {COLOR_LABELS[key].lower()} color")
        if chosen.isValid():
            self.theme_custom[self.theme_mode][key] = chosen.name().lower()
            self.save_theme()
            self.apply_theme()

    def reset_colors(self):
        self.theme_custom[self.theme_mode] = {}
        self.save_theme()
        self.apply_theme()

    # ---- navigation -------------------------------------------------------
    def on_main_tab(self, idx: int):
        if idx == 2:
            self.main_stack.setCurrentIndex(1)
        else:
            self.main_stack.setCurrentIndex(0)
            self.mode_stack.setCurrentIndex(idx)

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

        # While one mode runs the other can't be opened; Settings stays reachable
        self.main_seg.set_enabled_at(0, not file_on)
        self.main_seg.set_enabled_at(1, not live_on)

        # Engine toggle: only engines that passed the check, and not mid-session / mid-load
        for i, key in enumerate(ENGINE_KEYS):
            st = self.statuses.get(key)
            self.engine_seg.set_enabled_at(i, st is not None and st.level != "error" and not busy and not self.loading)
            mark = {"warn": " ⚠", "error": " ✖"}.get(st.level, "") if st else ""
            self.engine_seg.set_text(i, ENGINE_NAMES[key] + mark)
            self.engine_seg.set_tooltip(i, f"{st.summary} {st.details}".strip() if st else "Checking…")
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
        for key in ENGINE_KEYS:
            card = self.cards[key]
            card.set_active(key == self.engine_key)
            st = self.statuses.get(key)
            if st is None:
                card.status.setText("Checking…")
                continue
            icon, color = LEVEL_ICON[st.level]
            extra = f" {escape(st.details)}" if st.level != "ok" and st.details else ""
            card.status.setText(f'<span style="color:{THEME_COLORS[color]}">{icon}</span> {escape(st.summary)}{extra}')

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
        self.refresh_engine_info()
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
            f"Could not load the {ENGINE_NAMES[key]} model:\n\n{msg}\n\nSelect the engine again (Settings → General) to retry.",
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

    def update_mic_caption(self, *_):
        self.mic_caption.setText(f"🎙  {self.device_combo.currentText()}  ·  gate: {self.sens_label.text()}")

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
        level = rms_to_meter(rms)
        for meter in self.meters:
            meter.set_level(level)

    def on_sensitivity_changed(self, value: int):
        db = gate_db(value)
        for meter in self.meters:
            meter.set_gate(db_to_meter(db))
        self.sens_label.setText("Gate off" if db <= GATE_QUIETEST_DB else f"{db:.0f} dB")
        self.update_mic_caption()
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
        for meter in self.meters:
            meter.reset()
        self.set_status(self.ready_text(), "ready")
        self.update_controls()

    # ---- transcript -------------------------------------------------------
    @staticmethod
    def segment_html(start: float, text: str) -> str:
        return f'<span style="color:{THEME_COLORS["accent"]}">[{fmt_time(start)}]</span> {escape(text)}'

    def render_transcript(self):
        """Redraw the whole transcript (needed when the accent color changes)."""
        self.text.clear()
        for s in self.segments:
            self.text.append(self.segment_html(s.start, s.text))

    def add_segment(self, start: float, end: float, text: str):
        self.segments.append(Segment(start, end, text))
        self.text.append(self.segment_html(start, text))
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
