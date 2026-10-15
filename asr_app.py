import os
import sys
import re
import math
import queue
import shutil
import hashlib
import subprocess
import threading
import urllib.request
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
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

# Speaker detection: Silero VAD finds utterances, a speaker-embedding model says who spoke.
SPEAKER_MODELS_DIR = (
    Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    / "pianissimo-transcriber"
    / "models"
)
_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
VAD_MODEL = "silero_vad.onnx"
EMBEDDING_MODEL = "wespeaker_en_voxceleb_resnet34.onnx"
SPEAKER_MODEL_FILES = {  # name -> (url, size in bytes, sha256)
    VAD_MODEL: (
        f"{_RELEASES}/asr-models/{VAD_MODEL}",
        643854,
        "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6",
    ),
    EMBEDDING_MODEL: (
        f"{_RELEASES}/speaker-recongition-models/{EMBEDDING_MODEL}",  # (sic) that is the tag's spelling
        26534365,
        "5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
    ),
}
VAD_MIN_SILENCE = 0.5       # seconds of silence that end an utterance
VAD_MAX_SPEECH = 20.0       # force a split after this long without a pause
SPEAKER_THRESHOLD = 0.45    # cosine similarity needed to count as an already-known speaker
MAX_SPEAKERS = 8
MIN_EMBED_SECONDS = 1.0     # shorter utterances are too short to tell voices apart reliably
SPEAKER_COLORS = ["#7c8cff", "#ffb86b", "#5fe08f", "#ff7b8f", "#6fd3ff", "#d49bff", "#f0e060", "#9aa3b8"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""  # "" when speaker detection is off


@dataclass
class EngineStatus:
    level: str  # "ok" | "warn" (works, with a caveat) | "error" (cannot run)
    summary: str
    details: str = ""
    fix: str = ""  # "" | "install" (torch + NeMo) | "reinstall_torch" (swap CPU build for GPU build)


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def remove_overlap(previous: str, current: str) -> str:
    """Drop words at the start of `current` that repeat the end of `previous`
    (live chunks overlap, so boundary words get transcribed twice)."""
    if not previous or not current:
        return current.strip()

    prev_words = normalize_text(previous).split()
    curr_words = current.split()
    if not prev_words or not curr_words:
        return current.strip()

    max_overlap = min(len(prev_words), len(curr_words), 12)
    best = 0
    for n in range(1, max_overlap + 1):
        if prev_words[-n:] == [normalize_text(w) for w in curr_words[:n]]:
            best = n
    if best:
        return " ".join(curr_words[best:]).strip()
    return current.strip()


def hyp_text(result) -> str:
    """NeMo returns different shapes depending on model type and version."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        result = result[0]
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()


# ----------------------------------------------------------------------------
# Engines: availability checks
# ----------------------------------------------------------------------------
def check_onnx() -> EngineStatus:
    if importlib.util.find_spec("onnx_asr") is None:
        return EngineStatus("error", "The onnx-asr package is not installed.", "Fix: pip install onnx-asr")
    if importlib.util.find_spec("onnxruntime") is None:
        return EngineStatus("error", "ONNX Runtime is not installed.", "Fix: pip install onnxruntime")
    try:
        import onnx_asr  # noqa: F401
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception as exc:  # noqa: BLE001
        return EngineStatus(
            "error",
            f"ONNX failed to import: {type(exc).__name__}: {exc}",
            "Try reinstalling: pip install --force-reinstall onnx-asr onnxruntime",
        )
    return EngineStatus(
        "ok",
        "Ready · int8 model via ONNX Runtime.",
        "ONNX Runtime providers: " + ", ".join(providers),
    )


def check_nemo() -> EngineStatus:
    if importlib.util.find_spec("torch") is None:
        return EngineStatus(
            "error",
            "PyTorch is not installed.",
            "Click the install button to set up PyTorch (matched to your GPU driver) and NeMo automatically.",
            fix="install",
        )
    if importlib.util.find_spec("nemo") is None:
        return EngineStatus(
            "error",
            "NVIDIA NeMo is not installed.",
            "Click the install button to install it automatically.",
            fix="install",
        )
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
    return EngineStatus(
        "warn",
        "Works, but on CPU only (slower).",
        reason,
        fix=fix,
    )


class EngineChecker(QThread):
    result = Signal(str, object)

    def run(self):
        for key, fn in (("onnx", check_onnx), ("nemo", check_nemo), ("speakers", check_speakers)):
            try:
                status = fn()
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
    if fix == "install_sherpa":
        return [[uv, "pip", "install", "--python", sys.executable, "sherpa-onnx"]]
    base = [uv, "pip", "install", "--python", sys.executable, "--torch-backend=auto"]
    if fix == "reinstall_torch":
        family = [n for n in ("torch", "torchaudio", "torchvision")
                  if n == "torch" or importlib.util.find_spec(n)]
        reinstall = [arg for n in family for arg in ("--reinstall-package", n)]
        return [base + reinstall + family]
    # torch first, so NeMo's own dependency on it is already satisfied by the right build
    return [base + ["torch"], base + ["nemo_toolkit[asr]"]]


def run_commands(cmds, on_line, should_stop) -> int:
    """Run commands in order, streaming output lines. 0 = ok, -1 = cancelled, else exit code."""
    for cmd in cmds:
        on_line("$ " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
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


class InstallThread(QThread):
    line = Signal(str)
    done = Signal(int)

    def __init__(self, cmds, parent=None):
        super().__init__(parent)
        self.cmds = cmds
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

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
        if code == 0:
            self.success = True
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
        audio = np.clip(np.asarray(audio_16k, dtype=np.float32), -1.0, 1.0)
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
# Speaker detection: VAD utterances + speaker embeddings + online clustering
# ----------------------------------------------------------------------------
@dataclass
class Utterance:
    start: float  # seconds from the start of the audio
    end: float
    samples: np.ndarray  # 16 kHz mono float32


class SpeakerTracker:
    """Gives each utterance a speaker number by comparing its voice embedding with
    the speakers heard so far (cosine similarity against a running centroid)."""

    def __init__(self, threshold=SPEAKER_THRESHOLD, max_speakers=MAX_SPEAKERS, min_seconds=MIN_EMBED_SECONDS):
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_seconds = min_seconds
        self.sums: list[np.ndarray] = []  # per speaker: duration-weighted sum of unit embeddings
        self.last = 0

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-9)

    def assign(self, embedding: np.ndarray | None, seconds: float) -> int:
        """Return a 1-based speaker number."""
        if embedding is None:  # too short to embed at all: assume the previous speaker continues
            self.last = self.last or 1
            return self.last

        e = self._unit(np.asarray(embedding, dtype=np.float32))
        if not self.sums:
            self.sums.append(e * min(seconds, 10.0))
            self.last = 1
            return 1

        sims = [float(np.dot(e, self._unit(s))) for s in self.sums]
        best = int(np.argmax(sims))
        reliable = seconds >= self.min_seconds
        if sims[best] >= self.threshold:
            if reliable:
                self.sums[best] += e * min(seconds, 10.0)
        elif reliable and len(self.sums) < self.max_speakers:
            self.sums.append(e * min(seconds, 10.0))
            best = len(self.sums) - 1
        # else: short or at the speaker cap -> closest known speaker, without changing it
        self.last = best + 1
        return self.last


def _set_if_exists(obj, name, value):
    try:
        setattr(obj, name, value)
    except AttributeError:  # older/newer sherpa-onnx without this option: keep its default
        pass


def make_vad(model_path: Path):
    import sherpa_onnx

    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(model_path)
    _set_if_exists(cfg.silero_vad, "threshold", 0.5)
    _set_if_exists(cfg.silero_vad, "min_silence_duration", VAD_MIN_SILENCE)
    _set_if_exists(cfg.silero_vad, "min_speech_duration", 0.25)
    _set_if_exists(cfg.silero_vad, "max_speech_duration", VAD_MAX_SPEECH)
    cfg.sample_rate = MODEL_SAMPLE_RATE
    window = cfg.silero_vad.window_size
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60), window


class SpeakerEmbedder:
    MIN_SECONDS = 0.5

    def __init__(self, model_path: Path, num_threads: int = 2):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=num_threads, debug=False, provider="cpu"
        )
        if not cfg.validate():
            raise RuntimeError(f"Invalid speaker model: {model_path}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, samples: np.ndarray):
        if len(samples) < self.MIN_SECONDS * MODEL_SAMPLE_RATE:
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=MODEL_SAMPLE_RATE, waveform=samples)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return np.array(self.extractor.compute(stream), dtype=np.float32)


class SpeakerPipeline:
    """Feed 16 kHz audio in; get utterances out whenever the VAD sees a pause."""

    def __init__(self, vad, window: int, embedder, tracker: SpeakerTracker | None = None):
        self.vad = vad
        self.window = window
        self.embedder = embedder
        self.tracker = tracker or SpeakerTracker()
        self.buffer = np.empty(0, dtype=np.float32)
        self.fed = 0  # samples handed to the VAD so far

    @classmethod
    def create(cls, models_dir: Path):
        vad, window = make_vad(models_dir / VAD_MODEL)
        return cls(vad, window, SpeakerEmbedder(models_dir / EMBEDDING_MODEL))

    def feed(self, samples: np.ndarray) -> list[Utterance]:
        self.buffer = np.concatenate((self.buffer, np.asarray(samples, dtype=np.float32)))
        out: list[Utterance] = []
        while len(self.buffer) >= self.window:
            self.vad.accept_waveform(self.buffer[: self.window])
            self.buffer = self.buffer[self.window :]
            self.fed += self.window
            out.extend(self._drain())
        return out

    def flush(self) -> list[Utterance]:
        if len(self.buffer):  # pad the tail up to one full VAD window
            pad = np.zeros(self.window - len(self.buffer), dtype=np.float32)
            self.vad.accept_waveform(np.concatenate((self.buffer, pad)))
            self.fed += len(self.buffer)
            self.buffer = np.empty(0, dtype=np.float32)
        self.vad.flush()
        return self._drain()

    def _drain(self) -> list[Utterance]:
        out = []
        while not self.vad.empty():
            seg = self.vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            start_sample = getattr(seg, "start", None)
            if start_sample is None:  # no start index available: approximate from what has been fed
                start_sample = max(0, self.fed - len(samples))
            self.vad.pop()
            start = start_sample / MODEL_SAMPLE_RATE
            out.append(Utterance(start, start + len(samples) / MODEL_SAMPLE_RATE, samples))
        return out

    def label(self, utt: Utterance) -> str:
        seconds = len(utt.samples) / MODEL_SAMPLE_RATE
        return f"Speaker {self.tracker.assign(self.embedder.embed(utt.samples), seconds)}"


def models_present(models_dir: Path = SPEAKER_MODELS_DIR) -> bool:
    return all(
        (models_dir / name).is_file() and (models_dir / name).stat().st_size == size
        for name, (_url, size, _sha) in SPEAKER_MODEL_FILES.items()
    )


def check_speakers() -> EngineStatus:
    if importlib.util.find_spec("sherpa_onnx") is None:
        return EngineStatus(
            "error",
            "Speaker detection needs the sherpa-onnx package.",
            "Click the button to install it.",
            fix="install_sherpa",
        )
    try:
        import sherpa_onnx  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return EngineStatus(
            "error",
            f"sherpa-onnx failed to import: {type(exc).__name__}: {exc}",
            "Try reinstalling it: uv pip install --reinstall sherpa-onnx",
        )
    if not models_present():
        mb = sum(size for _u, size, _s in SPEAKER_MODEL_FILES.values()) / 1e6
        return EngineStatus(
            "error",
            f"Speaker models are not downloaded yet (~{mb:.0f} MB, one time).",
            "Click the button to download them.",
            fix="download_models",
        )
    return EngineStatus("ok", "Ready · pauses split the audio, voices are matched by sound.")


def download_models(dest_dir: Path, files: dict, on_progress, should_stop) -> None:
    """Download and verify the model files. Raises RuntimeError on failure or cancel."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = sum(size for _u, size, _s in files.values())
    done = 0
    for name, (url, size, sha) in files.items():
        path = dest_dir / name
        if path.is_file() and path.stat().st_size == size:
            done += size
            on_progress(int(done * 100 / total))
            continue
        part = path.with_name(name + ".part")
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(url, timeout=30) as resp, open(part, "wb") as out:
                while True:
                    if should_stop():
                        raise RuntimeError("Cancelled")
                    buf = resp.read(1 << 18)
                    if not buf:
                        break
                    out.write(buf)
                    digest.update(buf)
                    done += len(buf)
                    on_progress(min(99, int(done * 100 / total)))
            if part.stat().st_size != size or digest.hexdigest() != sha:
                raise RuntimeError(f"{name}: the downloaded file is corrupt (checksum mismatch). Try again.")
            part.replace(path)
        except Exception:
            part.unlink(missing_ok=True)
            raise
    on_progress(100)


class SpeakerModelDownloader(QThread):
    progress = Signal(int)
    done = Signal(str)  # "" on success, otherwise an error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            download_models(SPEAKER_MODELS_DIR, SPEAKER_MODEL_FILES, self.progress.emit, self._stop.is_set)
            self.done.emit("")
        except Exception as exc:  # noqa: BLE001
            self.done.emit(str(exc) or type(exc).__name__)


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


class LiveWorker(QThread):
    segment = Signal(float, float, str, str)  # start, end, text, speaker ("" = not detected)
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, recognizer, device_index, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
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
            previous_text = ""

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

                    text = self.recognizer.transcribe(to_model_rate(chunk, input_rate))
                    if not text:
                        continue
                    new_text = remove_overlap(previous_text, text)
                    previous_text = text
                    if new_text:
                        self.segment.emit(start, end, new_text, "")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


def transcribe_utterance(recognizer, pipeline: SpeakerPipeline, utt: Utterance):
    """Returns (text, speaker label), or None if nothing was recognised."""
    text = recognizer.transcribe(utt.samples)
    if not text:
        return None
    return text, pipeline.label(utt)


class LiveSpeakerWorker(QThread):
    """Live mode with speaker detection: the VAD cuts the microphone stream at pauses,
    so each utterance is transcribed and labelled as soon as the speaker stops talking."""

    segment = Signal(float, float, str, str)
    level = Signal(float)
    failed = Signal(str)

    def __init__(self, recognizer, device_index, models_dir, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.device_index = device_index
        self.models_dir = models_dir
        self.pipeline: SpeakerPipeline | None = None
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _handle(self, utt: Utterance):
        result = transcribe_utterance(self.recognizer, self.pipeline, utt)
        if result:
            self.segment.emit(utt.start, utt.end, *result)

    def run(self):
        try:
            self.pipeline = SpeakerPipeline.create(self.models_dir)
            try:  # prefer opening the mic at 16 kHz so no resampling is needed
                sd.check_input_settings(device=self.device_index, samplerate=MODEL_SAMPLE_RATE, channels=1)
                rate = MODEL_SAMPLE_RATE
            except Exception:  # noqa: BLE001
                rate = int(sd.query_devices(self.device_index, "input")["default_samplerate"])

            audio_q: queue.Queue = queue.Queue()

            def callback(indata, frames, time_info, status):
                audio_q.put(indata.copy())

            block = max(1, rate // 10)  # feed the VAD in ~100 ms pieces
            pending = np.empty(0, dtype=np.float32)

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
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    self.level.emit(float(np.sqrt(np.mean(data**2))))
                    pending = np.concatenate((pending, data))

                    while len(pending) >= block:
                        piece, pending = pending[:block], pending[block:]
                        audio16 = piece if rate == MODEL_SAMPLE_RATE else to_model_rate(piece, rate)
                        for utt in self.pipeline.feed(audio16):
                            self._handle(utt)

            for utt in self.pipeline.flush():  # whatever was being said when Stop was pressed
                self._handle(utt)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class FileWorker(QThread):
    segment = Signal(float, float, str, str)
    progress = Signal(int)
    failed = Signal(str)

    def __init__(self, recognizer, path, speakers_dir=None, parent=None):
        super().__init__(parent)
        self.recognizer = recognizer
        self.path = path
        self.speakers_dir = speakers_dir  # set -> split at pauses and label speakers
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def _run_speakers(self, audio: np.ndarray):
        pipeline = SpeakerPipeline.create(self.speakers_dir)
        total = len(audio)
        step = MODEL_SAMPLE_RATE  # feed one second at a time so Cancel and progress stay responsive

        def handle(utt: Utterance):
            result = transcribe_utterance(self.recognizer, pipeline, utt)
            if result:
                self.segment.emit(utt.start, utt.end, *result)

        pos = 0
        while pos < total and not self._stop.is_set():
            for utt in pipeline.feed(audio[pos : pos + step]):
                handle(utt)
            pos += step
            self.progress.emit(min(99, int(pos * 100 / total)))
        if not self._stop.is_set():
            for utt in pipeline.flush():
                handle(utt)
        self.progress.emit(100)

    def run(self):
        try:
            audio, sr = sf.read(self.path, dtype="float32")
            audio = to_model_rate(audio, sr)
            if self.speakers_dir is not None:
                self._run_speakers(audio)
                return
            seg_len = FILE_SEGMENT_SECONDS * MODEL_SAMPLE_RATE
            total = len(audio)
            pos = 0
            while pos < total and not self._stop.is_set():
                piece = audio[pos : pos + seg_len]
                start = pos / MODEL_SAMPLE_RATE
                end = (pos + len(piece)) / MODEL_SAMPLE_RATE
                text = self.recognizer.transcribe(piece)
                if text:
                    self.segment.emit(start, end, text, "")
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
QProgressBar#level::chunk { background: #3ddc84; }
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
QCheckBox { color: #dfe3ee; spacing: 8px; }
QCheckBox:disabled { color: #566077; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #2a3047; background: #1d2231; }
QCheckBox::indicator:checked { background: #5b6cff; border-color: #5b6cff; }
QCheckBox::indicator:disabled { background: #171b27; border-color: #1f2434; }
"""

LEVEL_ICON = {"ok": ("✔", "#5fe08f"), "warn": ("⚠", "#f0c05a"), "error": ("✖", "#ff6b7d")}


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
        self.resize(860, 800)

        self.recognizers: dict[str, object] = {}  # loaded engines, kept so switching back is instant
        self.statuses: dict[str, EngineStatus] = {}
        self.engine_key = "onnx"
        self.loading = False
        self.loader: ModelLoader | None = None
        self.downloader: SpeakerModelDownloader | None = None
        self.speaker_status: EngineStatus | None = None
        self.spk_ready = False
        self.segments: list[Segment] = []
        self.engines_used: list[str] = []
        self.live_worker: LiveWorker | None = None
        self.file_worker: FileWorker | None = None
        self.file_path: str | None = None
        self.source_label = ""

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

        # Mode toggle (left) + engine toggle (right)
        toggles = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        mode_bar, (self.btn_live, self.btn_file) = self._make_tab_bar(
            ["🎙  Live", "📁  File"], self.mode_group
        )
        self.btn_live.setChecked(True)
        toggles.addWidget(mode_bar)
        toggles.addStretch()
        engine_caption = QLabel("Engine")
        engine_caption.setObjectName("subtitle")
        toggles.addWidget(engine_caption)
        self.engine_group = QButtonGroup(self)
        engine_bar, engine_btns = self._make_tab_bar(
            [ENGINE_NAMES[k] for k in ENGINE_KEYS], self.engine_group
        )
        self.engine_buttons = dict(zip(ENGINE_KEYS, engine_btns))
        self.engine_buttons[self.engine_key].setChecked(True)
        for b in engine_btns:
            b.setEnabled(False)  # enabled once the availability check has finished
        toggles.addWidget(engine_bar)
        outer.addLayout(toggles)

        # Engine availability details
        self.engine_info = QLabel()
        self.engine_info.setObjectName("engineinfo")
        self.engine_info.setTextFormat(Qt.RichText)
        self.engine_info.setWordWrap(True)
        self.nemo_fix = ""
        info_row = QHBoxLayout()
        info_row.addWidget(self.engine_info, stretch=1)
        self.btn_fix = QPushButton("")
        self.btn_fix.setCursor(Qt.PointingHandCursor)
        self.btn_fix.setVisible(False)
        self.btn_fix.clicked.connect(lambda: self.start_install(self.nemo_fix))
        info_row.addWidget(self.btn_fix, alignment=Qt.AlignTop)
        outer.addLayout(info_row)

        # Speaker detection (VAD + voice matching)
        spk_row = QHBoxLayout()
        self.chk_speakers = QCheckBox("Detect speakers")
        self.chk_speakers.setEnabled(False)  # enabled once the check says it can run
        self.chk_speakers.toggled.connect(self.on_speakers_toggled)
        spk_row.addWidget(self.chk_speakers)
        self.spk_label = QLabel("Checking speaker detection…")
        self.spk_label.setObjectName("engineinfo")
        self.spk_label.setTextFormat(Qt.RichText)
        self.spk_label.setWordWrap(True)
        spk_row.addWidget(self.spk_label, stretch=1)
        self.btn_spk = QPushButton("")
        self.btn_spk.setCursor(Qt.PointingHandCursor)
        self.btn_spk.setVisible(False)
        self.btn_spk.clicked.connect(self.fix_speakers)
        spk_row.addWidget(self.btn_spk)
        outer.addLayout(spk_row)

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
        self.engine_group.idClicked.connect(self.on_engine_clicked)

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
        self.refresh_engine_info()
        self.update_controls()

        # Check which engines can run (imports can be slow, so do it off the UI thread)
        self.set_status("Checking engines…", "busy")
        self.checker = EngineChecker()
        self.checker.result.connect(self.on_engine_checked)
        self.checker.start()

    # ---- widget builders --------------------------------------------------
    def _make_tab_bar(self, labels, group):
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

        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("level")
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        lay.addWidget(self.level_bar)

        self.live_hint = QLabel()
        self.live_hint.setObjectName("subtitle")
        self.live_hint.setWordWrap(True)
        lay.addWidget(self.live_hint)
        self.update_live_hint()
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

    def set_status(self, text: str, state: str):
        self.pill.setText(text)
        self.pill.setProperty("state", state)
        self.pill.style().unpolish(self.pill)
        self.pill.style().polish(self.pill)

    def ready_text(self) -> str:
        rec = self.recognizer
        return f"Ready · {rec.label}" if rec else "Ready"

    def is_busy(self) -> bool:
        return bool(
            (self.live_worker and self.live_worker.isRunning())
            or (self.file_worker and self.file_worker.isRunning())
        )

    def update_controls(self):
        ready = self.recognizer is not None and not self.loading
        live_on = bool(self.live_worker and self.live_worker.isRunning())
        file_on = bool(self.file_worker and self.file_worker.isRunning())
        busy = live_on or file_on

        self.btn_live.setEnabled(not busy)
        self.btn_file.setEnabled(not busy)

        # Engine toggle: only engines that passed the check, and not mid-session / mid-load
        for key, btn in self.engine_buttons.items():
            st = self.statuses.get(key)
            usable = st is not None and st.level != "error"
            btn.setEnabled(usable and not busy and not self.loading)
            mark = "" if st is None or st.level == "ok" else (" ⚠" if st.level == "warn" else " ✖")
            btn.setText(ENGINE_NAMES[key] + mark)
            btn.setToolTip(f"{st.summary} {st.details}".strip() if st else "Checking…")

        self.btn_fix.setEnabled(not busy and not self.loading)

        downloading = bool(self.downloader and self.downloader.isRunning())
        self.chk_speakers.setEnabled(self.spk_ready and not busy and not downloading)
        self.btn_spk.setEnabled(not busy and not downloading)

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
        self.btn_clear.setEnabled(bool(self.segments) and not busy)

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
            lines.append(
                f'<b>{name}</b> <span style="color:{color}">{icon}</span> {escape(st.summary)}{extra}'
            )
        self.engine_info.setText("<br>".join(lines))

        nemo = self.statuses.get("nemo")
        self.nemo_fix = nemo.fix if nemo else ""
        self.btn_fix.setVisible(bool(self.nemo_fix))
        self.btn_fix.setText(
            "Install GPU PyTorch" if self.nemo_fix == "reinstall_torch" else "Install NeMo support"
        )

    # ---- speaker detection ------------------------------------------------
    def speakers_on(self) -> bool:
        return self.spk_ready and self.chk_speakers.isChecked()

    def update_live_hint(self):
        if self.speakers_on():
            self.live_hint.setText(
                "Speech is split at pauses, so each sentence appears when the speaker stops talking, "
                "labelled with who said it."
            )
        else:
            self.live_hint.setText(
                f"Audio is transcribed in {LIVE_CHUNK_SECONDS}-second chunks, "
                "so text appears a few seconds after you speak."
            )

    def on_speakers_toggled(self, _checked: bool):
        self.update_live_hint()

    def refresh_speaker_row(self):
        st = self.speaker_status
        if st is None:
            return
        self.spk_ready = st.level == "ok"
        icon, color = LEVEL_ICON[st.level]
        extra = f" {escape(st.details)}" if st.level != "ok" and st.details else ""
        self.spk_label.setText(f'<span style="color:{color}">{icon}</span> {escape(st.summary)}{extra}')
        self.btn_spk.setVisible(bool(st.fix))
        self.btn_spk.setText("Install sherpa-onnx" if st.fix == "install_sherpa" else "Download speaker models")
        self.update_live_hint()

    def fix_speakers(self):
        fix = self.speaker_status.fix if self.speaker_status else ""
        if fix == "install_sherpa":
            self.start_install("install_sherpa")
        elif fix == "download_models":
            self.start_download()

    def start_download(self):
        self.spk_label.setText("Downloading speaker models… 0%")
        self.downloader = SpeakerModelDownloader()
        self.downloader.progress.connect(
            lambda p: self.spk_label.setText(f"Downloading speaker models… {p}%")
        )
        self.downloader.done.connect(self.on_download_done)
        self.downloader.finished.connect(self.update_controls)  # thread has fully ended -> unlock controls
        self.downloader.start()
        self.update_controls()

    def on_download_done(self, error: str):
        if error:
            QMessageBox.critical(self, "Download failed", error)
        self.speaker_status = check_speakers()
        self.refresh_speaker_row()
        if self.spk_ready:
            self.chk_speakers.setChecked(True)
        self.update_controls()

    # ---- install packages (uv) --------------------------------------------
    def start_install(self, fix: str):
        uv = find_uv()
        if not uv:
            QMessageBox.warning(
                self,
                "uv not found",
                "The installer needs uv, but it wasn't found. Start the app with "
                "`uv run asr_app.py`, or install uv from https://docs.astral.sh/uv/.",
            )
            return
        dlg = InstallDialog(self, install_commands(fix, uv))
        dlg.exec()
        if dlg.success:
            reply = QMessageBox.question(
                self,
                "Restart needed",
                "The packages were installed. Restart the app now to start using them?",
            )
            if reply == QMessageBox.Yes:
                os.execv(sys.executable, [sys.executable, *sys.argv])

    def on_engine_checked(self, key: str, status: EngineStatus):
        if key == "speakers":
            self.speaker_status = status
            self.refresh_speaker_row()
            self.update_controls()
            return
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
        name = ENGINE_NAMES[key]
        st = self.statuses.get(key)
        if self.loading:
            pass
        elif key in self.recognizers:
            self.set_status(self.ready_text(), "ready")
        elif st is None:
            self.set_status("Checking engines…", "busy")
        elif st.level == "error":
            self.set_status(f"{name} unavailable", "error")
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
            f"Could not load the {ENGINE_NAMES[key]} model:\n\n{msg}\n\n"
            "Click the engine button again to retry.",
        )

    def note_engine(self):
        label = self.recognizer.label
        if label not in self.engines_used:
            self.engines_used.append(label)

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
        self.note_engine()
        if self.speakers_on():
            self.live_worker = LiveSpeakerWorker(self.recognizer, device, SPEAKER_MODELS_DIR)
        else:
            self.live_worker = LiveWorker(self.recognizer, device)
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
        self.note_engine()
        self.file_progress.setValue(0)
        self.file_worker = FileWorker(
            self.recognizer, self.file_path, SPEAKER_MODELS_DIR if self.speakers_on() else None
        )
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
        self.set_status(self.ready_text(), "ready")
        self.update_controls()

    # ---- transcript -------------------------------------------------------
    def add_segment(self, start: float, end: float, text: str, speaker: str = ""):
        self.segments.append(Segment(start, end, text, speaker))
        who = ""
        if speaker:
            n = int(speaker.rsplit(" ", 1)[-1]) if speaker.rsplit(" ", 1)[-1].isdigit() else 1
            color = SPEAKER_COLORS[(n - 1) % len(SPEAKER_COLORS)]
            who = f'<b style="color:{color}">{escape(speaker)}:</b> '
        self.text.append(
            f'<span style="color:#7c8cff">[{fmt_time(start)}]</span> {who}{escape(text)}'
        )
        n = len(self.segments)
        self.count_label.setText(f"{n} segment{'s' if n != 1 else ''}")
        self.update_controls()

    def clear_transcript(self):
        self.segments.clear()
        self.engines_used.clear()
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
            f"Engine: {', '.join(self.engines_used) or 'unknown'}",
            f"Saved: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "-" * 40,
        ]
        if any(seg.speaker for seg in self.segments):
            lines.insert(3, "Speakers: detected automatically (labels are guesses based on voice similarity)")
        for seg in self.segments:
            who = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"[{fmt_time(seg.start)} - {fmt_time(seg.end)}] {who}{seg.text}")
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
        if self.downloader and self.downloader.isRunning():
            self.downloader.stop()
        for t in (self.checker, self.loader, self.downloader):
            if t and t.isRunning():
                if not t.wait(3000):  # an import / model load can't be interrupted politely
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
