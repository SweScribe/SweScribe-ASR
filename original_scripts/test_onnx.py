import numpy as np
import soundfile as sf
import onnx_asr


INPUT_FILE = "test.wav"
TARGET_SAMPLE_RATE = 16000


print("Loading audio...")

audio, sample_rate = sf.read(INPUT_FILE, dtype="float32")

# Convert stereo/multichannel audio to mono
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

# Resample to 16 kHz if necessary
if sample_rate != TARGET_SAMPLE_RATE:
    print(f"Resampling {sample_rate} Hz -> {TARGET_SAMPLE_RATE} Hz")

    import scipy.signal

    target_length = round(len(audio) * TARGET_SAMPLE_RATE / sample_rate)

    audio = scipy.signal.resample(
        audio,
        target_length
    ).astype(np.float32)

    sample_rate = TARGET_SAMPLE_RATE

print(f"Sample rate: {sample_rate} Hz")
print("Channels: mono")
print(f"Samples: {len(audio)}")

print("Loading Pianissimo...")

model = onnx_asr.load_model(
    "KlangAI/pianissimo-sv-onnx",
    quantization="int8",
)

print("Transcribing...")

result = model.recognize(
    audio,
    sample_rate=sample_rate,
)

print("\n--- Transcription ---")
print(result)
