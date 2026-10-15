import queue
import numpy as np
import sounddevice as sd
import onnx_asr
from scipy.signal import resample_poly


# Settings
INPUT_SAMPLE_RATE = 48000   # Change this if your microphone uses another rate
MODEL_SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 4
OVERLAP_SECONDS = 0.75

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")

    audio_queue.put(indata.copy())


print("Loading Pianissimo ONNX model...")

model = onnx_asr.load_model(
    "KlangAI/pianissimo-sv-onnx",
    quantization="int8",
)

print("Model loaded.")
print("Starting microphone...")
print("Press Ctrl+C to stop.\n")


chunk_size = int(INPUT_SAMPLE_RATE * CHUNK_SECONDS)
overlap_size = int(INPUT_SAMPLE_RATE * OVERLAP_SECONDS)

audio_buffer = np.empty(0, dtype=np.float32)


try:
    with sd.InputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=0,
    ):
        while True:

            # Get audio from microphone
            data = audio_queue.get()

            # Convert to mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            audio_buffer = np.concatenate(
                (audio_buffer, data)
            )

            # Wait until we have enough audio
            if len(audio_buffer) < chunk_size:
                continue

            # Take a chunk
            chunk = audio_buffer[:chunk_size]

            # Keep overlap for the next chunk
            audio_buffer = audio_buffer[
                chunk_size - overlap_size:
            ]

            # Resample 48 kHz -> 16 kHz
            chunk = resample_poly(
                chunk,
                MODEL_SAMPLE_RATE,
                INPUT_SAMPLE_RATE,
            ).astype(np.float32)

            print("Transcribing...", end="\r")

            result = model.recognize(
                chunk,
                sample_rate=MODEL_SAMPLE_RATE,
            )

            # Print transcription
            print(f"                 \r{result}")


except KeyboardInterrupt:
    print("\nStopped.")
