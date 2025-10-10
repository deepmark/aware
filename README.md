# AWARE: Audio Watermarking via Adversarial Resistance to Edits

## Installation
```bash
git clone https://github.com/deepmarkpy/aware.git
cd ./aware
python -m pip install -e .
```

## Basic Usage
```python
import numpy as np
import librosa
import soundfile
from aware.utils.models import load
from aware.service import embed_watermark, detect_watermark
from aware.metrics.audio import BER, PESQ

# 1.load model
embedder, detector = load()

# 2.create 20-bit watermark
watermark_bits = np.random.randint(0, 2, size=20, dtype=np.int32)


# 3.read host audio
# the audio should be sampled at 16kHz, you can read it using librosa:
signal, sample_rate = librosa.load("example.wav", sr=None, mono=True)


# 4.embed watermark
watermarked_signal = embed_watermark(signal, 16000, watermark_bits, embedder)
# you can save it as a new one:
# soundfile.write("output.wav", watermarked_signal, 16000)


# 5.detect watermark
detected_pattern = detect_watermark(watermarked_signal, 16000, detector)


# 6.check accuracy and perceptual quality
ber_metric = BER()
ber = ber_metric(watermark_bits, detected_pattern)
print(f"BER: {ber:.2f}")

pesq_metric = PESQ()
pesq = pesq_metric(watermarked_signal, signal, 16000)
print(f"PESQ: {pesq:.2f}")
```

## License

This project is licensed under the terms of the MIT license.
