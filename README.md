# AWARE: Audio Watermarking via Adversarial Resistance to Edits

## Installation
```bash
pip install git+https://github.com/deepmark/aware
```

AWARE runs on Python 3.9 to 3.12. `webrtcvad` (and, on Python 3.12, `matplotlib`)
is built from source during the install, so a C compiler and the Python headers
are needed (`sudo apt-get install build-essential python3-dev` on Debian/Ubuntu).

For development, install from a clone in editable mode:
```bash
git clone https://github.com/deepmark/aware.git
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

# 1. load model
# `name` selects the configuration profile:
#   - "AWARE" (default): embedding is computed over the full sequence
#   - "AWARE(20bps)": embedding is computed per second of audio
embedder, detector = load(name="AWARE")

# 2.create 20-bit watermark
watermark_bits = np.random.randint(0, 2, size=20, dtype=np.int32)


# 3.read host audio
# AWARE works at 16 kHz; librosa resamples the file on load:
signal, sample_rate = librosa.load("example.wav", sr=16000, mono=True)


# 4.embed watermark
watermarked_signal = embed_watermark(signal, 16000, watermark_bits, embedder)
# you can save it as a new one:
# soundfile.write("output.wav", watermarked_signal, 16000)


# 5. detect watermark
# Detect the watermark from the audio. The function returns:
#   - detected_pattern: the recovered watermark bits
#   - confidence: a value in [0, 1] representing the estimated probability;
#                 that the signal contains a watermark.
# If confidence >= tau (default tau = 0.5), the signal is considered watermarked;
# otherwise, it is treated as unwatermarked.
detected_pattern, confidence = detect_watermark(watermarked_signal, 16000, detector)


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

## Citation
If you find this repository useful, we’d greatly appreciate it if you could give it a star ⭐.

Please cite our work if you use it in your research.

```
@article{pavlović2025aware,
      title={AWARE: Audio Watermarking with Adversarial Resistance to Edits}, 
      author={Kosta Pavlović and Lazar Stanarević and Petar Nedić and Slavko Kovačević and Igor Djurović},
      year={2025},
      eprint={2510.17512},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.17512}, 
}
```
