from typing import Tuple, Optional
import torch
from aware.interfaces.audio import BaseAudioProcessor


class STFT(BaseAudioProcessor):
    """
    Short-time Fourier transform

    Args:
        data: torch.Tensor of shape (batch_size, channels, time)

    Returns:
        torch.Tensor of shape (batch_size, channels, freq, time) (complex)
    """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_type = window
        self.win_length = win_length

    def _make_window(self, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
        # If input is complex, choose the real dtype (window must be real)
        if dtype is None:
            dtype = torch.float32
        if self.window_type == "hann":
            return torch.hann_window(self.win_length, device=device, dtype=dtype)
        elif self.window_type == "hamming":
            return torch.hamming_window(self.win_length, device=device, dtype=dtype)
        else:
            raise ValueError(f"Invalid window type: {self.window_type}")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Determine window dtype: if data is complex use its real part dtype
        dtype = data.real.dtype if torch.is_complex(data) else data.dtype
        window = self._make_window(data.device, dtype)
        return torch.stft(
            data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            window=window,
            return_complex=True,
        )


class ISTFT(BaseAudioProcessor):
    """
    Inverse Short-time Fourier transform

    Accepts a complex STFT tensor and returns a time-domain signal.
    """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, window: str = "hann", win_length: int = 2048):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_type = window
        self.win_length = win_length

    def _make_window(self, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
        # Window must be a real floating dtype; if input is complex, use its real dtype
        if dtype is None:
            dtype = torch.float32
        if self.window_type == "hann":
            return torch.hann_window(self.win_length, device=device, dtype=dtype)
        elif self.window_type == "hamming":
            return torch.hamming_window(self.win_length, device=device, dtype=dtype)
        else:
            raise ValueError(f"Invalid window type: {self.window_type}")

    def __call__(self, data: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            data: complex STFT tensor (e.g. returned by torch.stft with return_complex=True)
            length: optional desired length of the output signal (passed to torch.istft)
        Returns:
            time-domain tensor (float) on the same device as `data`
        """
        dtype = data.real.dtype if torch.is_complex(data) else data.dtype
        window = self._make_window(data.device, dtype)
        return torch.istft(
            data,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            window=window,
            length=length,
        )


class STFTDecomposer(BaseAudioProcessor):
    """
    Decompose STFT into magnitude and phase
    """
    def __call__(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.abs(data), torch.angle(data)


class STFTAssembler(BaseAudioProcessor):
    """
    Assemble magnitude and phase into complex tensor
    """
    def __call__(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # magnitude should be real, phase real -> produce complex STFT
        return magnitude * torch.exp(1j * phase)


class STFTNormalizer(BaseAudioProcessor):
    """Per-sample normalization: (B,T) → normalized (B,T)."""
    def __init__(self, eps: float = 1e-8): self.eps = eps
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data / (torch.abs(data).reshape(data.shape[0], -1).amax(dim=1, keepdim=True) + self.eps)
    