"""Audio capture and processing from modem voice mode."""

import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np


# DLE (Data Link Escape) constants
DLE = 0x10
ETX = 0x03


def decode_dle_data(raw_data: bytes) -> bytes:
    """
    Decode DLE-shielded voice data from modem.

    In voice mode, modems use DLE shielding:
    - DLE (0x10) followed by another byte is a control sequence
    - DLE DLE (0x10 0x10) represents a literal 0x10 in the data
    - DLE ETX (0x10 0x03) marks end of data stream
    - Other DLE sequences are modem events (silence, etc.)

    Args:
        raw_data: Raw bytes from modem with DLE shielding

    Returns:
        Decoded audio samples
    """
    decoded = bytearray()
    i = 0
    length = len(raw_data)

    while i < length:
        byte = raw_data[i]

        if byte == DLE and i + 1 < length:
            next_byte = raw_data[i + 1]

            if next_byte == DLE:
                # Escaped DLE - output single DLE
                decoded.append(DLE)
                i += 2
            elif next_byte == ETX:
                # End of transmission
                break
            else:
                # Other DLE code (modem event) - skip it
                # Common codes: 's' = silence, 'q' = quiet, etc.
                i += 2
        else:
            # Regular data byte
            decoded.append(byte)
            i += 1

    return bytes(decoded)


def pcm_to_linear(pcm_data: bytes, signed: bool = False) -> np.ndarray:
    """
    Convert 8-bit PCM data to normalized linear samples.

    Args:
        pcm_data: Raw 8-bit PCM bytes
        signed: Whether data is signed (True) or unsigned (False)

    Returns:
        NumPy array of float samples in range [-1.0, 1.0]
    """
    if signed:
        # Signed 8-bit
        samples = np.frombuffer(pcm_data, dtype=np.int8)
        return samples.astype(np.float32) / 128.0
    else:
        # Unsigned 8-bit (more common for modem voice)
        samples = np.frombuffer(pcm_data, dtype=np.uint8)
        return (samples.astype(np.float32) - 128.0) / 128.0


def ulaw_to_linear(ulaw_data: bytes) -> np.ndarray:
    """
    Convert u-law encoded audio to linear samples.

    Some modems use u-law compression for voice data.

    Args:
        ulaw_data: u-law encoded bytes

    Returns:
        NumPy array of float samples in range [-1.0, 1.0]
    """
    # u-law decoding table
    ULAW_BIAS = 0x84
    ULAW_CLIP = 32635

    samples = []
    for byte in ulaw_data:
        # Complement and extract sign/exponent/mantissa
        byte = ~byte & 0xFF
        sign = byte & 0x80
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F

        # Decode
        sample = (mantissa << 3) + ULAW_BIAS
        sample <<= exponent
        sample -= ULAW_BIAS

        if sign:
            sample = -sample

        samples.append(sample)

    arr = np.array(samples, dtype=np.float32)
    return arr / 32768.0  # Normalize to [-1, 1]


def save_wav(
    samples: np.ndarray,
    filepath: Path,
    sample_rate: int = 8000,
    bits: int = 16
) -> None:
    """
    Save audio samples to WAV file.

    Args:
        samples: Audio samples (float array, normalized to [-1, 1])
        filepath: Output file path
        sample_rate: Sample rate in Hz
        bits: Bits per sample (8 or 16)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert to integer samples
    if bits == 16:
        int_samples = (samples * 32767).astype(np.int16)
        sample_width = 2
    else:
        int_samples = ((samples + 1.0) * 127.5).astype(np.uint8)
        sample_width = 1

    with wave.open(str(filepath), 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(int_samples.tobytes())


def load_wav(filepath: Path) -> tuple[np.ndarray, int]:
    """
    Load audio samples from WAV file.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (samples as float array, sample rate)
    """
    with wave.open(str(filepath), 'rb') as wav:
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        raw_data = wav.readframes(n_frames)

    if sample_width == 2:
        samples = np.frombuffer(raw_data, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0, sample_rate
    elif sample_width == 1:
        samples = np.frombuffer(raw_data, dtype=np.uint8)
        return (samples.astype(np.float32) - 128.0) / 128.0, sample_rate
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")


class AudioCapture:
    """Handles audio capture from modem and conversion to usable format."""

    def __init__(
        self,
        sample_rate: int = 8000,
        encoding: str = "pcm_unsigned"
    ):
        """
        Initialize audio capture handler.

        Args:
            sample_rate: Expected sample rate from modem
            encoding: Audio encoding ("pcm_unsigned", "pcm_signed", "ulaw")
        """
        self.sample_rate = sample_rate
        self.encoding = encoding

    def process_raw_data(self, raw_data: bytes) -> np.ndarray:
        """
        Process raw modem voice data into audio samples.

        Args:
            raw_data: Raw bytes from modem (DLE-encoded)

        Returns:
            NumPy array of normalized float samples
        """
        # First decode DLE shielding
        decoded = decode_dle_data(raw_data)

        if not decoded:
            return np.array([], dtype=np.float32)

        # Convert to linear samples based on encoding
        if self.encoding == "pcm_unsigned":
            return pcm_to_linear(decoded, signed=False)
        elif self.encoding == "pcm_signed":
            return pcm_to_linear(decoded, signed=True)
        elif self.encoding == "ulaw":
            return ulaw_to_linear(decoded)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def save(
        self,
        samples: np.ndarray,
        filepath: Path,
        bits: int = 16
    ) -> None:
        """Save captured audio to WAV file."""
        save_wav(samples, filepath, self.sample_rate, bits)


def get_audio_duration(samples: np.ndarray, sample_rate: int) -> float:
    """Calculate duration in seconds for audio samples."""
    return len(samples) / sample_rate


def trim_silence(
    samples: np.ndarray,
    threshold: float = 0.01,
    min_duration: float = 0.1,
    sample_rate: int = 8000
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        samples: Audio samples
        threshold: Amplitude threshold for silence detection
        min_duration: Minimum duration in seconds to keep
        sample_rate: Sample rate for duration calculation

    Returns:
        Trimmed audio samples
    """
    # Find first non-silent sample
    abs_samples = np.abs(samples)
    non_silent = abs_samples > threshold

    if not np.any(non_silent):
        return samples  # All silence, return as-is

    first_idx = np.argmax(non_silent)
    last_idx = len(samples) - np.argmax(non_silent[::-1])

    # Ensure minimum duration
    min_samples = int(min_duration * sample_rate)
    if last_idx - first_idx < min_samples:
        return samples

    return samples[first_idx:last_idx]
