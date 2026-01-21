"""ML-based audio classification using audio features and scikit-learn.

This module provides a trainable classifier that can learn to distinguish
between voice, fax, modem, busy signals, and other telephony audio types.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .signatures import LineType


@dataclass
class AudioFeatures:
    """Extracted features from an audio sample."""
    # Spectral features
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_flatness: float

    # Energy features
    rms_energy: float
    energy_variance: float
    zero_crossing_rate: float

    # Frequency features
    dominant_freq: float
    num_peaks: int
    freq_spread: float

    # Temporal features
    energy_entropy: float

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML."""
        return np.array([
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.spectral_rolloff,
            self.spectral_flatness,
            self.rms_energy,
            self.energy_variance,
            self.zero_crossing_rate,
            self.dominant_freq,
            self.num_peaks,
            self.freq_spread,
            self.energy_entropy,
        ])


class FeatureExtractor:
    """Extracts audio features for ML classification."""

    def __init__(self, sample_rate: int = 8000, frame_size: int = 512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size

    def extract(self, samples: np.ndarray) -> AudioFeatures:
        """Extract features from audio samples."""
        # Ensure samples are float and normalized
        if samples.dtype != np.float32 and samples.dtype != np.float64:
            samples = samples.astype(np.float32)
        if np.max(np.abs(samples)) > 1.0:
            samples = samples / np.max(np.abs(samples))

        # Compute FFT
        n = len(samples)
        fft_result = np.abs(rfft(samples))
        freqs = rfftfreq(n, 1 / self.sample_rate)

        # Spectral centroid (center of mass of spectrum)
        if np.sum(fft_result) > 0:
            spectral_centroid = np.sum(freqs * fft_result) / np.sum(fft_result)
        else:
            spectral_centroid = 0.0

        # Spectral bandwidth (spread around centroid)
        if np.sum(fft_result) > 0:
            spectral_bandwidth = np.sqrt(
                np.sum(((freqs - spectral_centroid) ** 2) * fft_result) / np.sum(fft_result)
            )
        else:
            spectral_bandwidth = 0.0

        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(fft_result)
        if cumsum[-1] > 0:
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]
        else:
            spectral_rolloff = 0.0

        # Spectral flatness (how noise-like vs tone-like)
        geometric_mean = np.exp(np.mean(np.log(fft_result + 1e-10)))
        arithmetic_mean = np.mean(fft_result)
        if arithmetic_mean > 0:
            spectral_flatness = geometric_mean / arithmetic_mean
        else:
            spectral_flatness = 0.0

        # RMS energy
        rms_energy = np.sqrt(np.mean(samples ** 2))

        # Energy variance (computed over frames)
        frame_energies = []
        for i in range(0, len(samples) - self.frame_size, self.frame_size):
            frame = samples[i:i + self.frame_size]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))
        energy_variance = np.var(frame_energies) if frame_energies else 0.0

        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples)))) / 2
        zero_crossing_rate = zero_crossings / len(samples)

        # Dominant frequency
        if len(fft_result) > 0:
            dominant_idx = np.argmax(fft_result)
            dominant_freq = freqs[dominant_idx]
        else:
            dominant_freq = 0.0

        # Number of significant peaks
        threshold = np.max(fft_result) * 0.1 if len(fft_result) > 0 else 0
        peaks, _ = signal.find_peaks(fft_result, height=threshold)
        num_peaks = len(peaks)

        # Frequency spread (std of frequencies weighted by magnitude)
        if np.sum(fft_result) > 0:
            freq_spread = np.sqrt(
                np.sum(((freqs - np.mean(freqs)) ** 2) * fft_result) / np.sum(fft_result)
            )
        else:
            freq_spread = 0.0

        # Energy entropy (measure of energy distribution)
        if len(frame_energies) > 0 and np.sum(frame_energies) > 0:
            probs = np.array(frame_energies) / np.sum(frame_energies)
            probs = probs[probs > 0]  # Remove zeros for log
            energy_entropy = -np.sum(probs * np.log2(probs))
        else:
            energy_entropy = 0.0

        return AudioFeatures(
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            rms_energy=rms_energy,
            energy_variance=energy_variance,
            zero_crossing_rate=zero_crossing_rate,
            dominant_freq=dominant_freq,
            num_peaks=num_peaks,
            freq_spread=freq_spread,
            energy_entropy=energy_entropy,
        )


class MLClassifier:
    """
    ML-based audio classifier using Random Forest.

    Can be trained with labeled audio samples to distinguish between
    voice, fax, modem, busy signals, etc.
    """

    # Default model path
    DEFAULT_MODEL_PATH = Path(__file__).parent / "trained_model.pkl"

    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_map: dict[int, LineType] = {}
        self.reverse_label_map: dict[LineType, int] = {}

    def train(
        self,
        samples_by_type: dict[LineType, list[np.ndarray]],
        n_estimators: int = 100
    ) -> float:
        """
        Train the classifier with labeled audio samples.

        Args:
            samples_by_type: Dictionary mapping LineType to list of audio samples
            n_estimators: Number of trees in Random Forest

        Returns:
            Training accuracy
        """
        # Extract features from all samples
        X = []
        y = []

        # Create label mapping
        self.label_map = {i: lt for i, lt in enumerate(samples_by_type.keys())}
        self.reverse_label_map = {lt: i for i, lt in self.label_map.items()}

        for line_type, sample_list in samples_by_type.items():
            label = self.reverse_label_map[line_type]
            for samples in sample_list:
                features = self.feature_extractor.extract(samples)
                X.append(features.to_array())
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        # Return training accuracy
        return self.model.score(X_scaled, y)

    def classify(self, samples: np.ndarray) -> tuple[LineType, float]:
        """
        Classify audio samples.

        Args:
            samples: Audio samples (float array)

        Returns:
            Tuple of (LineType, confidence)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call train() first or load a model.")

        # Extract features
        features = self.feature_extractor.extract(samples)
        X = features.to_array().reshape(1, -1)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        confidence = probabilities[prediction]

        return self.label_map[prediction], confidence

    def save(self, path: Optional[Path] = None) -> None:
        """Save trained model to file."""
        path = path or self.DEFAULT_MODEL_PATH
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_map": self.label_map,
            "reverse_label_map": self.reverse_label_map,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Optional[Path] = None) -> bool:
        """Load trained model from file."""
        path = path or self.DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.label_map = data["label_map"]
        self.reverse_label_map = data["reverse_label_map"]
        return True

    @property
    def is_trained(self) -> bool:
        """Check if model is trained or loaded."""
        return self.model is not None and self.scaler is not None


def create_default_training_data() -> dict[LineType, list[np.ndarray]]:
    """
    Create synthetic training data for common telephony signals.

    This provides a baseline model. For better accuracy, train with
    real recorded samples.
    """
    sample_rate = 8000
    duration = 2.0  # 2 seconds per sample
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)

    training_data: dict[LineType, list[np.ndarray]] = {}

    # Dial tone (350 + 440 Hz, continuous)
    dial_tones = []
    for _ in range(20):
        noise = np.random.normal(0, 0.02, n_samples)
        tone = 0.5 * (np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 440 * t)) + noise
        dial_tones.append(tone.astype(np.float32))
    training_data[LineType.DIAL_TONE] = dial_tones

    # Busy signal (480 + 620 Hz, pulsed)
    busy_tones = []
    for _ in range(20):
        noise = np.random.normal(0, 0.02, n_samples)
        # Pulse on/off at 0.5s intervals
        pulse = np.where((t % 1.0) < 0.5, 1.0, 0.0)
        tone = 0.5 * pulse * (np.sin(2 * np.pi * 480 * t) + np.sin(2 * np.pi * 620 * t)) + noise
        busy_tones.append(tone.astype(np.float32))
    training_data[LineType.BUSY] = busy_tones

    # Fax CED tone (2100 Hz)
    fax_tones = []
    for _ in range(20):
        noise = np.random.normal(0, 0.02, n_samples)
        tone = 0.7 * np.sin(2 * np.pi * 2100 * t) + noise
        fax_tones.append(tone.astype(np.float32))
    training_data[LineType.FAX] = fax_tones

    # Modem (2100 Hz + harmonics, more complex)
    modem_tones = []
    for _ in range(20):
        noise = np.random.normal(0, 0.03, n_samples)
        tone = (0.4 * np.sin(2 * np.pi * 2100 * t) +
                0.2 * np.sin(2 * np.pi * 1200 * t) +
                0.2 * np.sin(2 * np.pi * 2400 * t)) + noise
        modem_tones.append(tone.astype(np.float32))
    training_data[LineType.MODEM] = modem_tones

    # Voice (simulated with multiple frequencies + noise + variation)
    voice_samples = []
    for _ in range(30):  # More voice samples for better detection
        # Voice has varying frequencies and amplitude
        fundamental = np.random.uniform(100, 300)  # Fundamental frequency
        harmonics = [fundamental * i for i in range(1, 6)]

        # Create voice-like signal with varying amplitude
        envelope = 0.3 + 0.4 * np.sin(2 * np.pi * 3 * t) + 0.2 * np.sin(2 * np.pi * 7 * t)
        envelope = np.clip(envelope, 0, 1)

        signal_voice = np.zeros(n_samples)
        for i, h in enumerate(harmonics):
            amp = 0.3 / (i + 1)  # Decreasing amplitude for harmonics
            signal_voice += amp * np.sin(2 * np.pi * h * t + np.random.uniform(0, 2 * np.pi))

        signal_voice = signal_voice * envelope
        noise = np.random.normal(0, 0.05, n_samples)
        signal_voice = signal_voice + noise

        # Normalize
        signal_voice = signal_voice / np.max(np.abs(signal_voice)) * 0.8
        voice_samples.append(signal_voice.astype(np.float32))
    training_data[LineType.VOICE] = voice_samples

    # Silence
    silence_samples = []
    for _ in range(20):
        noise = np.random.normal(0, 0.005, n_samples)
        silence_samples.append(noise.astype(np.float32))
    training_data[LineType.SILENCE] = silence_samples

    # Ringback (440 + 480 Hz, 2s on, 4s off pattern)
    ringback_tones = []
    for _ in range(20):
        noise = np.random.normal(0, 0.02, n_samples)
        # Simplified ringback pattern
        pulse = np.where((t % 2.0) < 1.0, 1.0, 0.0)
        tone = 0.5 * pulse * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) + noise
        ringback_tones.append(tone.astype(np.float32))
    training_data[LineType.RINGBACK] = ringback_tones

    return training_data
