"""FFT-based frequency analysis for audio classification."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq


@dataclass
class FrequencyPeak:
    """A detected frequency peak."""
    frequency: float  # Hz
    magnitude: float  # Normalized magnitude
    rounded: int  # Rounded to resolution


@dataclass
class FrameAnalysis:
    """Analysis results for a single time frame."""
    start_time: float  # Seconds
    end_time: float  # Seconds
    peaks: list[FrequencyPeak]
    dominant_frequency: float
    energy: float


@dataclass
class AudioAnalysis:
    """Complete analysis of an audio recording."""
    duration: float
    sample_rate: int
    frames: list[FrameAnalysis]
    all_frequencies: list[int]  # All rounded frequencies across all frames
    frequency_histogram: dict[int, int]  # Frequency -> count


class FFTAnalyzer:
    """
    Performs FFT analysis on audio to extract frequency information.

    Uses the following approach:
    1. Divide audio into 1-second frames
    2. Apply FFT to each frame
    3. Extract top N frequency peaks per frame
    4. Round frequencies to specified resolution
    5. Build frequency histogram for classification
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        window_size: int = 1024,
        top_frequencies: int = 5,
        frequency_resolution: int = 100,
        min_frequency: int = 200,
        max_frequency: int = 4000
    ):
        """
        Initialize FFT analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            window_size: FFT window size in samples
            top_frequencies: Number of top frequencies to extract per frame
            frequency_resolution: Round frequencies to this value (Hz)
            min_frequency: Minimum frequency to consider (Hz)
            max_frequency: Maximum frequency to consider (Hz)
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.top_frequencies = top_frequencies
        self.frequency_resolution = frequency_resolution
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        # Pre-compute frequency bins
        self._freq_bins = rfftfreq(window_size, 1 / sample_rate)

    def _round_frequency(self, freq: float) -> int:
        """Round frequency to specified resolution."""
        return int(round(freq / self.frequency_resolution) * self.frequency_resolution)

    def _compute_fft(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of samples.

        Args:
            samples: Audio samples

        Returns:
            Tuple of (frequencies, magnitudes)
        """
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(samples))
        windowed = samples * window

        # Compute FFT
        fft_result = rfft(windowed)
        magnitudes = np.abs(fft_result)

        # Normalize
        magnitudes = magnitudes / len(samples)

        # Get corresponding frequencies
        freqs = rfftfreq(len(samples), 1 / self.sample_rate)

        return freqs, magnitudes

    def _find_peaks(
        self,
        freqs: np.ndarray,
        magnitudes: np.ndarray
    ) -> list[FrequencyPeak]:
        """
        Find top frequency peaks.

        Args:
            freqs: Frequency array
            magnitudes: Magnitude array

        Returns:
            List of FrequencyPeak objects, sorted by magnitude
        """
        # Filter to frequency range of interest
        mask = (freqs >= self.min_frequency) & (freqs <= self.max_frequency)
        filtered_freqs = freqs[mask]
        filtered_mags = magnitudes[mask]

        if len(filtered_mags) == 0:
            return []

        # Find peaks using scipy
        peak_indices, properties = signal.find_peaks(
            filtered_mags,
            height=np.max(filtered_mags) * 0.1,  # At least 10% of max
            distance=5  # Minimum distance between peaks
        )

        if len(peak_indices) == 0:
            # No peaks found, use highest magnitudes
            peak_indices = np.argsort(filtered_mags)[-self.top_frequencies:]

        # Get peak frequencies and magnitudes
        peaks = []
        for idx in peak_indices:
            freq = filtered_freqs[idx]
            mag = filtered_mags[idx]
            peaks.append(FrequencyPeak(
                frequency=float(freq),
                magnitude=float(mag),
                rounded=self._round_frequency(freq)
            ))

        # Sort by magnitude and take top N
        peaks.sort(key=lambda p: p.magnitude, reverse=True)
        return peaks[:self.top_frequencies]

    def analyze_frame(
        self,
        samples: np.ndarray,
        start_time: float
    ) -> FrameAnalysis:
        """
        Analyze a single frame of audio.

        Args:
            samples: Audio samples for this frame
            start_time: Start time of frame in seconds

        Returns:
            FrameAnalysis object
        """
        duration = len(samples) / self.sample_rate
        freqs, magnitudes = self._compute_fft(samples)
        peaks = self._find_peaks(freqs, magnitudes)

        # Find dominant frequency
        if len(peaks) > 0:
            dominant_freq = peaks[0].frequency
        else:
            # Use overall max if no peaks
            max_idx = np.argmax(magnitudes)
            dominant_freq = float(freqs[max_idx])

        # Calculate frame energy
        energy = float(np.sum(samples ** 2) / len(samples))

        return FrameAnalysis(
            start_time=start_time,
            end_time=start_time + duration,
            peaks=peaks,
            dominant_frequency=dominant_freq,
            energy=energy
        )

    def analyze(self, samples: np.ndarray) -> AudioAnalysis:
        """
        Perform complete analysis of audio recording.

        Divides audio into 1-second frames and analyzes each.

        Args:
            samples: Complete audio samples

        Returns:
            AudioAnalysis object with all frame results
        """
        duration = len(samples) / self.sample_rate
        frame_size = self.sample_rate  # 1 second frames

        frames = []
        all_frequencies = []
        frequency_counts: dict[int, int] = {}

        # Process each 1-second frame
        for i in range(0, len(samples), frame_size):
            frame_samples = samples[i:i + frame_size]

            # Skip very short frames
            if len(frame_samples) < self.window_size:
                continue

            start_time = i / self.sample_rate
            frame_analysis = self.analyze_frame(frame_samples, start_time)
            frames.append(frame_analysis)

            # Collect frequencies
            for peak in frame_analysis.peaks:
                rounded = peak.rounded
                all_frequencies.append(rounded)
                frequency_counts[rounded] = frequency_counts.get(rounded, 0) + 1

        return AudioAnalysis(
            duration=duration,
            sample_rate=self.sample_rate,
            frames=frames,
            all_frequencies=all_frequencies,
            frequency_histogram=frequency_counts
        )

    def get_prominent_frequencies(
        self,
        analysis: AudioAnalysis,
        min_occurrences: int = 2
    ) -> list[tuple[int, int]]:
        """
        Get frequencies that appear multiple times across frames.

        Args:
            analysis: AudioAnalysis from analyze()
            min_occurrences: Minimum times a frequency must appear

        Returns:
            List of (frequency, count) tuples, sorted by count descending
        """
        prominent = [
            (freq, count) for freq, count in analysis.frequency_histogram.items()
            if count >= min_occurrences
        ]
        return sorted(prominent, key=lambda x: x[1], reverse=True)

    def compute_spectrum(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute full frequency spectrum for visualization.

        Args:
            samples: Audio samples

        Returns:
            Tuple of (frequencies, magnitudes in dB)
        """
        freqs, magnitudes = self._compute_fft(samples)

        # Convert to dB scale
        magnitudes_db = 20 * np.log10(magnitudes + 1e-10)

        return freqs, magnitudes_db
