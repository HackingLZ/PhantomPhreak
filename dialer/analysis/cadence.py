"""Tone cadence (timing pattern) detection.

Detects timing patterns for various telephony tones:
- Busy signal: 480+620 Hz, 0.5s on / 0.5s off
- Ringback: 440+480 Hz, 2s on / 4s off
- Reorder/fast busy: 480+620 Hz, 0.25s on / 0.25s off
- Off-hook warning: 1400+2060+2450+2600 Hz, 0.1s on / 0.1s off
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class CadencePattern(Enum):
    """Known cadence patterns."""
    BUSY = "busy"  # 0.5s on / 0.5s off
    RINGBACK = "ringback"  # 2s on / 4s off
    REORDER = "reorder"  # 0.25s on / 0.25s off (fast busy)
    OFF_HOOK = "off_hook"  # 0.1s on / 0.1s off
    CONTINUOUS = "continuous"  # Always on (dial tone, fax, modem)
    UNKNOWN = "unknown"


@dataclass
class CadenceResult:
    """Result of cadence analysis."""
    pattern: CadencePattern
    confidence: float
    on_duration: float  # Average on duration in seconds
    off_duration: float  # Average off duration in seconds
    cycle_count: int  # Number of detected cycles
    reasoning: str


# Known cadence definitions (on_time, off_time, tolerance)
CADENCE_DEFINITIONS = {
    CadencePattern.BUSY: (0.5, 0.5, 0.15),
    CadencePattern.RINGBACK: (2.0, 4.0, 0.5),
    CadencePattern.REORDER: (0.25, 0.25, 0.1),
    CadencePattern.OFF_HOOK: (0.1, 0.1, 0.05),
}


class CadenceDetector:
    """Detects tone timing patterns in audio."""

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_size_ms: int = 50,  # 50ms frames for timing analysis
        energy_threshold: float = 0.001,
    ):
        """
        Initialize cadence detector.

        Args:
            sample_rate: Audio sample rate
            frame_size_ms: Frame size in milliseconds
            energy_threshold: Minimum energy to consider "on"
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.frame_duration = frame_size_ms / 1000
        self.energy_threshold = energy_threshold

    def _compute_energy_envelope(self, samples: np.ndarray) -> np.ndarray:
        """Compute energy envelope using short frames."""
        num_frames = len(samples) // self.frame_size
        if num_frames == 0:
            return np.array([])

        energies = []
        for i in range(num_frames):
            frame = samples[i * self.frame_size:(i + 1) * self.frame_size]
            energy = np.sqrt(np.mean(frame ** 2))  # RMS energy
            energies.append(energy)

        return np.array(energies)

    def _find_on_off_periods(
        self,
        energies: np.ndarray,
        threshold: Optional[float] = None
    ) -> tuple[list[float], list[float]]:
        """
        Find on and off periods from energy envelope.

        Returns:
            Tuple of (on_durations, off_durations) in seconds
        """
        if len(energies) == 0:
            return [], []

        # Auto-detect threshold if not provided
        if threshold is None:
            # Use adaptive threshold between min and max
            min_e = np.min(energies)
            max_e = np.max(energies)
            threshold = min_e + (max_e - min_e) * 0.3
            # But ensure it's at least our minimum
            threshold = max(threshold, self.energy_threshold)

        # Convert to binary on/off
        is_on = energies > threshold

        # Find transitions
        on_durations = []
        off_durations = []

        current_state = is_on[0]
        current_duration = 1

        for i in range(1, len(is_on)):
            if is_on[i] == current_state:
                current_duration += 1
            else:
                # State changed
                duration = current_duration * self.frame_duration
                if current_state:
                    on_durations.append(duration)
                else:
                    off_durations.append(duration)
                current_state = is_on[i]
                current_duration = 1

        # Don't forget the last period
        duration = current_duration * self.frame_duration
        if current_state:
            on_durations.append(duration)
        else:
            off_durations.append(duration)

        return on_durations, off_durations

    def _match_cadence(
        self,
        on_durations: list[float],
        off_durations: list[float]
    ) -> tuple[CadencePattern, float, str]:
        """Match detected periods against known cadences."""
        if len(on_durations) == 0:
            return CadencePattern.UNKNOWN, 0.0, "No on periods detected"

        # Check for continuous tone (no significant off periods)
        if len(off_durations) == 0 or all(d < 0.1 for d in off_durations):
            return CadencePattern.CONTINUOUS, 0.9, "Continuous tone detected"

        avg_on = np.mean(on_durations)
        avg_off = np.mean(off_durations) if off_durations else 0

        # Match against known patterns
        best_match = CadencePattern.UNKNOWN
        best_confidence = 0.0
        best_reason = ""

        for pattern, (expected_on, expected_off, tolerance) in CADENCE_DEFINITIONS.items():
            on_error = abs(avg_on - expected_on) / expected_on
            off_error = abs(avg_off - expected_off) / expected_off if expected_off > 0 else 0

            # Check if within tolerance
            if on_error <= tolerance / expected_on and off_error <= tolerance / expected_off:
                # Calculate confidence based on how close we are
                confidence = 1.0 - (on_error + off_error) / 2

                # Boost confidence if we have multiple cycles
                num_cycles = min(len(on_durations), len(off_durations))
                if num_cycles >= 3:
                    confidence = min(confidence + 0.1, 1.0)

                if confidence > best_confidence:
                    best_match = pattern
                    best_confidence = confidence
                    best_reason = f"Matched {pattern.value}: {avg_on:.2f}s on / {avg_off:.2f}s off ({num_cycles} cycles)"

        if best_match == CadencePattern.UNKNOWN:
            best_reason = f"No match: {avg_on:.2f}s on / {avg_off:.2f}s off"

        return best_match, best_confidence, best_reason

    def detect(self, samples: np.ndarray) -> CadenceResult:
        """
        Detect cadence pattern in audio samples.

        Args:
            samples: Audio samples (float array, normalized)

        Returns:
            CadenceResult with detected pattern
        """
        # Compute energy envelope
        energies = self._compute_energy_envelope(samples)

        if len(energies) < 10:  # Need at least 0.5s of audio
            return CadenceResult(
                pattern=CadencePattern.UNKNOWN,
                confidence=0.0,
                on_duration=0.0,
                off_duration=0.0,
                cycle_count=0,
                reasoning="Audio too short for cadence detection"
            )

        # Find on/off periods
        on_durations, off_durations = self._find_on_off_periods(energies)

        # Match against known patterns
        pattern, confidence, reasoning = self._match_cadence(on_durations, off_durations)

        # Calculate averages
        avg_on = np.mean(on_durations) if on_durations else 0.0
        avg_off = np.mean(off_durations) if off_durations else 0.0
        cycle_count = min(len(on_durations), len(off_durations))

        return CadenceResult(
            pattern=pattern,
            confidence=confidence,
            on_duration=avg_on,
            off_duration=avg_off,
            cycle_count=cycle_count,
            reasoning=reasoning
        )

    def is_busy_signal(self, samples: np.ndarray) -> tuple[bool, float]:
        """Quick check for busy signal cadence."""
        result = self.detect(samples)
        return result.pattern == CadencePattern.BUSY, result.confidence

    def is_ringback(self, samples: np.ndarray) -> tuple[bool, float]:
        """Quick check for ringback cadence."""
        result = self.detect(samples)
        return result.pattern == CadencePattern.RINGBACK, result.confidence
