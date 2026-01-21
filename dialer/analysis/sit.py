"""SIT (Special Information Tone) decoder.

SIT tones precede intercept messages to indicate why a call failed.
They consist of three tones in sequence, each with specific frequencies
and durations that encode the message type.

Standard SIT frequencies:
- Low:  913.8 Hz
- Mid:  1370.6 Hz
- High: 1776.7 Hz

Tone segments (each ~330ms):
- Segment 1: Indicates general category
- Segment 2: Indicates specific condition
- Segment 3: Indicates additional info

Common SIT patterns:
- H-L-H: Vacant code (number not in service)
- H-H-L: No circuit (all circuits busy)
- L-H-L: Reorder (call cannot be completed)
- H-L-L: Operator intercept
- L-L-H: Reserved for future use
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class SITCategory(Enum):
    """SIT message categories."""
    VACANT_CODE = "vacant_code"  # Number not in service
    NO_CIRCUIT = "no_circuit"  # All circuits busy
    REORDER = "reorder"  # Call cannot be completed as dialed
    OPERATOR_INTERCEPT = "operator_intercept"  # Operator assistance required
    RESERVED = "reserved"  # Reserved patterns
    UNKNOWN = "unknown"


@dataclass
class SITResult:
    """Result of SIT tone decoding."""
    is_sit: bool
    category: SITCategory
    pattern: str  # e.g., "H-L-H" or "913-1370-1776"
    confidence: float
    segment_frequencies: list[float]  # Detected frequencies for each segment
    reasoning: str


# SIT frequency definitions (Hz)
SIT_LOW = 913.8
SIT_MID = 1370.6
SIT_HIGH = 1776.7
SIT_TOLERANCE = 30  # Hz tolerance for matching

# SIT pattern definitions
# Pattern: (seg1, seg2, seg3) where L=Low, M=Mid, H=High
SIT_PATTERNS = {
    ('H', 'L', 'H'): SITCategory.VACANT_CODE,
    ('H', 'H', 'L'): SITCategory.NO_CIRCUIT,
    ('L', 'H', 'L'): SITCategory.REORDER,
    ('H', 'L', 'L'): SITCategory.OPERATOR_INTERCEPT,
    ('L', 'L', 'H'): SITCategory.RESERVED,
    ('H', 'H', 'H'): SITCategory.RESERVED,
    ('L', 'L', 'L'): SITCategory.RESERVED,
    ('L', 'H', 'H'): SITCategory.RESERVED,
}

# Human-readable descriptions
SIT_DESCRIPTIONS = {
    SITCategory.VACANT_CODE: "Number not in service or disconnected",
    SITCategory.NO_CIRCUIT: "All circuits are busy, try again later",
    SITCategory.REORDER: "Call cannot be completed as dialed",
    SITCategory.OPERATOR_INTERCEPT: "Operator assistance required",
    SITCategory.RESERVED: "Reserved/non-standard intercept",
    SITCategory.UNKNOWN: "Unknown SIT pattern",
}


class SITDecoder:
    """Decodes SIT (Special Information Tone) patterns."""

    def __init__(
        self,
        sample_rate: int = 8000,
        segment_duration_ms: int = 330,  # Standard SIT segment length
        segment_tolerance_ms: int = 100,
    ):
        """
        Initialize SIT decoder.

        Args:
            sample_rate: Audio sample rate
            segment_duration_ms: Expected segment duration
            segment_tolerance_ms: Tolerance for segment timing
        """
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_duration_ms / 1000)
        self.segment_tolerance = int(sample_rate * segment_tolerance_ms / 1000)

    def _classify_frequency(self, freq: float) -> Optional[str]:
        """Classify a frequency as Low, Mid, or High SIT tone."""
        if abs(freq - SIT_LOW) < SIT_TOLERANCE:
            return 'L'
        elif abs(freq - SIT_MID) < SIT_TOLERANCE:
            return 'M'
        elif abs(freq - SIT_HIGH) < SIT_TOLERANCE:
            return 'H'
        return None

    def _find_dominant_frequency(self, samples: np.ndarray) -> float:
        """Find the dominant frequency in a segment using FFT."""
        if len(samples) == 0:
            return 0.0

        # Apply window
        window = np.hanning(len(samples))
        windowed = samples * window

        # FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(samples), 1 / self.sample_rate)

        # Find peak in SIT frequency range (800-2000 Hz)
        mask = (freqs >= 800) & (freqs <= 2000)
        if not np.any(mask):
            return 0.0

        masked_fft = np.where(mask, fft, 0)
        peak_idx = np.argmax(masked_fft)

        return freqs[peak_idx]

    def _detect_sit_segments(
        self,
        samples: np.ndarray
    ) -> list[tuple[float, float, str]]:
        """
        Detect SIT tone segments in audio.

        Returns:
            List of (start_time, frequency, classification) tuples
        """
        segments = []

        # Look for tonal segments in the first 3 seconds
        max_samples = min(len(samples), self.sample_rate * 3)
        frame_size = self.segment_samples // 3  # Use smaller frames for detection

        current_tone = None
        tone_start = 0
        tone_samples = []

        for i in range(0, max_samples - frame_size, frame_size):
            frame = samples[i:i + frame_size]

            # Check if frame has significant energy
            energy = np.sqrt(np.mean(frame ** 2))
            if energy < 0.01:
                # Low energy - end of tone?
                if current_tone is not None and len(tone_samples) > 0:
                    # Save segment
                    combined = np.concatenate(tone_samples)
                    freq = self._find_dominant_frequency(combined)
                    classification = self._classify_frequency(freq)
                    if classification:
                        segments.append((
                            tone_start / self.sample_rate,
                            freq,
                            classification
                        ))
                    current_tone = None
                    tone_samples = []
                continue

            # Find dominant frequency
            freq = self._find_dominant_frequency(frame)
            classification = self._classify_frequency(freq)

            if classification:
                if current_tone is None:
                    # Start of new tone
                    current_tone = classification
                    tone_start = i
                    tone_samples = [frame]
                elif classification == current_tone:
                    # Continue same tone
                    tone_samples.append(frame)
                else:
                    # Different tone - save previous and start new
                    if tone_samples:
                        combined = np.concatenate(tone_samples)
                        seg_freq = self._find_dominant_frequency(combined)
                        segments.append((
                            tone_start / self.sample_rate,
                            seg_freq,
                            current_tone
                        ))
                    current_tone = classification
                    tone_start = i
                    tone_samples = [frame]
            else:
                # Not a SIT frequency - might be gap or different tone
                if current_tone is not None and tone_samples:
                    combined = np.concatenate(tone_samples)
                    seg_freq = self._find_dominant_frequency(combined)
                    segments.append((
                        tone_start / self.sample_rate,
                        seg_freq,
                        current_tone
                    ))
                    current_tone = None
                    tone_samples = []

        # Don't forget last segment
        if current_tone is not None and tone_samples:
            combined = np.concatenate(tone_samples)
            freq = self._find_dominant_frequency(combined)
            segments.append((
                tone_start / self.sample_rate,
                freq,
                current_tone
            ))

        return segments

    def decode(self, samples: np.ndarray) -> SITResult:
        """
        Decode SIT tones from audio.

        Args:
            samples: Audio samples

        Returns:
            SITResult with decoded information
        """
        # Detect segments
        segments = self._detect_sit_segments(samples)

        if len(segments) < 3:
            return SITResult(
                is_sit=False,
                category=SITCategory.UNKNOWN,
                pattern="",
                confidence=0.0,
                segment_frequencies=[],
                reasoning=f"Only {len(segments)} SIT segments detected (need 3)"
            )

        # Take first 3 segments as the SIT pattern
        sit_segments = segments[:3]
        pattern_tuple = tuple(s[2] for s in sit_segments)
        pattern_str = '-'.join(pattern_tuple)
        frequencies = [s[1] for s in sit_segments]

        # Look up category
        category = SIT_PATTERNS.get(pattern_tuple, SITCategory.UNKNOWN)

        # Calculate confidence based on frequency accuracy
        expected_freqs = []
        for cls in pattern_tuple:
            if cls == 'L':
                expected_freqs.append(SIT_LOW)
            elif cls == 'M':
                expected_freqs.append(SIT_MID)
            else:
                expected_freqs.append(SIT_HIGH)

        freq_errors = [abs(f - e) / e for f, e in zip(frequencies, expected_freqs)]
        avg_error = np.mean(freq_errors)
        confidence = max(0.0, 1.0 - avg_error * 5)

        # Build reasoning
        description = SIT_DESCRIPTIONS.get(category, "Unknown")
        freq_str = ', '.join(f"{f:.0f} Hz" for f in frequencies)
        reasoning = f"SIT {pattern_str}: {description} (frequencies: {freq_str})"

        return SITResult(
            is_sit=True,
            category=category,
            pattern=pattern_str,
            confidence=confidence,
            segment_frequencies=frequencies,
            reasoning=reasoning
        )

    def is_sit_tone(self, samples: np.ndarray) -> tuple[bool, float]:
        """Quick check for SIT tone presence."""
        result = self.decode(samples)
        return result.is_sit, result.confidence
