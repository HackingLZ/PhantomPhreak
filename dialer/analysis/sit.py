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
SIT_TOLERANCE = 20  # Hz tolerance for matching (stricter than before)

# SIT timing requirements
SIT_SEGMENT_MIN_MS = 250  # Minimum segment duration
SIT_SEGMENT_MAX_MS = 400  # Maximum segment duration
SIT_TOTAL_MAX_MS = 1500   # Maximum total duration for all 3 tones

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

    def _find_dominant_frequency(self, samples: np.ndarray) -> tuple[float, float]:
        """Find the dominant frequency in a segment using FFT.

        Returns:
            Tuple of (frequency, spectral_purity) where spectral_purity is 0-1.
            High purity (>0.5) indicates a clean tone, low purity indicates voice/noise.
        """
        if len(samples) == 0:
            return 0.0, 0.0

        # Apply window
        window = np.hanning(len(samples))
        windowed = samples * window

        # FFT
        fft = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(samples), 1 / self.sample_rate)

        # Find peak in SIT frequency range (800-2000 Hz)
        mask = (freqs >= 800) & (freqs <= 2000)
        if not np.any(mask):
            return 0.0, 0.0

        masked_fft = np.where(mask, fft, 0)
        peak_idx = np.argmax(masked_fft)
        peak_value = masked_fft[peak_idx]

        # Calculate spectral purity: ratio of peak energy to total energy in range
        # Pure tones have most energy at one frequency, voice is spread out
        total_energy = np.sum(masked_fft)
        if total_energy > 0:
            # Consider energy within ±50Hz of peak as "peak energy"
            peak_freq = freqs[peak_idx]
            peak_mask = mask & (np.abs(freqs - peak_freq) <= 50)
            peak_energy = np.sum(np.where(peak_mask, fft, 0))
            spectral_purity = peak_energy / total_energy
        else:
            spectral_purity = 0.0

        return freqs[peak_idx], spectral_purity

    def _check_energy_consistency(self, samples: np.ndarray) -> float:
        """Check if energy is consistent throughout samples (characteristic of tones).

        Returns:
            Consistency score 0-1. High = consistent (tone), Low = variable (voice).
        """
        if len(samples) < 100:
            return 0.0

        # Split into chunks and measure energy of each
        chunk_size = len(samples) // 8
        energies = []
        for i in range(0, len(samples) - chunk_size, chunk_size):
            chunk = samples[i:i + chunk_size]
            energy = np.sqrt(np.mean(chunk ** 2))
            energies.append(energy)

        if len(energies) < 2:
            return 0.0

        # Calculate coefficient of variation
        mean_energy = np.mean(energies)
        if mean_energy < 0.001:
            return 0.0

        std_energy = np.std(energies)
        cv = std_energy / mean_energy

        # Low CV = consistent = tone, High CV = variable = voice
        # CV < 0.2 is very consistent, CV > 0.5 is quite variable
        if cv < 0.15:
            return 1.0
        elif cv < 0.3:
            return 0.7
        elif cv < 0.5:
            return 0.3
        else:
            return 0.0

    def _detect_sit_segments(
        self,
        samples: np.ndarray
    ) -> list[tuple[float, float, float, str, float, float]]:
        """
        Detect SIT tone segments in audio.

        Returns:
            List of (start_time, end_time, frequency, classification, purity, consistency) tuples
        """
        segments = []

        # Look for tonal segments in the first 2 seconds (SIT is ~1 second)
        max_samples = min(len(samples), self.sample_rate * 2)
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
                    freq, purity = self._find_dominant_frequency(combined)
                    classification = self._classify_frequency(freq)
                    consistency = self._check_energy_consistency(combined)
                    if classification:
                        segments.append((
                            tone_start / self.sample_rate,
                            i / self.sample_rate,
                            freq,
                            classification,
                            purity,
                            consistency
                        ))
                    current_tone = None
                    tone_samples = []
                continue

            # Find dominant frequency
            freq, purity = self._find_dominant_frequency(frame)
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
                        seg_freq, seg_purity = self._find_dominant_frequency(combined)
                        consistency = self._check_energy_consistency(combined)
                        segments.append((
                            tone_start / self.sample_rate,
                            i / self.sample_rate,
                            seg_freq,
                            current_tone,
                            seg_purity,
                            consistency
                        ))
                    current_tone = classification
                    tone_start = i
                    tone_samples = [frame]
            else:
                # Not a SIT frequency - might be gap or different tone
                if current_tone is not None and tone_samples:
                    combined = np.concatenate(tone_samples)
                    seg_freq, seg_purity = self._find_dominant_frequency(combined)
                    consistency = self._check_energy_consistency(combined)
                    segments.append((
                        tone_start / self.sample_rate,
                        i / self.sample_rate,
                        seg_freq,
                        current_tone,
                        seg_purity,
                        consistency
                    ))
                    current_tone = None
                    tone_samples = []

        # Don't forget last segment
        if current_tone is not None and tone_samples:
            combined = np.concatenate(tone_samples)
            freq, purity = self._find_dominant_frequency(combined)
            consistency = self._check_energy_consistency(combined)
            end_time = min(len(samples), tone_start + len(combined)) / self.sample_rate
            segments.append((
                tone_start / self.sample_rate,
                end_time,
                freq,
                current_tone,
                purity,
                consistency
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
        # Segments are: (start_time, end_time, frequency, classification, purity, consistency)
        pattern_tuple = tuple(s[3] for s in sit_segments)
        pattern_str = '-'.join(pattern_tuple)
        frequencies = [s[2] for s in sit_segments]
        purities = [s[4] for s in sit_segments]
        consistencies = [s[5] for s in sit_segments]

        # STRICT VALIDATION 1: Check segment timing
        # Each segment should be 250-400ms, and all 3 should complete within 1.5 seconds
        timing_valid = True
        timing_issues = []

        for i, seg in enumerate(sit_segments):
            start_time, end_time = seg[0], seg[1]
            duration_ms = (end_time - start_time) * 1000

            if duration_ms < SIT_SEGMENT_MIN_MS:
                timing_valid = False
                timing_issues.append(f"Segment {i+1} too short ({duration_ms:.0f}ms)")
            elif duration_ms > SIT_SEGMENT_MAX_MS:
                timing_valid = False
                timing_issues.append(f"Segment {i+1} too long ({duration_ms:.0f}ms)")

        # Check total duration
        total_duration_ms = (sit_segments[2][1] - sit_segments[0][0]) * 1000
        if total_duration_ms > SIT_TOTAL_MAX_MS:
            timing_valid = False
            timing_issues.append(f"Total duration too long ({total_duration_ms:.0f}ms)")

        if not timing_valid:
            return SITResult(
                is_sit=False,
                category=SITCategory.UNKNOWN,
                pattern=pattern_str,
                confidence=0.0,
                segment_frequencies=frequencies,
                reasoning=f"Timing invalid: {'; '.join(timing_issues)}"
            )

        # STRICT VALIDATION 2: Check spectral purity (tones should be clean)
        avg_purity = np.mean(purities)
        if avg_purity < 0.4:
            return SITResult(
                is_sit=False,
                category=SITCategory.UNKNOWN,
                pattern=pattern_str,
                confidence=0.0,
                segment_frequencies=frequencies,
                reasoning=f"Low spectral purity ({avg_purity:.2f}) - likely voice, not tones"
            )

        # STRICT VALIDATION 3: Check energy consistency (tones have steady energy)
        avg_consistency = np.mean(consistencies)
        if avg_consistency < 0.3:
            return SITResult(
                is_sit=False,
                category=SITCategory.UNKNOWN,
                pattern=pattern_str,
                confidence=0.0,
                segment_frequencies=frequencies,
                reasoning=f"Energy too variable ({avg_consistency:.2f}) - likely voice, not tones"
            )

        # Look up category
        category = SIT_PATTERNS.get(pattern_tuple, SITCategory.UNKNOWN)

        # Calculate confidence based on multiple factors
        expected_freqs = []
        for cls in pattern_tuple:
            if cls == 'L':
                expected_freqs.append(SIT_LOW)
            elif cls == 'M':
                expected_freqs.append(SIT_MID)
            else:
                expected_freqs.append(SIT_HIGH)

        # Frequency accuracy component
        freq_errors = [abs(f - e) / e for f, e in zip(frequencies, expected_freqs)]
        avg_error = np.mean(freq_errors)
        freq_confidence = max(0.0, 1.0 - avg_error * 10)  # Stricter penalty

        # Purity and consistency components
        purity_confidence = avg_purity
        consistency_confidence = avg_consistency

        # Combined confidence (all factors must be good)
        confidence = (freq_confidence * 0.4 + purity_confidence * 0.3 + consistency_confidence * 0.3)

        # Require minimum overall confidence
        if confidence < 0.5:
            return SITResult(
                is_sit=False,
                category=SITCategory.UNKNOWN,
                pattern=pattern_str,
                confidence=confidence,
                segment_frequencies=frequencies,
                reasoning=f"Confidence too low ({confidence:.2f}): freq={freq_confidence:.2f}, purity={purity_confidence:.2f}, consistency={consistency_confidence:.2f}"
            )

        # Build reasoning
        description = SIT_DESCRIPTIONS.get(category, "Unknown")
        freq_str = ', '.join(f"{f:.0f} Hz" for f in frequencies)
        reasoning = f"SIT {pattern_str}: {description} (frequencies: {freq_str}, purity={avg_purity:.2f}, consistency={avg_consistency:.2f})"

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
