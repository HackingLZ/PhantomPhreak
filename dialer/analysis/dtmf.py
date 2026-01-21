"""DTMF (Dual-Tone Multi-Frequency) detection.

Detects touch-tone digits and IVR menus.

DTMF frequencies:
        1209 Hz  1336 Hz  1477 Hz  1633 Hz
697 Hz    1        2        3        A
770 Hz    4        5        6        B
852 Hz    7        8        9        C
941 Hz    *        0        #        D
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.fft import rfft, rfftfreq


# DTMF frequency definitions
DTMF_LOW_FREQS = [697, 770, 852, 941]
DTMF_HIGH_FREQS = [1209, 1336, 1477, 1633]

# DTMF digit mapping
DTMF_DIGITS = {
    (697, 1209): '1', (697, 1336): '2', (697, 1477): '3', (697, 1633): 'A',
    (770, 1209): '4', (770, 1336): '5', (770, 1477): '6', (770, 1633): 'B',
    (852, 1209): '7', (852, 1336): '8', (852, 1477): '9', (852, 1633): 'C',
    (941, 1209): '*', (941, 1336): '0', (941, 1477): '#', (941, 1633): 'D',
}


@dataclass
class DTMFDigit:
    """A detected DTMF digit."""
    digit: str
    start_time: float
    duration: float
    low_freq: int
    high_freq: int
    confidence: float


@dataclass
class DTMFResult:
    """Result of DTMF detection."""
    digits: list[DTMFDigit]
    sequence: str  # All digits as string
    has_ivr: bool  # Detected IVR/menu patterns
    ivr_confidence: float
    reasoning: str


class DTMFDetector:
    """Detects DTMF tones in audio."""

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_size_ms: int = 40,  # DTMF tones are at least 40ms
        frequency_tolerance: int = 20,  # Hz tolerance for frequency matching
        min_energy_ratio: float = 0.1,  # Min ratio of DTMF energy to total
    ):
        """
        Initialize DTMF detector.

        Args:
            sample_rate: Audio sample rate
            frame_size_ms: Frame size for detection
            frequency_tolerance: Tolerance for frequency matching in Hz
            min_energy_ratio: Minimum DTMF energy ratio
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.frame_duration = frame_size_ms / 1000
        self.frequency_tolerance = frequency_tolerance
        self.min_energy_ratio = min_energy_ratio

    def _goertzel(self, samples: np.ndarray, target_freq: float) -> float:
        """
        Goertzel algorithm for efficient single-frequency detection.
        More efficient than FFT when detecting specific frequencies.
        """
        n = len(samples)
        k = int(0.5 + n * target_freq / self.sample_rate)
        omega = 2 * np.pi * k / n
        coeff = 2 * np.cos(omega)

        s0, s1, s2 = 0.0, 0.0, 0.0
        for sample in samples:
            s0 = sample + coeff * s1 - s2
            s2 = s1
            s1 = s0

        power = s1 * s1 + s2 * s2 - coeff * s1 * s2
        return power

    def _detect_dtmf_in_frame(
        self,
        samples: np.ndarray
    ) -> Optional[tuple[str, int, int, float]]:
        """
        Detect DTMF digit in a single frame.

        Returns:
            Tuple of (digit, low_freq, high_freq, confidence) or None
        """
        # Calculate power at each DTMF frequency using Goertzel
        low_powers = [(f, self._goertzel(samples, f)) for f in DTMF_LOW_FREQS]
        high_powers = [(f, self._goertzel(samples, f)) for f in DTMF_HIGH_FREQS]

        # Find strongest in each group
        low_freq, low_power = max(low_powers, key=lambda x: x[1])
        high_freq, high_power = max(high_powers, key=lambda x: x[1])

        # Calculate total DTMF power
        total_dtmf_power = sum(p for _, p in low_powers) + sum(p for _, p in high_powers)
        if total_dtmf_power == 0:
            return None

        # STRICT CHECK 1: The two strongest must dominate other DTMF frequencies
        # Real DTMF has energy at exactly two frequencies, voice spreads across many
        dtmf_power = low_power + high_power
        other_dtmf_power = total_dtmf_power - dtmf_power

        if other_dtmf_power > 0 and dtmf_power / other_dtmf_power < 5.0:
            return None  # Not a clean DTMF tone - energy at multiple DTMF frequencies

        # STRICT CHECK 2: Check that low and high freqs have similar power (within 10dB)
        # Real DTMF tones have balanced power in both frequencies
        if low_power > 0 and high_power > 0:
            power_ratio = max(low_power, high_power) / min(low_power, high_power)
            if power_ratio > 10.0:  # More than 10x difference = not balanced
                return None

        # STRICT CHECK 3: Use FFT to verify these are the ONLY strong frequencies
        # Voice has many harmonics, DTMF has only two clean tones
        fft = np.abs(rfft(samples))
        freqs = rfftfreq(len(samples), 1 / self.sample_rate)

        # Find energy in voice-typical ranges that aren't DTMF
        # If there's significant energy elsewhere, it's probably voice
        voice_mask = (freqs >= 200) & (freqs <= 3500)
        dtmf_mask = np.zeros_like(freqs, dtype=bool)
        for f in DTMF_LOW_FREQS + DTMF_HIGH_FREQS:
            dtmf_mask |= (np.abs(freqs - f) <= 30)

        non_dtmf_voice_mask = voice_mask & ~dtmf_mask
        non_dtmf_energy = np.sum(fft[non_dtmf_voice_mask] ** 2) if np.any(non_dtmf_voice_mask) else 0
        dtmf_fft_energy = np.sum(fft[dtmf_mask] ** 2) if np.any(dtmf_mask) else 0

        if dtmf_fft_energy > 0:
            # DTMF energy should be much higher than non-DTMF voice energy
            if non_dtmf_energy > dtmf_fft_energy * 0.3:
                return None  # Too much energy at non-DTMF frequencies = voice

        # Check minimum power threshold
        frame_energy = np.sum(samples ** 2)
        if frame_energy > 0 and dtmf_power / frame_energy < self.min_energy_ratio:
            return None

        # Look up digit
        digit = DTMF_DIGITS.get((low_freq, high_freq))
        if digit is None:
            return None

        # Calculate confidence based on how clean the DTMF is
        dtmf_ratio = dtmf_power / (other_dtmf_power + 1)
        spectral_purity = dtmf_fft_energy / (dtmf_fft_energy + non_dtmf_energy + 1)
        confidence = min(dtmf_ratio * spectral_purity / 5.0, 1.0)

        # Require minimum confidence
        if confidence < 0.5:
            return None

        return digit, low_freq, high_freq, confidence

    def detect(self, samples: np.ndarray) -> DTMFResult:
        """
        Detect DTMF digits in audio.

        Args:
            samples: Audio samples

        Returns:
            DTMFResult with detected digits
        """
        digits = []
        current_digit = None
        digit_start = 0.0
        digit_frames = 0

        num_frames = len(samples) // self.frame_size

        for i in range(num_frames):
            frame = samples[i * self.frame_size:(i + 1) * self.frame_size]
            frame_time = i * self.frame_duration

            result = self._detect_dtmf_in_frame(frame)

            # Require minimum 3 consecutive frames (~120ms) for a valid DTMF digit
            MIN_DTMF_FRAMES = 3

            if result is not None:
                digit, low_freq, high_freq, confidence = result

                if current_digit is None:
                    # Start of new digit
                    current_digit = (digit, low_freq, high_freq, confidence)
                    digit_start = frame_time
                    digit_frames = 1
                elif current_digit[0] == digit:
                    # Continue same digit
                    digit_frames += 1
                else:
                    # Different digit - save previous and start new
                    if digit_frames >= MIN_DTMF_FRAMES:
                        digits.append(DTMFDigit(
                            digit=current_digit[0],
                            start_time=digit_start,
                            duration=digit_frames * self.frame_duration,
                            low_freq=current_digit[1],
                            high_freq=current_digit[2],
                            confidence=current_digit[3]
                        ))
                    current_digit = (digit, low_freq, high_freq, confidence)
                    digit_start = frame_time
                    digit_frames = 1
            else:
                # No DTMF - end current digit if any
                if current_digit is not None and digit_frames >= MIN_DTMF_FRAMES:
                    digits.append(DTMFDigit(
                        digit=current_digit[0],
                        start_time=digit_start,
                        duration=digit_frames * self.frame_duration,
                        low_freq=current_digit[1],
                        high_freq=current_digit[2],
                        confidence=current_digit[3]
                    ))
                current_digit = None
                digit_frames = 0

        # Don't forget last digit
        if current_digit is not None and digit_frames >= MIN_DTMF_FRAMES:
            digits.append(DTMFDigit(
                digit=current_digit[0],
                start_time=digit_start,
                duration=digit_frames * self.frame_duration,
                low_freq=current_digit[1],
                high_freq=current_digit[2],
                confidence=current_digit[3]
            ))

        # Build sequence string
        sequence = ''.join(d.digit for d in digits)

        # Detect IVR patterns
        has_ivr, ivr_confidence, reasoning = self._detect_ivr_pattern(digits, sequence)

        return DTMFResult(
            digits=digits,
            sequence=sequence,
            has_ivr=has_ivr,
            ivr_confidence=ivr_confidence,
            reasoning=reasoning
        )

    def _detect_ivr_pattern(
        self,
        digits: list[DTMFDigit],
        sequence: str
    ) -> tuple[bool, float, str]:
        """Detect IVR/phone menu patterns."""
        if len(digits) == 0:
            return False, 0.0, "No DTMF detected"

        # Check average confidence of detected digits
        avg_confidence = np.mean([d.confidence for d in digits])
        if avg_confidence < 0.6:
            return False, 0.0, f"Low confidence DTMF ({avg_confidence:.2f}): {sequence}"

        # IVR indicators:
        # 1. Multiple digits with gaps (user pressing menu options)
        # 2. Common patterns: "1" for English, "2" for Spanish, etc.
        # 3. Short sequences followed by pauses

        # Check for typical IVR patterns - require at least 3 digits for confidence
        if len(digits) >= 3:
            # Check for spaced-out digits (user responding to prompts)
            gaps = []
            for i in range(1, len(digits)):
                gap = digits[i].start_time - (digits[i-1].start_time + digits[i-1].duration)
                gaps.append(gap)

            avg_gap = np.mean(gaps) if gaps else 0

            if avg_gap > 1.5:  # Gaps > 1.5 seconds suggest IVR interaction
                return True, 0.8, f"IVR detected: {len(digits)} digits with {avg_gap:.1f}s avg gap"

        # Check for extension/PIN patterns (rapid sequence of 4+ digits)
        if len(sequence) >= 4 and sequence.isdigit():
            # Check that digits were pressed in quick succession
            total_duration = digits[-1].start_time + digits[-1].duration - digits[0].start_time
            if total_duration < 3.0:  # 4+ digits in under 3 seconds = likely extension
                return True, 0.7, f"Extension/PIN pattern: {sequence}"

        # Single digit or 2 digits is not enough to confirm IVR
        # Could be accidental detection in voice
        return False, 0.0, f"DTMF sequence: {sequence}" if sequence else "No IVR pattern"
