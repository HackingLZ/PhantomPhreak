"""Answering machine and voicemail detection.

Detects characteristics of answering machines vs live voice:
- Initial greeting speech duration
- Silence waiting for message
- Beep tone signaling recording start
- Speech patterns (continuous vs conversational)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal


@dataclass
class AnsweringMachineResult:
    """Result of answering machine detection."""
    is_answering_machine: bool
    is_voicemail: bool
    confidence: float
    beep_detected: bool
    beep_time: Optional[float]  # Time of beep in seconds
    greeting_duration: float  # Duration of initial speech
    silence_after_greeting: float  # Silence duration after greeting
    reasoning: str


class AnsweringMachineDetector:
    """Detects answering machines and voicemail systems."""

    # Common beep frequencies (Hz)
    BEEP_FREQUENCIES = [440, 480, 500, 800, 1000, 1400, 2000]
    BEEP_TOLERANCE = 50  # Hz

    def __init__(
        self,
        sample_rate: int = 8000,
        frame_size_ms: int = 100,
        min_beep_duration_ms: int = 100,
        max_beep_duration_ms: int = 2000,
        energy_threshold: float = 0.001,
    ):
        """
        Initialize detector.

        Args:
            sample_rate: Audio sample rate
            frame_size_ms: Frame size for analysis
            min_beep_duration_ms: Minimum beep duration
            max_beep_duration_ms: Maximum beep duration
            energy_threshold: Energy threshold for silence detection
        """
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.frame_duration = frame_size_ms / 1000
        self.min_beep_frames = int(min_beep_duration_ms / frame_size_ms)
        self.max_beep_frames = int(max_beep_duration_ms / frame_size_ms)
        self.energy_threshold = energy_threshold

    def _compute_frame_features(
        self,
        samples: np.ndarray
    ) -> list[dict]:
        """Compute features for each frame."""
        frames = []
        num_frames = len(samples) // self.frame_size

        for i in range(num_frames):
            frame = samples[i * self.frame_size:(i + 1) * self.frame_size]

            # Energy (RMS)
            energy = np.sqrt(np.mean(frame ** 2))

            # Dominant frequency using FFT
            fft = np.abs(np.fft.rfft(frame))
            freqs = np.fft.rfftfreq(len(frame), 1 / self.sample_rate)
            dominant_idx = np.argmax(fft[1:]) + 1  # Skip DC
            dominant_freq = freqs[dominant_idx] if len(freqs) > dominant_idx else 0

            # Spectral flatness (how tone-like vs noise-like)
            fft_power = fft ** 2 + 1e-10
            spectral_flatness = np.exp(np.mean(np.log(fft_power))) / np.mean(fft_power)

            # Zero crossing rate (speech vs tone indicator)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr = zero_crossings / len(frame)

            frames.append({
                'time': i * self.frame_duration,
                'energy': energy,
                'dominant_freq': dominant_freq,
                'spectral_flatness': spectral_flatness,
                'zcr': zcr,
                'is_speech': energy > self.energy_threshold and spectral_flatness > 0.1,
                'is_silence': energy < self.energy_threshold,
                'is_tonal': spectral_flatness < 0.1 and energy > self.energy_threshold,
            })

        return frames

    def _detect_beep(self, frames: list[dict]) -> tuple[bool, Optional[float]]:
        """
        Detect answering machine beep.

        Beeps are characterized by:
        - Tonal (low spectral flatness)
        - Consistent frequency throughout
        - Duration 100ms - 2s
        - Comes AFTER some speech (not at the start)
        - Often followed by silence
        """
        beep_start = None
        beep_frames = 0
        beep_freq = None

        # Beep should not be at the very beginning - need at least 1 second of audio first
        # Real voicemail: greeting -> beep -> silence (for recording)
        min_start_frame = int(1.0 / self.frame_duration)  # At least 1 second in

        for i, frame in enumerate(frames):
            # Skip the first second - beeps don't come at the very start
            if i < min_start_frame:
                continue

            # Check for tonal quality with STRICT spectral flatness threshold
            # Beeps are very pure tones with flatness < 0.05 (not just < 0.1)
            is_pure_tone = frame['spectral_flatness'] < 0.05 and frame['energy'] > self.energy_threshold

            is_potential_beep = (
                is_pure_tone and
                any(abs(frame['dominant_freq'] - bf) < self.BEEP_TOLERANCE
                    for bf in self.BEEP_FREQUENCIES)
            )

            if is_potential_beep:
                if beep_start is None:
                    beep_start = i
                    beep_freq = frame['dominant_freq']
                    beep_frames = 1
                else:
                    # Check that frequency is consistent (within 20 Hz of first frame)
                    if abs(frame['dominant_freq'] - beep_freq) < 20:
                        beep_frames += 1
                    else:
                        # Frequency changed too much - not a clean beep
                        beep_start = None
                        beep_frames = 0
                        beep_freq = None
            else:
                if beep_frames >= self.min_beep_frames:
                    # Check if followed by silence or low energy
                    if i < len(frames) - 1:
                        next_frames = frames[i:min(i + 5, len(frames))]
                        silence_count = sum(1 for f in next_frames if f['is_silence'])
                        if silence_count >= 2:  # At least 2 of next 5 frames are silent
                            return True, beep_start * self.frame_duration
                beep_start = None
                beep_frames = 0
                beep_freq = None

        # Check final beep (must also be followed by end of recording = implied silence)
        if beep_frames >= self.min_beep_frames:
            return True, beep_start * self.frame_duration

        return False, None

    def _analyze_speech_pattern(
        self,
        frames: list[dict]
    ) -> tuple[float, float, bool]:
        """
        Analyze speech pattern for AM vs live voice.

        Returns:
            Tuple of (greeting_duration, silence_after, is_am_pattern)
        """
        if not frames:
            return 0.0, 0.0, False

        # Find continuous speech segments
        speech_segments = []
        segment_start = None

        for i, frame in enumerate(frames):
            if frame['is_speech']:
                if segment_start is None:
                    segment_start = i
            else:
                if segment_start is not None:
                    duration = (i - segment_start) * self.frame_duration
                    speech_segments.append({
                        'start': segment_start * self.frame_duration,
                        'duration': duration
                    })
                    segment_start = None

        # Handle final segment
        if segment_start is not None:
            duration = (len(frames) - segment_start) * self.frame_duration
            speech_segments.append({
                'start': segment_start * self.frame_duration,
                'duration': duration
            })

        if not speech_segments:
            return 0.0, 0.0, False

        # Answering machine pattern:
        # - Long initial greeting (3-30 seconds)
        # - Followed by silence (waiting for message)
        # - No back-and-forth conversation

        first_segment = speech_segments[0]
        greeting_duration = first_segment['duration']

        # Find silence after first segment
        silence_after = 0.0
        if len(speech_segments) >= 2:
            gap = speech_segments[1]['start'] - (first_segment['start'] + first_segment['duration'])
            silence_after = gap
        else:
            # Only one segment - check silence at end
            end_time = first_segment['start'] + first_segment['duration']
            total_duration = len(frames) * self.frame_duration
            silence_after = total_duration - end_time

        # AM pattern: long greeting (3-30s) followed by silence (1s+)
        is_am_pattern = (
            3.0 <= greeting_duration <= 30.0 and
            silence_after >= 1.0 and
            len(speech_segments) <= 3  # Not conversational
        )

        return greeting_duration, silence_after, is_am_pattern

    def detect(self, samples: np.ndarray) -> AnsweringMachineResult:
        """
        Detect if audio is from an answering machine.

        Args:
            samples: Audio samples

        Returns:
            AnsweringMachineResult with detection details
        """
        frames = self._compute_frame_features(samples)

        if len(frames) < 10:
            return AnsweringMachineResult(
                is_answering_machine=False,
                is_voicemail=False,
                confidence=0.0,
                beep_detected=False,
                beep_time=None,
                greeting_duration=0.0,
                silence_after_greeting=0.0,
                reasoning="Audio too short for AM detection"
            )

        # Detect beep
        beep_detected, beep_time = self._detect_beep(frames)

        # Analyze speech pattern
        greeting_duration, silence_after, is_am_pattern = self._analyze_speech_pattern(frames)

        # Determine if answering machine
        confidence = 0.0
        reasons = []

        if beep_detected:
            confidence += 0.5
            reasons.append(f"beep at {beep_time:.1f}s")

        if is_am_pattern:
            confidence += 0.3
            reasons.append(f"AM speech pattern ({greeting_duration:.1f}s greeting)")

        if silence_after >= 2.0:
            confidence += 0.2
            reasons.append(f"{silence_after:.1f}s silence after speech")

        is_am = confidence >= 0.5
        is_voicemail = is_am and beep_detected

        reasoning = "Answering machine: " + ", ".join(reasons) if reasons else "No AM indicators"

        return AnsweringMachineResult(
            is_answering_machine=is_am,
            is_voicemail=is_voicemail,
            confidence=min(confidence, 1.0),
            beep_detected=beep_detected,
            beep_time=beep_time,
            greeting_duration=greeting_duration,
            silence_after_greeting=silence_after,
            reasoning=reasoning
        )
