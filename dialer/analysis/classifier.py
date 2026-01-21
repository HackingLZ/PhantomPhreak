"""Line type classification based on frequency analysis.

Uses a combination of:
1. FFT-based signature matching for known tones (fax, modem, busy, etc.)
2. Voice detection heuristics (energy variance, frequency diversity)
3. Cadence/timing analysis for busy signals, ringback
4. DTMF detection for IVR/phone menus
5. Answering machine detection (beep, speech patterns)
6. SIT tone decoding for intercept messages
7. Optional ML-based classification using trained models
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fft import AudioAnalysis, FFTAnalyzer
from .signatures import (
    SIGNATURES,
    SILENCE_MAX_ENERGY,
    SILENCE_MAX_PEAKS,
    VOICE_FREQUENCY_RANGE,
    VOICE_MIN_UNIQUE_FREQUENCIES,
    LineType,
    ToneSignature,
)
from .cadence import CadenceDetector, CadencePattern, CadenceResult
from .dtmf import DTMFDetector, DTMFResult
from .answering_machine import AnsweringMachineDetector, AnsweringMachineResult
from .sit import SITDecoder, SITResult, SITCategory

# Try to import ML classifier (optional dependency)
try:
    from .ml_classifier import MLClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


@dataclass
class SignatureMatch:
    """Result of matching audio against a signature."""
    signature: ToneSignature
    matched_frequencies: set[int]
    match_ratio: float  # 0.0 to 1.0
    confidence: float


@dataclass
class ClassificationResult:
    """Complete classification result for a phone line."""
    line_type: LineType
    confidence: float  # 0.0 to 1.0
    matches: list[SignatureMatch]
    is_voice: bool
    is_silence: bool
    unique_frequencies: int
    avg_energy: float
    reasoning: str
    # Extended detection results
    cadence: Optional[CadenceResult] = None
    dtmf: Optional[DTMFResult] = None
    answering_machine: Optional[AnsweringMachineResult] = None
    sit: Optional[SITResult] = None
    all_frequencies: list[int] = field(default_factory=list)  # All detected frequencies


class Classifier:
    """
    Classifies phone lines based on audio frequency analysis.

    Uses the following approach:
    1. Extract prominent frequencies from audio
    2. Match against known tone signatures
    3. Check for voice characteristics (broad spectrum)
    4. Check for silence/dead line
    5. Optionally use ML classification for improved accuracy
    6. Return best classification with confidence
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        sample_rate: int = 8000,
        use_ml: bool = True
    ):
        """
        Initialize classifier.

        Args:
            min_confidence: Minimum confidence to report a match
            sample_rate: Audio sample rate
            use_ml: Whether to use ML classification (if available)
        """
        self.min_confidence = min_confidence
        self.sample_rate = sample_rate
        self.analyzer = FFTAnalyzer(sample_rate=sample_rate)

        # Initialize extended detectors
        self.cadence_detector = CadenceDetector(sample_rate=sample_rate)
        self.dtmf_detector = DTMFDetector(sample_rate=sample_rate)
        self.am_detector = AnsweringMachineDetector(sample_rate=sample_rate)
        self.sit_decoder = SITDecoder(sample_rate=sample_rate)

        # Initialize ML classifier if available and requested
        self.ml_classifier: Optional['MLClassifier'] = None
        if use_ml and ML_AVAILABLE:
            self.ml_classifier = MLClassifier(sample_rate=sample_rate)
            # Try to load pre-trained model
            if not self.ml_classifier.load():
                # No pre-trained model, train with synthetic data
                from .ml_classifier import create_default_training_data
                training_data = create_default_training_data()
                accuracy = self.ml_classifier.train(training_data)
                # Save for future use
                try:
                    self.ml_classifier.save()
                except Exception:
                    pass  # Don't fail if we can't save

    def _match_signature(
        self,
        frequencies: set[int],
        frequency_counts: dict[int, int],
        signature: ToneSignature
    ) -> SignatureMatch:
        """
        Match detected frequencies against a signature.

        Args:
            frequencies: Set of detected frequencies
            frequency_counts: Frequency histogram
            signature: Signature to match against

        Returns:
            SignatureMatch with match details
        """
        # Find matches in primary frequencies
        primary_matches = frequencies & signature.frequencies
        optional_matches = frequencies & signature.optional_frequencies
        all_matches = primary_matches | optional_matches

        # Calculate match ratio (primary frequencies matched / required)
        if len(signature.frequencies) > 0:
            primary_ratio = len(primary_matches) / len(signature.frequencies)
        else:
            primary_ratio = 0.0

        # Calculate confidence based on:
        # 1. Primary match ratio
        # 2. Whether minimum matches met
        # 3. Frequency occurrence counts (more occurrences = higher confidence)
        meets_minimum = len(primary_matches) >= signature.min_matches

        if not meets_minimum:
            confidence = primary_ratio * 0.5
        else:
            # Base confidence from primary matches
            confidence = primary_ratio * 0.7

            # Boost for matching optional frequencies
            if signature.optional_frequencies:
                optional_ratio = len(optional_matches) / len(signature.optional_frequencies)
                confidence += optional_ratio * 0.15

            # Boost for high frequency counts (tone consistently present)
            if all_matches:
                avg_count = np.mean([frequency_counts.get(f, 0) for f in all_matches])
                if avg_count > 5:  # Present in 5+ frames
                    confidence += 0.15

        return SignatureMatch(
            signature=signature,
            matched_frequencies=all_matches,
            match_ratio=primary_ratio,
            confidence=min(confidence, 1.0)
        )

    def _check_voice(
        self,
        analysis: AudioAnalysis,
        frequencies: set[int]
    ) -> tuple[bool, float]:
        """
        Check if audio characteristics match voice.

        Voice typically has:
        - Many unique frequencies (broad spectrum)
        - Frequencies in 300-3400 Hz range
        - Varying energy over time
        - No single dominant tone

        Returns:
            Tuple of (is_voice, confidence)
        """
        # Count unique frequencies in voice range
        voice_freqs = {
            f for f in frequencies
            if VOICE_FREQUENCY_RANGE[0] <= f <= VOICE_FREQUENCY_RANGE[1]
        }

        # Voice has many unique frequencies - lower threshold for better detection
        min_freqs_for_voice = 5  # Reduced from VOICE_MIN_UNIQUE_FREQUENCIES
        if len(voice_freqs) < min_freqs_for_voice:
            return False, 0.0

        # Check for varying energy across frames (key voice characteristic)
        variance_score = 0.0
        if len(analysis.frames) >= 3:
            energies = [f.energy for f in analysis.frames]
            energy_variance = np.var(energies)
            energy_mean = np.mean(energies)

            # Voice has variable energy - this is KEY differentiator from tones
            if energy_mean > 0:
                cv = np.sqrt(energy_variance) / energy_mean  # Coefficient of variation
                if cv > 0.3:  # High variation = likely voice
                    variance_score = min(cv, 1.0)
                elif cv > 0.1:
                    variance_score = cv * 2  # Scale up moderate variation
                else:
                    variance_score = 0.0  # Low variation = likely tone, not voice
        else:
            variance_score = 0.3

        # Check that no single frequency dominates (tones have dominant frequencies)
        max_count = max(analysis.frequency_histogram.values()) if analysis.frequency_histogram else 0
        total_count = sum(analysis.frequency_histogram.values()) if analysis.frequency_histogram else 1
        dominance = max_count / total_count if total_count > 0 else 1.0

        # Low dominance = good for voice (voice is spread across frequencies)
        if dominance < 0.2:
            dominance_score = 1.0
        elif dominance < 0.4:
            dominance_score = 0.7
        elif dominance < 0.6:
            dominance_score = 0.3
        else:
            dominance_score = 0.0

        # Frequency diversity score
        frequency_score = min(len(voice_freqs) / 15, 1.0)  # More frequencies = more likely voice

        # Calculate overall voice confidence
        # Weight variance heavily - it's the best differentiator
        confidence = (frequency_score * 0.3 + variance_score * 0.5 + dominance_score * 0.2)

        # Boost confidence if we have many frequencies AND high variance
        if len(voice_freqs) >= 10 and variance_score > 0.5:
            confidence = min(confidence + 0.2, 1.0)

        return confidence > 0.4, confidence

    def _check_silence(self, analysis: AudioAnalysis) -> tuple[bool, float]:
        """
        Check if audio is silence/dead line.

        Returns:
            Tuple of (is_silence, confidence)
        """
        if not analysis.frames:
            return True, 1.0

        # Check average energy
        avg_energy = np.mean([f.energy for f in analysis.frames])
        if avg_energy > SILENCE_MAX_ENERGY:
            return False, 0.0

        # Check number of peaks detected
        total_peaks = sum(len(f.peaks) for f in analysis.frames)
        avg_peaks = total_peaks / len(analysis.frames)

        if avg_peaks <= SILENCE_MAX_PEAKS:
            confidence = 1.0 - (avg_energy / SILENCE_MAX_ENERGY)
            return True, confidence

        return False, 0.0

    def classify(self, samples: np.ndarray) -> ClassificationResult:
        """
        Classify audio samples to determine line type.

        Args:
            samples: Audio samples (float array, normalized)

        Returns:
            ClassificationResult with classification and confidence
        """
        # Run all extended detectors
        cadence_result = self.cadence_detector.detect(samples)
        dtmf_result = self.dtmf_detector.detect(samples)
        am_result = self.am_detector.detect(samples)
        sit_result = self.sit_decoder.decode(samples)

        # Try ML classification if available
        ml_result: Optional[tuple[LineType, float]] = None
        if self.ml_classifier is not None and self.ml_classifier.is_trained:
            try:
                ml_result = self.ml_classifier.classify(samples)
            except Exception:
                pass  # Fall back to FFT-based classification

        # Perform FFT analysis
        analysis = self.analyzer.analyze(samples)

        # Get unique frequencies and counts
        frequencies = set(analysis.all_frequencies)
        freq_counts = analysis.frequency_histogram
        all_freqs = sorted(list(frequencies))
        avg_energy = np.mean([f.energy for f in analysis.frames]) if analysis.frames else 0

        # Helper to create result with extended fields
        def make_result(line_type, confidence, is_voice, is_silence, reasoning):
            return ClassificationResult(
                line_type=line_type,
                confidence=confidence,
                matches=matches if 'matches' in dir() else [],
                is_voice=is_voice,
                is_silence=is_silence,
                unique_frequencies=len(frequencies),
                avg_energy=avg_energy,
                reasoning=reasoning,
                cadence=cadence_result,
                dtmf=dtmf_result,
                answering_machine=am_result,
                sit=sit_result,
                all_frequencies=all_freqs
            )

        # Check for silence first
        is_silence, silence_confidence = self._check_silence(analysis)
        if is_silence and silence_confidence > 0.8:
            return make_result(LineType.SILENCE, silence_confidence, False, True,
                             "Very low energy, minimal frequency content")

        # SIT TONE DETECTION: Check for intercept messages
        # Require high confidence AND verify it's not actually voice
        if sit_result.is_sit and sit_result.confidence > 0.75:
            # Double-check: SIT should NOT have voice characteristics
            # Voice has many unique frequencies and variable energy
            is_voice_check, voice_conf = self._check_voice(analysis, frequencies)
            if not is_voice_check or voice_conf < 0.4:
                # Not voice-like, accept as SIT
                return make_result(LineType.SIT_TONE, sit_result.confidence, False, False,
                                 sit_result.reasoning)
            # Has voice characteristics - likely misdetection, continue classification

        # Match against all signatures
        matches = []
        for signature in SIGNATURES:
            match = self._match_signature(frequencies, freq_counts, signature)
            if match.confidence > 0.3:
                matches.append(match)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        # CADENCE-BASED DETECTION: Use timing patterns
        if cadence_result.pattern == CadencePattern.BUSY and cadence_result.confidence > 0.7:
            return make_result(LineType.BUSY, cadence_result.confidence, False, False,
                             f"Busy signal: {cadence_result.on_duration:.2f}s on / {cadence_result.off_duration:.2f}s off")

        if cadence_result.pattern == CadencePattern.RINGBACK and cadence_result.confidence > 0.7:
            return make_result(LineType.RINGBACK, cadence_result.confidence, False, False,
                             f"Ringback: {cadence_result.on_duration:.2f}s on / {cadence_result.off_duration:.2f}s off")

        # DTMF/IVR DETECTION: Check for phone menus
        # Require high confidence AND multiple digits to avoid false positives from voice
        if dtmf_result.has_ivr and dtmf_result.ivr_confidence > 0.7 and len(dtmf_result.digits) >= 3:
            return make_result(LineType.IVR, dtmf_result.ivr_confidence, False, False,
                             f"IVR/Phone menu: {dtmf_result.reasoning}")

        # EARLY FAX/MODEM DETECTION
        has_2100 = 2100 in frequencies
        has_2200 = 2200 in frequencies
        has_1100 = 1100 in frequencies

        # Calculate recording duration from number of frames
        # Each frame is typically 1 second of audio with default FFT settings
        recording_duration = len(analysis.frames) if analysis.frames else 0

        # Modem: 2100 + 2200 Hz together
        if has_2100 and has_2200:
            count_2100 = freq_counts.get(2100, 0)
            count_2200 = freq_counts.get(2200, 0)
            if count_2100 >= 2 and count_2200 >= 2:
                modem_match = next((m for m in matches if m.signature.line_type == LineType.MODEM), None)
                confidence = max(modem_match.confidence if modem_match else 0.8, 0.85)
                return make_result(LineType.MODEM, confidence, False, False,
                                 f"Modem detected: 2100 Hz ({count_2100} frames) + 2200 Hz ({count_2200} frames)")

        # Fax: 2100 Hz alone (CED) or 1100 Hz (CNG)
        # Be more conservative: modems also start with 2100 Hz, then add 2200 Hz
        # If recording is short, we might not have captured the 2200 Hz yet
        if has_2100 or has_1100:
            fax_freq = 2100 if has_2100 else 1100
            fax_count = freq_counts.get(fax_freq, 0)

            # For 2100 Hz, require more evidence since modems also use it
            # 1100 Hz (CNG calling tone) is more distinctive to fax
            if fax_freq == 2100:
                # Need more frames AND longer recording for confidence with 2100 Hz
                min_frames_needed = 5  # Require 5+ frames of 2100 Hz for fax
                if fax_count >= min_frames_needed and recording_duration >= 5:
                    # Long enough recording with consistent 2100 Hz and NO 2200 Hz = fax
                    fax_match = next((m for m in matches if m.signature.line_type == LineType.FAX), None)
                    confidence = max(fax_match.confidence if fax_match else 0.7, 0.80)
                    return make_result(LineType.FAX, confidence, False, False,
                                     f"Fax detected: {fax_freq} Hz tone present in {fax_count} frames")
                elif fax_count >= 3:
                    # Recording has 2100 Hz but either not enough frames or not long enough
                    # Could be fax OR modem - mark as INTERESTING for manual review
                    return make_result(LineType.INTERESTING, 0.6, False, False,
                                     f"Possible fax/modem: 2100 Hz detected ({fax_count} frames, {recording_duration}s), needs longer capture")
            else:
                # 1100 Hz CNG is tricky - voice also has energy around 1100 Hz
                # CNG is a pulsed tone (0.5s on, 3s off), voice is continuous with many frequencies
                # Only classify as fax if:
                # 1. High frame count (lots of 1100 Hz energy)
                # 2. Few other frequencies (CNG is a pure tone, voice is broadband)
                # 3. No voice-like characteristics
                voice_like, voice_conf = self._check_voice(analysis, frequencies)

                # CNG should NOT have voice characteristics
                # If we have many frequencies (>8) or voice-like energy patterns, it's probably voice
                if fax_count >= 5 and len(frequencies) <= 8 and not voice_like:
                    fax_match = next((m for m in matches if m.signature.line_type == LineType.FAX), None)
                    confidence = max(fax_match.confidence if fax_match else 0.7, 0.80)
                    return make_result(LineType.FAX, confidence, False, False,
                                     f"Fax detected: {fax_freq} Hz CNG tone present in {fax_count} frames")
                # If we have 1100 Hz but also voice characteristics, don't classify as fax here
                # Let it fall through to voice detection

        # Check for voice
        is_voice, voice_confidence = self._check_voice(analysis, frequencies)

        # Get best signature match
        best_match = matches[0] if matches else None

        # Use ML result to boost confidence or break ties
        ml_type, ml_confidence = ml_result if ml_result else (None, 0.0)

        # If ML says voice with high confidence, trust it
        if ml_type == LineType.VOICE and ml_confidence > 0.7:
            # But check for answering machine first - require HIGH confidence
            if am_result.is_answering_machine and am_result.confidence > 0.7 and am_result.beep_detected:
                line_type = LineType.VOICEMAIL if am_result.is_voicemail else LineType.ANSWERING_MACHINE
                return make_result(line_type, am_result.confidence, True, False,
                                 f"ML voice + {am_result.reasoning}")
            return make_result(LineType.VOICE, ml_confidence, True, False,
                             f"ML classifier: voice detected ({ml_confidence:.0%} confidence)")

        # PRIORITY: Voice detection over weak tone matches
        # If we detect voice characteristics AND the tone match is weak/ambiguous
        # (like dial_tone which shares frequencies with voice), prefer voice
        if is_voice and voice_confidence >= 0.5:
            # Check for answering machine - require beep AND high confidence
            if am_result.is_answering_machine and am_result.confidence > 0.7 and am_result.beep_detected:
                line_type = LineType.VOICEMAIL if am_result.is_voicemail else LineType.ANSWERING_MACHINE
                return make_result(line_type, am_result.confidence, True, False,
                                 am_result.reasoning)

            # Check if best match is a "weak" match that could be confused with voice
            weak_tone_types = {LineType.DIAL_TONE, LineType.RINGBACK}  # Tones that share voice frequencies

            if best_match is None:
                # No tone match, definitely voice
                return make_result(LineType.VOICE, voice_confidence, True, False,
                                 f"Voice detected: {len(frequencies)} unique frequencies, variable energy")
            elif best_match.signature.line_type in weak_tone_types:
                # Tone match is a type that's often confused with voice
                # Prefer voice if we have strong voice characteristics
                if voice_confidence > best_match.confidence or len(frequencies) >= 10:
                    return make_result(LineType.VOICE, voice_confidence, True, False,
                                     f"Voice detected (overriding {best_match.signature.name}): {len(frequencies)} frequencies, high variance")

        # Special handling for fax vs modem disambiguation
        # If 2200 Hz is present alongside 2100 Hz, it's modem (not fax)
        has_2100 = 2100 in frequencies
        has_2200 = 2200 in frequencies
        if has_2100 and has_2200:
            # This is a modem handshake pattern, not pure fax CED
            modem_match = next((m for m in matches if m.signature.line_type == LineType.MODEM), None)
            if modem_match:
                return make_result(LineType.MODEM, max(modem_match.confidence, 0.85), False, False,
                                 "Modem detected: 2100 Hz + 2200 Hz handshake pattern")

        # Strong signature matches (fax, modem, busy, SIT) take priority
        strong_tone_types = {LineType.FAX, LineType.MODEM, LineType.CARRIER, LineType.BUSY, LineType.SIT_TONE}
        if best_match and best_match.confidence >= self.min_confidence:
            if best_match.signature.line_type in strong_tone_types:
                # Special case: FAX CNG (1100 Hz) overlaps with voice frequencies
                # Only trust FAX if it's CED (2100 Hz) OR if not voice-like
                if best_match.signature.line_type == LineType.FAX:
                    # Check if this is a 2100 Hz match (reliable) or 1100 Hz match (could be voice)
                    has_2100_match = 2100 in best_match.matched_frequencies
                    if not has_2100_match and is_voice:
                        # 1100 Hz "fax" with voice characteristics - probably not fax
                        # Let it fall through to voice detection
                        pass
                    else:
                        # 2100 Hz fax OR 1100 Hz without voice characteristics
                        return make_result(best_match.signature.line_type, best_match.confidence, is_voice, is_silence,
                                         f"Matched {best_match.signature.name}: frequencies {best_match.matched_frequencies}")
                else:
                    # Other strong tones (modem, carrier, busy, SIT) - trust the match
                    return make_result(best_match.signature.line_type, best_match.confidence, is_voice, is_silence,
                                     f"Matched {best_match.signature.name}: frequencies {best_match.matched_frequencies}")
            elif not is_voice:
                # Weak tone type but no voice detected
                return make_result(best_match.signature.line_type, best_match.confidence, is_voice, is_silence,
                                 f"Matched {best_match.signature.name}: frequencies {best_match.matched_frequencies}")

        # Voice detected
        if is_voice and voice_confidence >= 0.4:
            # Check for answering machine - require beep AND high confidence
            if am_result.is_answering_machine and am_result.confidence > 0.7 and am_result.beep_detected:
                line_type = LineType.VOICEMAIL if am_result.is_voicemail else LineType.ANSWERING_MACHINE
                return make_result(line_type, am_result.confidence, True, False, am_result.reasoning)
            return make_result(LineType.VOICE, voice_confidence, True, False,
                             f"Broad spectrum with {len(frequencies)} unique frequencies, variable energy")

        # Silence/dead line
        if is_silence:
            return make_result(LineType.SILENCE, silence_confidence, False, True,
                             "Low energy, minimal frequency content")

        # Fall back to best match if any
        if best_match and best_match.confidence >= 0.3:
            return make_result(best_match.signature.line_type, best_match.confidence, is_voice, is_silence,
                             f"Weak match: {best_match.signature.name}")

        # Unknown
        return make_result(LineType.UNKNOWN, 0.0, is_voice, is_silence,
                         f"No confident match. Best guess: {best_match.signature.name if best_match else 'none'}")

    def classify_from_file(self, filepath: str) -> ClassificationResult:
        """
        Classify audio from a WAV file.

        Args:
            filepath: Path to WAV file

        Returns:
            ClassificationResult
        """
        from ..core.audio import load_wav

        samples, sample_rate = load_wav(filepath)

        # Resample if necessary
        if sample_rate != self.sample_rate:
            from scipy import signal
            num_samples = int(len(samples) * self.sample_rate / sample_rate)
            samples = signal.resample(samples, num_samples)

        return self.classify(samples)
