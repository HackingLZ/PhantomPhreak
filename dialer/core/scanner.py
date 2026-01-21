"""Main scanning orchestration."""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..analysis.classifier import Classifier, ClassificationResult
from ..analysis.signatures import LineType
from ..storage.database import Database
from ..storage.models import Scan, ScanResult, ScanStatus
from .audio import AudioCapture, save_wav
from .modem import CallResult as ModemCallResult
from .provider import CallResult, ProviderError, TelephonyProvider

logger = logging.getLogger(__name__)


class ScannerError(Exception):
    """Exception raised for scanner errors."""
    pass


class ScannerConfig:
    """Configuration for scanner behavior."""

    def __init__(
        self,
        # Provider selection
        provider_type: str = "modem",  # "modem" or "iax2"
        # Modem settings
        modem_device: Optional[str] = None,
        baud_rate: int = 460800,
        dial_mode: str = "tone",
        detect_dial_tone: bool = False,
        init_string: Optional[str] = None,
        # IAX2 settings
        iax2_host: Optional[str] = None,
        iax2_port: int = 4569,
        iax2_username: Optional[str] = None,
        iax2_password: Optional[str] = None,
        iax2_caller_id: Optional[str] = None,
        # Common settings
        call_timeout: float = 45.0,
        record_duration: float = 30.0,
        call_delay: float = 3.0,
        max_retries: int = 2,
        sample_rate: int = 8000,
        audio_output_dir: str = "./recordings",
        keep_audio_types: Optional[list[LineType]] = None,
        db_path: str = "./dialer.db",
        early_hangup: bool = True,
        early_hangup_interval: float = 3.0,
        early_hangup_types: Optional[list[LineType]] = None,
        early_hangup_min_confidence: float = 0.8,
    ):
        # Provider selection
        self.provider_type = provider_type
        # Modem settings
        self.modem_device = modem_device
        self.baud_rate = baud_rate
        self.dial_mode = dial_mode
        self.detect_dial_tone = detect_dial_tone
        self.init_string = init_string
        # IAX2 settings
        self.iax2_host = iax2_host
        self.iax2_port = iax2_port
        self.iax2_username = iax2_username
        self.iax2_password = iax2_password
        self.iax2_caller_id = iax2_caller_id
        # Common settings
        self.call_timeout = call_timeout
        self.record_duration = record_duration
        self.call_delay = call_delay
        self.max_retries = max_retries
        self.sample_rate = sample_rate
        self.audio_output_dir = Path(audio_output_dir)
        self.keep_audio_types = keep_audio_types or [
            LineType.UNKNOWN,
            LineType.VOICE,
            LineType.FAX,
            LineType.MODEM,
        ]
        self.db_path = db_path
        # Early hangup for quick fax/modem detection
        self.early_hangup = early_hangup
        self.early_hangup_interval = early_hangup_interval
        self.early_hangup_types = early_hangup_types or [
            LineType.FAX,
            LineType.MODEM,
            LineType.CARRIER,
            LineType.BUSY,
            LineType.SIT_TONE,
        ]
        self.early_hangup_min_confidence = early_hangup_min_confidence


class Scanner:
    """
    Main scanner that orchestrates the wardialing process.

    Workflow:
    1. Load phone numbers from file or range
    2. For each number:
       a. Check if already scanned
       b. Dial in voice mode
       c. Record audio on answer
       d. Analyze audio with FFT
       e. Classify line type
       f. Store results
    3. Generate summary
    """

    def __init__(
        self,
        config: ScannerConfig,
        progress_callback: Optional[Callable[[int, int, ScanResult], None]] = None,
    ):
        """
        Initialize scanner.

        Args:
            config: Scanner configuration
            progress_callback: Called after each number with (current, total, result)
        """
        self.config = config
        self.progress_callback = progress_callback

        self.provider: Optional[TelephonyProvider] = None
        self.db = Database(config.db_path)
        self.classifier = Classifier(sample_rate=config.sample_rate)
        self.audio_capture = AudioCapture(sample_rate=config.sample_rate)

        self._current_scan: Optional[Scan] = None
        self._stop_requested = False
        self._voice_mode_available = False

        # Apply IAX2-specific settings (faster connection = need longer capture)
        if config.provider_type == "iax2":
            # IAX2 connects in 1-2s vs 10-15s for modem, so we need more time
            # to capture full handshake for fax/modem distinction (2100 Hz + 2200 Hz)
            if config.early_hangup_interval < 10.0:
                config.early_hangup_interval = 10.0
            # Remove SIT_TONE from early hangup - often confused with IVR on fast connects
            if LineType.SIT_TONE in config.early_hangup_types:
                config.early_hangup_types = [t for t in config.early_hangup_types if t != LineType.SIT_TONE]
            # Use 20s record duration for IAX2 (vs 25s for modem)
            if config.record_duration > 20:
                config.record_duration = 20

    def _create_provider(self) -> TelephonyProvider:
        """Create a telephony provider based on configuration."""
        if self.config.provider_type == "iax2":
            from .iax2_provider import IAX2Provider

            if not self.config.iax2_host:
                raise ScannerError("IAX2 host is required for IAX2 provider")
            if not self.config.iax2_username:
                raise ScannerError("IAX2 username is required for IAX2 provider")
            if not self.config.iax2_password:
                raise ScannerError("IAX2 password is required for IAX2 provider")

            return IAX2Provider(
                host=self.config.iax2_host,
                port=self.config.iax2_port,
                username=self.config.iax2_username,
                password=self.config.iax2_password,
                caller_id=self.config.iax2_caller_id or '',
            )
        else:
            # Default to modem provider
            from .modem_provider import ModemProvider

            return ModemProvider(
                device=self.config.modem_device,
                baud_rate=self.config.baud_rate,
                dial_mode=self.config.dial_mode,
                detect_dial_tone=self.config.detect_dial_tone,
                sample_rate=self.config.sample_rate,
            )

    def _ensure_provider(self) -> None:
        """Ensure provider is connected and initialized."""
        if self.provider is None:
            self.provider = self._create_provider()

        try:
            self.provider.connect()
            info = self.provider.get_info()
            self._voice_mode_available = info.voice_supported
            logger.debug(f"Provider connected: {info.name}, voice={info.voice_supported}")
        except ProviderError as e:
            raise ScannerError(f"Failed to connect provider: {e}") from e

    def _create_early_check_callback(self, number: str) -> Optional[Callable[[np.ndarray, float], bool]]:
        """
        Create early check callback for fax/modem detection.

        Returns a callback function that returns False to stop recording early
        when a fax/modem/SIT tone is detected with high confidence.
        """
        if not self.config.early_hangup:
            return None

        def check_for_early_hangup(samples: np.ndarray, elapsed: float) -> bool:
            """Check partial audio for clear fax/modem signatures. Return False to stop."""
            try:
                if len(samples) < self.config.sample_rate:  # Need at least 1 second
                    return True  # Continue recording

                # Classify partial audio
                classification = self.classifier.classify(samples)

                # Check if we detected a type that warrants early hangup
                if (classification.line_type in self.config.early_hangup_types and
                    classification.confidence >= self.config.early_hangup_min_confidence):
                    logger.info(
                        f"[{number}] Early detection: {classification.line_type.value} "
                        f"({classification.confidence:.0%}) at {elapsed:.1f}s"
                    )
                    return False  # Stop recording

                return True  # Continue recording
            except Exception as e:
                logger.debug(f"[{number}] Early check error: {e}")
                return True  # Continue on error

        return check_for_early_hangup

    def _dial_and_record(
        self,
        number: str
    ) -> tuple[CallResult, Optional[np.ndarray], float]:
        """
        Dial a number and optionally record audio.

        Returns:
            Tuple of (call_result, audio_samples or None, duration)
        """
        logger.debug(f"[{number}] Starting dial_and_record")
        self._ensure_provider()
        logger.debug(f"[{number}] Provider ready")

        audio_samples = None
        duration = 0.0

        # Use voice-aware dialing if voice mode is available
        if self._voice_mode_available:
            logger.debug(f"[{number}] Calling dial_for_voice...")
            result = self.provider.dial_for_voice(number, timeout=self.config.call_timeout)
            logger.debug(f"[{number}] dial_for_voice returned: {result}")

            if result == CallResult.VOICE:
                try:
                    # Create early check callback for fax/modem detection
                    early_callback = self._create_early_check_callback(number)

                    logger.debug(f"[{number}] Reading voice data (max {self.config.record_duration}s)...")
                    # Provider's read_voice_data returns decoded samples directly
                    audio_samples = self.provider.read_voice_data(
                        duration=self.config.record_duration,
                        silence_timeout=5.0,
                        early_check_callback=early_callback,
                        early_check_interval=self.config.early_hangup_interval,
                    )

                    if audio_samples is not None and len(audio_samples) > 0:
                        duration = len(audio_samples) / self.config.sample_rate
                        logger.debug(f"[{number}] Got {len(audio_samples)} samples ({duration:.1f}s)")
                    else:
                        logger.debug(f"[{number}] No audio samples received")

                except ProviderError as e:
                    logger.debug(f"[{number}] ProviderError during recording: {e}")
                    # Failed to record, but we still got a connection
                    pass
        else:
            logger.debug(f"[{number}] No voice mode available")
            # Without voice mode we can only get call result without audio
            result = self.provider.dial_for_voice(number, timeout=self.config.call_timeout)
            logger.debug(f"[{number}] dial_for_voice returned: {result}")

        logger.debug(f"[{number}] dial_and_record complete, result={result}")
        return result, audio_samples, duration

    def _classify_result(
        self,
        call_result: CallResult,
        audio_samples: Optional[np.ndarray]
    ) -> tuple[LineType, float, list[int], str]:
        """
        Classify the line type based on call result and audio.

        Returns:
            Tuple of (line_type, confidence, frequencies, notes)
        """
        # Non-answer results have predetermined classifications
        if call_result == CallResult.BUSY:
            return LineType.BUSY, 1.0, [], "Busy signal from result code"

        if call_result == CallResult.NO_ANSWER:
            return LineType.UNKNOWN, 0.0, [], "No answer"

        if call_result == CallResult.NO_DIALTONE:
            return LineType.UNKNOWN, 0.0, [], "No dial tone"

        if call_result == CallResult.ERROR:
            return LineType.UNKNOWN, 0.0, [], "Provider error"

        if call_result == CallResult.TIMEOUT:
            return LineType.UNKNOWN, 0.0, [], "Timeout"

        if call_result == CallResult.UNREACHABLE:
            # Could be fax/modem that dropped when no handshake
            return LineType.UNKNOWN, 0.3, [], "Unreachable - possible fax/modem"

        if call_result == CallResult.REJECTED:
            return LineType.UNKNOWN, 0.0, [], "Call rejected"

        # For answered calls, classify based on audio
        if audio_samples is not None and len(audio_samples) > 0:
            classification = self.classifier.classify(audio_samples)
            return (
                classification.line_type,
                classification.confidence,
                list(set(f for f in (
                    m.matched_frequencies
                    for m in classification.matches
                ) for f in f)),
                classification.reasoning,
            )

        # Answered but no audio (voice mode not available)
        if call_result == CallResult.VOICE:
            return LineType.UNKNOWN, 0.5, [], "Answered but no audio capture"

        return LineType.UNKNOWN, 0.0, [], "Unknown result"

    def _save_audio(
        self,
        samples: np.ndarray,
        scan_id: int,
        phone_number: str,
        line_type: LineType
    ) -> Optional[str]:
        """Save audio recording if configured to keep it."""
        if line_type not in self.config.keep_audio_types:
            return None

        if len(samples) == 0:
            return None

        # Create output directory
        output_dir = self.config.audio_output_dir / str(scan_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{phone_number}_{line_type.value}.wav"
        filepath = output_dir / filename

        # Save WAV
        save_wav(samples, filepath, self.config.sample_rate)

        return str(filepath)

    def scan_number(self, number: str, scan_id: int) -> ScanResult:
        """
        Scan a single phone number.

        Args:
            number: Phone number to scan
            scan_id: Parent scan ID

        Returns:
            ScanResult with classification
        """
        # Dial and record
        call_result, audio_samples, duration = self._dial_and_record(number)

        # Classify
        line_type, confidence, frequencies, notes = self._classify_result(
            call_result, audio_samples
        )

        # Save audio if applicable
        audio_file = None
        if audio_samples is not None:
            audio_file = self._save_audio(audio_samples, scan_id, number, line_type)

        # Create result
        result = ScanResult(
            scan_id=scan_id,
            phone_number=number,
            call_result=call_result,
            line_type=line_type,
            confidence=confidence,
            audio_file=audio_file,
            duration=duration,
            frequencies=frequencies,
            notes=notes,
            scanned_at=datetime.now(),
        )

        return result

    def _reset_provider(self) -> None:
        """Fully disconnect and reset provider state for clean next call."""
        logger.debug("_reset_provider: Disconnecting provider completely")
        if self.provider:
            try:
                self.provider.disconnect()
            except Exception as e:
                logger.debug(f"_reset_provider: Disconnect error (ignored): {e}")
        # Clear provider reference so _ensure_provider creates fresh connection
        self.provider = None
        self._voice_mode_available = False
        logger.debug("_reset_provider: Provider reset complete")

    def run_scan(
        self,
        numbers: list[str],
        name: str = "Scan",
        description: str = "",
        resume_scan_id: Optional[int] = None,
    ) -> Scan:
        """
        Run a full scan on a list of phone numbers.

        Each number is scanned with a fresh provider connection to avoid
        state issues between calls.

        Args:
            numbers: List of phone numbers to scan
            name: Name for this scan
            description: Optional description
            resume_scan_id: If provided, resume this existing scan

        Returns:
            Completed Scan object
        """
        self._stop_requested = False

        # Create or resume scan
        if resume_scan_id:
            scan = self.db.get_scan(resume_scan_id)
            if scan is None:
                raise ScannerError(f"Scan {resume_scan_id} not found")
        else:
            scan = Scan(
                name=name,
                description=description,
                status=ScanStatus.PENDING,
                total_numbers=len(numbers),
                created_at=datetime.now(),
                modem_device=self.config.modem_device,
            )
            scan = self.db.create_scan(scan)

        # Mark as running
        scan.status = ScanStatus.RUNNING
        scan.started_at = datetime.now()
        self.db.update_scan(scan)
        self._current_scan = scan

        try:
            # Process each number
            for i, number in enumerate(numbers):
                logger.debug(f"=== Starting number {i+1}/{len(numbers)}: {number} ===")

                if self._stop_requested:
                    scan.status = ScanStatus.PAUSED
                    break

                # Skip if already scanned
                if self.db.number_exists_in_scan(scan.id, number):
                    logger.debug(f"[{number}] Already scanned, skipping")
                    continue

                # Scan the number with retry logic
                result = None
                retry_count = 0
                retryable_results = {
                    CallResult.ERROR,
                    CallResult.NO_DIALTONE,
                    CallResult.TIMEOUT,
                }

                while retry_count <= self.config.max_retries:
                    logger.debug(f"[{number}] Attempt {retry_count + 1}/{self.config.max_retries + 1}")

                    # Scan this number
                    try:
                        result = self.scan_number(number, scan.id)
                        result.retry_count = retry_count
                        logger.debug(f"[{number}] scan_number returned: {result.call_result}, {result.line_type}")
                    finally:
                        # CRITICAL: Disconnect provider after EVERY call attempt
                        # This ensures clean state for next call
                        logger.debug(f"[{number}] Resetting provider after call...")
                        self._reset_provider()

                    # Check if we should retry
                    if result.call_result in retryable_results and retry_count < self.config.max_retries:
                        retry_count += 1
                        logger.debug(f"[{number}] Retrying in {self.config.call_delay}s...")
                        time.sleep(self.config.call_delay)
                        continue

                    break  # Success or non-retryable result

                # Save result
                logger.debug(f"[{number}] Saving result to database...")
                self.db.add_result(result)

                # Update scan progress
                scan.completed_numbers = i + 1
                self.db.update_scan(scan)
                logger.debug(f"[{number}] Progress updated: {i+1}/{len(numbers)}")

                # Callback
                if self.progress_callback:
                    logger.debug(f"[{number}] Calling progress callback...")
                    self.progress_callback(i + 1, len(numbers), result)

                # Delay between calls
                if i < len(numbers) - 1 and not self._stop_requested:
                    logger.debug(f"[{number}] Waiting {self.config.call_delay}s before next call...")
                    time.sleep(self.config.call_delay)
                    logger.debug(f"[{number}] Delay complete, moving to next number")

            # Mark complete
            if not self._stop_requested:
                scan.status = ScanStatus.COMPLETED
            scan.completed_at = datetime.now()
            self.db.update_scan(scan)

        except Exception as e:
            scan.status = ScanStatus.FAILED
            scan.completed_at = datetime.now()
            self.db.update_scan(scan)
            raise ScannerError(f"Scan failed: {e}") from e

        finally:
            # Final cleanup
            self._reset_provider()
            self._current_scan = None

        return scan

    def stop(self) -> None:
        """Request scan to stop after current number."""
        self._stop_requested = True

    def get_summary(self, scan_id: int):
        """Get summary for a scan."""
        return self.db.get_scan_summary(scan_id)

    @property
    def is_running(self) -> bool:
        """Check if a scan is currently running."""
        return self._current_scan is not None and not self._stop_requested


def create_scanner_from_config(config_path: str) -> Scanner:
    """Create a Scanner from a YAML config file."""
    import yaml

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    modem_cfg = config_data.get("modem", {})
    iax2_cfg = config_data.get("iax2", {})
    audio_cfg = config_data.get("audio", {})
    scanner_cfg = config_data.get("scanner", {})
    db_cfg = config_data.get("database", {})

    config = ScannerConfig(
        # Provider selection
        provider_type=config_data.get("provider", "modem"),
        # Modem settings
        modem_device=modem_cfg.get("device"),
        baud_rate=modem_cfg.get("baud_rate", 460800),
        dial_mode=modem_cfg.get("dial_mode", "tone"),
        detect_dial_tone=modem_cfg.get("detect_dial_tone", False),
        init_string=modem_cfg.get("init_string"),
        # IAX2 settings
        iax2_host=iax2_cfg.get("host"),
        iax2_port=iax2_cfg.get("port", 4569),
        iax2_username=iax2_cfg.get("username"),
        iax2_password=iax2_cfg.get("password"),
        iax2_caller_id=iax2_cfg.get("caller_id"),
        # Common settings
        call_timeout=modem_cfg.get("call_timeout", 45),
        record_duration=modem_cfg.get("record_duration", 25),
        sample_rate=audio_cfg.get("sample_rate", 8000),
        audio_output_dir=audio_cfg.get("output_dir", "./recordings"),
        call_delay=scanner_cfg.get("call_delay", 3),
        max_retries=scanner_cfg.get("max_retries", 2),
        db_path=db_cfg.get("path", "./dialer.db"),
    )

    return Scanner(config)
