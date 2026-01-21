"""USB Modem telephony provider implementation."""

import logging
from typing import Callable, Optional

import numpy as np

from .audio import AudioCapture
from .modem import CallResult as ModemCallResult
from .modem import Modem, ModemError
from .provider import CallResult, ProviderError, ProviderInfo, TelephonyProvider

logger = logging.getLogger(__name__)


class ModemProvider(TelephonyProvider):
    """Telephony provider using USB modem hardware.

    Wraps the existing Modem class to implement the TelephonyProvider
    interface. Handles DLE decoding internally and returns numpy samples.
    """

    # Map modem CallResult to provider CallResult
    RESULT_MAP = {
        ModemCallResult.VOICE: CallResult.VOICE,
        ModemCallResult.OK: CallResult.VOICE,
        ModemCallResult.CONNECT: CallResult.VOICE,
        ModemCallResult.BUSY: CallResult.BUSY,
        ModemCallResult.NO_ANSWER: CallResult.NO_ANSWER,
        ModemCallResult.NO_DIALTONE: CallResult.NO_DIALTONE,
        ModemCallResult.NO_CARRIER: CallResult.UNREACHABLE,
        ModemCallResult.ERROR: CallResult.ERROR,
        ModemCallResult.TIMEOUT: CallResult.TIMEOUT,
        ModemCallResult.UNKNOWN: CallResult.ERROR,
        ModemCallResult.RING: CallResult.ERROR,  # Unexpected state
    }

    def __init__(
        self,
        device: Optional[str] = None,
        baud_rate: int = 460800,
        timeout: float = 2.0,
        dial_mode: str = "tone",
        detect_dial_tone: bool = False,
        sample_rate: int = 8000,
        encoding: str = "pcm_unsigned",
    ):
        """Initialize the modem provider.

        Args:
            device: Modem device path (e.g., /dev/ttyACM0). Auto-detected if None.
            baud_rate: Serial baud rate.
            timeout: Serial timeout in seconds.
            dial_mode: "tone" (ATDT) or "pulse" (ATDP).
            detect_dial_tone: Wait for dial tone before dialing.
            sample_rate: Audio sample rate in Hz.
            encoding: Audio encoding ("pcm_unsigned", "pcm_signed", "ulaw").
        """
        self.device = device
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.dial_mode = dial_mode
        self.detect_dial_tone = detect_dial_tone
        self.sample_rate = sample_rate
        self.encoding = encoding

        self._modem: Optional[Modem] = None
        self._audio_capture = AudioCapture(sample_rate=sample_rate, encoding=encoding)
        self._in_voice_receive = False

    def connect(self) -> None:
        """Connect to the USB modem."""
        if self._modem is not None:
            return

        try:
            self._modem = Modem(
                device=self.device,
                baud_rate=self.baud_rate,
                timeout=self.timeout,
                dial_mode=self.dial_mode,
                detect_dial_tone=self.detect_dial_tone,
            )
            self._modem.connect()
            self._modem.initialize()

            # Check for voice capability
            info = self._modem.get_info()
            if not info.voice_supported:
                logger.warning(
                    f"Modem {info.model} does not support voice mode. "
                    "Audio capture will not be available."
                )
        except ModemError as e:
            self._modem = None
            raise ProviderError(f"Failed to connect to modem: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the USB modem."""
        if self._modem is not None:
            try:
                self._modem.disconnect()
            except Exception:
                pass
            self._modem = None
        self._in_voice_receive = False

    def get_info(self) -> ProviderInfo:
        """Get information about the modem provider."""
        if self._modem is None:
            return ProviderInfo(
                name="USB Modem",
                description="USB modem (not connected)",
                protocol="usb_modem",
                voice_supported=False,
            )

        try:
            modem_info = self._modem.get_info()
            return ProviderInfo(
                name=f"{modem_info.manufacturer} {modem_info.model}",
                description=f"USB modem on {modem_info.device}",
                protocol="usb_modem",
                voice_supported=modem_info.voice_supported,
            )
        except ModemError:
            return ProviderInfo(
                name="USB Modem",
                description="USB modem (info unavailable)",
                protocol="usb_modem",
                voice_supported=False,
            )

    def dial_for_voice(
        self,
        number: str,
        timeout: float = 45.0,
    ) -> CallResult:
        """Dial a number and prepare for voice recording."""
        if self._modem is None:
            raise ProviderError("Not connected")

        try:
            result = self._modem.dial_for_voice(
                number=number,
                timeout=timeout,
                wait_for_dial_tone=self.detect_dial_tone,
            )
            return self.RESULT_MAP.get(result, CallResult.ERROR)
        except ModemError as e:
            logger.error(f"Dial failed: {e}")
            return CallResult.ERROR

    def read_voice_data(
        self,
        duration: float,
        silence_timeout: float = 5.0,
        early_check_callback: Optional[Callable[[np.ndarray, float], bool]] = None,
        early_check_interval: float = 2.0,
    ) -> np.ndarray:
        """Read voice data and return decoded samples.

        This method starts voice receive mode, reads raw data from the modem,
        then decodes DLE encoding and converts to numpy samples.
        """
        if self._modem is None:
            raise ProviderError("Not connected")

        # Start voice receive mode if not already active
        if not self._in_voice_receive:
            if not self._modem.start_voice_receive():
                logger.error("Failed to start voice receive mode")
                return np.array([], dtype=np.float32)
            self._in_voice_receive = True

        # Create an internal callback that converts raw bytes to samples
        # before passing to the user's callback
        internal_callback = None
        if early_check_callback is not None:
            def internal_callback(raw_data: bytes, elapsed: float) -> bool:
                samples = self._audio_capture.process_raw_data(raw_data)
                return early_check_callback(samples, elapsed)

        try:
            # Read raw voice data from modem
            raw_data = self._modem.read_voice_data(
                duration=duration,
                silence_timeout=silence_timeout,
                silence_threshold=3,  # Low level for silence detection
                min_record_time=5.0,  # Don't stop early in first 5 seconds
                early_check_callback=internal_callback,
                early_check_interval=early_check_interval,
            )

            # Stop voice receive mode
            self._modem.stop_voice_receive()
            self._in_voice_receive = False

            # Decode DLE and convert to samples
            samples = self._audio_capture.process_raw_data(raw_data)
            return samples

        except ModemError as e:
            logger.error(f"Error reading voice data: {e}")
            self._in_voice_receive = False
            return np.array([], dtype=np.float32)

    def hangup(self) -> bool:
        """Hang up the current call."""
        if self._modem is None:
            return True

        try:
            self._in_voice_receive = False
            return self._modem.hangup()
        except ModemError as e:
            logger.error(f"Hangup failed: {e}")
            return False

    def reset_for_next_call(self) -> bool:
        """Reset modem state for the next call.

        For maximum reliability, we do a full disconnect/reconnect
        between calls. This ensures clean state.
        """
        self._in_voice_receive = False

        if self._modem is not None:
            try:
                return self._modem.reset_for_next_call()
            except ModemError:
                return False

        return True
