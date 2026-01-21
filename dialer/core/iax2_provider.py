"""IAX2 telephony provider implementation for VOIP.MS and compatible servers."""

import logging
from typing import Callable, Optional

import numpy as np

from .iax2 import CallState, IAX2Client
from .provider import CallResult, ProviderError, ProviderInfo, TelephonyProvider

logger = logging.getLogger(__name__)


class IAX2Provider(TelephonyProvider):
    """Telephony provider using IAX2 protocol (VOIP.MS, Asterisk, etc.).

    IAX2 is the Inter-Asterisk eXchange protocol (RFC 5456), a VoIP signaling
    protocol that combines signaling and media in a single UDP stream. This
    provider enables making calls through VOIP.MS or any IAX2-compatible server.
    """

    # Map IAX2 CallState to provider CallResult
    STATE_MAP = {
        CallState.ANSWERED: CallResult.VOICE,
        CallState.RINGING: CallResult.VOICE,  # Treat ringing that times out as no answer
        CallState.REJECTED: CallResult.REJECTED,
        CallState.HANGUP: CallResult.REJECTED,
        CallState.ERROR: CallResult.ERROR,
        CallState.IDLE: CallResult.ERROR,
    }

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        caller_id: str = '',
        port: int = 4569,
    ):
        """Initialize the IAX2 provider.

        Args:
            host: IAX2 server hostname (e.g., 'chicago3.voip.ms')
            username: IAX2 account username
            password: IAX2 account password
            caller_id: Caller ID to send (optional)
            port: IAX2 server port (default 4569)
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.caller_id = caller_id

        self._client: Optional[IAX2Client] = None

    def connect(self) -> None:
        """Connect to the IAX2 server."""
        if self._client is not None:
            return

        try:
            self._client = IAX2Client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                caller_id=self.caller_id,
            )
            self._client.connect()
            logger.info(f"Connected to IAX2 server {self.host}:{self.port}")
        except Exception as e:
            self._client = None
            raise ProviderError(f"Failed to connect to IAX2 server: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from the IAX2 server."""
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    def get_info(self) -> ProviderInfo:
        """Get information about this provider."""
        connected = self._client is not None and self._client.is_connected
        return ProviderInfo(
            name=f"IAX2 ({self.host})",
            description=f"VOIP via IAX2 server {self.host}:{self.port}",
            protocol="iax2",
            voice_supported=True,
        )

    def dial_for_voice(
        self,
        number: str,
        timeout: float = 45.0,
    ) -> CallResult:
        """Dial a number via IAX2 and wait for answer."""
        if self._client is None:
            raise ProviderError("Not connected")

        try:
            state = self._client.call(number=number, timeout=timeout)

            # Map call state to result
            if state == CallState.ANSWERED:
                return CallResult.VOICE
            elif state == CallState.REJECTED:
                cause = self._client.call_cause.upper()
                if 'BUSY' in cause:
                    return CallResult.BUSY
                elif 'CONGESTION' in cause:
                    return CallResult.UNREACHABLE
                return CallResult.REJECTED
            elif state == CallState.HANGUP:
                return CallResult.NO_ANSWER
            else:
                return CallResult.TIMEOUT

        except Exception as e:
            logger.error(f"IAX2 dial failed: {e}")
            return CallResult.ERROR

    def read_voice_data(
        self,
        duration: float,
        silence_timeout: float = 5.0,
        early_check_callback: Optional[Callable[[np.ndarray, float], bool]] = None,
        early_check_interval: float = 2.0,
    ) -> np.ndarray:
        """Read voice data from the active IAX2 call."""
        if self._client is None:
            raise ProviderError("Not connected")

        try:
            samples = self._client.receive_audio(
                duration=duration,
                silence_timeout=silence_timeout,
                early_check_callback=early_check_callback,
                early_check_interval=early_check_interval,
            )
            return samples
        except Exception as e:
            logger.error(f"Error reading IAX2 voice data: {e}")
            return np.array([], dtype=np.float32)

    def hangup(self) -> bool:
        """Hang up the current call."""
        if self._client is None:
            return True

        try:
            self._client.hangup()
            return True
        except Exception as e:
            logger.error(f"IAX2 hangup failed: {e}")
            return False

    def reset_for_next_call(self) -> bool:
        """Reset state for the next call.

        For IAX2, we keep the UDP connection open but clear any call state.
        """
        # The IAX2Client automatically clears call state when starting a new call
        return True
