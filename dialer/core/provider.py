"""Abstract telephony provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np


class CallResult(Enum):
    """Unified call result codes for all providers."""
    VOICE = "VOICE"           # Connected, ready for audio
    BUSY = "BUSY"             # Line busy
    NO_ANSWER = "NO_ANSWER"   # No answer after timeout
    NO_DIALTONE = "NO_DIALTONE"  # No dial tone detected
    REJECTED = "REJECTED"     # Call rejected/declined
    UNREACHABLE = "UNREACHABLE"  # Network/routing failure
    ERROR = "ERROR"           # General error
    TIMEOUT = "TIMEOUT"       # Operation timed out
    UNKNOWN = "UNKNOWN"       # Unknown/unclassified result


@dataclass
class ProviderInfo:
    """Information about a telephony provider."""
    name: str
    description: str
    protocol: str  # "usb_modem", "iax2", etc.
    voice_supported: bool


class ProviderError(Exception):
    """Exception raised for provider errors."""
    pass


class TelephonyProvider(ABC):
    """Abstract base class for telephony backends.

    Implementations handle the specifics of connecting, dialing,
    and capturing audio for their respective protocols. The Scanner
    uses this interface to remain protocol-agnostic.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the telephony backend.

        Raises:
            ProviderError: If connection fails.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the telephony backend."""
        ...

    @abstractmethod
    def get_info(self) -> ProviderInfo:
        """Get information about this provider.

        Returns:
            ProviderInfo with name, description, and capabilities.
        """
        ...

    @abstractmethod
    def dial_for_voice(
        self,
        number: str,
        timeout: float = 45.0,
    ) -> CallResult:
        """Dial a number and wait for voice connection.

        Args:
            number: Phone number to dial.
            timeout: Maximum seconds to wait for answer.

        Returns:
            CallResult indicating outcome.
        """
        ...

    @abstractmethod
    def read_voice_data(
        self,
        duration: float,
        silence_timeout: float = 5.0,
        early_check_callback: Optional[Callable[[np.ndarray, float], bool]] = None,
        early_check_interval: float = 2.0,
    ) -> np.ndarray:
        """Read voice data from an active call.

        Called after dial_for_voice() returns CallResult.VOICE.

        Args:
            duration: Maximum seconds to record.
            silence_timeout: Stop after this many seconds of silence.
            early_check_callback: Optional callback(samples, elapsed) -> bool.
                                  Called periodically with accumulated audio.
                                  Return False to stop recording early.
            early_check_interval: Seconds between early check callbacks.

        Returns:
            Audio samples as float32 numpy array, normalized to [-1.0, 1.0].
        """
        ...

    @abstractmethod
    def hangup(self) -> bool:
        """Hang up the current call.

        Returns:
            True if hangup succeeded.
        """
        ...

    @abstractmethod
    def reset_for_next_call(self) -> bool:
        """Reset provider state for the next call.

        Called between calls to ensure clean state.

        Returns:
            True if reset succeeded.
        """
        ...
