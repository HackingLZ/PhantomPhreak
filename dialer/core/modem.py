"""Modem control via AT commands."""

import glob
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import serial

logger = logging.getLogger(__name__)


class ModemCapability(Enum):
    """Modem feature capabilities."""
    DATA_ONLY = "data_only"
    VOICE = "voice"
    FAX = "fax"


class CallResult(Enum):
    """Result codes from dialing."""
    OK = "OK"
    CONNECT = "CONNECT"
    RING = "RING"
    NO_CARRIER = "NO CARRIER"
    ERROR = "ERROR"
    NO_DIALTONE = "NO DIALTONE"
    BUSY = "BUSY"
    NO_ANSWER = "NO ANSWER"
    VOICE = "VCON"  # Voice connection (voice mode)
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class ModemInfo:
    """Information about a detected modem."""
    device: str
    manufacturer: str
    model: str
    capabilities: list[ModemCapability]
    voice_supported: bool


class ModemError(Exception):
    """Exception raised for modem communication errors."""
    pass


class Modem:
    """Controls a USB modem via AT commands."""

    # DLE (Data Link Escape) codes for voice mode
    DLE = b'\x10'  # ASCII DLE character
    ETX = b'\x03'  # ASCII ETX (end of text)
    DLE_ETX = b'\x10\x03'  # End voice data stream
    DLE_SHIELDED = b'\x10\x10'  # Escaped DLE in data stream

    def __init__(
        self,
        device: Optional[str] = None,
        baud_rate: int = 460800,
        timeout: float = 2.0,
        dial_mode: str = "tone",  # "tone" (ATDT) or "pulse" (ATDP)
        detect_dial_tone: bool = False,
    ):
        self.device = device
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.dial_mode = dial_mode
        self.detect_dial_tone = detect_dial_tone
        self._serial: Optional[serial.Serial] = None
        self._voice_supported: bool = False
        self._in_voice_mode: bool = False

    @staticmethod
    def detect_devices() -> list[str]:
        """Find potential modem devices on the system."""
        patterns = [
            '/dev/ttyACM*',
            '/dev/ttyUSB*',
            '/dev/cu.usbmodem*',
            '/dev/cu.usbserial*',
        ]
        devices = []
        for pattern in patterns:
            devices.extend(glob.glob(pattern))
        return sorted(devices)

    def connect(self) -> None:
        """Open serial connection to modem."""
        if self._serial and self._serial.is_open:
            return

        if not self.device:
            devices = self.detect_devices()
            if not devices:
                raise ModemError("No modem devices found")
            self.device = devices[0]

        try:
            self._serial = serial.Serial(
                port=self.device,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False,
            )
            time.sleep(0.5)  # Allow modem to initialize
            self._flush()
        except serial.SerialException as e:
            raise ModemError(f"Failed to open {self.device}: {e}")

    def disconnect(self) -> None:
        """Close serial connection (hard reset - just close the port)."""
        if self._serial and self._serial.is_open:
            try:
                # Quick flush and close - no need for complex hangup
                # since closing serial port resets the modem
                self._serial.reset_input_buffer()
                self._serial.reset_output_buffer()
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._in_voice_mode = False

    def _flush(self) -> None:
        """Flush input/output buffers."""
        if self._serial:
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

    def _send(self, command: str) -> None:
        """Send AT command to modem."""
        if not self._serial:
            raise ModemError("Not connected")
        self._flush()
        cmd = f"{command}\r".encode('ascii')
        self._serial.write(cmd)
        time.sleep(0.1)

    def _read_response(self, timeout: Optional[float] = None) -> str:
        """Read response from modem until OK/ERROR or timeout."""
        if not self._serial:
            raise ModemError("Not connected")

        old_timeout = self._serial.timeout
        effective_timeout = timeout or self.timeout or 2.0
        self._serial.timeout = min(effective_timeout, 2.0)  # Max 2s per readline

        response_lines = []
        start_time = time.time()
        empty_reads = 0
        max_empty_reads = 3  # Give up after 3 empty reads with no content

        try:
            while True:
                # Check overall timeout
                if (time.time() - start_time) > effective_timeout:
                    logger.debug(f"_read_response: Timeout after {effective_timeout}s")
                    break

                line = self._serial.readline().decode('ascii', errors='ignore').strip()
                if line:
                    logger.debug(f"_read_response: Got line: {line}")
                    response_lines.append(line)
                    empty_reads = 0  # Reset empty counter
                    # Check for terminal responses
                    if line in ('OK', 'ERROR'):
                        break
                    if line.startswith(('CONNECT', 'NO CARRIER', 'BUSY',
                                       'NO DIALTONE', 'NO ANSWER', 'VCON')):
                        break
                else:
                    empty_reads += 1
                    if not response_lines:
                        # Haven't received anything yet
                        if empty_reads >= max_empty_reads:
                            logger.debug(f"_read_response: No response after {empty_reads} reads")
                            break
                    else:
                        # Empty line after content, might be done
                        if self._serial.in_waiting == 0:
                            break
        finally:
            self._serial.timeout = old_timeout

        return '\n'.join(response_lines)

    def send_command(self, command: str, timeout: Optional[float] = None) -> str:
        """Send AT command and return response."""
        self._send(command)
        return self._read_response(timeout)

    def reset(self) -> bool:
        """Reset modem to default settings."""
        response = self.send_command("ATZ")
        return "OK" in response

    def initialize(self) -> bool:
        """
        Initialize modem with standard settings (matching wvdial defaults).

        Sends: ATQ0 V1 E1 S0=0 &C1 &D2 +FCLASS=0
        - Q0: Enable result codes
        - V1: Verbose result codes (words instead of numbers)
        - E1: Echo commands back
        - S0=0: Don't auto-answer incoming calls
        - &C1: DCD follows carrier state
        - &D2: Hangup on DTR drop
        - +FCLASS=0: Data mode (not fax/voice initially)
        """
        # Reset first
        if not self.reset():
            return False

        # Send initialization string
        response = self.send_command("ATQ0 V1 E1 S0=0 &C1 &D2 +FCLASS=0")
        return "OK" in response

    def check_dial_tone(self, timeout: float = 5.0) -> bool:
        """
        Check if dial tone is present on the line.

        Uses ATX4 to enable dial tone detection, then attempts to go off-hook
        and detect the dial tone. This is useful for verifying line connectivity
        before making calls.

        Args:
            timeout: Seconds to wait for dial tone

        Returns:
            True if dial tone detected, False otherwise
        """
        if not self._serial:
            raise ModemError("Not connected")

        # ATX4 enables full result codes including NO DIALTONE detection
        # ATX3 would disable dial tone detection
        response = self.send_command("ATX4")
        if "OK" not in response:
            # Some modems may not support ATX4, try anyway
            pass

        # Go off-hook and wait for dial tone
        # ATH1 = go off-hook (pick up the line)
        response = self.send_command("ATH1", timeout=timeout)

        # Wait briefly for dial tone to stabilize
        time.sleep(0.5)

        # Try to dial with 'W' which waits for dial tone
        # ATDTW, dials nothing but checks for dial tone
        # Using comma for pause, then check response
        self._send("ATDT,")
        response = self._read_response(timeout=timeout)

        # Hang up
        self.send_command("ATH")

        # Check result - NO DIALTONE means no dial tone
        if "NO DIALTONE" in response.upper():
            return False

        # If we got OK or any other response, dial tone was likely present
        return "OK" in response or "NO DIALTONE" not in response.upper()

    def wait_for_dial_tone(self, timeout: float = 10.0) -> bool:
        """
        Wait for dial tone using voice mode audio analysis.

        This method uses the modem's voice mode to listen for the
        characteristic 350+440 Hz dial tone frequencies.

        Args:
            timeout: Maximum seconds to wait for dial tone

        Returns:
            True if dial tone detected, False if timeout or no dial tone
        """
        if not self._voice_supported:
            info = self.get_info()
            if not info.voice_supported:
                # Fall back to simple check if voice not supported
                return self.check_dial_tone(timeout)

        # Enter voice mode
        if not self._in_voice_mode:
            if not self.init_voice_mode():
                return self.check_dial_tone(timeout)

        # Go off-hook
        self.send_command("ATH1")
        time.sleep(0.3)

        # Start voice receive to capture dial tone
        self.start_voice_receive()

        # Read a short sample (1-2 seconds should be enough)
        raw_data = self.read_voice_data(2.0)

        # Stop and hang up
        self.stop_voice_receive()
        self.send_command("ATH")

        if not raw_data or len(raw_data) < 1000:
            return False

        # Analyze for dial tone frequencies (350 Hz and 440 Hz)
        # This is a simple check - the full classifier can be used for more accuracy
        return self._detect_dial_tone_frequencies(raw_data)

    def _detect_dial_tone_frequencies(self, raw_data: bytes) -> bool:
        """
        Analyze raw audio data for dial tone frequencies.

        North American dial tone is 350 Hz + 440 Hz.
        We check for energy in the 300-500 Hz band.

        Args:
            raw_data: Raw audio bytes from modem (DLE-encoded)

        Returns:
            True if dial tone frequencies detected
        """
        import numpy as np
        from scipy.fft import rfft, rfftfreq

        # Decode DLE-encoded data
        decoded = self._decode_dle_simple(raw_data)
        if len(decoded) < 512:
            return False

        # Convert to float samples (assume unsigned 8-bit PCM)
        samples = np.frombuffer(decoded, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128) / 128.0  # Normalize to -1 to 1

        # Compute FFT
        n = len(samples)
        sample_rate = 8000
        fft_result = np.abs(rfft(samples))
        freqs = rfftfreq(n, 1 / sample_rate)

        # Check for energy at dial tone frequencies (350 and 440 Hz)
        # Allow some tolerance (±50 Hz)
        dial_tone_bands = [(300, 400), (400, 500)]  # 350 Hz and 440 Hz bands
        total_energy = np.sum(fft_result ** 2)

        if total_energy < 0.01:
            return False  # Too quiet, no signal

        dial_tone_energy = 0
        for low, high in dial_tone_bands:
            mask = (freqs >= low) & (freqs <= high)
            dial_tone_energy += np.sum(fft_result[mask] ** 2)

        # Dial tone should have significant energy in these bands
        dial_tone_ratio = dial_tone_energy / total_energy
        return dial_tone_ratio > 0.3  # At least 30% of energy in dial tone bands

    def _decode_dle_simple(self, data: bytes) -> bytes:
        """Simple DLE decoding for dial tone detection."""
        result = bytearray()
        i = 0
        while i < len(data):
            if data[i:i+1] == self.DLE:
                if i + 1 < len(data):
                    next_byte = data[i+1:i+2]
                    if next_byte == self.DLE:
                        # Escaped DLE - include one DLE byte
                        result.append(0x10)
                        i += 2
                    elif next_byte == self.ETX:
                        # End of data
                        break
                    else:
                        # Other DLE sequence, skip
                        i += 2
                else:
                    i += 1
            else:
                result.append(data[i])
                i += 1
        return bytes(result)

    def get_info(self) -> ModemInfo:
        """Get modem identification and capabilities."""
        if not self._serial:
            self.connect()

        # Get manufacturer
        mfr_response = self.send_command("AT+GMI")
        manufacturer = mfr_response.split('\n')[0] if mfr_response else "Unknown"

        # Get model
        model_response = self.send_command("AT+GMM")
        model = model_response.split('\n')[0] if model_response else "Unknown"

        # Get capabilities (FCLASS support)
        fclass_response = self.send_command("AT+FCLASS=?")
        capabilities = []
        voice_supported = False

        if "0" in fclass_response:
            capabilities.append(ModemCapability.DATA_ONLY)
        if "1" in fclass_response or "1.0" in fclass_response:
            capabilities.append(ModemCapability.FAX)
        if "8" in fclass_response:
            capabilities.append(ModemCapability.VOICE)
            voice_supported = True

        self._voice_supported = voice_supported

        return ModemInfo(
            device=self.device or "",
            manufacturer=manufacturer.replace("OK", "").strip(),
            model=model.replace("OK", "").strip(),
            capabilities=capabilities,
            voice_supported=voice_supported,
        )

    def init_voice_mode(self) -> bool:
        """
        Initialize modem for voice mode recording.

        Note: This should be called AFTER dialing with dial_for_voice(),
        not before dialing. Many modems don't support dialing while in voice mode.
        """
        if not self._voice_supported:
            info = self.get_info()
            if not info.voice_supported:
                return False

        # Set voice mode
        response = self.send_command("AT+FCLASS=8")
        if "OK" not in response:
            return False

        # Initialize voice parameters (USR-specific command)
        self.send_command("AT+VIP")

        # Configure voice sampling: 128 = 8-bit linear PCM at 8000 Hz
        # Note: Format is AT+VSM=<compression>,<sample_rate>
        # USR modems use: 128=8-bit linear, 129=16-bit linear, 130=8-bit A-law, 131=8-bit u-law
        self.send_command("AT+VSM=128,8000")

        # Select telephone line for voice I/O
        self.send_command("AT+VLS=1")

        self._in_voice_mode = True
        return True

    def dial(
        self,
        number: str,
        timeout: float = 45.0,
        wait_for_dial_tone: bool = False
    ) -> CallResult:
        """
        Dial a phone number in data mode (waits for answer).

        Args:
            number: Phone number to dial
            timeout: Seconds to wait for answer
            wait_for_dial_tone: If True, wait for dial tone before dialing (adds 'W' prefix)

        Returns:
            CallResult indicating the outcome
        """
        # Clean number - remove non-digits except + for international
        clean_number = ''.join(c for c in number if c.isdigit() or c == '+')
        if not clean_number:
            raise ModemError(f"Invalid phone number: {number}")

        # Always use data mode for dialing (ATDT/ATDP)
        # Many modems don't support dialing in voice mode
        dial_char = "P" if self.dial_mode == "pulse" else "T"
        dial_prefix = "W" if (wait_for_dial_tone or self.detect_dial_tone) else ""
        command = f"ATD{dial_char}{dial_prefix}{clean_number}"

        self._send(command)
        response = self._read_response(timeout=timeout)

        return self._parse_dial_result(response)

    def dial_for_voice(
        self,
        number: str,
        timeout: float = 45.0,
        wait_for_dial_tone: bool = False
    ) -> CallResult:
        """
        Dial a phone number and prepare for voice recording.

        This method dials using the semicolon suffix to stay in command mode,
        then switches to voice mode. This works with modems that don't support
        dialing while already in voice mode.

        Args:
            number: Phone number to dial
            timeout: Seconds to wait for dial initiation
            wait_for_dial_tone: If True, wait for dial tone before dialing

        Returns:
            CallResult indicating the outcome (VOICE if ready for recording)
        """
        # Clean number
        clean_number = ''.join(c for c in number if c.isdigit() or c == '+')
        if not clean_number:
            raise ModemError(f"Invalid phone number: {number}")

        logger.debug(f"dial_for_voice: Starting dial to {clean_number}")

        # Ensure we're in data mode before dialing
        if self._in_voice_mode:
            logger.debug("dial_for_voice: Exiting voice mode first")
            self.send_command("AT+FCLASS=0")
            self._in_voice_mode = False

        # Dial with semicolon to return to command mode after dialing
        # This allows us to switch to voice mode while the call is ringing
        dial_char = "P" if self.dial_mode == "pulse" else "T"
        dial_prefix = "W" if (wait_for_dial_tone or self.detect_dial_tone) else ""
        command = f"ATD{dial_char}{dial_prefix}{clean_number};"

        logger.debug(f"dial_for_voice: Sending {command}")
        self._send(command)
        response = self._read_response(timeout=10)  # Short timeout for dial initiation
        logger.debug(f"dial_for_voice: Got response: {response[:100] if response else 'None'}")

        # Validate response is a real modem response, not garbage audio data
        # Real modem responses are short and mostly printable ASCII
        is_valid_response = (
            response and
            len(response) < 200 and  # Real responses are short
            sum(1 for c in response if c.isprintable() or c in '\r\n') > len(response) * 0.8  # Mostly printable
        )

        if not is_valid_response:
            logger.debug(f"dial_for_voice: Response looks like garbage audio data (len={len(response) if response else 0})")
            return CallResult.ERROR

        if "OK" not in response.upper():
            # Dial initiation failed
            result = self._parse_dial_result(response)
            logger.debug(f"dial_for_voice: Dial failed with {result}")
            return result

        # Dial initiated successfully, now switch to voice mode
        logger.debug("dial_for_voice: Dial OK, switching to voice mode...")
        if not self.init_voice_mode():
            # Failed to switch to voice mode, try to cancel the call
            logger.debug("dial_for_voice: Voice mode init failed, hanging up")
            self.hangup()
            return CallResult.ERROR

        # Voice mode active, ready for recording
        logger.debug("dial_for_voice: Success, voice mode active")
        return CallResult.VOICE

    def _parse_dial_result(self, response: str) -> CallResult:
        """Parse modem response to determine call result."""
        response_upper = response.upper()

        if "VCON" in response_upper:
            return CallResult.VOICE
        elif "CONNECT" in response_upper:
            return CallResult.CONNECT
        elif "BUSY" in response_upper:
            return CallResult.BUSY
        elif "NO CARRIER" in response_upper:
            return CallResult.NO_CARRIER
        elif "NO DIALTONE" in response_upper:
            return CallResult.NO_DIALTONE
        elif "NO ANSWER" in response_upper:
            return CallResult.NO_ANSWER
        elif "ERROR" in response_upper:
            return CallResult.ERROR
        else:
            return CallResult.TIMEOUT

    def start_voice_receive(self) -> bool:
        """Start receiving voice data from the line."""
        if not self._in_voice_mode:
            if not self.init_voice_mode():
                return False

        # Enter voice receive mode
        response = self.send_command("AT+VRX")
        return "CONNECT" in response or "OK" in response

    # DLE codes that indicate call/voice events
    DLE_BUSY = b'\x10b'      # Busy tone detected
    DLE_SILENCE = b'\x10s'   # Silence detected
    DLE_QUIET = b'\x10q'     # Quiet (ringback ended)
    DLE_HANGUP = b'\x10h'    # Remote hangup
    DLE_RING = b'\x10R'      # Ring detected

    def read_voice_data(
        self,
        duration: float,
        silence_timeout: float = 5.0,
        silence_threshold: int = 3,
        min_record_time: float = 5.0,
        early_check_callback: 'Callable[[bytes, float], bool] | None' = None,
        early_check_interval: float = 3.0
    ) -> bytes:
        """
        Read raw voice data from modem with early termination on hangup/silence.

        Args:
            duration: Maximum seconds of audio to capture
            silence_timeout: Seconds of continuous silence before assuming hangup
            silence_threshold: Audio level below which is considered silence (0-255)
            min_record_time: Minimum seconds to record before silence detection activates
            early_check_callback: Optional callback(data, elapsed) -> bool. Return False to stop early.
            early_check_interval: Seconds between early check callbacks

        Returns:
            Raw audio bytes (DLE-encoded from modem)
        """
        if not self._serial:
            raise ModemError("Not connected")

        data = bytearray()
        old_timeout = self._serial.timeout
        self._serial.timeout = 0.5

        start_time = time.time()
        silence_start = None
        last_chunk_time = start_time
        last_early_check = start_time

        # DLE codes that indicate end of call
        end_codes = [self.DLE_ETX, self.DLE_HANGUP, self.DLE_BUSY]

        try:
            while (time.time() - start_time) < duration:
                chunk = self._serial.read(1024)
                elapsed = time.time() - start_time

                if chunk:
                    last_chunk_time = time.time()
                    data.extend(chunk)

                    # Check for DLE codes indicating call end
                    for code in end_codes:
                        if code in chunk:
                            return bytes(data)

                    # Early check callback for fax/modem detection
                    if early_check_callback and (elapsed - last_early_check) >= early_check_interval:
                        last_early_check = elapsed
                        try:
                            if not early_check_callback(bytes(data), elapsed):
                                logger.debug(f"Early check callback returned False at {elapsed:.1f}s, stopping")
                                return bytes(data)
                        except Exception as e:
                            logger.debug(f"Early check callback error: {e}")

                    # Only check for silence after minimum recording time
                    # This prevents cutting off calls too early
                    if elapsed >= min_record_time:
                        # Check for silence (low audio levels)
                        # Calculate average level of this chunk (ignore DLE-encoded bytes)
                        audio_bytes = [b for b in chunk if b != 0x10]  # Skip DLE bytes
                        if audio_bytes:
                            # Audio is centered at 128 for unsigned 8-bit
                            avg_level = sum(abs(b - 128) for b in audio_bytes) / len(audio_bytes)

                            if avg_level < silence_threshold:
                                if silence_start is None:
                                    silence_start = time.time()
                                elif (time.time() - silence_start) > silence_timeout:
                                    # Extended silence - likely hangup
                                    return bytes(data)
                            else:
                                silence_start = None  # Reset silence timer

                else:
                    # No data received - check for timeout (only after min time)
                    if elapsed >= min_record_time and (time.time() - last_chunk_time) > 3.0:
                        # No data for 3 seconds - connection likely dropped
                        break

        finally:
            self._serial.timeout = old_timeout

        return bytes(data)

    def stop_voice_receive(self) -> None:
        """Stop voice receive mode and drain all buffered audio data."""
        if not self._serial:
            return

        logger.debug("stop_voice_receive: Starting")

        old_timeout = self._serial.timeout
        self._serial.timeout = 0.1  # Very short timeout for draining

        try:
            # Send DLE+ETX multiple times to ensure voice mode exits
            for i in range(3):
                logger.debug(f"stop_voice_receive: Sending DLE+ETX (attempt {i+1})")
                self._serial.write(self.DLE_ETX)
                time.sleep(0.2)

            # Aggressively drain ALL buffered audio data
            # The modem can buffer several seconds of audio
            logger.debug("stop_voice_receive: Draining audio buffer...")
            total_drained = 0
            empty_count = 0
            max_drain_time = 3.0  # Max 3 seconds draining
            drain_start = time.time()

            while (time.time() - drain_start) < max_drain_time:
                data = self._serial.read(8192)  # Large reads
                if data:
                    total_drained += len(data)
                    empty_count = 0
                    # Check if we got an "OK" - modem is back in command mode
                    if b'OK' in data:
                        logger.debug("stop_voice_receive: Got OK, modem back in command mode")
                        break
                else:
                    empty_count += 1
                    if empty_count >= 5:  # 5 empty reads = buffer is clear
                        break

            logger.debug(f"stop_voice_receive: Drained {total_drained} bytes")

            # Final buffer flush
            self._serial.reset_input_buffer()

        except Exception as e:
            logger.debug(f"stop_voice_receive: Error: {e}")
        finally:
            self._serial.timeout = old_timeout

        logger.debug("stop_voice_receive: Complete")

    def hangup(self) -> bool:
        """Hang up the current call and ensure modem is back in command mode."""
        logger.debug("hangup: Starting")
        if not self._serial:
            logger.debug("hangup: No serial connection")
            return False

        old_timeout = self._serial.timeout
        self._serial.timeout = 0.1

        # CRITICAL: Exit voice mode aggressively
        # The modem may still be streaming audio data
        logger.debug("hangup: Exiting voice mode aggressively")

        # Send DLE+ETX repeatedly while draining
        for attempt in range(5):
            self._serial.write(self.DLE_ETX)
            time.sleep(0.1)

        # Drain ALL audio data - this is critical
        # Keep draining until we get silence (multiple empty reads)
        logger.debug("hangup: Draining audio buffer...")
        total_drained = 0
        empty_count = 0
        drain_start = time.time()
        max_drain_time = 5.0  # Up to 5 seconds of draining

        while (time.time() - drain_start) < max_drain_time:
            data = self._serial.read(8192)
            if data:
                total_drained += len(data)
                empty_count = 0
                # Keep sending DLE+ETX while we're getting data
                if total_drained % 16384 == 0:  # Every 16KB
                    self._serial.write(self.DLE_ETX)
            else:
                empty_count += 1
                # Need several consecutive empty reads to confirm silence
                if empty_count >= 10:
                    break

        logger.debug(f"hangup: Drained {total_drained} bytes")
        self._in_voice_mode = False

        # Flush buffers completely
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

        # Now try to get modem back to command mode
        # Try multiple approaches until we get OK
        for reset_attempt in range(3):
            logger.debug(f"hangup: Reset attempt {reset_attempt + 1}/3")

            # Send escape sequence
            self._serial.write(b'+++')
            time.sleep(1.0)
            self._serial.reset_input_buffer()

            # Exit voice/fax mode
            self._serial.write(b'AT+FCLASS=0\r')
            time.sleep(0.3)
            self._serial.reset_input_buffer()

            # Hangup
            self._serial.write(b'ATH\r')
            time.sleep(0.5)
            self._serial.reset_input_buffer()

            # Full reset
            self._serial.write(b'ATZ\r')
            time.sleep(1.0)

            # Drain any response/remaining audio
            drain_count = 0
            self._serial.timeout = 0.2
            for _ in range(10):
                data = self._serial.read(4096)
                if data:
                    drain_count += len(data)
                else:
                    break
            if drain_count > 0:
                logger.debug(f"hangup: Drained {drain_count} more bytes after ATZ")

            self._serial.reset_input_buffer()

            # Verify modem is responsive
            self._serial.write(b'AT\r')
            time.sleep(0.3)
            self._serial.timeout = 1.0
            response = self._serial.read(256)

            if b'OK' in response:
                logger.debug("hangup: Modem responsive (got OK)")
                break
            else:
                logger.debug(f"hangup: No OK response, got: {response[:30] if response else 'None'}")
                # More aggressive drain before retry
                self._serial.timeout = 0.1
                for _ in range(20):
                    data = self._serial.read(8192)
                    if not data:
                        break
                self._serial.reset_input_buffer()

        self._serial.timeout = old_timeout
        self._serial.reset_input_buffer()
        logger.debug("hangup: Complete")

        return True

    def reset_for_next_call(self) -> bool:
        """
        Reset modem to a clean state ready for the next call.

        Call this between calls to ensure proper state cleanup.
        """
        if not self._serial:
            return False

        # Ensure we're out of voice mode
        if self._in_voice_mode:
            self._serial.write(self.DLE_ETX)
            time.sleep(0.2)
            self._in_voice_mode = False

        # Flush buffers
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

        # Switch to data mode and reset
        self._serial.write(b'AT+FCLASS=0\r')
        time.sleep(0.3)

        # Send ATZ to reset modem to defaults
        self._serial.write(b'ATZ\r')
        time.sleep(0.5)

        # Clear any remaining data
        old_timeout = self._serial.timeout
        self._serial.timeout = 0.3
        try:
            for _ in range(5):
                data = self._serial.read(1024)
                if not data:
                    break
        except Exception:
            pass
        finally:
            self._serial.timeout = old_timeout

        # Re-initialize essential settings
        self._serial.write(b'ATE0\r')  # Echo off
        time.sleep(0.1)
        self._serial.read(256)

        return True

    def __enter__(self) -> 'Modem':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
