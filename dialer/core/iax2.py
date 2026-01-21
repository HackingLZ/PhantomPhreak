"""Minimal IAX2 protocol implementation for voice calls.

Based on RFC 5456 - Inter-Asterisk eXchange Version 2.
This implements just enough IAX2 to make outbound calls and receive audio.
"""

import hashlib
import logging
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# IAX2 Frame Types (RFC 5456 Section 8.2)
class FrameType(IntEnum):
    """IAX2 frame types."""
    DTMF_END = 0x01      # DTMF end
    VOICE = 0x02         # Voice data
    VIDEO = 0x03         # Video data
    CONTROL = 0x04       # Session control
    NULL = 0x05          # Null frame (keepalive)
    IAX = 0x06           # IAX protocol control
    TEXT = 0x07          # Text data
    IMAGE = 0x08         # Image data
    HTML = 0x09          # HTML data
    COMFORT_NOISE = 0x0A # Comfort noise


# IAX2 IAX Subclass (RFC 5456 Section 8.4)
class IAXSubclass(IntEnum):
    """IAX frame subclass codes."""
    NEW = 0x01           # Initiate new call
    PING = 0x02          # Ping request
    PONG = 0x03          # Ping response
    ACK = 0x04           # Acknowledgment
    HANGUP = 0x05        # Hangup
    REJECT = 0x06        # Reject
    ACCEPT = 0x07        # Accept
    AUTHREQ = 0x08       # Authentication request
    AUTHREP = 0x09       # Authentication reply
    INVAL = 0x0A         # Invalid
    LAGRQ = 0x0B         # Lag request
    LAGRP = 0x0C         # Lag reply
    REGREQ = 0x0D        # Registration request
    REGAUTH = 0x0E       # Registration auth challenge
    REGACK = 0x0F        # Registration acknowledge
    REGREJ = 0x10        # Registration reject
    REGREL = 0x11        # Registration release
    VNAK = 0x12          # Voice negative ack
    DPREQ = 0x13         # Dialplan request
    DPREP = 0x14         # Dialplan reply
    DIAL = 0x15          # Dial
    TXREQ = 0x16         # Transfer request
    TXCNT = 0x17         # Transfer connect
    TXACC = 0x18         # Transfer accepted
    TXREADY = 0x19       # Transfer ready
    TXREL = 0x1A         # Transfer release
    TXREJ = 0x1B         # Transfer reject
    QUELCH = 0x1C        # Stop audio
    UNQUELCH = 0x1D      # Resume audio
    POKE = 0x1E          # Poke request
    MWI = 0x20           # Message waiting indicator
    UNSUPPORT = 0x21     # Unsupported message
    TRANSFER = 0x22      # Transfer
    CALLTOKEN = 0x28     # Call token request/response


# Control Frame Subclass (RFC 5456 Section 8.3)
class ControlSubclass(IntEnum):
    """Control frame subclass codes."""
    HANGUP = 0x01        # Hangup
    RING = 0x02          # Ringing
    RINGING = 0x03       # Remote is ringing
    ANSWER = 0x04        # Call answered
    BUSY = 0x05          # Line busy
    CONGESTION = 0x08    # Congestion
    FLASH = 0x09         # Flash hook
    OPTION = 0x0B        # Option
    KEY = 0x0C           # Key
    UNKEY = 0x0D         # Unkey
    PROGRESS = 0x0E      # Call progress
    PROCEEDING = 0x0F    # Call proceeding
    HOLD = 0x10          # Hold
    UNHOLD = 0x11        # Unhold
    VIDUPDATE = 0x12     # Video update


# IAX2 Information Elements (RFC 5456 Section 8.6)
class InfoElement(IntEnum):
    """Information element types."""
    CALLED_NUMBER = 0x01      # Called number
    CALLING_NUMBER = 0x02     # Calling number
    CALLING_ANI = 0x03        # ANI
    CALLING_NAME = 0x04       # Calling name
    CALLED_CONTEXT = 0x05     # Called context
    USERNAME = 0x06           # Username
    PASSWORD = 0x07           # Password
    CAPABILITY = 0x08         # Codec capability
    FORMAT = 0x09             # Desired codec
    LANGUAGE = 0x0A           # Language
    VERSION = 0x0B            # Protocol version
    ADSICPE = 0x0C            # ADSI CPE ID
    DNID = 0x0D               # Dialed Number ID
    AUTHMETHODS = 0x0E        # Auth methods
    CHALLENGE = 0x0F          # Challenge
    MD5_RESULT = 0x10         # MD5 result
    RSA_RESULT = 0x11         # RSA result
    APPARENT_ADDR = 0x12      # Apparent address
    REFRESH = 0x13            # Registration refresh
    DPSTATUS = 0x14           # Dialplan status
    CALLNO = 0x15             # Call number
    CAUSE = 0x16              # Cause
    IAX_UNKNOWN = 0x17        # Unknown IAX
    MSGCOUNT = 0x18           # Message count
    AUTOANSWER = 0x19         # Auto answer
    MUSICONHOLD = 0x1A        # Music on hold
    TRANSFERID = 0x1B         # Transfer ID
    RDNIS = 0x1C              # Redirecting number
    DATETIME = 0x1F           # Date/time
    CALLINGPRES = 0x26        # Calling presentation
    CALLINGTON = 0x27         # Calling type of number
    CALLINGTNS = 0x28         # Calling TNS
    SAMPLINGRATE = 0x29       # Sampling rate
    CAUSECODE = 0x2A          # Cause code
    ENCRYPTION = 0x2B         # Encryption format
    ENCKEY = 0x2C             # Encryption key
    CODEC_PREFS = 0x2D        # Codec preferences
    RR_JITTER = 0x2E          # Jitter
    RR_LOSS = 0x2F            # Packet loss
    RR_PKTS = 0x30            # Packets
    RR_DELAY = 0x31           # Delay
    RR_DROPPED = 0x32         # Dropped
    RR_OOO = 0x33             # Out of order
    CALLTOKEN = 0x36          # Call token


# Audio Codecs (RFC 5456 Section 8.7)
class AudioCodec(IntEnum):
    """Audio codec format flags."""
    G723_1 = 0x00000001
    GSM = 0x00000002
    ULAW = 0x00000004       # G.711 u-law
    ALAW = 0x00000008       # G.711 a-law
    G726_ADPCM = 0x00000010
    ADPCM = 0x00000020
    SLINEAR = 0x00000040    # 16-bit linear
    LPC10 = 0x00000080
    G729A = 0x00000100
    SPEEX = 0x00000200
    ILBC = 0x00000400
    G726_AAL2 = 0x00000800
    G722 = 0x00001000
    SLINEAR16 = 0x00002000
    OPUS = 0x00040000


# Authentication methods
class AuthMethod(IntEnum):
    """Authentication methods."""
    PLAINTEXT = 0x0001
    MD5 = 0x0002
    RSA = 0x0004


# Call states
class CallState(IntEnum):
    """Call state machine states."""
    IDLE = 0
    WAITING_AUTHREQ = 1
    WAITING_ACCEPT = 2
    RINGING = 3
    ANSWERED = 4
    HANGUP = 5
    REJECTED = 6
    ERROR = 7


@dataclass
class IAX2Frame:
    """Represents an IAX2 frame."""
    source_call: int = 0
    dest_call: int = 0
    timestamp: int = 0
    oseqno: int = 0
    iseqno: int = 0
    frame_type: int = 0
    subclass: int = 0
    data: bytes = b''
    is_full_frame: bool = True

    def encode(self) -> bytes:
        """Encode frame to bytes for transmission."""
        if self.is_full_frame:
            # Full frame: 12-byte header
            # Bit 15 (F) = 1 for full frame
            scall = self.source_call | 0x8000
            header = struct.pack(
                '>HHIBBBB',
                scall,
                self.dest_call,
                self.timestamp,
                self.oseqno,
                self.iseqno,
                self.frame_type,
                self.subclass,
            )
            return header + self.data
        else:
            # Mini frame: 4-byte header
            # Bit 15 (F) = 0 for mini frame
            scall = self.source_call & 0x7FFF
            ts_low = self.timestamp & 0xFFFF
            header = struct.pack('>HH', scall, ts_low)
            return header + self.data

    @classmethod
    def decode(cls, data: bytes) -> 'IAX2Frame':
        """Decode bytes to IAX2Frame."""
        if len(data) < 4:
            raise ValueError("Frame too short")

        scall = struct.unpack('>H', data[0:2])[0]
        is_full = bool(scall & 0x8000)
        source_call = scall & 0x7FFF

        if is_full:
            if len(data) < 12:
                raise ValueError("Full frame too short")

            dest_call, timestamp, oseqno, iseqno, frame_type, subclass = struct.unpack(
                '>HIBBBB', data[2:12]
            )
            frame_data = data[12:]
        else:
            # Mini frame
            ts_low = struct.unpack('>H', data[2:4])[0]
            dest_call = 0
            timestamp = ts_low
            oseqno = 0
            iseqno = 0
            frame_type = FrameType.VOICE
            subclass = 0
            frame_data = data[4:]

        return cls(
            source_call=source_call,
            dest_call=dest_call,
            timestamp=timestamp,
            oseqno=oseqno,
            iseqno=iseqno,
            frame_type=frame_type,
            subclass=subclass,
            data=frame_data,
            is_full_frame=is_full,
        )


def encode_ie(ie_type: int, value: bytes) -> bytes:
    """Encode an information element."""
    return bytes([ie_type, len(value)]) + value


def encode_ie_string(ie_type: int, value: str) -> bytes:
    """Encode a string information element."""
    return encode_ie(ie_type, value.encode('utf-8'))


def encode_ie_int(ie_type: int, value: int, size: int = 4) -> bytes:
    """Encode an integer information element."""
    if size == 1:
        return encode_ie(ie_type, bytes([value & 0xFF]))
    elif size == 2:
        return encode_ie(ie_type, struct.pack('>H', value))
    else:
        return encode_ie(ie_type, struct.pack('>I', value))


def decode_ies(data: bytes) -> dict[int, bytes]:
    """Decode information elements from frame data."""
    ies = {}
    i = 0
    while i + 1 < len(data):
        ie_type = data[i]
        ie_len = data[i + 1]
        if i + 2 + ie_len > len(data):
            break
        ies[ie_type] = data[i + 2:i + 2 + ie_len]
        i += 2 + ie_len
    return ies


# u-law decoding table (ITU G.711)
ULAW_TABLE = [
    -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
    -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
    -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
    -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
    -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
    -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
    -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
    -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
    -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
    -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
    -876, -844, -812, -780, -748, -716, -684, -652,
    -620, -588, -556, -524, -492, -460, -428, -396,
    -372, -356, -340, -324, -308, -292, -276, -260,
    -244, -228, -212, -196, -180, -164, -148, -132,
    -120, -112, -104, -96, -88, -80, -72, -64,
    -56, -48, -40, -32, -24, -16, -8, 0,
    32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
    23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
    15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
    11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
    7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
    5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
    3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
    2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
    1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
    1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
    876, 844, 812, 780, 748, 716, 684, 652,
    620, 588, 556, 524, 492, 460, 428, 396,
    372, 356, 340, 324, 308, 292, 276, 260,
    244, 228, 212, 196, 180, 164, 148, 132,
    120, 112, 104, 96, 88, 80, 72, 64,
    56, 48, 40, 32, 24, 16, 8, 0,
]

# a-law decoding table (ITU G.711)
ALAW_TABLE = [
    -5504, -5248, -6016, -5760, -4480, -4224, -4992, -4736,
    -7552, -7296, -8064, -7808, -6528, -6272, -7040, -6784,
    -2752, -2624, -3008, -2880, -2240, -2112, -2496, -2368,
    -3776, -3648, -4032, -3904, -3264, -3136, -3520, -3392,
    -22016, -20992, -24064, -23040, -17920, -16896, -19968, -18944,
    -30208, -29184, -32256, -31232, -26112, -25088, -28160, -27136,
    -11008, -10496, -12032, -11520, -8960, -8448, -9984, -9472,
    -15104, -14592, -16128, -15616, -13056, -12544, -14080, -13568,
    -344, -328, -376, -360, -280, -264, -312, -296,
    -472, -456, -504, -488, -408, -392, -440, -424,
    -88, -72, -120, -104, -24, -8, -56, -40,
    -216, -200, -248, -232, -152, -136, -184, -168,
    -1376, -1312, -1504, -1440, -1120, -1056, -1248, -1184,
    -1888, -1824, -2016, -1952, -1632, -1568, -1760, -1696,
    -688, -656, -752, -720, -560, -528, -624, -592,
    -944, -912, -1008, -976, -816, -784, -880, -848,
    5504, 5248, 6016, 5760, 4480, 4224, 4992, 4736,
    7552, 7296, 8064, 7808, 6528, 6272, 7040, 6784,
    2752, 2624, 3008, 2880, 2240, 2112, 2496, 2368,
    3776, 3648, 4032, 3904, 3264, 3136, 3520, 3392,
    22016, 20992, 24064, 23040, 17920, 16896, 19968, 18944,
    30208, 29184, 32256, 31232, 26112, 25088, 28160, 27136,
    11008, 10496, 12032, 11520, 8960, 8448, 9984, 9472,
    15104, 14592, 16128, 15616, 13056, 12544, 14080, 13568,
    344, 328, 376, 360, 280, 264, 312, 296,
    472, 456, 504, 488, 408, 392, 440, 424,
    88, 72, 120, 104, 24, 8, 56, 40,
    216, 200, 248, 232, 152, 136, 184, 168,
    1376, 1312, 1504, 1440, 1120, 1056, 1248, 1184,
    1888, 1824, 2016, 1952, 1632, 1568, 1760, 1696,
    688, 656, 752, 720, 560, 528, 624, 592,
    944, 912, 1008, 976, 816, 784, 880, 848,
]


def decode_ulaw(data: bytes) -> np.ndarray:
    """Decode u-law audio to linear samples."""
    samples = np.array([ULAW_TABLE[b] for b in data], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def decode_alaw(data: bytes) -> np.ndarray:
    """Decode a-law audio to linear samples."""
    samples = np.array([ALAW_TABLE[b] for b in data], dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


@dataclass
class IAX2Call:
    """Represents an active IAX2 call."""
    local_call_number: int = 0
    remote_call_number: int = 0
    state: CallState = CallState.IDLE
    oseqno: int = 0
    iseqno: int = 0
    timestamp_base: float = 0.0
    codec: int = AudioCodec.ULAW
    challenge: str = ''
    cause: str = ''
    audio_buffer: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        if self.timestamp_base == 0:
            self.timestamp_base = time.time()
        return int((time.time() - self.timestamp_base) * 1000) & 0xFFFFFFFF


class IAX2Client:
    """Minimal IAX2 client for making outbound voice calls."""

    DEFAULT_PORT = 4569

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        username: str = '',
        password: str = '',
        caller_id: str = '',
    ):
        """Initialize IAX2 client.

        Args:
            host: IAX2 server hostname (e.g., 'chicago3.voip.ms')
            port: IAX2 port (default 4569)
            username: IAX2 username
            password: IAX2 password
            caller_id: Caller ID to send
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.caller_id = caller_id

        self._socket: Optional[socket.socket] = None
        self._call: Optional[IAX2Call] = None
        self._receive_thread: Optional[threading.Thread] = None
        self._running = False
        self._recv_lock = threading.Lock()

    def connect(self) -> None:
        """Create UDP socket and start receive thread."""
        if self._socket is not None:
            return

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(0.5)

        # Resolve hostname
        logger.debug(f"Resolving hostname: {self.host}")
        try:
            resolved_ip = socket.gethostbyname(self.host)
            self._server_addr = (resolved_ip, self.port)
            logger.debug(f"Resolved {self.host} to {resolved_ip}")
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {self.host}: {e}")
            raise ConnectionError(f"Failed to resolve {self.host}: {e}")

        # Start receive thread
        self._running = True
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

        logger.info(f"Connected to IAX2 server {self.host}:{self.port} ({resolved_ip})")

    def disconnect(self) -> None:
        """Close socket and stop receive thread."""
        self._running = False

        if self._call and self._call.state not in (CallState.IDLE, CallState.HANGUP):
            try:
                self.hangup()
            except Exception:
                pass

        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
            self._receive_thread = None

        if self._socket:
            self._socket.close()
            self._socket = None

        self._call = None
        logger.info("Disconnected from IAX2 server")

    def _send_frame(self, frame: IAX2Frame) -> None:
        """Send a frame to the server."""
        if not self._socket:
            raise ConnectionError("Not connected")
        data = frame.encode()
        self._socket.sendto(data, self._server_addr)

        # Debug logging for sent frames
        if frame.is_full_frame and frame.frame_type == FrameType.IAX:
            try:
                subclass_name = IAXSubclass(frame.subclass).name
            except ValueError:
                subclass_name = f"0x{frame.subclass:02x}"
            logger.debug(f"TX IAX {subclass_name} scall={frame.source_call} dcall={frame.dest_call} oseq={frame.oseqno} iseq={frame.iseqno}")
        elif frame.is_full_frame and frame.frame_type == FrameType.CONTROL:
            try:
                subclass_name = ControlSubclass(frame.subclass).name
            except ValueError:
                subclass_name = f"0x{frame.subclass:02x}"
            logger.debug(f"TX CONTROL {subclass_name}")

    def _receive_loop(self) -> None:
        """Background thread to receive frames."""
        while self._running:
            try:
                data, addr = self._socket.recvfrom(4096)
                if data:
                    self._handle_frame(data)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.debug(f"Receive error: {e}")

    def _handle_frame(self, data: bytes) -> None:
        """Process a received frame."""
        try:
            frame = IAX2Frame.decode(data)
        except ValueError as e:
            logger.debug(f"Failed to decode frame: {e}")
            return

        if not self._call:
            return

        # Debug logging for received frames
        if frame.is_full_frame:
            if frame.frame_type == FrameType.IAX:
                try:
                    subclass_name = IAXSubclass(frame.subclass).name
                except ValueError:
                    subclass_name = f"0x{frame.subclass:02x}"
                logger.debug(f"RX IAX {subclass_name} scall={frame.source_call} dcall={frame.dest_call} oseq={frame.oseqno} iseq={frame.iseqno}")
            elif frame.frame_type == FrameType.CONTROL:
                try:
                    subclass_name = ControlSubclass(frame.subclass).name
                except ValueError:
                    subclass_name = f"0x{frame.subclass:02x}"
                logger.debug(f"RX CONTROL {subclass_name}")
            elif frame.frame_type == FrameType.VOICE:
                logger.debug(f"RX VOICE {len(frame.data)} bytes")

        with self._recv_lock:
            # Update remote call number
            if frame.source_call and not self._call.remote_call_number:
                self._call.remote_call_number = frame.source_call
                logger.debug(f"Remote call number set to {frame.source_call}")

            # Update sequence numbers for full frames
            if frame.is_full_frame:
                self._call.iseqno = (frame.oseqno + 1) & 0xFF

            # Handle by frame type
            if frame.frame_type == FrameType.IAX:
                self._handle_iax_frame(frame)
            elif frame.frame_type == FrameType.CONTROL:
                self._handle_control_frame(frame)
            elif frame.frame_type == FrameType.VOICE:
                self._handle_voice_frame(frame)

    def _handle_iax_frame(self, frame: IAX2Frame) -> None:
        """Handle IAX protocol frames."""
        subclass = frame.subclass

        if subclass == IAXSubclass.AUTHREQ:
            # Authentication challenge received
            ies = decode_ies(frame.data)
            self._call.challenge = ies.get(InfoElement.CHALLENGE, b'').decode('utf-8')
            auth_methods = struct.unpack('>H', ies.get(InfoElement.AUTHMETHODS, b'\x00\x02'))[0]
            logger.debug(f"Auth challenge received: {self._call.challenge[:20]}...")

            # Send ACK first
            self._send_ack(frame)

            # Respond with MD5 auth
            self._send_auth_reply()
            self._call.state = CallState.WAITING_ACCEPT

        elif subclass == IAXSubclass.ACCEPT:
            # Call accepted
            logger.debug("Call accepted")
            self._send_ack(frame)
            self._call.state = CallState.RINGING

            # Parse codec from ACCEPT
            ies = decode_ies(frame.data)
            if InfoElement.FORMAT in ies:
                self._call.codec = struct.unpack('>I', ies[InfoElement.FORMAT])[0]
                logger.debug(f"Negotiated codec: {self._call.codec}")

        elif subclass == IAXSubclass.REJECT:
            # Call rejected
            ies = decode_ies(frame.data)
            self._call.cause = ies.get(InfoElement.CAUSE, b'').decode('utf-8', errors='replace')
            logger.info(f"Call rejected: {self._call.cause}")
            self._send_ack(frame)
            self._call.state = CallState.REJECTED

        elif subclass == IAXSubclass.HANGUP:
            # Remote hangup
            ies = decode_ies(frame.data)
            self._call.cause = ies.get(InfoElement.CAUSE, b'').decode('utf-8', errors='replace')
            logger.debug(f"Remote hangup: {self._call.cause}")
            self._send_ack(frame)
            self._call.state = CallState.HANGUP

        elif subclass == IAXSubclass.ACK:
            # Acknowledgment - just log it
            pass

        elif subclass == IAXSubclass.PING:
            # Respond to ping
            self._send_pong(frame)

        elif subclass == IAXSubclass.LAGRQ:
            # Respond to lag request
            self._send_lagrp(frame)

    def _handle_control_frame(self, frame: IAX2Frame) -> None:
        """Handle control frames."""
        subclass = frame.subclass

        if subclass == ControlSubclass.RINGING:
            logger.debug("Remote is ringing")
            self._send_ack(frame)
            self._call.state = CallState.RINGING

        elif subclass == ControlSubclass.ANSWER:
            logger.info("Call answered")
            self._send_ack(frame)
            self._call.state = CallState.ANSWERED

        elif subclass == ControlSubclass.BUSY:
            logger.info("Line busy")
            self._send_ack(frame)
            self._call.state = CallState.REJECTED
            self._call.cause = "BUSY"

        elif subclass == ControlSubclass.CONGESTION:
            logger.info("Network congestion")
            self._send_ack(frame)
            self._call.state = CallState.REJECTED
            self._call.cause = "CONGESTION"

        elif subclass == ControlSubclass.HANGUP:
            logger.debug("Control hangup")
            self._send_ack(frame)
            self._call.state = CallState.HANGUP

        elif subclass == ControlSubclass.PROCEEDING:
            logger.debug("Call proceeding")
            self._send_ack(frame)

        elif subclass == ControlSubclass.PROGRESS:
            logger.debug("Call progress")
            self._send_ack(frame)

    def _handle_voice_frame(self, frame: IAX2Frame) -> None:
        """Handle voice data frames."""
        if self._call and frame.data:
            with self._call.lock:
                self._call.audio_buffer.append(frame.data)

    def _send_ack(self, frame: IAX2Frame) -> None:
        """Send ACK for a received frame."""
        ack = IAX2Frame(
            source_call=self._call.local_call_number,
            dest_call=self._call.remote_call_number,
            timestamp=self._call.get_timestamp(),
            oseqno=self._call.oseqno,
            iseqno=self._call.iseqno,
            frame_type=FrameType.IAX,
            subclass=IAXSubclass.ACK,
        )
        self._send_frame(ack)

    def _send_pong(self, frame: IAX2Frame) -> None:
        """Send PONG response to PING."""
        pong = IAX2Frame(
            source_call=self._call.local_call_number,
            dest_call=self._call.remote_call_number,
            timestamp=frame.timestamp,
            oseqno=self._call.oseqno,
            iseqno=self._call.iseqno,
            frame_type=FrameType.IAX,
            subclass=IAXSubclass.PONG,
        )
        self._call.oseqno = (self._call.oseqno + 1) & 0xFF
        self._send_frame(pong)

    def _send_lagrp(self, frame: IAX2Frame) -> None:
        """Send LAGRP response to LAGRQ."""
        lagrp = IAX2Frame(
            source_call=self._call.local_call_number,
            dest_call=self._call.remote_call_number,
            timestamp=frame.timestamp,
            oseqno=self._call.oseqno,
            iseqno=self._call.iseqno,
            frame_type=FrameType.IAX,
            subclass=IAXSubclass.LAGRP,
        )
        self._call.oseqno = (self._call.oseqno + 1) & 0xFF
        self._send_frame(lagrp)

    def _send_auth_reply(self) -> None:
        """Send authentication reply with MD5 hash."""
        # MD5(challenge + password)
        md5_input = self._call.challenge + self.password
        md5_hash = hashlib.md5(md5_input.encode('utf-8')).hexdigest()

        data = b''
        data += encode_ie_string(InfoElement.USERNAME, self.username)
        data += encode_ie_string(InfoElement.MD5_RESULT, md5_hash)

        auth_reply = IAX2Frame(
            source_call=self._call.local_call_number,
            dest_call=self._call.remote_call_number,
            timestamp=self._call.get_timestamp(),
            oseqno=self._call.oseqno,
            iseqno=self._call.iseqno,
            frame_type=FrameType.IAX,
            subclass=IAXSubclass.AUTHREP,
            data=data,
        )
        self._call.oseqno = (self._call.oseqno + 1) & 0xFF
        self._send_frame(auth_reply)
        logger.debug("Sent auth reply")

    def call(self, number: str, timeout: float = 45.0) -> CallState:
        """Place an outbound call.

        Args:
            number: Phone number to dial
            timeout: Maximum seconds to wait for answer

        Returns:
            Final CallState (ANSWERED, REJECTED, HANGUP, or ERROR)
        """
        if not self._socket:
            raise ConnectionError("Not connected")

        # Clean the number
        clean_number = ''.join(c for c in number if c.isdigit() or c == '+')
        if not clean_number:
            raise ValueError(f"Invalid phone number: {number}")

        # Create new call
        self._call = IAX2Call(
            local_call_number=random.randint(1, 0x7FFE),
            state=CallState.WAITING_AUTHREQ,
        )

        # Build NEW frame with information elements
        data = b''
        data += encode_ie_int(InfoElement.VERSION, 2, size=2)
        data += encode_ie_string(InfoElement.CALLED_NUMBER, clean_number)
        data += encode_ie_string(InfoElement.CALLING_NUMBER, self.caller_id or 'anonymous')
        data += encode_ie_string(InfoElement.CALLING_NAME, self.caller_id or 'Anonymous')
        data += encode_ie_string(InfoElement.USERNAME, self.username)
        # Request u-law and a-law codecs
        codec_cap = AudioCodec.ULAW | AudioCodec.ALAW
        data += encode_ie_int(InfoElement.CAPABILITY, codec_cap)
        data += encode_ie_int(InfoElement.FORMAT, AudioCodec.ULAW)

        new_frame = IAX2Frame(
            source_call=self._call.local_call_number,
            dest_call=0,
            timestamp=self._call.get_timestamp(),
            oseqno=self._call.oseqno,
            iseqno=self._call.iseqno,
            frame_type=FrameType.IAX,
            subclass=IAXSubclass.NEW,
            data=data,
        )
        self._call.oseqno = (self._call.oseqno + 1) & 0xFF

        logger.info(f"Calling {clean_number}")
        logger.debug(f"Local call number: {self._call.local_call_number}, waiting for AUTHREQ")
        self._send_frame(new_frame)

        # Wait for call to complete or timeout
        start_time = time.time()
        last_state = self._call.state
        last_log_time = start_time

        while (time.time() - start_time) < timeout:
            current_state = self._call.state

            # Log state changes
            if current_state != last_state:
                logger.debug(f"Call state: {last_state.name} -> {current_state.name}")
                last_state = current_state

            # Periodic progress logging every 5 seconds
            if (time.time() - last_log_time) >= 5.0:
                elapsed = time.time() - start_time
                logger.debug(f"Waiting... state={current_state.name}, elapsed={elapsed:.1f}s")
                last_log_time = time.time()

            if current_state == CallState.ANSWERED:
                logger.info("Call answered")
                return CallState.ANSWERED
            elif current_state in (CallState.REJECTED, CallState.HANGUP, CallState.ERROR):
                logger.info(f"Call ended: {current_state.name}, cause: {self._call.cause}")
                return current_state
            time.sleep(0.1)

        # Timeout
        logger.info(f"Call timed out after {timeout}s in state {self._call.state.name}")
        self._call.state = CallState.ERROR
        return CallState.ERROR

    def hangup(self) -> None:
        """Hang up the current call."""
        if not self._call or not self._socket:
            return

        # Send HANGUP frame
        data = encode_ie_string(InfoElement.CAUSE, "Normal Clearing")
        hangup_frame = IAX2Frame(
            source_call=self._call.local_call_number,
            dest_call=self._call.remote_call_number,
            timestamp=self._call.get_timestamp(),
            oseqno=self._call.oseqno,
            iseqno=self._call.iseqno,
            frame_type=FrameType.IAX,
            subclass=IAXSubclass.HANGUP,
            data=data,
        )
        self._call.oseqno = (self._call.oseqno + 1) & 0xFF
        self._send_frame(hangup_frame)
        self._call.state = CallState.HANGUP
        logger.debug("Sent hangup")

    def receive_audio(
        self,
        duration: float,
        silence_timeout: float = 5.0,
        early_check_callback: Optional[Callable[[np.ndarray, float], bool]] = None,
        early_check_interval: float = 2.0,
    ) -> np.ndarray:
        """Receive audio from the current call.

        Args:
            duration: Maximum seconds to receive
            silence_timeout: Stop after this many seconds of silence
            early_check_callback: Optional callback(samples, elapsed) -> bool
            early_check_interval: Seconds between early check calls

        Returns:
            Audio samples as float32 numpy array
        """
        if not self._call or self._call.state != CallState.ANSWERED:
            return np.array([], dtype=np.float32)

        all_samples = []
        start_time = time.time()
        last_audio_time = start_time
        last_check_time = start_time
        min_record_time = 5.0  # Don't stop early in first 5 seconds

        while (time.time() - start_time) < duration:
            # Check call state
            if self._call.state in (CallState.HANGUP, CallState.REJECTED):
                logger.debug("Call ended during audio receive")
                break

            # Get buffered audio
            with self._call.lock:
                audio_data = b''.join(self._call.audio_buffer)
                self._call.audio_buffer.clear()

            if audio_data:
                last_audio_time = time.time()

                # Decode based on codec
                if self._call.codec & AudioCodec.ULAW:
                    samples = decode_ulaw(audio_data)
                elif self._call.codec & AudioCodec.ALAW:
                    samples = decode_alaw(audio_data)
                else:
                    # Assume u-law as default
                    samples = decode_ulaw(audio_data)

                all_samples.append(samples)

            elapsed = time.time() - start_time

            # Early check callback
            if early_check_callback and (time.time() - last_check_time) >= early_check_interval:
                last_check_time = time.time()
                if all_samples:
                    current_samples = np.concatenate(all_samples)
                    try:
                        if not early_check_callback(current_samples, elapsed):
                            logger.debug("Early check callback returned False, stopping")
                            break
                    except Exception as e:
                        logger.debug(f"Early check callback error: {e}")

            # Check for silence timeout (after minimum record time)
            if elapsed >= min_record_time:
                silence_duration = time.time() - last_audio_time
                if silence_duration > silence_timeout:
                    logger.debug(f"Silence timeout after {silence_duration:.1f}s")
                    break

            time.sleep(0.02)  # Small sleep to prevent busy loop

        # Concatenate all samples
        if all_samples:
            return np.concatenate(all_samples)
        return np.array([], dtype=np.float32)

    @property
    def call_cause(self) -> str:
        """Get the cause/reason for call end."""
        if self._call:
            return self._call.cause
        return ""

    @property
    def is_connected(self) -> bool:
        """Check if connected to IAX2 server."""
        return self._socket is not None and self._running
