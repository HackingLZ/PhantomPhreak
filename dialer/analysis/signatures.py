"""Known audio signatures for line type classification.

This module defines frequency patterns for various telecommunications tones
and equipment. Frequencies are rounded to 100Hz resolution to match the
FFT analysis output.

Reference frequencies from ITU-T and North American standards.
"""

from dataclasses import dataclass
from enum import Enum


class LineType(Enum):
    """Classification of phone line types."""
    FAX = "fax"
    MODEM = "modem"
    VOICE = "voice"
    VOICEMAIL = "voicemail"
    ANSWERING_MACHINE = "answering_machine"
    IVR = "ivr"  # Interactive Voice Response / phone menu
    DIAL_TONE = "dial_tone"
    BUSY = "busy"
    RINGBACK = "ringback"
    SIT_TONE = "sit"  # Special Information Tone
    DTMF = "dtmf"
    SILENCE = "silence"
    UNKNOWN = "unknown"
    CARRIER = "carrier"  # Generic data carrier
    # Specific modem protocols
    MODEM_V21 = "modem_v21"  # 300 baud
    MODEM_V22 = "modem_v22"  # 1200 baud
    MODEM_V22BIS = "modem_v22bis"  # 2400 baud
    MODEM_V32 = "modem_v32"  # 9600 baud
    MODEM_V34 = "modem_v34"  # 28800/33600 baud


@dataclass
class ToneSignature:
    """Definition of a recognizable tone pattern."""
    line_type: LineType
    name: str
    frequencies: set[int]  # Primary frequencies (rounded to 100Hz)
    optional_frequencies: set[int]  # Secondary/optional frequencies
    min_matches: int  # Minimum primary frequencies that must match
    description: str


# Standard telephony tones (North American)
DIAL_TONE = ToneSignature(
    line_type=LineType.DIAL_TONE,
    name="Dial Tone",
    frequencies={400},  # 350 + 440 Hz combined
    optional_frequencies={300, 500},
    min_matches=1,
    description="Standard dial tone (350 + 440 Hz)"
)

BUSY_TONE = ToneSignature(
    line_type=LineType.BUSY,
    name="Busy Signal",
    frequencies={500, 600},  # 480 + 620 Hz
    optional_frequencies=set(),
    min_matches=1,
    description="Busy tone (480 + 620 Hz, 0.5s on/off)"
)

RINGBACK_TONE = ToneSignature(
    line_type=LineType.RINGBACK,
    name="Ringback",
    frequencies={400, 500},  # 440 + 480 Hz
    optional_frequencies=set(),
    min_matches=1,
    description="Ringback tone (440 + 480 Hz)"
)

# Special Information Tones (intercept messages)
SIT_TONE = ToneSignature(
    line_type=LineType.SIT_TONE,
    name="Special Info Tone",
    frequencies={900, 1400, 1800},  # 913.8, 1370.6, 1776.7 Hz
    optional_frequencies={1000, 1300, 1700},
    min_matches=2,
    description="SIT tone indicating disconnected/changed number"
)

# Fax machine tones
FAX_CNG = ToneSignature(
    line_type=LineType.FAX,
    name="Fax CNG",
    frequencies={1100},  # 1100 Hz calling tone
    optional_frequencies={1000, 1200},
    min_matches=1,
    description="Fax CNG tone (1100 Hz calling signal)"
)

FAX_CED = ToneSignature(
    line_type=LineType.FAX,
    name="Fax CED",
    frequencies={2100},  # 2100 Hz answer tone ONLY
    optional_frequencies=set(),  # No optional - fax is pure 2100 Hz
    min_matches=1,
    description="Fax CED/ANS tone (pure 2100 Hz answer)"
)

# Modem tones - requires multiple frequencies (handshake pattern)
MODEM_ANSWER = ToneSignature(
    line_type=LineType.MODEM,
    name="Modem Answer",
    frequencies={2100, 2200},  # V.22bis/V.32 answer tones - BOTH required
    optional_frequencies={1200, 2300, 2400},
    min_matches=2,  # Must have BOTH 2100 AND 2200 to be modem
    description="Modem answer tone (2100 + 2200 Hz handshake)"
)

MODEM_CARRIER = ToneSignature(
    line_type=LineType.CARRIER,
    name="Data Carrier",
    frequencies={1200, 2400},  # Common carrier frequencies
    optional_frequencies={1800, 2100, 2200},
    min_matches=1,
    description="Data modem carrier signal"
)

# DTMF tones (for reference - usually too short to detect reliably)
DTMF_LOW = ToneSignature(
    line_type=LineType.DTMF,
    name="DTMF",
    frequencies={700, 800, 900, 1000},  # Low group: 697, 770, 852, 941
    optional_frequencies={1200, 1300, 1500, 1600},  # High group
    min_matches=2,
    description="DTMF touch tones"
)

# Specific modem protocol signatures
# V.21 (300 baud) - ITU-T standard
MODEM_V21 = ToneSignature(
    line_type=LineType.MODEM_V21,
    name="V.21 Modem",
    frequencies={1080, 1750},  # Channel 1: 980/1180, Channel 2: 1650/1850
    optional_frequencies={1000, 1200, 1700, 1900},
    min_matches=1,
    description="V.21 300 baud modem (1080/1750 Hz)"
)

# V.22 (1200 baud)
MODEM_V22 = ToneSignature(
    line_type=LineType.MODEM_V22,
    name="V.22 Modem",
    frequencies={1200, 2400},  # 1200 Hz originate, 2400 Hz answer
    optional_frequencies={600, 1800},
    min_matches=1,
    description="V.22 1200 baud modem"
)

# V.22bis (2400 baud) - most common legacy modem
MODEM_V22BIS = ToneSignature(
    line_type=LineType.MODEM_V22BIS,
    name="V.22bis Modem",
    frequencies={1200, 2400},
    optional_frequencies={600, 1800, 2100},
    min_matches=2,
    description="V.22bis 2400 baud modem"
)

# V.32 (9600 baud)
MODEM_V32 = ToneSignature(
    line_type=LineType.MODEM_V32,
    name="V.32 Modem",
    frequencies={1800},  # Carrier at 1800 Hz
    optional_frequencies={600, 1200, 2400, 3000},
    min_matches=1,
    description="V.32 9600 baud modem (1800 Hz carrier)"
)

# V.34 (28800/33600 baud) - high-speed analog
MODEM_V34 = ToneSignature(
    line_type=LineType.MODEM_V34,
    name="V.34 Modem",
    frequencies={1800, 2400},  # Uses multiple carriers
    optional_frequencies={600, 1200, 3000, 3200, 3400},
    min_matches=1,
    description="V.34 28800+ baud modem"
)

# Bell 103 (300 baud - US standard, predecessor to V.21)
MODEM_BELL103 = ToneSignature(
    line_type=LineType.MODEM_V21,  # Compatible with V.21
    name="Bell 103",
    frequencies={1070, 1270, 2025, 2225},
    optional_frequencies={1100, 1200, 2000, 2100},
    min_matches=2,
    description="Bell 103 300 baud modem"
)

# Bell 212A (1200 baud - US standard)
MODEM_BELL212A = ToneSignature(
    line_type=LineType.MODEM_V22,  # Compatible with V.22
    name="Bell 212A",
    frequencies={1200, 2400},
    optional_frequencies={1800, 2200},
    min_matches=1,
    description="Bell 212A 1200 baud modem"
)


# All known signatures for classification
SIGNATURES: list[ToneSignature] = [
    FAX_CNG,
    FAX_CED,
    MODEM_ANSWER,
    MODEM_CARRIER,
    MODEM_V21,
    MODEM_V22,
    MODEM_V22BIS,
    MODEM_V32,
    MODEM_V34,
    MODEM_BELL103,
    MODEM_BELL212A,
    DIAL_TONE,
    BUSY_TONE,
    RINGBACK_TONE,
    SIT_TONE,
    DTMF_LOW,
]


# Frequency to possible line type mapping for quick lookup
FREQUENCY_MAP: dict[int, list[LineType]] = {}
for sig in SIGNATURES:
    for freq in sig.frequencies | sig.optional_frequencies:
        if freq not in FREQUENCY_MAP:
            FREQUENCY_MAP[freq] = []
        if sig.line_type not in FREQUENCY_MAP[freq]:
            FREQUENCY_MAP[freq].append(sig.line_type)


def get_possible_types(frequency: int) -> list[LineType]:
    """Get line types that could produce this frequency."""
    return FREQUENCY_MAP.get(frequency, [])


def get_signature(line_type: LineType) -> list[ToneSignature]:
    """Get all signatures for a line type."""
    return [s for s in SIGNATURES if s.line_type == line_type]


# Voice characteristics
# Voice typically has:
# - Broad spectrum with no dominant single frequency
# - Energy distributed across 300-3400 Hz
# - Varying patterns over time (not monotonic)
VOICE_FREQUENCY_RANGE = (300, 3400)
VOICE_MIN_UNIQUE_FREQUENCIES = 10  # Voices usually have many different frequencies


# Silence/dead line characteristics
SILENCE_MAX_ENERGY = 0.001  # Very low RMS energy
SILENCE_MAX_PEAKS = 2  # Few to no frequency peaks
