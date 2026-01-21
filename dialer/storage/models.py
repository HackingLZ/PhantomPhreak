"""Data models for scan results."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from ..analysis.signatures import LineType
from ..core.provider import CallResult


class ScanStatus(Enum):
    """Status of a scan job."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScanResult:
    """Result of scanning a single phone number."""
    id: Optional[int] = None
    scan_id: Optional[int] = None
    phone_number: str = ""
    call_result: CallResult = CallResult.UNKNOWN
    line_type: LineType = LineType.UNKNOWN
    confidence: float = 0.0
    audio_file: Optional[str] = None
    duration: float = 0.0
    frequencies: list[int] = field(default_factory=list)
    notes: str = ""
    scanned_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "scan_id": self.scan_id,
            "phone_number": self.phone_number,
            "call_result": self.call_result.value,
            "line_type": self.line_type.value,
            "confidence": self.confidence,
            "audio_file": self.audio_file,
            "duration": self.duration,
            "frequencies": ",".join(map(str, self.frequencies)),
            "notes": self.notes,
            "scanned_at": self.scanned_at.isoformat(),
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScanResult":
        """Create from dictionary."""
        # Parse frequencies
        freq_str = data.get("frequencies", "")
        if freq_str:
            frequencies = [int(f) for f in freq_str.split(",") if f]
        else:
            frequencies = []

        # Parse datetime
        scanned_at = data.get("scanned_at")
        if isinstance(scanned_at, str):
            scanned_at = datetime.fromisoformat(scanned_at)
        elif scanned_at is None:
            scanned_at = datetime.now()

        return cls(
            id=data.get("id"),
            scan_id=data.get("scan_id"),
            phone_number=data.get("phone_number", ""),
            call_result=CallResult(data.get("call_result", "UNKNOWN")),
            line_type=LineType(data.get("line_type", "unknown")),
            confidence=float(data.get("confidence", 0.0)),
            audio_file=data.get("audio_file"),
            duration=float(data.get("duration", 0.0)),
            frequencies=frequencies,
            notes=data.get("notes", ""),
            scanned_at=scanned_at,
            retry_count=int(data.get("retry_count", 0)),
        )


@dataclass
class Scan:
    """A scanning job/session."""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    status: ScanStatus = ScanStatus.PENDING
    total_numbers: int = 0
    completed_numbers: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    modem_device: Optional[str] = None
    config: dict = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Return progress as percentage."""
        if self.total_numbers == 0:
            return 0.0
        return (self.completed_numbers / self.total_numbers) * 100

    @property
    def is_active(self) -> bool:
        """Check if scan is currently running."""
        return self.status == ScanStatus.RUNNING

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "total_numbers": self.total_numbers,
            "completed_numbers": self.completed_numbers,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "modem_device": self.modem_device,
            "config": str(self.config),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scan":
        """Create from dictionary."""
        def parse_dt(val):
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(val)

        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=ScanStatus(data.get("status", "pending")),
            total_numbers=int(data.get("total_numbers", 0)),
            completed_numbers=int(data.get("completed_numbers", 0)),
            created_at=parse_dt(data.get("created_at")) or datetime.now(),
            started_at=parse_dt(data.get("started_at")),
            completed_at=parse_dt(data.get("completed_at")),
            modem_device=data.get("modem_device"),
            config=eval(data.get("config", "{}")),
        )


@dataclass
class ScanSummary:
    """Summary statistics for a scan."""
    scan: Scan
    total: int = 0
    completed: int = 0
    by_line_type: dict[LineType, int] = field(default_factory=dict)
    by_call_result: dict[CallResult, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    interesting_count: int = 0  # Fax, modem, etc.

    @property
    def fax_count(self) -> int:
        return self.by_line_type.get(LineType.FAX, 0)

    @property
    def modem_count(self) -> int:
        return self.by_line_type.get(LineType.MODEM, 0)

    @property
    def voice_count(self) -> int:
        return self.by_line_type.get(LineType.VOICE, 0)


def parse_number_range(range_spec: str) -> list[str]:
    """
    Parse a number range specification into individual numbers.

    Formats supported:
    - "555-1000,555-1010" -> 555-1000 through 555-1010
    - "5551000-5551010" -> single range
    - "555-100X" -> 555-1000 through 555-1009

    Args:
        range_spec: Range specification string

    Returns:
        List of phone number strings
    """
    numbers = []

    for part in range_spec.split(","):
        part = part.strip()
        if not part:
            continue

        # Check for X wildcard (e.g., 555-100X)
        if "X" in part.upper():
            base = part.upper().replace("X", "")
            x_count = part.upper().count("X")
            for i in range(10 ** x_count):
                num = base + str(i).zfill(x_count)
                numbers.append(num.replace("-", ""))
            continue

        # Check for hyphenated range within a number (e.g., 555-1000,555-1010)
        if part.count("-") == 2:
            # Format: prefix-start,prefix-end
            parts = part.rsplit("-", 1)
            if len(parts) == 2:
                prefix = parts[0]  # e.g., "555-1000" or "555"
                # This needs more context, skip for now
                numbers.append(part.replace("-", ""))
            continue

        # Check for simple range (e.g., 5551000-5551010)
        if "-" in part:
            range_parts = part.split("-")
            if len(range_parts) == 2:
                try:
                    start = int(range_parts[0].replace("-", ""))
                    end = int(range_parts[1].replace("-", ""))
                    for num in range(start, end + 1):
                        numbers.append(str(num))
                    continue
                except ValueError:
                    pass

        # Single number
        numbers.append(part.replace("-", ""))

    return numbers


def load_numbers_from_file(filepath: str) -> list[str]:
    """
    Load phone numbers from a text file.

    Expects one number per line. Lines starting with # are comments.
    Empty lines are skipped.

    Args:
        filepath: Path to file

    Returns:
        List of phone number strings
    """
    numbers = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove common formatting
            clean = "".join(c for c in line if c.isdigit() or c == "+")
            if clean:
                numbers.append(clean)
    return numbers
