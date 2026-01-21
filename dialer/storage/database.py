"""SQLite database for scan persistence."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from ..analysis.signatures import LineType
from ..core.provider import CallResult
from .models import Scan, ScanResult, ScanStatus, ScanSummary


class Database:
    """SQLite database for storing scan results."""

    def __init__(self, db_path: str = "./dialer.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    status TEXT DEFAULT 'pending',
                    total_numbers INTEGER DEFAULT 0,
                    completed_numbers INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    modem_device TEXT,
                    config TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    phone_number TEXT NOT NULL,
                    call_result TEXT NOT NULL,
                    line_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    audio_file TEXT,
                    duration REAL DEFAULT 0.0,
                    frequencies TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    scanned_at TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (scan_id) REFERENCES scans(id)
                );

                CREATE INDEX IF NOT EXISTS idx_results_scan_id ON scan_results(scan_id);
                CREATE INDEX IF NOT EXISTS idx_results_line_type ON scan_results(line_type);
                CREATE INDEX IF NOT EXISTS idx_results_phone ON scan_results(phone_number);
            """)

    # Scan operations

    def create_scan(self, scan: Scan) -> Scan:
        """Create a new scan record."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scans (name, description, status, total_numbers,
                    completed_numbers, created_at, started_at, completed_at,
                    modem_device, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan.name,
                    scan.description,
                    scan.status.value,
                    scan.total_numbers,
                    scan.completed_numbers,
                    scan.created_at.isoformat(),
                    scan.started_at.isoformat() if scan.started_at else None,
                    scan.completed_at.isoformat() if scan.completed_at else None,
                    scan.modem_device,
                    str(scan.config),
                )
            )
            scan.id = cursor.lastrowid
        return scan

    def update_scan(self, scan: Scan) -> None:
        """Update an existing scan record."""
        if scan.id is None:
            raise ValueError("Scan must have an ID to update")

        with self._connection() as conn:
            conn.execute(
                """
                UPDATE scans SET
                    name = ?,
                    description = ?,
                    status = ?,
                    total_numbers = ?,
                    completed_numbers = ?,
                    started_at = ?,
                    completed_at = ?,
                    modem_device = ?,
                    config = ?
                WHERE id = ?
                """,
                (
                    scan.name,
                    scan.description,
                    scan.status.value,
                    scan.total_numbers,
                    scan.completed_numbers,
                    scan.started_at.isoformat() if scan.started_at else None,
                    scan.completed_at.isoformat() if scan.completed_at else None,
                    scan.modem_device,
                    str(scan.config),
                    scan.id,
                )
            )

    def get_scan(self, scan_id: int) -> Optional[Scan]:
        """Get a scan by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scans WHERE id = ?", (scan_id,)
            ).fetchone()

        if row is None:
            return None

        return Scan.from_dict(dict(row))

    def get_all_scans(self) -> list[Scan]:
        """Get all scans, most recent first."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scans ORDER BY created_at DESC"
            ).fetchall()

        return [Scan.from_dict(dict(row)) for row in rows]

    def get_active_scan(self) -> Optional[Scan]:
        """Get the currently running scan, if any."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scans WHERE status = 'running' LIMIT 1"
            ).fetchone()

        if row is None:
            return None

        return Scan.from_dict(dict(row))

    def delete_scan(self, scan_id: int) -> None:
        """Delete a scan and all its results."""
        with self._connection() as conn:
            conn.execute("DELETE FROM scan_results WHERE scan_id = ?", (scan_id,))
            conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))

    # Scan result operations

    def add_result(self, result: ScanResult) -> ScanResult:
        """Add a scan result."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scan_results (scan_id, phone_number, call_result,
                    line_type, confidence, audio_file, duration, frequencies,
                    notes, scanned_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.scan_id,
                    result.phone_number,
                    result.call_result.value,
                    result.line_type.value,
                    result.confidence,
                    result.audio_file,
                    result.duration,
                    ",".join(map(str, result.frequencies)),
                    result.notes,
                    result.scanned_at.isoformat(),
                    result.retry_count,
                )
            )
            result.id = cursor.lastrowid

            # Update scan completed count
            conn.execute(
                """
                UPDATE scans SET completed_numbers = completed_numbers + 1
                WHERE id = ?
                """,
                (result.scan_id,)
            )

        return result

    def get_results(
        self,
        scan_id: int,
        line_type: Optional[LineType] = None,
        limit: Optional[int] = None
    ) -> list[ScanResult]:
        """Get results for a scan, optionally filtered by line type."""
        query = "SELECT * FROM scan_results WHERE scan_id = ?"
        params: list = [scan_id]

        if line_type is not None:
            query += " AND line_type = ?"
            params.append(line_type.value)

        query += " ORDER BY scanned_at DESC"

        if limit is not None:
            query += f" LIMIT {limit}"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [ScanResult.from_dict(dict(row)) for row in rows]

    def get_result_by_number(
        self,
        scan_id: int,
        phone_number: str
    ) -> Optional[ScanResult]:
        """Get result for a specific phone number in a scan."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM scan_results
                WHERE scan_id = ? AND phone_number = ?
                ORDER BY scanned_at DESC LIMIT 1
                """,
                (scan_id, phone_number)
            ).fetchone()

        if row is None:
            return None

        return ScanResult.from_dict(dict(row))

    def number_exists_in_scan(self, scan_id: int, phone_number: str) -> bool:
        """Check if a number has already been scanned in this scan."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM scan_results
                WHERE scan_id = ? AND phone_number = ?
                LIMIT 1
                """,
                (scan_id, phone_number)
            ).fetchone()
        return row is not None

    def get_interesting_results(self, scan_id: int) -> list[ScanResult]:
        """Get results classified as fax, modem, or other interesting types."""
        interesting_types = [
            LineType.FAX.value,
            LineType.MODEM.value,
            LineType.CARRIER.value,
        ]
        placeholders = ",".join("?" * len(interesting_types))

        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM scan_results
                WHERE scan_id = ? AND line_type IN ({placeholders})
                ORDER BY confidence DESC, scanned_at DESC
                """,
                [scan_id] + interesting_types
            ).fetchall()

        return [ScanResult.from_dict(dict(row)) for row in rows]

    # Summary and statistics

    def get_scan_summary(self, scan_id: int) -> Optional[ScanSummary]:
        """Get summary statistics for a scan."""
        scan = self.get_scan(scan_id)
        if scan is None:
            return None

        with self._connection() as conn:
            # Count by line type
            type_counts = conn.execute(
                """
                SELECT line_type, COUNT(*) as count
                FROM scan_results WHERE scan_id = ?
                GROUP BY line_type
                """,
                (scan_id,)
            ).fetchall()

            # Count by call result
            result_counts = conn.execute(
                """
                SELECT call_result, COUNT(*) as count
                FROM scan_results WHERE scan_id = ?
                GROUP BY call_result
                """,
                (scan_id,)
            ).fetchall()

            # Average confidence
            avg_conf = conn.execute(
                """
                SELECT AVG(confidence) as avg_conf
                FROM scan_results WHERE scan_id = ?
                """,
                (scan_id,)
            ).fetchone()

            # Total count
            total = conn.execute(
                "SELECT COUNT(*) as total FROM scan_results WHERE scan_id = ?",
                (scan_id,)
            ).fetchone()

        by_line_type = {
            LineType(row["line_type"]): row["count"]
            for row in type_counts
        }
        by_call_result = {
            CallResult(row["call_result"]): row["count"]
            for row in result_counts
        }

        interesting_count = sum(
            by_line_type.get(lt, 0)
            for lt in [LineType.FAX, LineType.MODEM, LineType.CARRIER]
        )

        return ScanSummary(
            scan=scan,
            total=total["total"] if total else 0,
            completed=scan.completed_numbers,
            by_line_type=by_line_type,
            by_call_result=by_call_result,
            avg_confidence=avg_conf["avg_conf"] or 0.0 if avg_conf else 0.0,
            interesting_count=interesting_count,
        )

    # Export

    def export_csv(self, scan_id: int, filepath: str) -> int:
        """
        Export scan results to CSV file.

        Returns:
            Number of rows exported
        """
        import csv

        results = self.get_results(scan_id)
        if not results:
            return 0

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "phone_number", "call_result", "line_type", "confidence",
                "duration", "frequencies", "audio_file", "notes", "scanned_at"
            ])

            # Data
            for r in results:
                writer.writerow([
                    r.phone_number,
                    r.call_result.value,
                    r.line_type.value,
                    f"{r.confidence:.2f}",
                    f"{r.duration:.1f}",
                    " ".join(map(str, r.frequencies)),
                    r.audio_file or "",
                    r.notes,
                    r.scanned_at.isoformat(),
                ])

        return len(results)

    def export_json(self, scan_id: int, filepath: str) -> int:
        """
        Export scan results to JSON file.

        Returns:
            Number of rows exported
        """
        import json

        results = self.get_results(scan_id)
        scan = self.get_scan(scan_id)

        if not scan:
            return 0

        data = {
            "scan": scan.to_dict(),
            "results": [r.to_dict() for r in results],
            "exported_at": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return len(results)
