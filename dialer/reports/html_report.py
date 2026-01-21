"""HTML report generator with audio playback and spectrograms."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..analysis.signatures import LineType
from ..storage.database import Database
from ..storage.models import Scan, ScanResult


class HTMLReportGenerator:
    """Generates interactive HTML reports for scan results."""

    def __init__(self, db: Database, audio_dir: Path):
        """
        Initialize report generator.

        Args:
            db: Database connection
            audio_dir: Base directory for audio recordings
        """
        self.db = db
        self.audio_dir = Path(audio_dir)

    def generate(
        self,
        scan_id: int,
        output_path: str,
        include_spectrograms: bool = True
    ) -> int:
        """
        Generate HTML report for a scan.

        Creates a folder structure:
          output_path/
            index.html
            audio/
              phone_type.wav
            spectrograms/
              phone_type.png

        Args:
            scan_id: Scan ID to report on
            output_path: Path to report folder (will be created)
            include_spectrograms: Whether to generate spectrograms

        Returns:
            Number of results included in report
        """
        scan = self.db.get_scan(scan_id)
        if not scan:
            raise ValueError(f"Scan {scan_id} not found")

        results = self.db.get_results(scan_id, limit=10000)
        summary = self.db.get_scan_summary(scan_id)

        # Create output folder structure
        output_dir = Path(output_path)
        if output_dir.suffix == '.html':
            # If user passed a .html file, use parent dir with name
            output_dir = output_dir.parent / output_dir.stem

        output_dir.mkdir(parents=True, exist_ok=True)
        audio_out = output_dir / "audio"
        audio_out.mkdir(exist_ok=True)

        if include_spectrograms:
            spec_out = output_dir / "spectrograms"
            spec_out.mkdir(exist_ok=True)
        else:
            spec_out = None

        # Copy audio files and generate spectrograms
        audio_files = {}  # phone -> relative path
        spectrogram_files = {}  # phone -> relative path

        for r in results:
            if r.audio_file:
                audio_path = self._find_audio_file(r.audio_file, scan_id)
                if audio_path and audio_path.exists():
                    # Copy audio
                    dest_name = f"{r.phone_number}_{r.line_type.value}.wav"
                    dest_path = audio_out / dest_name
                    shutil.copy2(audio_path, dest_path)
                    audio_files[r.phone_number] = f"audio/{dest_name}"

                    # Generate spectrogram
                    if include_spectrograms and spec_out:
                        spec_name = f"{r.phone_number}_{r.line_type.value}.png"
                        spec_path = spec_out / spec_name
                        if self._generate_spectrogram_file(audio_path, spec_path):
                            spectrogram_files[r.phone_number] = f"spectrograms/{spec_name}"

        # Build HTML
        html = self._generate_html(scan, results, summary, audio_files, spectrogram_files)

        # Write index.html
        index_path = output_dir / "index.html"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return len(results)

    def _find_audio_file(self, audio_file: str, scan_id: int) -> Optional[Path]:
        """Find the audio file, trying multiple path patterns."""
        # Try the path as stored in DB first
        audio_path = Path(audio_file)
        if audio_path.exists():
            return audio_path

        # Try relative to audio_dir with just filename
        audio_path = self.audio_dir / str(scan_id) / Path(audio_file).name
        if audio_path.exists():
            return audio_path

        # Try in scan folder
        audio_path = self.audio_dir / str(scan_id) / audio_file
        if audio_path.exists():
            return audio_path

        return None

    def _generate_spectrogram_file(self, audio_path: Path, output_path: Path) -> bool:
        """Generate spectrogram PNG file."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from scipy.io import wavfile

            # Read audio
            sample_rate, samples = wavfile.read(audio_path)
            if len(samples.shape) > 1:
                samples = samples[:, 0]

            # Normalize
            samples = samples.astype(np.float32)
            if samples.max() > 0:
                samples = samples / np.abs(samples).max()

            # Generate spectrogram
            fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
            ax.specgram(samples, Fs=sample_rate, cmap='viridis', NFFT=256, noverlap=128)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Freq (Hz)', fontsize=9)
            ax.tick_params(labelsize=8)
            plt.tight_layout()

            # Save
            plt.savefig(output_path, format='png', bbox_inches='tight', facecolor='#16213e')
            plt.close(fig)
            return True

        except ImportError:
            return False
        except Exception:
            return False

    def _generate_html(
        self,
        scan: Scan,
        results: list[ScanResult],
        summary,
        audio_files: dict,
        spectrogram_files: dict
    ) -> str:
        """Generate the complete HTML document."""
        # Group results by type for summary chart
        type_counts = {}
        for r in results:
            lt = r.line_type.value
            type_counts[lt] = type_counts.get(lt, 0) + 1

        # Generate results rows
        rows_html = []
        for r in results:
            row = self._generate_result_row(r, audio_files, spectrogram_files)
            rows_html.append(row)

        results_html = "\n".join(rows_html)

        # Generate chart data
        chart_data = json.dumps(type_counts)

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scan Report: {scan.name}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --accent: #0f3460;
            --text: #e4e4e4;
            --text-muted: #8a8a8a;
            --success: #4ecca3;
            --warning: #ffc107;
            --danger: #ff6b6b;
            --info: #45b7d1;
            --purple: #9b59b6;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-color);
            color: var(--text);
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1, h2 {{
            margin-bottom: 20px;
        }}
        .header {{
            background: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: var(--info);
            font-size: 2em;
        }}
        .meta {{
            color: var(--text-muted);
            margin-top: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--info);
        }}
        .stat-label {{
            color: var(--text-muted);
            margin-top: 5px;
        }}
        .chart-container {{
            background: var(--card-bg);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            display: flex;
            justify-content: center;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
        }}
        .results-table th {{
            background: var(--accent);
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .results-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--accent);
            vertical-align: top;
        }}
        .results-table tr:hover {{
            background: rgba(69, 183, 209, 0.1);
        }}
        .type-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .type-fax, .type-modem, .type-carrier {{ background: var(--purple); }}
        .type-voice, .type-voicemail, .type-answering_machine {{ background: var(--success); }}
        .type-busy, .type-ringback {{ background: var(--warning); color: #000; }}
        .type-sit {{ background: var(--danger); }}
        .type-ivr {{ background: var(--info); }}
        .type-silence, .type-unknown {{ background: var(--text-muted); }}
        .confidence {{
            display: inline-block;
            width: 100px;
            height: 8px;
            background: var(--accent);
            border-radius: 4px;
            overflow: hidden;
        }}
        .confidence-bar {{
            height: 100%;
            background: var(--success);
            transition: width 0.3s;
        }}
        .audio-player {{
            margin: 10px 0;
        }}
        .audio-player audio {{
            width: 100%;
            height: 40px;
        }}
        .spectrogram {{
            max-width: 400px;
            height: auto;
            border-radius: 8px;
            margin-top: 10px;
        }}
        .notes {{
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 5px;
        }}
        .frequencies {{
            font-family: monospace;
            font-size: 0.8em;
            color: var(--info);
        }}
        .filter-bar {{
            background: var(--card-bg);
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        .filter-bar label {{
            color: var(--text-muted);
        }}
        .filter-bar select, .filter-bar input {{
            background: var(--accent);
            color: var(--text);
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
        }}
        .hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{scan.name}</h1>
            <div class="meta">
                <div>Scan ID: {scan.id}</div>
                <div>Created: {scan.created_at.strftime("%Y-%m-%d %H:%M:%S")}</div>
                <div>Status: {scan.status.value}</div>
            </div>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{scan.total_numbers}</div>
                <div class="stat-label">Total Numbers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{scan.completed_numbers}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.interesting_count if summary else 0}</div>
                <div class="stat-label">Interesting Finds</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{type_counts.get('fax', 0) + type_counts.get('modem', 0)}</div>
                <div class="stat-label">Fax/Modem</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="typeChart" width="400" height="400"></canvas>
        </div>

        <div class="filter-bar">
            <label>Filter by Type:</label>
            <select id="typeFilter" onchange="filterResults()">
                <option value="">All Types</option>
                {self._generate_type_options(type_counts)}
            </select>
            <label>Search:</label>
            <input type="text" id="searchBox" placeholder="Phone number..." oninput="filterResults()">
        </div>

        <table class="results-table">
            <thead>
                <tr>
                    <th>Phone Number</th>
                    <th>Type</th>
                    <th>Confidence</th>
                    <th>Result</th>
                    <th>Duration</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody id="resultsBody">
                {results_html}
            </tbody>
        </table>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        // Chart
        const chartData = {chart_data};
        const ctx = document.getElementById('typeChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(chartData),
                datasets: [{{
                    data: Object.values(chartData),
                    backgroundColor: [
                        '#9b59b6', '#4ecca3', '#ffc107', '#ff6b6b',
                        '#45b7d1', '#8a8a8a', '#e74c3c', '#3498db'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'right',
                        labels: {{ color: '#e4e4e4' }}
                    }}
                }}
            }}
        }});

        // Filtering
        function filterResults() {{
            const typeFilter = document.getElementById('typeFilter').value.toLowerCase();
            const searchText = document.getElementById('searchBox').value.toLowerCase();
            const rows = document.querySelectorAll('#resultsBody tr');

            rows.forEach(row => {{
                const type = row.dataset.type || '';
                const phone = row.dataset.phone || '';
                const matchesType = !typeFilter || type === typeFilter;
                const matchesSearch = !searchText || phone.includes(searchText);
                row.classList.toggle('hidden', !(matchesType && matchesSearch));
            }});
        }}
    </script>
</body>
</html>'''

    def _generate_type_options(self, type_counts: dict) -> str:
        """Generate HTML option tags for type filter."""
        options = []
        for lt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            options.append(f'<option value="{lt}">{lt} ({count})</option>')
        return "\n".join(options)

    def _generate_result_row(
        self,
        result: ScanResult,
        audio_files: dict,
        spectrogram_files: dict
    ) -> str:
        """Generate a table row for a single result."""
        # Type badge
        type_class = f"type-{result.line_type.value}"
        type_badge = f'<span class="type-badge {type_class}">{result.line_type.value}</span>'

        # Confidence bar
        conf_pct = int(result.confidence * 100)
        confidence_html = f'''
            <div class="confidence">
                <div class="confidence-bar" style="width: {conf_pct}%"></div>
            </div>
            <span style="margin-left: 8px">{conf_pct}%</span>
        '''

        # Duration
        duration = f"{result.duration:.1f}s" if result.duration else "-"

        # Details section
        details = []

        # Notes
        if result.notes:
            details.append(f'<div class="notes">{result.notes}</div>')

        # Frequencies
        if result.frequencies:
            freq_str = ", ".join(f"{f} Hz" for f in sorted(result.frequencies))
            details.append(f'<div class="frequencies">Frequencies: {freq_str}</div>')

        # Audio player (linked, not embedded)
        if result.phone_number in audio_files:
            audio_src = audio_files[result.phone_number]
            details.append(f'<div class="audio-player"><audio controls src="{audio_src}"></audio></div>')

        # Spectrogram (linked image)
        if result.phone_number in spectrogram_files:
            spec_src = spectrogram_files[result.phone_number]
            details.append(f'<img class="spectrogram" src="{spec_src}" alt="Spectrogram">')

        details_html = "".join(details) if details else "-"

        return f'''<tr data-type="{result.line_type.value}" data-phone="{result.phone_number}">
            <td>{result.phone_number}</td>
            <td>{type_badge}</td>
            <td>{confidence_html}</td>
            <td>{result.call_result.value}</td>
            <td>{duration}</td>
            <td>{details_html}</td>
        </tr>'''
