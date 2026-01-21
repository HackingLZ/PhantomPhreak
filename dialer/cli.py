"""Command-line interface for the phone line scanner."""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .analysis.signatures import LineType
from .core.modem import Modem, ModemCapability, ModemError
from .core.scanner import Scanner, ScannerConfig, ScannerError
from .storage.database import Database
from .storage.models import ScanResult, ScanStatus, load_numbers_from_file, parse_number_range


console = Console()

# Setup logging
def setup_logging(log_file: str = "dialer.log", level: str = "INFO"):
    """Configure logging to file and console."""
    log_level = getattr(logging, os.environ.get("DIALER_LOG_LEVEL", level).upper())

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler],
    )

    return logging.getLogger("dialer")

logger = setup_logging()


def load_config(config_path: Optional[str]) -> dict:
    """Load configuration from YAML file."""
    import yaml

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    # Check default locations
    for default in ["./config.yaml", "~/.config/dialer/config.yaml"]:
        path = Path(default).expanduser()
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}

    return {}


@click.group()
@click.option("--config", "-c", help="Path to config file", default=None)
@click.pass_context
def cli(ctx, config):
    """Phone Line Scanner - Wardialer using USB modems."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)


@cli.command()
@click.option("--device", "-d", help="Modem device (e.g., /dev/ttyACM0)")
@click.pass_context
def detect(ctx, device):
    """Detect and test modem connectivity."""
    config = ctx.obj["config"]
    modem_config = config.get("modem", {})

    device = device or modem_config.get("device")

    console.print("\n[bold]Phone Line Scanner - Modem Detection[/bold]\n")

    # Find devices
    console.print("[cyan]Searching for modem devices...[/cyan]")
    devices = Modem.detect_devices()

    if not devices:
        console.print("[red]No modem devices found![/red]")
        console.print("\nCheck that your USB modem is connected.")
        sys.exit(1)

    console.print(f"Found {len(devices)} device(s):\n")
    for d in devices:
        console.print(f"  - {d}")

    # Test specified or first device
    test_device = device or devices[0]
    console.print(f"\n[cyan]Testing {test_device}...[/cyan]\n")

    try:
        modem = Modem(
            device=test_device,
            baud_rate=modem_config.get("baud_rate", 460800),
        )
        modem.connect()

        # Reset
        if modem.reset():
            console.print("[green]OK[/green] - Modem reset successful")
        else:
            console.print("[yellow]WARN[/yellow] - Modem reset returned unexpected response")

        # Get info
        info = modem.get_info()

        table = Table(title="Modem Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Device", info.device)
        table.add_row("Manufacturer", info.manufacturer)
        table.add_row("Model", info.model)
        table.add_row(
            "Capabilities",
            ", ".join(c.value for c in info.capabilities) or "None detected"
        )
        table.add_row(
            "Voice Mode",
            "[green]Supported[/green]" if info.voice_supported else "[yellow]Not supported[/yellow]"
        )

        console.print(table)

        if not info.voice_supported:
            console.print("\n[yellow]Note:[/yellow] Voice mode not supported.")
            console.print("Scanner will use result codes only (BUSY, NO CARRIER, etc.)")
            console.print("Audio-based classification will not be available.\n")
        else:
            console.print("\n[green]Voice mode supported![/green]")
            console.print("Full audio analysis will be available.\n")

        modem.disconnect()

    except ModemError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("number")
@click.option("--device", "-d", help="Modem device")
@click.option("--pulse", "-p", is_flag=True, help="Use pulse dialing (ATDP) instead of tone (ATDT)")
@click.option("--detect-dial-tone", "-w", is_flag=True, help="Wait for dial tone before dialing")
@click.pass_context
def call(ctx, number, device, pulse, detect_dial_tone):
    """Test dial a single phone number."""
    config = ctx.obj["config"]
    modem_config = config.get("modem", {})
    audio_config = config.get("audio", {})

    console.print(f"\n[bold]Testing: {number}[/bold]\n")
    logger.info(f"Test call to {number}")

    # Determine dial mode
    dial_mode = "pulse" if pulse else modem_config.get("dial_mode", "tone")
    use_dial_tone = detect_dial_tone or modem_config.get("detect_dial_tone", False)

    if dial_mode == "pulse":
        console.print("[dim]Using pulse dialing[/dim]")
    if use_dial_tone:
        console.print("[dim]Waiting for dial tone before dialing[/dim]")

    # Create minimal scanner config
    scanner_cfg = ScannerConfig(
        modem_device=device or modem_config.get("device"),
        baud_rate=modem_config.get("baud_rate", 460800),
        call_timeout=modem_config.get("call_timeout", 45),
        record_duration=modem_config.get("record_duration", 10),
        sample_rate=audio_config.get("sample_rate", 8000),
        audio_output_dir=audio_config.get("output_dir", "./recordings"),
        db_path=":memory:",  # Don't persist test calls
        dial_mode=dial_mode,
        detect_dial_tone=use_dial_tone,
        init_string=modem_config.get("init_string"),
    )

    scanner = Scanner(scanner_cfg)

    try:
        console.print("[cyan]Dialing...[/cyan]")
        start_time = time.time()

        # Scan the single number (use scan_id=0 for test)
        result = scanner.scan_number(number, scan_id=0)

        total_time = time.time() - start_time

        # Display results
        type_color = {
            LineType.FAX: "magenta",
            LineType.MODEM: "magenta",
            LineType.VOICE: "green",
            LineType.BUSY: "yellow",
        }.get(result.line_type, "white")

        console.print()
        table = Table(title="Call Result")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Phone Number", result.phone_number)
        table.add_row("Call Result", result.call_result.value)
        table.add_row("Line Type", f"[{type_color}]{result.line_type.value}[/{type_color}]")
        table.add_row("Confidence", f"{result.confidence:.0%}")
        table.add_row("Audio Duration", f"{result.duration:.1f}s" if result.duration else "-")
        table.add_row("Total Time", f"[bold]{total_time:.1f}s[/bold]")
        table.add_row("Audio File", result.audio_file or "-")

        console.print(table)

        if result.notes:
            console.print(f"\n[dim]Notes: {result.notes}[/dim]")

        if result.frequencies:
            console.print(f"[dim]Frequencies: {result.frequencies}[/dim]")

        logger.info(f"Test call result: {result.line_type.value} ({result.confidence:.0%})")

    except ScannerError as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Test call failed: {e}")
        sys.exit(1)
    except ModemError as e:
        console.print(f"[red]Modem error:[/red] {e}")
        logger.error(f"Modem error: {e}")
        sys.exit(1)
    finally:
        if scanner.modem:
            scanner.modem.disconnect()


@cli.command()
@click.option("--numbers", "-n", help="File containing phone numbers (one per line)")
@click.option("--range", "-r", "number_range", help="Number range (e.g., 5551000-5551010)")
@click.option("--blacklist", "-b", help="File containing numbers to skip")
@click.option("--name", default="Scan", help="Name for this scan")
@click.option("--device", "-d", help="Modem device")
@click.option("--resume", type=int, help="Resume scan by ID")
@click.option("--pulse", "-p", is_flag=True, help="Use pulse dialing (ATDP) instead of tone (ATDT)")
@click.option("--detect-dial-tone", "-w", is_flag=True, help="Wait for dial tone before each call")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose debug output")
@click.pass_context
def scan(ctx, numbers, number_range, blacklist, name, device, resume, pulse, detect_dial_tone, verbose):
    """Scan phone numbers for line type classification."""
    config = ctx.obj["config"]
    modem_config = config.get("modem", {})
    audio_config = config.get("audio", {})
    scanner_config = config.get("scanner", {})
    db_config = config.get("database", {})

    # Parse numbers
    phone_numbers = []
    blacklisted = set()

    # Load blacklist if provided
    if blacklist:
        blacklisted = set(load_numbers_from_file(blacklist))
        console.print(f"[dim]Loaded {len(blacklisted)} blacklisted numbers[/dim]")
        logger.info(f"Loaded {len(blacklisted)} blacklisted numbers")

    if resume:
        # Get numbers from existing scan
        db = Database(db_config.get("path", "./dialer.db"))
        existing_scan = db.get_scan(resume)
        if not existing_scan:
            console.print(f"[red]Scan {resume} not found[/red]")
            sys.exit(1)
        name = existing_scan.name
        console.print(f"[cyan]Resuming scan: {name}[/cyan]")
        logger.info(f"Resuming scan {resume}: {name}")
    elif numbers:
        phone_numbers = load_numbers_from_file(numbers)
    elif number_range:
        phone_numbers = parse_number_range(number_range)
    else:
        console.print("[red]Error:[/red] Must provide --numbers file or --range")
        sys.exit(1)

    # Apply blacklist
    if blacklisted and phone_numbers:
        original_count = len(phone_numbers)
        phone_numbers = [n for n in phone_numbers if n not in blacklisted]
        skipped = original_count - len(phone_numbers)
        if skipped > 0:
            console.print(f"[dim]Skipping {skipped} blacklisted numbers[/dim]")
            logger.info(f"Skipped {skipped} blacklisted numbers")

    if not resume and not phone_numbers:
        console.print("[red]Error:[/red] No phone numbers to scan")
        sys.exit(1)

    # Determine dial mode
    dial_mode = "pulse" if pulse else modem_config.get("dial_mode", "tone")
    use_dial_tone = detect_dial_tone or modem_config.get("detect_dial_tone", False)

    console.print(f"\n[bold]Phone Line Scanner[/bold]")
    console.print(f"Scan: {name}")
    if not resume:
        console.print(f"Numbers to scan: {len(phone_numbers)}")
    if dial_mode == "pulse":
        console.print("[dim]Using pulse dialing[/dim]")
    if use_dial_tone:
        console.print("[dim]Dial tone detection enabled[/dim]")
    if verbose:
        console.print("[dim]Verbose logging enabled[/dim]")
        # Enable debug logging for scanner and modem with console output
        debug_handler = logging.StreamHandler()
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

        scanner_logger = logging.getLogger("dialer.core.scanner")
        scanner_logger.setLevel(logging.DEBUG)
        scanner_logger.addHandler(debug_handler)

        modem_logger = logging.getLogger("dialer.core.modem")
        modem_logger.setLevel(logging.DEBUG)
        modem_logger.addHandler(debug_handler)
    console.print()

    logger.info(f"Starting scan '{name}' with {len(phone_numbers)} numbers")

    # Create scanner config
    scanner_cfg = ScannerConfig(
        modem_device=device or modem_config.get("device"),
        baud_rate=modem_config.get("baud_rate", 460800),
        call_timeout=modem_config.get("call_timeout", 45),
        record_duration=modem_config.get("record_duration", 10),
        sample_rate=audio_config.get("sample_rate", 8000),
        audio_output_dir=audio_config.get("output_dir", "./recordings"),
        call_delay=scanner_config.get("call_delay", 3),
        max_retries=scanner_config.get("max_retries", 2),
        db_path=db_config.get("path", "./dialer.db"),
        dial_mode=dial_mode,
        detect_dial_tone=use_dial_tone,
        init_string=modem_config.get("init_string"),
    )

    # Track results for live display
    results_by_type = {lt: 0 for lt in LineType}
    total_numbers = len(phone_numbers) if not resume else 100
    scan_start_time = time.time()

    def make_progress_callback(progress, task_id, is_verbose):
        def callback(current: int, total: int, result: ScanResult):
            results_by_type[result.line_type] += 1
            progress.update(task_id, completed=current)

            # Show result with stats
            color = {
                LineType.FAX: "magenta",
                LineType.MODEM: "magenta",
                LineType.VOICE: "green",
                LineType.BUSY: "yellow",
                LineType.UNKNOWN: "white",
            }.get(result.line_type, "white")

            remaining = total - current
            elapsed = time.time() - scan_start_time
            avg_time = elapsed / current if current > 0 else 0

            # Build stats line
            stats = []
            if results_by_type[LineType.VOICE] > 0:
                stats.append(f"[green]Voice:{results_by_type[LineType.VOICE]}[/green]")
            if results_by_type[LineType.FAX] > 0:
                stats.append(f"[magenta]Fax:{results_by_type[LineType.FAX]}[/magenta]")
            if results_by_type[LineType.MODEM] > 0:
                stats.append(f"[magenta]Modem:{results_by_type[LineType.MODEM]}[/magenta]")
            if results_by_type[LineType.BUSY] > 0:
                stats.append(f"[yellow]Busy:{results_by_type[LineType.BUSY]}[/yellow]")
            stats.append(f"[dim]Remaining:{remaining}[/dim]")
            stats.append(f"[dim]Avg:{avg_time:.1f}s/call[/dim]")

            stats_str = " | ".join(stats)

            console.print(
                f"  {result.phone_number}: "
                f"[{color}]{result.line_type.value}[/{color}] "
                f"({result.confidence:.0%}) "
                f"[dim]({result.duration:.1f}s)[/dim]"
            )
            # Show notes/reasoning
            if result.notes:
                console.print(f"  [dim]Notes: {result.notes}[/dim]")
            # Show detected frequencies
            if result.frequencies:
                freq_str = ", ".join(f"{f} Hz" for f in sorted(result.frequencies))
                console.print(f"  [dim]Frequencies: {freq_str}[/dim]")
            console.print(f"  {stats_str}")

        return callback

    # Run scan with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(
            "Scanning...",
            total=len(phone_numbers) if not resume else 100
        )

        scanner = Scanner(
            scanner_cfg,
            progress_callback=make_progress_callback(progress, task_id, verbose),
        )

        try:
            scan_result = scanner.run_scan(
                phone_numbers,
                name=name,
                resume_scan_id=resume,
            )
        except ScannerError as e:
            console.print(f"\n[red]Scan error:[/red] {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Scan interrupted[/yellow]")
            scanner.stop()
            return

    # Show summary
    console.print("\n[bold]Scan Complete[/bold]\n")

    # Show recordings folder
    recordings_dir = Path(scanner_cfg.audio_output_dir) / str(scan_result.id)
    if recordings_dir.exists():
        recording_count = len(list(recordings_dir.glob("*.wav")))
        console.print(f"[cyan]Recordings:[/cyan] {recordings_dir.absolute()} ({recording_count} files)\n")

    summary = scanner.get_summary(scan_result.id)
    if summary:
        table = Table(title="Results Summary")
        table.add_column("Line Type", style="cyan")
        table.add_column("Count", justify="right")

        for lt, count in sorted(summary.by_line_type.items(), key=lambda x: -x[1]):
            if count > 0:
                table.add_row(lt.value, str(count))

        console.print(table)

        if summary.interesting_count > 0:
            console.print(
                f"\n[green]Found {summary.interesting_count} interesting result(s)[/green]"
            )
            console.print(f"Run 'dialer results {scan_result.id}' to view details")


@cli.command()
@click.argument("scan_id", type=int, required=False)
@click.option("--type", "-t", "line_type", help="Filter by line type")
@click.option("--limit", "-l", type=int, default=50, help="Maximum results to show")
@click.pass_context
def results(ctx, scan_id, line_type, limit):
    """View scan results."""
    config = ctx.obj["config"]
    db_config = config.get("database", {})
    db = Database(db_config.get("path", "./dialer.db"))

    # If no scan_id, show list of scans
    if not scan_id:
        scans = db.get_all_scans()
        if not scans:
            console.print("No scans found.")
            return

        table = Table(title="Scans")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Status")
        table.add_column("Progress")
        table.add_column("Created")

        for s in scans:
            status_color = {
                ScanStatus.COMPLETED: "green",
                ScanStatus.RUNNING: "yellow",
                ScanStatus.FAILED: "red",
            }.get(s.status, "white")

            table.add_row(
                str(s.id),
                s.name,
                f"[{status_color}]{s.status.value}[/{status_color}]",
                f"{s.completed_numbers}/{s.total_numbers}",
                s.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)
        console.print("\nUse 'dialer results <ID>' to view details")
        return

    # Show results for specific scan
    filter_type = LineType(line_type) if line_type else None
    results_list = db.get_results(scan_id, line_type=filter_type, limit=limit)

    if not results_list:
        console.print(f"No results found for scan {scan_id}")
        return

    scan = db.get_scan(scan_id)
    console.print(f"\n[bold]Results for: {scan.name}[/bold]\n")

    table = Table()
    table.add_column("Phone", style="cyan")
    table.add_column("Type")
    table.add_column("Confidence", justify="right")
    table.add_column("Result")
    table.add_column("Duration", justify="right")

    for r in results_list:
        type_color = {
            LineType.FAX: "magenta",
            LineType.MODEM: "magenta",
            LineType.VOICE: "green",
            LineType.BUSY: "yellow",
        }.get(r.line_type, "white")

        table.add_row(
            r.phone_number,
            f"[{type_color}]{r.line_type.value}[/{type_color}]",
            f"{r.confidence:.0%}",
            r.call_result.value,
            f"{r.duration:.1f}s" if r.duration else "-",
        )

    console.print(table)

    # Show summary
    summary = db.get_scan_summary(scan_id)
    if summary and summary.interesting_count > 0:
        console.print(f"\n[bold]Interesting finds:[/bold]")
        interesting = db.get_interesting_results(scan_id)
        for r in interesting[:10]:
            console.print(f"  {r.phone_number}: {r.line_type.value} ({r.confidence:.0%})")


@cli.command()
@click.argument("scan_id", type=int)
@click.option("--format", "-f", "fmt", type=click.Choice(["csv", "json"]), default="csv")
@click.option("--output", "-o", help="Output file path")
@click.pass_context
def export(ctx, scan_id, fmt, output):
    """Export scan results to file."""
    config = ctx.obj["config"]
    db_config = config.get("database", {})
    db = Database(db_config.get("path", "./dialer.db"))

    scan = db.get_scan(scan_id)
    if not scan:
        console.print(f"[red]Scan {scan_id} not found[/red]")
        sys.exit(1)

    # Default output path
    if not output:
        output = f"scan_{scan_id}_results.{fmt}"

    if fmt == "csv":
        count = db.export_csv(scan_id, output)
    else:
        count = db.export_json(scan_id, output)

    console.print(f"[green]Exported {count} results to {output}[/green]")


@cli.command()
@click.argument("scan_id", type=int)
@click.option("--output", "-o", help="Output folder path (default: scan_<id>_report/)")
@click.option("--no-spectrograms", is_flag=True, help="Skip spectrogram generation")
@click.pass_context
def report(ctx, scan_id, output, no_spectrograms):
    """Generate HTML report with audio playback and spectrograms.

    Creates a folder with:
      - index.html (the report)
      - audio/ (wav files)
      - spectrograms/ (png images)
    """
    from .reports.html_report import HTMLReportGenerator

    config = ctx.obj["config"]
    db_config = config.get("database", {})
    modem_config = config.get("modem", {})

    db = Database(db_config.get("path", "./dialer.db"))
    audio_dir = Path(modem_config.get("audio_output_dir", "./recordings"))

    scan = db.get_scan(scan_id)
    if not scan:
        console.print(f"[red]Scan {scan_id} not found[/red]")
        sys.exit(1)

    # Default output path
    if not output:
        output = f"scan_{scan_id}_report"

    console.print(f"\n[bold]Generating HTML Report for Scan {scan_id}[/bold]\n")
    console.print(f"[dim]Output folder: {output}/[/dim]")
    console.print(f"[dim]Include spectrograms: {not no_spectrograms}[/dim]\n")

    try:
        generator = HTMLReportGenerator(db, audio_dir)
        count = generator.generate(
            scan_id,
            output,
            include_spectrograms=not no_spectrograms
        )
        console.print(f"[green]Generated report with {count} results: {output}/index.html[/green]")
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--device", "-d", help="Modem device")
@click.option("--method", "-m", type=click.Choice(["simple", "audio"]), default="simple",
              help="Detection method: simple (ATX4) or audio (voice mode FFT)")
@click.pass_context
def dialtone(ctx, device, method):
    """Test dial tone detection on the phone line.

    This command checks if a dial tone is present on the connected phone line.
    Useful for verifying line connectivity before running a scan.

    Methods:
    - simple: Uses ATX4 and dial command (fast, works on all modems)
    - audio: Uses voice mode to analyze frequencies (requires voice-capable modem)
    """
    config = ctx.obj["config"]
    modem_config = config.get("modem", {})

    console.print("\n[bold]Dial Tone Detection Test[/bold]\n")

    try:
        modem = Modem(
            device=device or modem_config.get("device"),
            baud_rate=modem_config.get("baud_rate", 460800),
        )
        modem.connect()
        modem.initialize()

        console.print(f"[cyan]Testing dial tone on {modem.device}...[/cyan]")
        console.print(f"[dim]Method: {method}[/dim]\n")

        if method == "audio":
            # Check if voice mode is supported
            info = modem.get_info()
            if not info.voice_supported:
                console.print("[yellow]Voice mode not supported, falling back to simple method[/yellow]")
                method = "simple"

        if method == "simple":
            result = modem.check_dial_tone(timeout=5.0)
        else:
            result = modem.wait_for_dial_tone(timeout=10.0)

        if result:
            console.print("[green]✓ Dial tone detected![/green]")
            console.print("Phone line is connected and ready for dialing.")
        else:
            console.print("[red]✗ No dial tone detected[/red]")
            console.print("\nPossible causes:")
            console.print("  - Phone line not connected to modem")
            console.print("  - Phone line is disconnected or out of service")
            console.print("  - Another device is using the line")
            console.print("  - PBX or VoIP system without standard dial tone")

        modem.disconnect()

    except ModemError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.pass_context
def analyze(ctx, audio_file):
    """Analyze an audio file for line type classification."""
    from .analysis.classifier import Classifier

    console.print(f"\n[bold]Analyzing: {audio_file}[/bold]\n")

    classifier = Classifier()
    result = classifier.classify_from_file(audio_file)

    console.print(f"Line Type: [cyan]{result.line_type.value}[/cyan]")
    console.print(f"Confidence: {result.confidence:.0%}")
    console.print(f"Unique Frequencies: {result.unique_frequencies}")
    console.print(f"Average Energy: {result.avg_energy:.4f}")
    console.print(f"\nReasoning: {result.reasoning}")

    if result.matches:
        console.print("\n[bold]Signature Matches:[/bold]")
        for match in result.matches[:5]:
            console.print(
                f"  {match.signature.name}: "
                f"{match.confidence:.0%} "
                f"(frequencies: {match.matched_frequencies})"
            )


@cli.command()
@click.argument("scan_id", type=int)
@click.option("--update", "-u", is_flag=True, help="Update database with new classifications")
@click.pass_context
def reanalyze(ctx, scan_id, update):
    """Re-analyze audio files from a previous scan.

    Useful after updating classification signatures or to review results.
    """
    from .analysis.classifier import Classifier

    config = ctx.obj["config"]
    db_config = config.get("database", {})
    db = Database(db_config.get("path", "./dialer.db"))

    scan = db.get_scan(scan_id)
    if not scan:
        console.print(f"[red]Scan {scan_id} not found[/red]")
        sys.exit(1)

    results = db.get_results(scan_id)
    if not results:
        console.print(f"No results found for scan {scan_id}")
        return

    # Filter to results with audio files
    with_audio = [r for r in results if r.audio_file and Path(r.audio_file).exists()]

    if not with_audio:
        console.print("No audio files found for this scan")
        return

    console.print(f"\n[bold]Re-analyzing {len(with_audio)} audio files[/bold]\n")
    logger.info(f"Re-analyzing {len(with_audio)} files from scan {scan_id}")

    classifier = Classifier()
    changes = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=len(with_audio))

        for result in with_audio:
            try:
                new_classification = classifier.classify_from_file(result.audio_file)

                # Check if classification changed
                if new_classification.line_type != result.line_type:
                    changes.append({
                        "number": result.phone_number,
                        "old_type": result.line_type,
                        "new_type": new_classification.line_type,
                        "old_conf": result.confidence,
                        "new_conf": new_classification.confidence,
                        "result": result,
                        "classification": new_classification,
                    })

            except Exception as e:
                logger.warning(f"Failed to analyze {result.audio_file}: {e}")

            progress.advance(task)

    # Report changes
    if not changes:
        console.print("[green]No classification changes detected[/green]")
        return

    console.print(f"\n[yellow]Found {len(changes)} classification change(s):[/yellow]\n")

    table = Table()
    table.add_column("Phone", style="cyan")
    table.add_column("Old Type")
    table.add_column("New Type")
    table.add_column("Old Conf")
    table.add_column("New Conf")

    for change in changes:
        table.add_row(
            change["number"],
            change["old_type"].value,
            change["new_type"].value,
            f"{change['old_conf']:.0%}",
            f"{change['new_conf']:.0%}",
        )

    console.print(table)

    # Update database if requested
    if update:
        console.print("\n[cyan]Updating database...[/cyan]")
        # Note: Would need to add an update_result method to Database
        # For now, just report what would change
        console.print(f"[yellow]Database update not yet implemented[/yellow]")
        console.print("Results shown above would be updated.")
        logger.info(f"Would update {len(changes)} results (not implemented)")
    else:
        console.print("\n[dim]Use --update to save changes to database[/dim]")


@cli.command()
@click.argument("number")
@click.option("--device", "-d", help="Modem device")
@click.pass_context
def dialtest(ctx, number, device):
    """Test different dialing methods to debug modem issues.

    Tries multiple dialing approaches and shows raw AT responses to help
    identify what works with your modem.
    """
    import time

    config = ctx.obj["config"]
    modem_config = config.get("modem", {})

    console.print("\n[bold]Modem Dial Test[/bold]")
    console.print(f"Testing number: {number}\n")

    # Clean number
    clean_number = ''.join(c for c in number if c.isdigit() or c == '+')

    try:
        modem = Modem(
            device=device or modem_config.get("device"),
            baud_rate=modem_config.get("baud_rate", 460800),
        )
        modem.connect()

        console.print(f"[cyan]Connected to {modem.device}[/cyan]\n")

        # Helper to send command and show response
        def test_command(cmd: str, description: str, timeout: float = 10.0) -> str:
            console.print(f"[dim]Test: {description}[/dim]")
            console.print(f"  → Sending: [yellow]{cmd}[/yellow]")
            try:
                modem._send(cmd)
                response = modem._read_response(timeout=timeout)
                console.print(f"  ← Response: [green]{repr(response)}[/green]\n")
                return response
            except Exception as e:
                console.print(f"  ← Error: [red]{e}[/red]\n")
                return ""

        # Test 1: Reset and initialize
        console.print("[bold]═══ Phase 1: Reset & Initialize ═══[/bold]\n")
        test_command("ATZ", "Reset modem")
        test_command("ATE1V1", "Enable echo and verbose responses")
        test_command("ATI", "Modem identification")

        # Test 2: Check capabilities
        console.print("[bold]═══ Phase 2: Check Capabilities ═══[/bold]\n")
        test_command("AT+FCLASS=?", "Query supported modes")
        test_command("AT+VCID=?", "Query caller ID support")
        test_command("ATX?", "Query result code mode")

        # Test 3: Data mode dialing
        console.print("[bold]═══ Phase 3: Data Mode Dial Tests ═══[/bold]\n")

        # Standard init
        test_command("ATZ", "Reset")
        test_command("AT+FCLASS=0", "Set data mode (FCLASS=0)")
        test_command("ATX4", "Enable extended result codes")

        # Try ATDT (tone dial)
        console.print("[yellow]Attempting ATDT (tone dial)...[/yellow]")
        console.print("[dim]Will wait up to 15 seconds for response[/dim]\n")
        response = test_command(f"ATDT{clean_number}", "Tone dial in data mode", timeout=15.0)

        # Hang up
        time.sleep(1)
        modem._serial.write(b"+++")
        time.sleep(1)
        test_command("ATH", "Hang up")

        # Test 4: Voice mode (if supported)
        console.print("[bold]═══ Phase 4: Voice Mode Tests ═══[/bold]\n")

        test_command("ATZ", "Reset")
        fclass_resp = test_command("AT+FCLASS=8", "Set voice mode (FCLASS=8)")

        if "OK" in fclass_resp:
            console.print("[green]Voice mode supported![/green]\n")

            test_command("AT+VSM=?", "Query voice compression methods")
            test_command("AT+VLS=?", "Query voice line select options")

            # Try voice dial
            console.print("[yellow]Attempting ATD in voice mode...[/yellow]\n")
            response = test_command(f"ATD{clean_number}", "Voice mode dial", timeout=15.0)

            # Hang up
            time.sleep(1)
            modem._serial.write(b"+++")
            time.sleep(1)
            test_command("ATH", "Hang up")

            # Test alternative: dial first, then switch to voice
            console.print("\n[bold]═══ Phase 5: Hybrid Approach ═══[/bold]")
            console.print("[dim]Dial in data mode, then switch to voice[/dim]\n")

            test_command("ATZ", "Reset")
            test_command("AT+FCLASS=0", "Set data mode")
            test_command("ATX4", "Enable extended result codes")

            console.print("[yellow]Dialing in data mode first...[/yellow]\n")
            response = test_command(f"ATDT{clean_number};", "Dial with semicolon (return to command mode)", timeout=15.0)

            if "OK" in response:
                console.print("[green]Dial initiated, modem in command mode[/green]")
                test_command("AT+FCLASS=8", "Switch to voice mode")
                test_command("AT+VRX", "Start voice receive")
                time.sleep(2)
                modem._serial.write(bytes([0x10, 0x03]))  # DLE ETX
                time.sleep(0.5)

            # Final hangup
            time.sleep(1)
            modem._serial.write(b"+++")
            time.sleep(1)
            test_command("ATH", "Final hang up")

        else:
            console.print("[yellow]Voice mode not supported, skipping voice tests[/yellow]\n")

        # Summary
        console.print("[bold]═══ Test Complete ═══[/bold]\n")
        console.print("Review the responses above to determine:")
        console.print("  • Which dial method works (ATDT vs ATD)")
        console.print("  • Whether voice mode is properly supported")
        console.print("  • What responses your modem returns\n")

        modem.disconnect()

    except ModemError as e:
        console.print(f"[red]Modem error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--scan-id", "-s", type=int, help="Train from scan results (uses recordings)")
@click.option("--directory", "-d", type=click.Path(exists=True), help="Train from directory of audio files")
@click.option("--min-confidence", type=float, default=0.8, help="Min confidence to use as training sample")
@click.option("--output", "-o", help="Output model path (default: built-in location)")
@click.pass_context
def train(ctx, scan_id, directory, min_confidence, output):
    """Train ML classifier on real audio samples.

    Train from scan results (--scan-id) or a directory of audio files (--directory).

    For directory training, organize files like:
      training/
        voice/
          sample1.wav
          sample2.wav
        fax/
          sample1.wav
        modem/
          sample1.wav
        ...
    """
    from pathlib import Path

    from .analysis.ml_classifier import MLClassifier
    from .analysis.signatures import LineType
    from .core.audio import load_wav

    config = ctx.obj["config"]
    console.print("\n[bold]ML Classifier Training[/bold]\n")

    samples_by_type: dict[LineType, list] = {}

    if scan_id:
        # Train from scan results
        db_config = config.get("database", {})
        modem_config = config.get("modem", {})
        db = Database(db_config.get("path", "./dialer.db"))
        audio_base = Path(modem_config.get("audio_output_dir", "./recordings"))

        scan = db.get_scan(scan_id)
        if not scan:
            console.print(f"[red]Scan {scan_id} not found[/red]")
            sys.exit(1)

        results = db.get_results(scan_id)
        console.print(f"Loading samples from scan {scan_id}...")

        for r in results:
            if r.confidence < min_confidence:
                continue

            if not r.audio_file:
                continue

            # Try to find audio file
            audio_path = audio_base / str(scan_id) / r.audio_file
            if not audio_path.exists():
                audio_path = Path(r.audio_file)
            if not audio_path.exists():
                continue

            try:
                samples, sample_rate = load_wav(str(audio_path))
                if r.line_type not in samples_by_type:
                    samples_by_type[r.line_type] = []
                samples_by_type[r.line_type].append(samples)
            except Exception as e:
                console.print(f"[yellow]Skip {audio_path}: {e}[/yellow]")

    elif directory:
        # Train from directory structure
        dir_path = Path(directory)
        console.print(f"Loading samples from {directory}...")

        for type_dir in dir_path.iterdir():
            if not type_dir.is_dir():
                continue

            # Map directory name to LineType
            try:
                line_type = LineType(type_dir.name.lower())
            except ValueError:
                console.print(f"[yellow]Unknown type directory: {type_dir.name}[/yellow]")
                continue

            samples_by_type[line_type] = []
            for audio_file in type_dir.glob("*.wav"):
                try:
                    samples, sample_rate = load_wav(str(audio_file))
                    samples_by_type[line_type].append(samples)
                except Exception as e:
                    console.print(f"[yellow]Skip {audio_file}: {e}[/yellow]")

    else:
        console.print("[red]Specify --scan-id or --directory[/red]")
        sys.exit(1)

    # Report what we found
    total_samples = 0
    console.print("\n[bold]Training Samples:[/bold]")
    for lt, samples in sorted(samples_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        count = len(samples)
        total_samples += count
        console.print(f"  {lt.value}: {count} samples")

    if total_samples < 10:
        console.print(f"\n[red]Too few samples ({total_samples}). Need at least 10 for training.[/red]")
        sys.exit(1)

    # Train
    console.print(f"\n[cyan]Training on {total_samples} samples...[/cyan]")
    classifier = MLClassifier()
    accuracy = classifier.train(samples_by_type)
    console.print(f"[green]Training complete![/green]")
    console.print(f"Training accuracy: {accuracy:.1%}")

    # Save
    if output:
        model_path = Path(output)
    else:
        model_path = classifier.DEFAULT_MODEL_PATH

    classifier.save(model_path)
    console.print(f"\n[green]Model saved to: {model_path}[/green]")


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
