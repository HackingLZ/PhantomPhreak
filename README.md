# War Dialer

A phone line scanner that dials numbers and classifies what answers (fax, modem, voice, etc.) using audio frequency analysis, tone cadence detection, and machine learning.

Supports two telephony backends:
- **USB Modem** - Traditional hardware modem (voice-capable recommended)
- **IAX2 VOIP** - Internet telephony via VOIP.MS or other IAX2 providers

## Features

- **Dual provider support** - Use USB modems or IAX2 VOIP (e.g., voip.ms)
- **Voice mode audio capture** - Records audio for analysis
- **FFT-based classification** - Analyzes frequencies to identify fax, modem, dial tone, busy signals
- **Tone cadence detection** - Detects busy (0.5s on/off) and ringback (2s on/4s off) timing patterns
- **Answering machine detection** - Distinguishes voicemail/AM from live human voice
- **DTMF/IVR detection** - Identifies phone menus and IVR systems
- **SIT tone decoding** - Decodes intercept messages (disconnected, circuits busy, etc.)
- **Modem protocol detection** - Identifies V.21, V.22, V.22bis, V.32, V.34 modems
- **Early hangup** - Automatically hangs up when fax/modem detected to save time
- **ML classification** - Train classifier on real samples for improved accuracy
- **HTML reports** - Interactive reports with audio playback and spectrograms
- **SQLite storage** - Persists results with export to CSV/JSON
- **Resume support** - Can pause and resume scans
- **Blacklist support** - Skip specified numbers

## Requirements

- Python 3.10+
- **One of:**
  - USB modem (voice-capable recommended) on Linux/macOS
  - IAX2 VOIP account (e.g., voip.ms) - no hardware needed

## Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
cd dialer
uv sync

# Run commands with uv
uv run dialer detect
uv run dialer scan --range "555-0000:555-0099" --name "Test"
```

### Using pip

```bash
cd dialer
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Option A: Using USB Modem (Default)

#### 1. Detect and Test Your Modem

```bash
dialer detect
```

This will:
- Find connected modem devices
- Test AT commands
- Report voice mode capability

**Expected output for voice-capable modem:**
```
Found 1 device(s):
  - /dev/ttyACM0

Testing /dev/ttyACM0...
OK - Modem reset successful

┌─────────────────────────────────────┐
│         Modem Information           │
├─────────────┬───────────────────────┤
│ Device      │ /dev/ttyACM0          │
│ Manufacturer│ USRobotics            │
│ Model       │ USR5637               │
│ Capabilities│ data_only, fax, voice │
│ Voice Mode  │ Supported             │
└─────────────┴───────────────────────┘
```

#### 2. Test Dial Tone

```bash
dialer dialtone
```

Verifies phone line is connected and has dial tone.

#### 3. Test a Single Number

```bash
dialer call 5551234
```

#### 4. Run a Scan

```bash
dialer scan --numbers numbers.txt --name "Office PBX"
```

### Option B: Using IAX2 VOIP (e.g., voip.ms)

#### 1. Configure IAX2 Credentials

Edit `config.yaml`:
```yaml
provider: iax2

iax2:
  host: chicago3.voip.ms    # Your nearest POP
  port: 4569
  username: "your_username"
  password: "your_password"
  caller_id: "+15551234567"
```

Or use command-line options:
```bash
dialer call 5551234 --provider iax2 \
  --iax2-host chicago3.voip.ms \
  --iax2-user myuser \
  --iax2-pass mypass
```

#### 2. Test a Single Number

```bash
dialer call 5551234 --provider iax2 -v
```

The `-v` flag shows verbose IAX2 protocol output for debugging.

#### 3. Run a Scan

```bash
dialer scan --numbers numbers.txt --provider iax2 --name "VOIP Scan"
```

### Scan Options (Both Providers)

**From a file (one number per line):**
```bash
dialer scan --numbers numbers.txt --name "Office PBX"
```

**From a range:**
```bash
dialer scan --range "5551000-5551050" --name "Test Range"
```

**With a blacklist:**
```bash
dialer scan --numbers numbers.txt --blacklist skip.txt --name "Filtered Scan"
```

### 5. View Results

```bash
# List all scans
dialer results

# View specific scan
dialer results 1

# Filter by type
dialer results 1 --type fax
```

### 6. Export Results

```bash
# CSV export
dialer export 1 -f csv -o results.csv

# JSON export
dialer export 1 -f json -o results.json

# HTML report with spectrograms
dialer report 1 -o report.html

# HTML report without embedded audio (smaller file)
dialer report 1 --no-audio --no-spectrograms -o report.html
```

### 7. Train ML Classifier

```bash
# Train from scan results (uses high-confidence classifications)
dialer train --scan-id 1

# Train from organized directory
dialer train --directory ./training_samples/

# Training directory structure:
# training_samples/
#   voice/
#     sample1.wav
#   fax/
#     sample1.wav
#   modem/
#     sample1.wav
```

## Commands

| Command | Description |
|---------|-------------|
| `detect` | Detect and test modem connectivity |
| `dialtone` | Test dial tone detection |
| `call` | Test dial a single number |
| `scan` | Scan a list or range of numbers |
| `results` | View scan results |
| `export` | Export results to CSV/JSON |
| `report` | Generate HTML report with spectrograms |
| `analyze` | Classify an existing audio file |
| `reanalyze` | Re-classify all audio files in a scan |
| `train` | Train ML classifier on real samples |
| `dialtest` | Debug dialing methods |

## Configuration

Copy `config.example.yaml` to `config.yaml` and customize:

```yaml
# Choose provider: "modem" or "iax2"
provider: modem

# USB Modem settings
modem:
  device: /dev/ttyACM0     # Or null for auto-detect
  baud_rate: 460800
  call_timeout: 45         # Seconds to wait for answer
  record_duration: 25      # Seconds of audio to capture
  dial_mode: tone          # "tone" (ATDT) or "pulse" (ATDP)

# IAX2 VOIP settings (for provider: iax2)
iax2:
  host: chicago3.voip.ms   # VOIP.MS POP or other IAX2 server
  port: 4569
  username: "your_username"
  password: "your_password"
  caller_id: "+15551234567"

audio:
  sample_rate: 8000
  output_dir: "./recordings"

scanner:
  call_delay: 3            # Seconds between calls
  max_retries: 2           # Retry failed calls

database:
  path: "./dialer.db"
```

**Note:** `config.yaml` is in `.gitignore` to protect credentials. Use `config.example.yaml` as a template.

## Line Type Detection

The scanner identifies lines by matching audio frequencies and patterns:

| Line Type | Description |
|-----------|-------------|
| `fax` | Fax machine (CNG 1100Hz / CED 2100Hz) |
| `modem` | Data modem (generic) |
| `modem_v21` | V.21 300 baud modem |
| `modem_v22` | V.22 1200 baud modem |
| `modem_v22bis` | V.22bis 2400 baud modem |
| `modem_v32` | V.32 9600 baud modem |
| `modem_v34` | V.34 28800+ baud modem |
| `carrier` | Generic data carrier |
| `voice` | Live human voice |
| `voicemail` | Voicemail system (with beep) |
| `answering_machine` | Answering machine |
| `ivr` | IVR/phone menu (DTMF detected) |
| `dial_tone` | Dial tone (350+440Hz) |
| `busy` | Busy signal (480+620Hz, 0.5s cadence) |
| `ringback` | Ringback tone (440+480Hz, 2s on/4s off) |
| `sit` | SIT tone (intercept message) |
| `dtmf` | DTMF touch tones |
| `silence` | Dead line / no signal |
| `unknown` | Unclassified |

## SIT Tone Categories

SIT (Special Information Tone) decoding identifies why calls fail:

| Pattern | Category | Meaning |
|---------|----------|---------|
| H-L-H | Vacant Code | Number not in service |
| H-H-L | No Circuit | All circuits busy |
| L-H-L | Reorder | Call cannot be completed |
| H-L-L | Operator Intercept | Operator assistance needed |

## Project Structure

```
dialer/
├── analysis/           # Audio analysis modules
│   ├── classifier.py   # Main classifier orchestration
│   ├── fft.py          # FFT frequency analysis
│   ├── signatures.py   # Tone signatures and patterns
│   ├── cadence.py      # Tone timing/cadence detection
│   ├── dtmf.py         # DTMF tone detection
│   ├── answering_machine.py  # AM/voicemail detection
│   ├── sit.py          # SIT tone decoder
│   └── ml_classifier.py  # ML-based classifier
├── core/               # Core functionality
│   ├── provider.py     # Abstract telephony provider interface
│   ├── modem.py        # USB modem AT command control
│   ├── modem_provider.py   # Modem provider wrapper
│   ├── iax2.py         # IAX2 protocol implementation
│   ├── iax2_provider.py    # IAX2 provider wrapper
│   ├── scanner.py      # Scan orchestration
│   └── audio.py        # Audio processing/DLE decode
├── storage/            # Data persistence
│   ├── database.py     # SQLite database
│   └── models.py       # Data models
├── reports/            # Report generation
│   └── html_report.py  # HTML reports with spectrograms
└── cli.py              # Command-line interface
```

## Modem Compatibility

**Recommended (voice mode):**
- US Robotics USR5637 (tested)
- Conexant CX93010-based modems
- Most "voice modems" or "TAM modems"

**Fallback (result codes only):**
- Any Hayes-compatible modem
- Classification limited to BUSY/NO CARRIER/CONNECT

To check if your modem supports voice mode:
```bash
dialer detect
# Look for "Voice Mode: Supported"
```

## IAX2 VOIP Setup

IAX2 (Inter-Asterisk eXchange v2) lets you dial without hardware using VOIP providers like voip.ms.

### Setting Up voip.ms

1. Create account at [voip.ms](https://voip.ms)
2. Add funds and purchase a DID (phone number)
3. Create a sub-account with IAX2 enabled:
   - Main Menu → Sub Accounts → Create Sub Account
   - Enable "IAX2" protocol
   - Note the username and set a password
4. Find your nearest POP server (e.g., `chicago3.voip.ms`, `atlanta1.voip.ms`)

### IAX2 Configuration

```yaml
provider: iax2

iax2:
  host: atlanta1.voip.ms    # Use POP nearest to your DID
  port: 4569
  username: "123456_subaccount"
  password: "your_password"
  caller_id: "+15551234567"  # Your DID
```

### IAX2 vs USB Modem

| Feature | USB Modem | IAX2 VOIP |
|---------|-----------|-----------|
| Hardware required | Yes | No |
| Setup complexity | Medium | Easy |
| Call quality | Depends on line | Good |
| Cost per call | Phone line rates | VOIP rates (~$0.01/min) |
| Connection speed | 10-15 seconds | 1-2 seconds |
| Geographic flexibility | Fixed location | Anywhere with internet |

## Logging

Logs are written to `dialer.log` in the current directory. Set log level in config or via environment:

```bash
DIALER_LOG_LEVEL=DEBUG dialer scan ...
```

## Number File Format

```
# Comments start with #
5551234
555-1235      # Dashes are stripped
+15551236     # International format OK
```

## Troubleshooting

### "No modem devices found"
- Check USB connection: `ls -la /dev/ttyACM* /dev/ttyUSB*`
- Check dmesg: `dmesg | tail -20`
- On Linux, add user to dialout group: `sudo usermod -a -G dialout $USER`

### "Permission denied" on device
```bash
sudo chmod 666 /dev/ttyACM0
# Or add to dialout group (permanent fix)
sudo usermod -a -G dialout $USER
```

### Voice mode not working
- Run `detect` to verify FCLASS=8 support
- Some modems need specific init strings
- Check modem manual for voice AT commands

### Poor classification accuracy
- Ensure good phone line quality
- Increase `record_duration` for longer samples
- Check `recordings/` folder for captured audio
- Use `analyze` command to test individual files
- Train ML classifier with real samples: `dialer train --scan-id <id>`

### Scan gets stuck between calls
- The scanner now disconnects/reconnects between calls for reliability
- Check `dialer.log` for detailed debug output

## Legal Notice

This tool is intended for authorized security testing, telecommunications research, and educational purposes. Ensure you have proper authorization before scanning phone numbers. Comply with all applicable laws and regulations in your jurisdiction.

## License

MIT License
