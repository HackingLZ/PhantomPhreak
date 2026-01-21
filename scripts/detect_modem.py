#!/usr/bin/env python3
"""
Utility script to detect and test USB modem connectivity.

Usage:
    python scripts/detect_modem.py [device]

Example:
    python scripts/detect_modem.py
    python scripts/detect_modem.py /dev/ttyACM0
"""

import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

try:
    import serial
except ImportError:
    print("Error: pyserial not installed")
    print("Run: pip install pyserial")
    sys.exit(1)


def find_modem_devices():
    """Find potential modem devices."""
    import glob

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


def test_at_command(ser, command, description=""):
    """Send AT command and print response."""
    print(f"\n{'=' * 50}")
    print(f"Command: {command}")
    if description:
        print(f"Purpose: {description}")
    print("-" * 50)

    ser.reset_input_buffer()
    ser.write(f"{command}\r".encode())
    time.sleep(0.3)

    response = b""
    while ser.in_waiting:
        response += ser.read(ser.in_waiting)
        time.sleep(0.1)

    decoded = response.decode('ascii', errors='ignore').strip()
    print(f"Response:\n{decoded}")

    return decoded


def main():
    device = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("  USB Modem Detection and Testing")
    print("=" * 60)

    # Find devices
    print("\nSearching for modem devices...")
    devices = find_modem_devices()

    if not devices:
        print("\nNo modem devices found!")
        print("\nTroubleshooting:")
        print("  1. Check that the modem is plugged in")
        print("  2. Check USB connection")
        print("  3. Check dmesg output: dmesg | tail -20")
        print("  4. On Linux, you may need to be in 'dialout' group")
        sys.exit(1)

    print(f"\nFound {len(devices)} device(s):")
    for d in devices:
        print(f"  - {d}")

    # Select device
    if device:
        if device not in devices:
            print(f"\nWarning: {device} not in detected devices, trying anyway...")
        test_device = device
    else:
        test_device = devices[0]

    print(f"\nTesting: {test_device}")

    # Open connection
    try:
        ser = serial.Serial(
            port=test_device,
            baudrate=460800,
            timeout=2,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
        )
        time.sleep(0.5)
        print("Connection opened successfully")
    except serial.SerialException as e:
        print(f"\nError opening {test_device}: {e}")
        print("\nTroubleshooting:")
        print("  - Check permissions: ls -la", test_device)
        print("  - On Linux: sudo usermod -a -G dialout $USER")
        print("  - Try a different baud rate")
        sys.exit(1)

    # Test basic AT commands
    try:
        # Reset
        test_at_command(ser, "ATZ", "Reset modem to defaults")

        # Get manufacturer
        test_at_command(ser, "AT+GMI", "Get manufacturer")

        # Get model
        test_at_command(ser, "AT+GMM", "Get model")

        # Get IMEI (for cellular modems)
        test_at_command(ser, "AT+GSN", "Get serial number/IMEI")

        # Query supported FCLASS modes
        response = test_at_command(
            ser,
            "AT+FCLASS=?",
            "Query supported modes (0=data, 1=fax, 8=voice)"
        )

        # Parse voice support
        voice_supported = "8" in response
        print("\n" + "=" * 50)
        print("CAPABILITY SUMMARY")
        print("=" * 50)

        if "0" in response:
            print("  [X] Data mode (FCLASS=0)")
        if "1" in response:
            print("  [X] Fax mode (FCLASS=1)")
        if "8" in response:
            print("  [X] Voice mode (FCLASS=8) <-- Required for audio capture")
        else:
            print("  [ ] Voice mode (FCLASS=8) <-- NOT SUPPORTED")

        # If voice supported, get more details
        if voice_supported:
            print("\n" + "-" * 50)
            print("Voice Mode Details:")

            # Query voice sample modes
            test_at_command(ser, "AT+VSM=?", "Supported voice compression modes")

            # Query voice line select options
            test_at_command(ser, "AT+VLS=?", "Voice line select options")

        # Test dialing capability
        print("\n" + "=" * 50)
        print("DIALING TEST (will not actually dial)")
        print("=" * 50)

        test_at_command(ser, "ATX4", "Enable all result codes")
        test_at_command(ser, "ATS7=45", "Set wait for carrier to 45 seconds")

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)

        if voice_supported:
            print("\n[SUCCESS] Modem supports voice mode!")
            print("Full audio-based scanning will work.")
            print("\nNext steps:")
            print("  1. Install requirements: pip install -r requirements.txt")
            print("  2. Configure config.yaml with device path")
            print("  3. Run: python -m dialer scan --numbers numbers.txt")
        else:
            print("\n[WARNING] Voice mode not supported")
            print("Scanner will work in fallback mode (result codes only).")
            print("Classification will be limited to BUSY/NO CARRIER/etc.")
            print("\nNext steps:")
            print("  1. Consider getting a voice-capable modem")
            print("  2. Or use fallback mode for basic scanning")

    except Exception as e:
        print(f"\nError during testing: {e}")

    finally:
        ser.close()
        print("\nConnection closed.")


if __name__ == "__main__":
    main()
