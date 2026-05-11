#!/usr/bin/env python3
# =============================================================================
# QUADRUPED-K1 // K230 UART Interface
# =============================================================================
# Sends 12 joint angles to Arduino Mega over UART
# Receives toe microswitch contact state from Arduino on change
#
# PACKET FORMAT (K230 → Arduino):
#   "<a0,a1,a2,...,a11>\n"   angles in degrees (floats rounded to int)
#
# PACKET FORMAT (Arduino → K230, on switch change):
#   "C:L1,R1,L2,R2\n"       each value 0 or 1
#
# USAGE:
#   Standalone test:   python3 uart_interface.py
#   As module:         from uart_interface import ArduinoLink
# =============================================================================

import serial
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("uart")

# ---------------------------------------------------------------------------
# CONFIG — adjust port for your K230 OS
# ---------------------------------------------------------------------------
SERIAL_PORT  = "/dev/ttyS1"   # K230 UART1; change to /dev/ttyUSB0 for PC testing
BAUD_RATE    = 115200
SEND_RATE_HZ = 50             # joint angle transmit rate
NUM_SERVOS   = 12
ANGLE_MIN    = 10
ANGLE_MAX    = 170

# Leg / joint index reference
#  0  L1 Hip Rotation      3  R1 Hip Rotation      6  L2 Hip Rotation      9  R2 Hip Rotation
#  1  L1 Hip Abduction     4  R1 Hip Abduction      7  L2 Hip Abduction    10  R2 Hip Abduction
#  2  L1 Knee              5  R1 Knee               8  L2 Knee             11  R2 Knee
LEG_INDICES = {
    "L1": (0, 1, 2),
    "R1": (3, 4, 5),
    "L2": (6, 7, 8),
    "R2": (9, 10, 11),
}

# ---------------------------------------------------------------------------
# CONTACT STATE
# ---------------------------------------------------------------------------
@dataclass
class ContactState:
    L1: bool = False
    R1: bool = False
    L2: bool = False
    R2: bool = False
    timestamp: float = field(default_factory=time.time)

    def as_list(self) -> list[int]:
        return [int(self.L1), int(self.R1), int(self.L2), int(self.R2)]

    def __str__(self):
        def sym(v): return "●" if v else "○"
        return f"L1{sym(self.L1)} R1{sym(self.R1)} L2{sym(self.L2)} R2{sym(self.R2)}"


# ---------------------------------------------------------------------------
# ARDUINO LINK
# ---------------------------------------------------------------------------
class ArduinoLink:
    """
    Thread-safe UART link to the Arduino Mega.

    Usage:
        link = ArduinoLink()
        link.start()

        angles = [90] * 12
        link.send_angles(angles)

        contact = link.contact          # ContactState dataclass
        link.stop()
    """

    def __init__(self, port: str = SERIAL_PORT, baud: int = BAUD_RATE):
        self._port    = port
        self._baud    = baud
        self._ser: Optional[serial.Serial] = None

        self._angles       = [90] * NUM_SERVOS   # current target angles
        self._angles_lock  = threading.Lock()

        self.contact       = ContactState()
        self._contact_lock = threading.Lock()

        self._running   = False
        self._tx_thread = None
        self._rx_thread = None

        # Callbacks — set externally if needed
        # e.g. link.on_contact_change = my_fn
        self.on_contact_change = None   # fn(ContactState)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def start(self):
        """Open serial port and start TX/RX threads."""
        log.info(f"Opening {self._port} @ {self._baud} baud")
        self._ser = serial.Serial(
            port=self._port,
            baudrate=self._baud,
            timeout=0.1
        )
        time.sleep(0.1)   # let Arduino reset if DTR toggled
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        self._running = True
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True, name="uart-tx")
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True, name="uart-rx")
        self._tx_thread.start()
        self._rx_thread.start()
        log.info("ArduinoLink started")

    def stop(self):
        """Stop threads and close serial port."""
        self._running = False
        if self._tx_thread: self._tx_thread.join(timeout=1.0)
        if self._rx_thread: self._rx_thread.join(timeout=1.0)
        if self._ser and self._ser.is_open:
            self._ser.close()
        log.info("ArduinoLink stopped")

    def send_angles(self, angles: list[float]):
        """
        Set 12 joint angle targets (degrees).
        Values are clamped to [ANGLE_MIN, ANGLE_MAX].
        Thread-safe — call freely from RL inference loop.
        """
        if len(angles) != NUM_SERVOS:
            raise ValueError(f"Expected {NUM_SERVOS} angles, got {len(angles)}")
        clamped = [int(max(ANGLE_MIN, min(ANGLE_MAX, a))) for a in angles]
        with self._angles_lock:
            self._angles = clamped

    def get_contact(self) -> ContactState:
        """Return latest contact state. Thread-safe."""
        with self._contact_lock:
            return self.contact

    # ------------------------------------------------------------------
    # IK HELPERS — convert (t1, t2) per leg to servo angles
    # ------------------------------------------------------------------

    @staticmethod
    def ik_to_servo(t1_rad: float, t2_rad: float,
                    home_offset: float = 0.0,
                    mirror: bool = False) -> tuple[int, int]:
        """
        Convert IK joint angles to MG996R servo degrees.

        t1_rad      : hip rotation angle (radians)
        t2_rad      : knee angle (radians)
        home_offset : per-leg servo home trim (degrees)
        mirror      : True for right-side legs (flips hip direction)

        Returns (knee_servo_deg, hip_servo_deg)
        """
        import math
        knee_deg = 90 - math.degrees(t2_rad)
        hip_raw  = math.degrees(t1_rad)
        hip_deg  = home_offset + (-hip_raw if mirror else hip_raw)
        return (
            int(max(ANGLE_MIN, min(ANGLE_MAX, knee_deg))),
            int(max(ANGLE_MIN, min(ANGLE_MAX, hip_deg))),
        )

    def set_leg_ik(self, leg: str, t1_rad: float, t2_rad: float,
                   home_offset: float = 0.0):
        """
        Convenience: set one leg's angles from IK output.
        leg: "L1" | "R1" | "L2" | "R2"
        """
        mirror = leg.startswith("R")
        knee_deg, hip_deg = self.ik_to_servo(t1_rad, t2_rad, home_offset, mirror)
        hip_rot_idx, hip_abd_idx, knee_idx = LEG_INDICES[leg]

        with self._angles_lock:
            self._angles[hip_rot_idx] = hip_deg
            # hip_abd handled separately (abduction IK not shown here)
            self._angles[knee_idx]    = knee_deg

    # ------------------------------------------------------------------
    # TX THREAD — sends joint angles at SEND_RATE_HZ
    # ------------------------------------------------------------------

    def _tx_loop(self):
        interval = 1.0 / SEND_RATE_HZ
        while self._running:
            t_start = time.monotonic()

            with self._angles_lock:
                angles = list(self._angles)

            packet = self._build_packet(angles)
            try:
                self._ser.write(packet)
            except serial.SerialException as e:
                log.error(f"TX error: {e}")
                break

            elapsed = time.monotonic() - t_start
            sleep_t = interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    @staticmethod
    def _build_packet(angles: list[int]) -> bytes:
        """Build "<a0,a1,...,a11>\n" packet."""
        body = ",".join(str(a) for a in angles)
        return f"<{body}>\n".encode("ascii")

    # ------------------------------------------------------------------
    # RX THREAD — parses contact state from Arduino
    # ------------------------------------------------------------------

    def _rx_loop(self):
        buf = ""
        while self._running:
            try:
                raw = self._ser.readline()
            except serial.SerialException as e:
                log.error(f"RX error: {e}")
                break

            if not raw:
                continue

            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError:
                log.warning("RX decode error, skipping line")
                continue

            if line.startswith("C:"):
                self._parse_contact(line)
            elif line.startswith("[WARN]") or line.startswith("OK:"):
                log.debug(f"Arduino: {line}")
            elif line:
                log.debug(f"Arduino: {line}")

    def _parse_contact(self, line: str):
        """
        Parse "C:1,0,1,0" into ContactState.
        Order: L1, R1, L2, R2
        """
        try:
            values_str = line[2:]   # strip "C:"
            values = [int(v.strip()) for v in values_str.split(",")]
            if len(values) != 4:
                raise ValueError(f"Expected 4 contact values, got {len(values)}")

            new_state = ContactState(
                L1=bool(values[0]),
                R1=bool(values[1]),
                L2=bool(values[2]),
                R2=bool(values[3]),
                timestamp=time.time()
            )

            with self._contact_lock:
                self.contact = new_state

            log.debug(f"Contact: {new_state}")

            if self.on_contact_change:
                self.on_contact_change(new_state)

        except (ValueError, IndexError) as e:
            log.warning(f"Bad contact packet '{line}': {e}")


# ---------------------------------------------------------------------------
# STANDALONE TEST — runs without K230/Arduino connected
# Sends a slow sine-wave sweep on all knees to verify packet format,
# then prints any contact packets received.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math
    import sys

    # Allow port override from command line
    port = sys.argv[1] if len(sys.argv) > 1 else SERIAL_PORT

    link = ArduinoLink(port=port)

    # Print contact changes to console
    def on_contact(state: ContactState):
        print(f"  >> CONTACT CHANGE: {state}")
    link.on_contact_change = on_contact

    try:
        link.start()
    except serial.SerialException as e:
        log.error(f"Could not open serial port: {e}")
        log.error(f"Try: python3 uart_interface.py /dev/ttyUSB0")
        raise SystemExit(1)

    log.info("Running knee sweep test — Ctrl+C to stop")
    log.info(f"Packet example: {ArduinoLink._build_packet([90]*12)}")

    t = 0.0
    try:
        while True:
            # Sweep knees between 60° and 110° on a sine wave
            knee_angle = 85 + 25 * math.sin(t)
            angles = [90] * NUM_SERVOS
            for knee_idx in (2, 5, 8, 11):   # all knee channels
                angles[knee_idx] = int(knee_angle)

            link.send_angles(angles)

            contact = link.get_contact()
            print(
                f"\r  t={t:5.1f}s  knee={knee_angle:5.1f}°  contact={contact}   ",
                end="", flush=True
            )

            t += 1.0 / SEND_RATE_HZ
            time.sleep(1.0 / SEND_RATE_HZ)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        link.stop()
