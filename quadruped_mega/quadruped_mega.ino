// =============================================================================
// QUADRUPED-K1 // Arduino Mega 2560 Firmware
// =============================================================================
// Receives 12 joint angles from K230 over Serial1 (UART)
// Drives 12 MG996R servos via Servo.h (hardware PWM)
// Reads 4 toe microswitches, reports state to K230 on change
//
// PACKET FORMAT (K230 → Arduino, ASCII/CSV):
//   "<a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11>\n"
//   angles in degrees (0–180), comma-separated, wrapped in < >
//   example: "<90,90,90,85,90,90,90,90,90,95,90,90>\n"
//
// PACKET FORMAT (Arduino → K230, on switch change):
//   "C:<L1>,<R1>,<L2>,<R2>\n"
//   each value is 0 (airborne) or 1 (contact)
//   example: "C:1,0,1,0\n"
//
// SERVO CHANNEL MAP (Mega PWM pins):
//   CH  0 → Pin 2  → L1 Hip Rotation
//   CH  1 → Pin 3  → L1 Hip Abduction
//   CH  2 → Pin 4  → L1 Knee
//   CH  3 → Pin 5  → R1 Hip Rotation
//   CH  4 → Pin 6  → R1 Hip Abduction
//   CH  5 → Pin 7  → R1 Knee
//   CH  6 → Pin 8  → L2 Hip Rotation
//   CH  7 → Pin 9  → L2 Hip Abduction
//   CH  8 → Pin 10 → L2 Knee
//   CH  9 → Pin 11 → R2 Hip Rotation
//   CH 10 → Pin 12 → R2 Hip Abduction
//   CH 11 → Pin 13 → R2 Knee
//
// MICROSWITCH PINS (INPUT_PULLUP, LOW = contact):
//   L1 toe → Pin 22
//   R1 toe → Pin 23
//   L2 toe → Pin 24
//   R2 toe → Pin 25
//
// UART:
//   Serial1 (pins 18/19) ← K230 TX/RX  @ 115200 baud
//   Serial0 (USB)        ← debug monitor @ 115200 baud
// =============================================================================

#include <Servo.h>

// ---------------------------------------------------------------------------
// CONFIG
// ---------------------------------------------------------------------------
#define NUM_SERVOS        12
#define NUM_SWITCHES      4
#define UART_BAUD         115200
#define PACKET_TIMEOUT_MS 100     // reset parser if no '\n' within this window

// Servo angle limits (mechanical safety clamp)
#define SERVO_MIN_DEG     10
#define SERVO_MAX_DEG     170

// Startup pose: all knees slightly bent, hips centred
const int HOME_ANGLES[NUM_SERVOS] = {
  90, 90, 70,   // L1: hip-rot, hip-abd, knee
  90, 90, 70,   // R1: hip-rot, hip-abd, knee
  90, 90, 70,   // L2: hip-rot, hip-abd, knee
  90, 90, 70    // R2: hip-rot, hip-abd, knee
};

// ---------------------------------------------------------------------------
// PIN ASSIGNMENTS
// ---------------------------------------------------------------------------
const uint8_t SERVO_PINS[NUM_SERVOS] = {
  2, 3, 4,      // L1
  5, 6, 7,      // R1
  8, 9, 10,     // L2
  11, 12, 13    // R2
};

const uint8_t SWITCH_PINS[NUM_SWITCHES] = {
  22, 23, 24, 25   // L1, R1, L2, R2 toes
};

const char* SWITCH_LABELS[NUM_SWITCHES] = {
  "L1", "R1", "L2", "R2"
};

// ---------------------------------------------------------------------------
// GLOBALS
// ---------------------------------------------------------------------------
Servo servos[NUM_SERVOS];
int   targetAngles[NUM_SERVOS];

// Parser state
char    rxBuf[128];
uint8_t rxIdx       = 0;
bool    inPacket    = false;
uint32_t lastByteMs = 0;

// Switch state tracking
uint8_t switchState[NUM_SWITCHES];
uint8_t lastSwitchState[NUM_SWITCHES];

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
void setup() {
  // Debug serial (USB)
  Serial.begin(UART_BAUD);
  Serial.println(F("QUADRUPED-K1 // Mega 2560 firmware boot"));

  // K230 serial
  Serial1.begin(UART_BAUD);
  Serial.println(F("Serial1 ready @ 115200 (K230 link)"));

  // Attach servos and move to home
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    targetAngles[i] = HOME_ANGLES[i];
    servos[i].write(HOME_ANGLES[i]);
  }
  Serial.println(F("Servos attached — moving to home pose"));

  // Microswitches with internal pullup (LOW = pressed/contact)
  for (int i = 0; i < NUM_SWITCHES; i++) {
    pinMode(SWITCH_PINS[i], INPUT_PULLUP);
    switchState[i]     = digitalRead(SWITCH_PINS[i]);
    lastSwitchState[i] = switchState[i];
  }
  Serial.println(F("Microswitches ready"));
  Serial.println(F("Waiting for K230 packets..."));
}

// ---------------------------------------------------------------------------
// MAIN LOOP
// ---------------------------------------------------------------------------
void loop() {
  parseUART();
  readSwitches();
}

// ---------------------------------------------------------------------------
// UART PARSER
// Expects: "<a0,a1,...,a11>\n"
// Tolerant of partial packets; resets on timeout.
// ---------------------------------------------------------------------------
void parseUART() {
  // Timeout guard: reset parser if mid-packet and nothing arrives
  if (inPacket && (millis() - lastByteMs > PACKET_TIMEOUT_MS)) {
    Serial.println(F("[WARN] Packet timeout, resetting parser"));
    rxIdx    = 0;
    inPacket = false;
  }

  while (Serial1.available()) {
    char c = Serial1.read();
    lastByteMs = millis();

    if (c == '<') {
      // Start of packet
      rxIdx    = 0;
      inPacket = true;
      continue;
    }

    if (!inPacket) continue;

    if (c == '>') {
      // End of payload — null-terminate and parse
      rxBuf[rxIdx] = '\0';
      inPacket = false;
      processPacket(rxBuf);
      rxIdx = 0;
      continue;
    }

    if (c == '\n' || c == '\r') continue;  // ignore line endings inside packet

    // Guard against buffer overrun
    if (rxIdx < sizeof(rxBuf) - 1) {
      rxBuf[rxIdx++] = c;
    } else {
      // Overrun — discard and reset
      Serial.println(F("[WARN] RX buffer overrun, resetting parser"));
      rxIdx    = 0;
      inPacket = false;
    }
  }
}

// ---------------------------------------------------------------------------
// PACKET PROCESSOR
// Parses comma-separated angle list, clamps, and writes to servos
// ---------------------------------------------------------------------------
void processPacket(const char* payload) {
  int angles[NUM_SERVOS];
  int count = 0;

  char buf[128];
  strncpy(buf, payload, sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';

  char* token = strtok(buf, ",");
  while (token != NULL && count < NUM_SERVOS) {
    int val = atoi(token);
    // Clamp to safe range
    val = constrain(val, SERVO_MIN_DEG, SERVO_MAX_DEG);
    angles[count++] = val;
    token = strtok(NULL, ",");
  }

  if (count != NUM_SERVOS) {
    Serial.print(F("[WARN] Expected "));
    Serial.print(NUM_SERVOS);
    Serial.print(F(" angles, got "));
    Serial.println(count);
    return;
  }

  // Write to servos
  for (int i = 0; i < NUM_SERVOS; i++) {
    targetAngles[i] = angles[i];
    servos[i].write(angles[i]);
  }

  // Echo to debug serial (can comment out once stable)
  Serial.print(F("OK: "));
  for (int i = 0; i < NUM_SERVOS; i++) {
    Serial.print(angles[i]);
    if (i < NUM_SERVOS - 1) Serial.print(',');
  }
  Serial.println();
}

// ---------------------------------------------------------------------------
// MICROSWITCH READER
// Sends "C:L1,R1,L2,R2\n" to K230 only when state changes
// LOW pin = contact (INPUT_PULLUP), reported as 1
// HIGH pin = airborne,             reported as 0
// ---------------------------------------------------------------------------
void readSwitches() {
  bool changed = false;

  for (int i = 0; i < NUM_SWITCHES; i++) {
    switchState[i] = (digitalRead(SWITCH_PINS[i]) == LOW) ? 1 : 0;
    if (switchState[i] != lastSwitchState[i]) {
      changed = true;
    }
  }

  if (changed) {
    // Send to K230
    Serial1.print(F("C:"));
    for (int i = 0; i < NUM_SWITCHES; i++) {
      Serial1.print(switchState[i]);
      if (i < NUM_SWITCHES - 1) Serial1.print(',');
    }
    Serial1.print('\n');

    // Mirror to debug
    Serial.print(F("CONTACT: "));
    for (int i = 0; i < NUM_SWITCHES; i++) {
      Serial.print(SWITCH_LABELS[i]);
      Serial.print('=');
      Serial.print(switchState[i]);
      if (i < NUM_SWITCHES - 1) Serial.print(' ');
    }
    Serial.println();

    // Update last state
    for (int i = 0; i < NUM_SWITCHES; i++) {
      lastSwitchState[i] = switchState[i];
    }
  }
}
