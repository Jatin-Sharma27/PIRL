// ESP32 swing-up + balance controller for the motor-driven pendulum.
//
// Policy: a flash-resident int8 lookup table over (theta, omega), exported by
//         scripts/deploy_lut.py -> policy_lut.h.  Inference is a bilinear
//         lookup with no matrix multiplies.
// Sensing: quadrature encoder decoded in GPIO interrupts (x4), velocity from a
//          finite difference + EMA filter (same filter as pirl/deploy.py).
// Actuation: H-bridge (e.g. TB6612 / DRV8871 / L298N) driven by LEDC PWM + DIR pin.
//
// Arduino-ESP32 core 3.x.  Angle convention matches the simulator: theta = 0 is
// upright, so power the board with the pendulum hanging still (theta = pi).

#include <Arduino.h>
#include "policy_lut.h"

// ---------------- pins (change to your wiring) ----------------
static const int PIN_ENC_A = 34;
static const int PIN_ENC_B = 35;
static const int PIN_PWM   = 25;
static const int PIN_DIR   = 26;
static const int PIN_DIR2  = 27;   // second H-bridge input (set to -1 for PWM+DIR drivers)

// ---------------- parameters (must match the simulator) ----------------
static const int    ENC_CPR     = 2400;      // counts per rev after x4 decoding
static const float  CTRL_HZ     = 50.0f;     // sys.dt = 0.02 s
static const float  EMA_ALPHA   = 0.35f;
static const int    PWM_FREQ    = 20000;
static const int    PWM_BITS    = 8;
static const float  OMEGA_TRIP  = 25.0f;     // rad/s safety cut-off
static const float  U_SIGN      = 1.0f;      // flip if positive command spins the wrong way

// ---------------- encoder ----------------
static volatile int32_t enc_count = 0;
static volatile uint8_t enc_state = 0;
// state transition table for x4 quadrature decoding
static const int8_t QDEC[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};

void IRAM_ATTR enc_isr() {
  uint8_t s = (digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B);
  enc_state = ((enc_state << 2) | s) & 0x0F;
  enc_count += QDEC[enc_state];
}

// ---------------- helpers ----------------
static inline float wrap_pi(float a) {
  a = fmodf(a + PI, 2.0f * PI);
  if (a < 0) a += 2.0f * PI;
  return a - PI;
}

static float lut_policy(float theta, float omega) {
  float ti = (wrap_pi(theta) + PI) / (2.0f * PI) * LUT_N_TH;
  float wi = (omega + LUT_W_LIM) / (2.0f * LUT_W_LIM) * (LUT_N_W - 1);
  if (wi < 0) wi = 0;
  if (wi > LUT_N_W - 1 - 1e-4f) wi = LUT_N_W - 1 - 1e-4f;
  int t0 = (int)floorf(ti); float ft = ti - t0; t0 %= LUT_N_TH; int t1 = (t0 + 1) % LUT_N_TH;
  int w0 = (int)floorf(wi); float fw = wi - w0; int w1 = w0 + 1 < LUT_N_W ? w0 + 1 : LUT_N_W - 1;
  float u = (1 - ft) * (1 - fw) * POLICY_LUT[t0][w0] + ft * (1 - fw) * POLICY_LUT[t1][w0]
          + (1 - ft) * fw * POLICY_LUT[t0][w1] + ft * fw * POLICY_LUT[t1][w1];
  return u / 127.0f;
}

static void motor_write(float u) {
  u = constrain(u * U_SIGN, -1.0f, 1.0f);
  int duty = (int)roundf(fabsf(u) * ((1 << PWM_BITS) - 1));
  if (PIN_DIR2 >= 0) {
    digitalWrite(PIN_DIR, u >= 0 ? HIGH : LOW);
    digitalWrite(PIN_DIR2, u >= 0 ? LOW : HIGH);
  } else {
    digitalWrite(PIN_DIR, u >= 0 ? HIGH : LOW);
  }
  ledcWrite(PIN_PWM, duty);
}

// ---------------- main loop ----------------
static float omega_hat = 0.0f;
static int32_t last_count = 0;
static float theta0 = PI;  // hanging at boot
static bool tripped = false;
static uint32_t next_us;
static const uint32_t PERIOD_US = (uint32_t)(1e6f / CTRL_HZ);

void setup() {
  Serial.begin(921600);
  pinMode(PIN_ENC_A, INPUT_PULLUP);  // GPIO34/35 have no pull-ups: use external 10k
  pinMode(PIN_ENC_B, INPUT_PULLUP);
  pinMode(PIN_DIR, OUTPUT);
  if (PIN_DIR2 >= 0) pinMode(PIN_DIR2, OUTPUT);
  ledcAttach(PIN_PWM, PWM_FREQ, PWM_BITS);
  motor_write(0);
  enc_state = (digitalRead(PIN_ENC_A) << 1) | digitalRead(PIN_ENC_B);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), enc_isr, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_B), enc_isr, CHANGE);
  delay(1000);
  noInterrupts(); last_count = enc_count; interrupts();
  next_us = micros() + PERIOD_US;
  Serial.println("t_ms,theta,omega_hat,u");
}

void loop() {
  if ((int32_t)(micros() - next_us) < 0) return;
  next_us += PERIOD_US;

  noInterrupts(); int32_t c = enc_count; interrupts();
  const float res = 2.0f * PI / ENC_CPR;
  float theta = wrap_pi(theta0 + c * res);
  float dtheta = (c - last_count) * res;
  last_count = c;
  omega_hat = EMA_ALPHA * (dtheta * CTRL_HZ) + (1.0f - EMA_ALPHA) * omega_hat;

  if (fabsf(omega_hat) > OMEGA_TRIP) tripped = true;
  float u = tripped ? 0.0f : lut_policy(theta, omega_hat);
  motor_write(u);

  static uint8_t div = 0;
  if (++div >= 2) {  // log at 25 Hz
    div = 0;
    Serial.printf("%lu,%.4f,%.3f,%.3f\n", (unsigned long)millis(), theta, omega_hat, u);
  }
}
