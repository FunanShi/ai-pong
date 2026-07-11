#include "pong_core.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pong {
namespace {
constexpr double kPi = 3.14159265358979323846;
double clampd(double v, double lo, double hi){ return std::max(lo, std::min(hi, v)); }
double deg2rad(double d){ return d * kPi / 180.0; }
double speed(const Vec2& v){ return std::sqrt(v.x*v.x + v.y*v.y); }

// Deterministic per-point hash → [0,1). splitmix64 finalizer; no global RNG state,
// so stepOnce stays pure and a (seed, pointIndex) pair fully determines each serve.
double hash01(unsigned seed, int pointIndex){
  uint64_t z = (uint64_t(seed) << 32) ^ uint64_t(uint32_t(pointIndex));
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z =  z ^ (z >> 31);
  return double(z >> 11) * (1.0 / 9007199254740992.0);   // top 53 bits → [0,1)
}

void serve(State& s){
  s.ball.pos = {0.5, 0.5};
  s.ball.r   = k::BallR;
  double angDeg = (s.seed == 0)
      ? k::ServeAngleDeg
      : k::ServeAngleMin + (k::ServeAngleMax - k::ServeAngleMin) * hash01(s.seed, s.pointIndex);
  double ang = deg2rad(angDeg);
  double sx  = (s.servingTo >= 0) ? +1.0 : -1.0;
  double sy  = (s.pointIndex % 2 == 0) ? +1.0 : -1.0;
  s.ball.vel = { k::ServeSpeed*std::cos(ang)*sx, k::ServeSpeed*std::sin(ang)*sy };
  s.phase      = Phase::Serving;
  s.serveTimer = s.serveHold;
  s.rallyHits  = 0;
}

void movePaddle(Paddle& p, Move m){
  double dy   = (m==Move::Up ? -1.0 : m==Move::Down ? +1.0 : 0.0) * k::PaddleSpeed * k::Dt;
  double half = p.h * 0.5;
  p.y = clampd(p.y + dy, half, 1.0 - half);
}
void collideWalls(State& s){
  if (s.ball.pos.y - s.ball.r < 0.0){ s.ball.pos.y = s.ball.r;       s.ball.vel.y = -s.ball.vel.y; }
  if (s.ball.pos.y + s.ball.r > 1.0){ s.ball.pos.y = 1.0 - s.ball.r; s.ball.vel.y = -s.ball.vel.y; }
}
void bounceOffPaddle(State& s, const Paddle& p, double front, double away){
  double u  = clampd((s.ball.pos.y - p.y) / (p.h*0.5), -1.0, 1.0);
  double th = deg2rad(k::MaxBounceDeg) * u;
  double sp = std::min(speed(s.ball.vel) * s.speedup, k::BallSpeedMax);
  s.ball.vel   = { away*sp*std::cos(th), sp*std::sin(th) };
  s.ball.pos.x = front + away*(s.ball.r + 1e-4);        // nudge just off the struck face
  s.rallyHits++;
  if (s.rallyHits >= s.rallyMax){                        // marathon rally: nobody wins
    s.phase   = Phase::GameOver;
    s.outcome = Outcome::RallyCap;
  }
}
void collidePaddles(State& s){
  const Ball& b = s.ball;
  if (b.vel.x < 0){                                     // toward left paddle
    double front = s.left.x + s.left.w*0.5, back = s.left.x - s.left.w*0.5;
    if (b.pos.x - b.r <= front && b.pos.x + b.r >= back &&
        std::abs(b.pos.y - s.left.y) <= s.left.h*0.5 + b.r)
      bounceOffPaddle(s, s.left, front, +1.0);
  } else if (b.vel.x > 0){                              // toward right paddle
    double front = s.right.x - s.right.w*0.5, back = s.right.x + s.right.w*0.5;
    if (b.pos.x + b.r >= front && b.pos.x - b.r <= back &&
        std::abs(b.pos.y - s.right.y) <= s.right.h*0.5 + b.r)
      bounceOffPaddle(s, s.right, front, -1.0);
  }
}
bool scoreIfOut(State& s){
  if (s.ball.pos.x < 0.0){                                // ball past the left edge → left conceded
    s.missSum[0] += std::abs(s.ball.pos.y - s.left.y);    // |Δy| at the concession [cu]
    s.missN[0]++;
    s.score.right++; s.servingTo = -1; return true;       // serve toward left
  }
  if (s.ball.pos.x > 1.0){                                // ball past the right edge → right conceded
    s.missSum[1] += std::abs(s.ball.pos.y - s.right.y);
    s.missN[1]++;
    s.score.left++;  s.servingTo = +1; return true;       // serve toward right
  }
  return false;
}

// --- conservative advancement (perf; DECISIONS.md #11) -----------------------------------
// Between collisions the flight is exactly linear, so fine substeps buy nothing except
// contact DETECTION. We fine-step (≤ MaxSubDisp) only inside a band around collidable
// surfaces, and leap directly to the nearest band edge otherwise. Contact resolution is
// unchanged (same overlap checks, same ≤ MaxSubDisp quantization near surfaces); only the
// partition of the flight differs, so trajectories match the fixed-step scheme up to
// floating-point summation order. Within-version determinism is exact.
constexpr double kFineBand = 2.0 * k::MaxSubDisp;   // cu; ≥ one fine step of headroom

// Largest time the ball can fly from the current state without entering any approachable
// surface's fine band. Returns ≤ 0 when already inside a band (caller fine-steps).
double safeFlightTime(const State& s){
  const Ball& b = s.ball;
  double t = k::Dt;                                  // caller clamps to remaining tick time
  auto consider = [&](double dist, double speedTowards){
    double tt = dist / speedTowards;                 // dist < 0 → inside band → tt ≤ 0
    if (tt < t) t = tt;
  };
  if (b.vel.y < 0) consider(b.pos.y - b.r - kFineBand, -b.vel.y);          // top wall
  if (b.vel.y > 0) consider(1.0 - b.r - kFineBand - b.pos.y, b.vel.y);     // bottom wall
  if (b.vel.x < 0){
    // left paddle slab band [back − r − band, front + r + band]; ignore once fully passed
    double front = s.left.x + s.left.w * 0.5, back = s.left.x - s.left.w * 0.5;
    if (b.pos.x >= back - b.r - kFineBand)
      consider(b.pos.x - (front + b.r + kFineBand), -b.vel.x);
    consider(b.pos.x - kFineBand, -b.vel.x);                               // score plane x=0
  }
  if (b.vel.x > 0){
    double front = s.right.x - s.right.w * 0.5, back = s.right.x + s.right.w * 0.5;
    if (b.pos.x <= back + b.r + kFineBand)
      consider((front - b.r - kFineBand) - b.pos.x, b.vel.x);
    consider(1.0 - kFineBand - b.pos.x, b.vel.x);                          // score plane x=1
  }
  return t;
}
} // namespace

State initialState(unsigned seed){
  State s;
  s.seed  = seed;
  s.left  = { k::LeftX,  0.5, k::PaddleH, k::PaddleW };
  s.right = { k::RightX, 0.5, k::PaddleH, k::PaddleW };
  s.pointIndex = 0;
  s.servingTo  = -1;              // first serve toward the left player
  serve(s);
  return s;
}

void stepOnce(State& s, const Input& in){
  if (s.phase == Phase::GameOver) return;
  movePaddle(s.left,  in.left);
  movePaddle(s.right, in.right);
  if (s.phase == Phase::Serving){
    s.serveTimer -= k::Dt;
    if (s.serveTimer <= 0.0) s.phase = Phase::Playing;
    return;
  }
  // Ball integration with substeps for collision accuracy: at high speed the per-tick
  // displacement far exceeds paddle width (100 cu/s × 1/60 s ≈ 1.67 cu), so naive Euler
  // tunnels. Conservative advancement keeps the fine ≤ MaxSubDisp steps only near
  // surfaces and leaps across open court (see safeFlightTime above). A mid-tick bounce
  // raises speed ×1.03; the fine step stays under BallR, so contact still can't be skipped.
  double remaining = k::Dt;
  while (remaining > 1e-12 && s.phase == Phase::Playing){
    double v = speed(s.ball.vel);
    if (v <= 0.0) break;                            // defensive; serve always sets v > 0
    double hFine = k::MaxSubDisp / v;
    double tSafe = safeFlightTime(s);
    double h = (tSafe > hFine) ? tSafe : hFine;     // never below a fine step: progress guaranteed
    if (h > remaining) h = remaining;
    s.ball.pos.x += s.ball.vel.x * h;
    s.ball.pos.y += s.ball.vel.y * h;
    collideWalls(s);
    collidePaddles(s);                    // may end the game via RallyCap
    if (s.phase != Phase::Playing) break; // RallyCap fired mid-tick
    if (scoreIfOut(s)){
      s.pointIndex++;
      if (s.score.left >= s.winScore || s.score.right >= s.winScore){
        s.phase   = Phase::GameOver;
        s.outcome = (s.score.left > s.score.right) ? Outcome::LeftWin : Outcome::RightWin;
      } else {
        serve(s);
      }
      break;
    }
    remaining -= h;
  }
}

Snapshot view(const State& s){
  Snapshot v;
  v.ball = s.ball; v.left = s.left; v.right = s.right; v.score = s.score;
  v.phase = s.phase; v.outcome = s.outcome; v.rallyHits = s.rallyHits; v.rallyMax = s.rallyMax;
  v.missSum[0] = s.missSum[0]; v.missSum[1] = s.missSum[1];
  v.missN[0]   = s.missN[0];   v.missN[1]   = s.missN[1];
  return v;
}

std::array<double, 6> observation(const Snapshot& s, Side side){
  if (side == Side::Left)
    return { s.ball.pos.x, s.ball.pos.y, s.ball.vel.x, s.ball.vel.y, s.left.y, s.right.y };
  return { 1.0 - s.ball.pos.x, s.ball.pos.y, -s.ball.vel.x, s.ball.vel.y, s.right.y, s.left.y };
}

std::array<double, 6> observation(const State& s, Side side){   // view-free overload (hot paths)
  if (side == Side::Left)
    return { s.ball.pos.x, s.ball.pos.y, s.ball.vel.x, s.ball.vel.y, s.left.y, s.right.y };
  return { 1.0 - s.ball.pos.x, s.ball.pos.y, -s.ball.vel.x, s.ball.vel.y, s.right.y, s.left.y };
}
} // namespace pong
