#pragma once
#include <array>

namespace pong {

// Court frame C: x∈[0,1] right, y∈[0,1] down, unit "cu"; velocity cu/s; dt s.
struct Vec2   { double x = 0, y = 0; };            // cu or cu/s
enum class Move { None, Up, Down };                // Up = visually up = y decreases
enum class Side { Left, Right };
struct Input  { Move left = Move::None; Move right = Move::None; };
struct Ball   { Vec2 pos, vel; double r = 0; };
struct Paddle { double x = 0, y = 0, h = 0, w = 0; };  // x fixed; y = center height
struct Score  { int left = 0, right = 0; };
enum class Phase   { Serving, Playing, GameOver };
enum class Outcome { None, LeftWin, RightWin, RallyCap };  // RallyCap: no winner — scored as a loss for BOTH
struct Snapshot {
  Ball ball; Paddle left, right; Score score;
  Phase   phase   = Phase::Serving;
  Outcome outcome = Outcome::None;
  int     rallyHits = 0;                           // paddle hits in the current rally
  int     rallyMax  = 0;                           // this game's rally cap (k::RallyMax default)
  double  missSum[2] = {0, 0};                      // cu; Σ|ball.y−paddle.y| at concessions ([0]=left,[1]=right)
  int     missN[2]   = {0, 0};
};

namespace k {   // tunable design defaults (units in comments)
  constexpr double Dt            = 1.0 / 60.0;  // s
  constexpr double PaddleH       = 0.16;        // cu
  constexpr double PaddleW       = 0.02;        // cu
  constexpr double LeftX         = 0.03;        // cu (paddle center x)
  constexpr double RightX        = 0.97;        // cu
  constexpr double BallR         = 0.012;       // cu
  constexpr double ServeSpeed    = 0.75;        // cu/s
  constexpr double ServeAngleDeg = 20.0;        // deg (seed==0 legacy serve)
  constexpr double ServeAngleMin = 10.0;        // deg (seeded serve jitter range)
  constexpr double ServeAngleMax = 35.0;        // deg
  constexpr double ServeHold     = 0.75;        // s
  constexpr double Speedup       = 1.03;        // - (per paddle hit)
  constexpr double BallSpeedMax  = 100.0;       // cu/s — "super high": court crossed in ~10 ms; unreturnable long before this
  constexpr double MaxSubDisp    = 0.008;       // cu — max ball travel per physics substep (< BallR, < PaddleW)
  constexpr double PaddleSpeed   = 1.2;         // cu/s — SAME for both sides (symmetric actuation)
  constexpr double MaxBounceDeg  = 60.0;        // deg
  constexpr int    WinScore      = 11;          // pts
  constexpr int    RallyMax      = 1000;        // paddle hits; reaching it ends the game, no winner
}

// Internal sim state (superset of Snapshot; never crosses to renderers).
struct State {
  Ball     ball;
  Paddle   left, right;
  Score    score;
  Phase    phase      = Phase::Serving;
  Outcome  outcome    = Outcome::None;
  double   serveTimer = 0;    // s remaining in Serving
  int      pointIndex = 0;    // total points played (serve vy sign parity)
  int      servingTo  = -1;   // -1 serve toward left, +1 toward right
  int      rallyHits  = 0;    // paddle hits this rally (drives RallyCap)
  int      rallyMax   = k::RallyMax;  // runtime cap: training configs may shorten (never lengthen for ladder play)
  double   speedup    = k::Speedup;   // per-hit speed multiplier: training may raise (canonical play = k::Speedup)
  double   serveHold  = k::ServeHold; // s; training may zero it (canonical play = k::ServeHold)
  int      winScore   = k::WinScore;  // first-to-N; training may shorten games (canonical = k::WinScore)
  unsigned seed       = 0;    // 0 = legacy fixed serve angle; else deterministic per-point jitter

  // Concession telemetry (passive — never feeds back into physics; drives loss-shaping fitness).
  // Σ|ball.y − paddle.y| [cu] and count at each conceded point, per side ([0]=left, [1]=right).
  double   missSum[2] = {0, 0};   // cu, court frame C (y down)
  int      missN[2]   = {0, 0};
};

State       initialState(unsigned seed = 0);
void        stepOnce(State&, const Input&);   // one fixed tick; pure, no RNG/IO (seeded jitter is hashed, not sampled)
Snapshot    view(const State&);

// Policy-facing state vector. Mirrored so every agent "plays left" regardless of side:
//   { ball.x, ball.y, ball.vx, ball.vy, own paddle y, opponent paddle y }
// Units: cu / cu/s, court frame C (y down). For Side::Right, x→1−x and vx→−vx.
std::array<double, 6> observation(const Snapshot&, Side);
std::array<double, 6> observation(const State&, Side);      // view-free overload for hot paths

class PongCore {                              // convenience wrapper for frontends
public:
  explicit PongCore(unsigned seed = 0, int rallyCap = k::RallyMax)
      : cap_(rallyCap), s_(initialState(seed)) { s_.rallyMax = cap_; }
  void     step(const Input& in)  { stepOnce(s_, in); }
  Snapshot snapshot() const       { return view(s_); }
  void     reset(unsigned seed = 0) {
    s_ = initialState(seed);
    s_.rallyMax = cap_; s_.speedup = spd_; s_.winScore = win_;
    s_.serveHold = hold_; s_.serveTimer = hold_;   // re-serve honors the configured hold
  }
  void     setRallyCap(int cap)    { cap_ = cap; s_.rallyMax = cap; }    // all persist across reset()
  void     setSpeedup(double g)    { spd_ = g; s_.speedup = g; }
  void     setServeHold(double s)  { hold_ = s; s_.serveHold = s; }
  void     setWinScore(int w)      { win_ = w; s_.winScore = w; }
private:
  int    cap_  = k::RallyMax;
  double spd_  = k::Speedup;
  double hold_ = k::ServeHold;
  int    win_  = k::WinScore;
  State  s_;
};

} // namespace pong
