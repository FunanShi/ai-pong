#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "test_util.hpp"
#include <cmath>
#include <vector>
using namespace pong;
using testutil::playing;
using testutil::spd;

// ---------------- game flow ----------------

TEST_CASE("initial state: serving, centered, clean outcome"){
  State s = initialState();
  CHECK(s.phase == Phase::Serving);
  CHECK(s.outcome == Outcome::None);
  CHECK(s.rallyHits == 0);
  CHECK(s.ball.pos.x == doctest::Approx(0.5));
  CHECK(s.ball.pos.y == doctest::Approx(0.5));
  CHECK(s.left.y  == doctest::Approx(0.5));
  CHECK(s.right.y == doctest::Approx(0.5));
  CHECK(spd(s.ball.vel) == doctest::Approx(k::ServeSpeed));
}

TEST_CASE("serve hold elapses into Playing"){
  State s = initialState();
  int ticks = (int)std::ceil(k::ServeHold / k::Dt) + 1;
  for (int i=0;i<ticks;i++) stepOnce(s, {});
  CHECK(s.phase == Phase::Playing);
}

TEST_CASE("left paddle: Up decreases y and clamps at the top"){
  State s = initialState();
  double y0 = s.left.y;
  stepOnce(s, {Move::Up, Move::None});
  CHECK(s.left.y < y0);
  for (int i=0;i<10000;i++) stepOnce(s, {Move::Up, Move::None});
  CHECK(s.left.y == doctest::Approx(s.left.h*0.5));
}
TEST_CASE("right paddle obeys in.right with the same speed as the left (symmetric actuation)"){
  State s = initialState();
  double l0 = s.left.y, r0 = s.right.y;
  stepOnce(s, {Move::Up, Move::Down});
  CHECK(s.left.y  == doctest::Approx(l0 - k::PaddleSpeed*k::Dt));
  CHECK(s.right.y == doctest::Approx(r0 + k::PaddleSpeed*k::Dt));
}

// ---------------- ball physics ----------------

TEST_CASE("ball reflects off the top wall, |vy| preserved, stays inside"){
  State s = playing();
  s.ball ={{0.5, k::BallR+0.001},{0.0,-0.5},k::BallR};   // up into the top wall
  stepOnce(s, {});
  CHECK(s.ball.vel.y == doctest::Approx(+0.5));
  CHECK(s.ball.pos.y >= k::BallR - 1e-9);
}
TEST_CASE("ball reflects off the bottom wall"){
  State s = playing();
  s.ball ={{0.5, 1.0-k::BallR-0.001},{0.0,+0.5},k::BallR};
  stepOnce(s, {});
  CHECK(s.ball.vel.y == doctest::Approx(-0.5));
  CHECK(s.ball.pos.y <= 1.0 - k::BallR + 1e-9);
}

TEST_CASE("center hit returns the ball horizontally and speeds it up"){
  State s = playing();
  double sp0=0.8, front=k::RightX-k::PaddleW*0.5;
  s.ball={{ front-k::BallR-1e-4, 0.5 },{ +sp0, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(s.ball.vel.x < 0);
  CHECK(std::abs(s.ball.vel.y) < 1e-6);                 // u=0 → horizontal
  CHECK(spd(s.ball.vel) == doctest::Approx(sp0*k::Speedup));
  CHECK(s.rallyHits == 1);
}
TEST_CASE("low-edge hit deflects the ball downward"){
  State s = playing();
  double front=k::RightX-k::PaddleW*0.5;
  s.ball={{ front-k::BallR-1e-4, 0.5+k::PaddleH*0.5 },{ +0.8, 0.0 }, k::BallR};  // hit low (u>0)
  stepOnce(s, {});
  CHECK(s.ball.vel.x < 0);
  CHECK(s.ball.vel.y > 0);
}

TEST_CASE("speed multiplies by Speedup each hit (uncapped region grows geometrically)"){
  State s = playing();
  double sp=0.5, front=k::RightX-k::PaddleW*0.5;
  for (int i=0;i<80;i++){
    s = playing();
    s.ball={{ front-k::BallR-1e-4, 0.5 },{ +sp, 0.0 }, k::BallR};
    stepOnce(s, {});
    double out = spd(s.ball.vel);
    CHECK(out == doctest::Approx(sp*k::Speedup));       // 0.5·1.03^80 ≈ 5.3 — far below the 100 cap
    sp = out;
  }
}
TEST_CASE("BallSpeedMax caps the rally: 99 cu/s hit pins to exactly 100, multi-bounce within one tick"){
  State s = playing();
  double front=k::RightX-k::PaddleW*0.5;
  s.ball={{ front-k::BallR-1e-4, 0.5 },{ +99.0, 0.0 }, k::BallR};
  stepOnce(s, {});
  // In one 1/60 s tick a 99 cu/s ball travels ~1.65 cu: it bounces off the right paddle
  // (99·1.03 → capped 100), crosses the court, and bounces off the left paddle too.
  CHECK(spd(s.ball.vel) == doctest::Approx(k::BallSpeedMax));
  CHECK(s.rallyHits >= 2);
  CHECK(s.phase == Phase::Playing);
}

TEST_CASE("no tunneling: a 50 cu/s ball bounces off a paddle it would previously have skipped"){
  State s = playing();
  s.ball ={{0.5, 0.5},{+50.0, 0.0}, k::BallR};          // one legacy Euler tick would jump 0.83 cu past the paddle
  stepOnce(s, {});
  CHECK(s.ball.vel.x < 0);                              // bounced, not tunneled
  CHECK(s.score.left == 0);
  CHECK(s.score.right == 0);
  CHECK(s.rallyHits >= 1);
}
TEST_CASE("no phantom walls: a 50 cu/s ball that misses the paddle still scores"){
  State s = playing();
  s.right.y = 0.1;                                      // paddle far from the ball's path
  s.ball ={{0.5, 0.6},{+50.0, 0.0}, k::BallR};
  stepOnce(s, {});
  CHECK(s.score.left == 1);
  CHECK(s.phase == Phase::Serving);                     // point ended, re-serving
}

// ---------------- scoring, win, rally cap ----------------

TEST_CASE("concession distance: scoreIfOut accumulates |ball.y - paddle.y| for the conceding side"){
  State s = playing();
  s.left.y = 0.30;
  s.ball = {{ -0.001, 0.55 },{ -0.5, 0.0 }, k::BallR};   // past the left edge, vy=0 → left concedes at y=0.55
  stepOnce(s, {});
  CHECK(s.score.right == 1);
  CHECK(s.missN[0] == 1);
  CHECK(s.missSum[0] == doctest::Approx(0.25));          // |0.55 − 0.30|
  CHECK(s.missN[1] == 0);                                // right never conceded

  State t = playing();                                   // mirror: past the right edge → right concedes
  t.right.y = 0.40;
  t.ball = {{ 1.001, 0.90 },{ +0.5, 0.0 }, k::BallR};
  stepOnce(t, {});
  CHECK(t.score.left == 1);
  CHECK(t.missN[1] == 1);
  CHECK(t.missSum[1] == doctest::Approx(0.50));          // |0.90 − 0.40|
  CHECK(t.missN[0] == 0);
}

TEST_CASE("ball past the left edge: right scores, re-serves centered & Serving"){
  State s = playing();
  s.ball ={{ -0.001, 0.5 },{ -0.5, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(s.score.right == 1);
  CHECK(s.phase == Phase::Serving);
  CHECK(s.ball.pos.x == doctest::Approx(0.5));
  CHECK(s.rallyHits == 0);                              // rally counter resets with the serve
}
TEST_CASE("reaching WinScore ends the game with a winner outcome and freezes play"){
  State s = playing();
  s.score.left = k::WinScore - 1;
  s.ball ={{ 1.001, 0.5 },{ +0.5, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(s.score.left == k::WinScore);
  CHECK(s.phase == Phase::GameOver);
  CHECK(s.outcome == Outcome::LeftWin);
  int before = s.score.left;
  stepOnce(s, {Move::Up, Move::Down});
  CHECK(s.score.left == before);                        // frozen
}
TEST_CASE("runtime serve hold: zero hold enters Playing on the first tick"){
  State s = initialState(4);
  s.serveHold = 0.0; s.serveTimer = 0.0;
  stepOnce(s, {});
  CHECK(s.phase == Phase::Playing);
  PongCore core(4);
  core.setServeHold(0.0);
  core.reset(4);
  core.step({});
  CHECK(core.snapshot().phase == Phase::Playing);      // persists across reset
}
TEST_CASE("runtime win score: first-to-3 ends the game at 3"){
  State s = playing();
  s.winScore = 3;
  s.score.left = 2;
  s.ball = {{ 1.001, 0.5 },{ +0.5, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(s.score.left == 3);
  CHECK(s.phase == Phase::GameOver);
  CHECK(s.outcome == Outcome::LeftWin);
}

TEST_CASE("runtime speedup: a raised multiplier applies per hit and survives reset"){
  PongCore core(2);
  core.setSpeedup(1.06);
  core.reset(2);
  State s = playing();
  s.speedup = 1.06;
  double sp0 = 0.8, front = k::RightX - k::PaddleW*0.5;
  s.ball = {{ front-k::BallR-1e-4, 0.5 },{ +sp0, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(spd(s.ball.vel) == doctest::Approx(sp0 * 1.06));
}

TEST_CASE("runtime rally cap: a shortened cap ends the game early and survives reset"){
  PongCore core(3, /*rallyCap=*/5);
  CHECK(core.snapshot().rallyMax == 5);
  core.reset(3);
  CHECK(core.snapshot().rallyMax == 5);               // cap persists across reset
  State s = playing();
  s.rallyMax = 5; s.rallyHits = 4;
  double front=k::RightX-k::PaddleW*0.5;
  s.ball={{ front-k::BallR-1e-4, 0.5 },{ +0.8, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(s.phase == Phase::GameOver);
  CHECK(s.outcome == Outcome::RallyCap);
}

TEST_CASE("rally limit: the 1000th paddle hit ends the game with no winner"){
  State s = playing();
  s.rallyHits = k::RallyMax - 1;
  double front=k::RightX-k::PaddleW*0.5;
  s.ball={{ front-k::BallR-1e-4, 0.5 },{ +0.8, 0.0 }, k::BallR};
  stepOnce(s, {});
  CHECK(s.phase == Phase::GameOver);
  CHECK(s.outcome == Outcome::RallyCap);
  CHECK(s.score.left == 0);
  CHECK(s.score.right == 0);                            // nobody is awarded the point
  CHECK(view(s).rallyHits == k::RallyMax);
}

// ---------------- seeding & determinism ----------------

TEST_CASE("seed 0 keeps the legacy fixed serve angle"){
  State s = initialState(0);
  CHECK(std::abs(s.ball.vel.y / s.ball.vel.x) == doctest::Approx(std::tan(k::ServeAngleDeg * 3.14159265358979323846 / 180.0)));
}
TEST_CASE("same seed → identical serves and runs; different seed → different serve"){
  State a = initialState(7), b = initialState(7), c = initialState(8);
  CHECK(a.ball.vel.x == b.ball.vel.x);
  CHECK(a.ball.vel.y == b.ball.vel.y);
  CHECK(a.ball.vel.y != c.ball.vel.y);                  // angle jitter differs by seed
  auto run = [](unsigned seed){
    PongCore core(seed); std::vector<double> log;
    for (int i=0;i<1200;i++){
      core.step(Input{ (i/30)%2 ? Move::Up : Move::Down, (i/17)%2 ? Move::Down : Move::None });
      Snapshot s = core.snapshot();
      log.push_back(s.ball.pos.x); log.push_back(s.ball.pos.y);
      log.push_back(s.left.y);     log.push_back(s.right.y);
    }
    return log;
  };
  CHECK(run(9) == run(9));
}

// ---------------- observation ----------------

TEST_CASE("observation: left is identity, right is x-mirrored with own paddle first"){
  State s = playing();
  s.ball = {{0.3, 0.7},{+0.4, -0.2}, k::BallR};
  s.left.y = 0.2; s.right.y = 0.8;
  Snapshot v = view(s);
  auto L = observation(v, Side::Left);
  CHECK(L[0] == doctest::Approx(0.3));  CHECK(L[1] == doctest::Approx(0.7));
  CHECK(L[2] == doctest::Approx(+0.4)); CHECK(L[3] == doctest::Approx(-0.2));
  CHECK(L[4] == doctest::Approx(0.2));  CHECK(L[5] == doctest::Approx(0.8));
  auto R = observation(v, Side::Right);
  CHECK(R[0] == doctest::Approx(0.7));  CHECK(R[1] == doctest::Approx(0.7));
  CHECK(R[2] == doctest::Approx(-0.4)); CHECK(R[3] == doctest::Approx(-0.2));
  CHECK(R[4] == doctest::Approx(0.8));  CHECK(R[5] == doctest::Approx(0.2));
}
