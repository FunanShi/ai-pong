#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "agents.hpp"
#include "test_util.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace pong;
using testutil::playing;

// ---------------- scripted agents ----------------

TEST_CASE("PAgent tracks an incoming ball and respects its deadband"){
  State s = playing();
  s.right.y = 0.2;
  s.ball = {{0.6, 0.9},{+0.5, 0.0}, k::BallR};          // incoming toward right, ball below paddle
  PAgent a("p", 0.02, 1.0);
  CHECK(a.act(view(s), Side::Right) == Move::Down);     // y must increase toward the ball
  s.right.y = 0.9;                                      // within deadband of ball.y
  s.ball.pos.y = 0.905;
  CHECK(a.act(view(s), Side::Right) == Move::None);
}
TEST_CASE("PAgent recenters when the ball is receding"){
  State s = playing();
  s.right.y = 0.9;
  s.ball = {{0.6, 0.1},{-0.5, 0.0}, k::BallR};          // receding from the right side
  PAgent a("p", 0.02, 1.0);
  CHECK(a.act(view(s), Side::Right) == Move::Up);       // toward center 0.5
}
TEST_CASE("PAgent duty cycle gates moves on a deterministic schedule"){
  State s = playing();
  s.right.y = 0.2;
  s.ball = {{0.6, 0.9},{+0.5, 0.0}, k::BallR};
  PAgent a("p", 0.02, 0.5);
  CHECK(a.act(view(s), Side::Right) == Move::None);     // acc 0.5 — gated
  CHECK(a.act(view(s), Side::Right) == Move::Down);     // acc 1.0 — allowed
  CHECK(a.act(view(s), Side::Right) == Move::None);     // gated again
}

TEST_CASE("predictInterceptY: straight shot, one-fold shot, and receding ball"){
  Ball b{{0.5, 0.5},{+0.4, +0.2}, k::BallR};
  CHECK(predictInterceptY(b, 0.96) == doctest::Approx(0.73));            // no wall contact
  Ball f{{0.5, 0.5},{+0.4, +1.6}, k::BallR};
  CHECK(predictInterceptY(f, 0.96) == doctest::Approx(0.388));           // folds once off the bottom
  Ball r{{0.5, 0.5},{-0.4, +0.2}, k::BallR};
  CHECK(predictInterceptY(r, 0.96) == doctest::Approx(0.5));             // not moving toward plane
}
TEST_CASE("InterceptorAgent drives toward the predicted intercept"){
  State s = playing();
  s.right.y = 0.2;
  s.ball = {{0.5, 0.5},{+0.4, +0.2}, k::BallR};         // intercept at y≈0.73 > paddle y
  InterceptorAgent a("icept", 0.015, 1);
  CHECK(a.act(view(s), Side::Right) == Move::Down);
}

// ---------------- MLP model agent ----------------

TEST_CASE("MlpAgent: loads v1 format and argmaxes its logits"){
  std::ostringstream m;
  m << "aipong-mlp 1\n" << "6 4 3\n";
  for (int i=0;i<24;i++) m << "0 ";                     // W1 4x6
  m << "\n";
  for (int i=0;i<4;i++)  m << "0 ";                     // b1
  m << "\n";
  for (int i=0;i<12;i++) m << "0 ";                     // W2 3x4
  m << "\n";
  m << "0 1 -1\n";                                      // b2 → logits {0,1,-1} → argmax 1 → Up
  std::istringstream in(m.str());
  MlpAgent a("mlp");
  CHECK(a.loadStream(in));
  CHECK(a.loaded());
  CHECK(a.act(view(playing()), Side::Right) == Move::Up);
}
TEST_CASE("MlpAgent rejects bad magic and wrong input width"){
  MlpAgent a("mlp");
  std::istringstream bad1("not-a-model 1\n6 3\n");
  CHECK_FALSE(a.loadStream(bad1));
  std::istringstream bad2("aipong-mlp 1\n5 3\n");
  CHECK_FALSE(a.loadStream(bad2));
  CHECK_FALSE(a.loaded());
}

TEST_CASE("MlpAgent v2: frame stacking assembles oldest->newest and pads fresh episodes"){
  // Single linear layer, 12 -> 3. Logit Up reads input[0] (OLDEST frame ball.x);
  // logit Down reads input[6] (NEWEST frame ball.x). Everything else zero.
  std::ostringstream m;
  m << "aipong-mlp 2\nobs stack 2\n12 3\n";
  std::vector<double> W(36, 0.0);
  W[12*1 + 0] = 1.0;                                    // Up  <- oldest ball.x
  W[12*2 + 6] = 1.0;                                    // Down <- newest ball.x
  for (double v : W) m << v << ' ';
  m << "\n0 0 0\n";
  MlpAgent a("mlp2");
  std::istringstream in(m.str());
  CHECK(a.loadStream(in));
  CHECK(a.stack() == 2);
  State s = playing();
  s.ball = {{0.3, 0.5},{+0.4, 0.0}, k::BallR};
  CHECK(a.act(view(s), Side::Left) == Move::Up);        // padded: both frames 0.3 -> tie -> first max (Up)
  s.ball.pos.x = 0.8;
  CHECK(a.act(view(s), Side::Left) == Move::Down);      // oldest 0.3 (Up) < newest 0.8 (Down)
  a.reset();                                            // new match: history cleared
  s.ball.pos.x = 0.3;
  CHECK(a.act(view(s), Side::Left) == Move::Up);        // padding behavior again
}
TEST_CASE("MlpAgent v2: stride token parses; caps enforced; stride-1 emission unchanged"){
  std::ostringstream m;
  m << "aipong-mlp 2\nobs stack 2 stride 6\n12 3\n";
  for (int i = 0; i < 12*3; ++i) m << "0 ";
  m << "\n0 0 0\n";
  MlpAgent a("mlp");
  std::istringstream in(m.str());
  CHECK(a.loadStream(in));
  CHECK(a.stack() == 2);
  CHECK(a.stride() == 6);

  MlpAgent b("bad");
  std::istringstream bad1("aipong-mlp 2\nobs stack 2 stride 0\n12 3\n");
  CHECK_FALSE(b.loadStream(bad1));                      // stride < 1
  std::istringstream bad2("aipong-mlp 2\nobs stack 33\n198 3\n");
  CHECK_FALSE(b.loadStream(bad2));                      // K > 32
  std::istringstream bad3("aipong-mlp 2\nobs stack 32 stride 60\n192 3\n");
  CHECK_FALSE(b.loadStream(bad3));                      // window (32-1)*60+1 > 512
  std::istringstream bad4("aipong-mlp 2\nobs stack 2 wobble 6\n12 3\n");
  CHECK_FALSE(b.loadStream(bad4));                      // unknown token

  std::vector<int> sizes{12, 3};
  std::vector<double> flat(12*3 + 3, 0.25);
  std::ostringstream o1; writeModelText(o1, sizes, flat, 2, 6);
  CHECK(o1.str().find("obs stack 2 stride 6") != std::string::npos);
  std::ostringstream o2; writeModelText(o2, sizes, flat, 2, 1);
  CHECK(o2.str().find("stride") == std::string::npos);  // stride-1 files look like today's
}

TEST_CASE("MlpAgent strided stacking samples every S-th frame, oldest->newest"){
  std::ostringstream m;
  m << "aipong-mlp 2\nobs stack 2 stride 3\n12 3\n";
  std::vector<double> W(36, 0.0);
  W[12*1 + 0] = 1.0;                                    // Up  <- oldest (t-3) ball.x
  W[12*2 + 6] = 1.0;                                    // Down <- newest (t) ball.x
  for (double v : W) m << v << ' ';
  m << "\n0 0 0\n";
  MlpAgent a("mlp");
  std::istringstream in(m.str());
  REQUIRE(a.loadStream(in));
  State s = playing();
  s.ball = {{0.0, 0.5},{+0.4, 0.0}, k::BallR};
  for (int t = 1; t <= 6; ++t){                         // rising ball.x: newest > t-3 sample
    s.ball.pos.x = 0.10 * t;
    Move mv = a.act(view(s), Side::Left);
    if (t >= 2) CHECK(mv == Move::Down);
  }
  a.reset();
  for (int t = 6; t >= 1; --t){                         // falling: the old sample dominates
    s.ball.pos.x = 0.10 * t;
    Move mv = a.act(view(s), Side::Left);
    if (t <= 4) CHECK(mv == Move::Up);
  }
}

TEST_CASE("setAblateHistory: removes temporal info from a memory-using policy"){
  // stack-2 linear net: Up-logit reads OLDEST ball.x (col 0), Down-logit reads NEWEST (col 6).
  std::ostringstream m;
  m << "aipong-mlp 2\nobs stack 2\n12 3\n";
  std::vector<double> W(36, 0.0);
  W[12*1 + 0] = 1.0;                                    // Up   <- oldest ball.x
  W[12*2 + 6] = 1.0;                                    // Down <- newest ball.x
  for (double v : W) m << v << ' ';
  m << "\n0 0 0\n";
  MlpAgent a("mem");
  std::istringstream in(m.str());
  REQUIRE(a.loadStream(in));
  State s = playing();
  s.ball = {{0.2, 0.5},{+0.4, 0.0}, k::BallR};
  a.act(view(s), Side::Left);                           // seed history at x=0.2
  s.ball.pos.x = 0.8;
  CHECK(a.act(view(s), Side::Left) == Move::Down);      // true history: oldest 0.2 < newest 0.8
  a.reset();
  a.setAblateHistory(true);
  CHECK(a.act(view(s), Side::Left) == Move::Up);        // ablated: both slots 0.8 -> tie -> Up
}

TEST_CASE("MlpAgent v2: rejects size/stack mismatch and unknown obs kind"){
  MlpAgent a("mlp");
  std::istringstream bad1("aipong-mlp 2\nobs stack 2\n6 3\n");   // input must be 12
  CHECK_FALSE(a.loadStream(bad1));
  std::istringstream bad2("aipong-mlp 2\nobs wobble 2\n12 3\n");
  CHECK_FALSE(a.loadStream(bad2));
  std::istringstream bad3("aipong-mlp 3\n6 3\n");                // unknown version
  CHECK_FALSE(a.loadStream(bad3));
}

TEST_CASE("MlpAgent: malformed sizes line fails cleanly before any allocation"){
  const char* p = "/tmp/aipong_bad_sizes.txt";
  { std::ofstream f(p); f << "aipong-mlp 1\n6 2000000000 3\n0.1 0.2\n"; }   // absurd hidden width
  MlpAgent a("bad");
  CHECK_FALSE(a.loadFile(p));                          // must return false, not bad_alloc
  { std::ofstream f(p); f << "aipong-mlp 1\n6 -8 3\n0.1\n"; }              // negative size
  CHECK_FALSE(a.loadFile(p));
  std::filesystem::remove(p);
}

TEST_CASE("sizesInBounds: the shared pre-allocation sanity gate"){
  CHECK(sizesInBounds({6, 32, 32, 3}));
  CHECK(sizesInBounds({36, 32, 32, 3}));
  CHECK_FALSE(sizesInBounds({6}));                       // too few layers
  CHECK_FALSE(sizesInBounds({6, 0, 3}));                 // zero width
  CHECK_FALSE(sizesInBounds({6, -8, 3}));                // negative
  CHECK_FALSE(sizesInBounds({6, 2000000000, 3}));        // absurd width
  CHECK_FALSE(sizesInBounds({100000, 100000, 3}));       // param count over 8M
}

// ---------------- numerics pin (Decision 11/13/23) ----------------

// MlpAgent::act() exposes no logit accessor (agents.hpp — deliberately not adding public API
// just for this test), so this pins the Move SEQUENCE act() returns instead of raw logits:
// FNV-1a folded over every Move across 200 LCG-perturbed states (same randomizer as
// test_train.cpp's widenGenome test) x {stack 1, stack 6}, deterministic weight fill
// (i*17)%31 — a wider modulus than the (i*7)%11 used elsewhere (e.g. bench.cpp): that fill
// collapses THIS shape to a 200/200 "Down" landslide (checked numerically), which would give
// the pin zero sensitivity, whereas (i*17)%31 spreads across all three Moves for both stacks
// (see kArgmax1/kArgmax6). A change here is a cross-version numerics change: see DECISIONS.md
// #11 (float32 + -ffast-math), #13 (batched SoA path), #23 (act()'s per-layer loop flattened
// to transposed-weight/flat-MAC shape so libmvec vectorizes tanh — confirmed by disassembly:
// agents.cpp.o now links _ZGVbN4v_tanhf/_ZGVdN8v_tanhf alongside scalar tanhf).
// Decision 23 measurement: this hash/table pair was regenerated against the flattened act()
// and came back BYTE-IDENTICAL to the pre-#23 constants below — 0/400 sampled Moves flipped.
// That is a legitimate reassociation-class result, not a no-op change: a throwaway logit-diff
// probe (old row-major-serial-reduce vs new transposed-flat forward, same weights/inputs,
// same float32/-ffast-math/-march=native codegen, see DECISIONS.md #23) measured real drift —
// max |Δlogit| ~8.6e-7, mean ~1.6e-7, same order as #11/#13's ~1e-7 class — that simply never
// exceeded any sampled decision's margin. The 20-state tables are a SEPARATE, coarser
// stability sentinel drawn from the same sequence: if either ever changes on a future
// numerics edit, STOP — that is a decision flip, not reassociation noise (mirrors "BatchedMlp
// matches the scalar agent on well-separated decisions" in test_train.cpp, which tolerates
// the same class of drift for the same reason). Constants were generated by running this test
// once (hash/tables printed to stdout) and pasting the output; regenerate ONLY as a recorded
// numerics decision (Decision 23 did — and reconfirmed the same values).
TEST_CASE("MlpAgent::act: bit-exact logit pin (regenerate ONLY as a recorded numerics decision)"){
  static const Move kArgmax1[20] = {
    Move::Up,   Move::Up,   Move::Up,   Move::Up,   Move::Down, Move::Down, Move::Down,
    Move::Up,   Move::Down, Move::Down, Move::Down, Move::Up,   Move::Down, Move::Down,
    Move::Up,   Move::Up,   Move::Up,   Move::Down, Move::Down, Move::Down,
  };
  static const Move kArgmax6[20] = {
    Move::None, Move::None, Move::None, Move::None, Move::None, Move::None, Move::None,
    Move::None, Move::None, Move::None, Move::Down, Move::Up,   Move::None, Move::Down,
    Move::Up,   Move::Down, Move::None, Move::Up,   Move::Down, Move::Up,
  };
  uint64_t h = 14695981039346656037ull;                  // FNV-1a 64 offset basis
  const uint64_t prime = 1099511628211ull;                // FNV-1a 64 prime
  for (int stack : {1, 6}){
    std::vector<int> sizes{6 * stack, 32, 32, 3};
    std::vector<double> flat;
    int n = sizes[0]*32 + 32 + 32*32 + 32 + 32*3 + 3;
    for (int i = 0; i < n; ++i) flat.push_back(0.025 * ((i * 17) % 31) - 0.4);
    MlpAgent a("pin");
    REQUIRE(a.loadWeights(sizes, flat, stack, stack));
    State s = testutil::playing();
    uint64_t r = 20260710;                                 // fixed stream, own test's namespace
    for (int t = 0; t < 200; ++t){
      r = r * 6364136223846793005ull + 1442695040888963407ull;
      s.ball.pos = {0.05 + 0.9 * double((r >> 11) % 1000) / 1000.0,
                    0.05 + 0.9 * double((r >> 21) % 1000) / 1000.0};
      s.ball.vel = {((r >> 31) % 2 ? 1.0 : -1.0) * 0.8, (double((r >> 33) % 200) / 100.0) - 1.0};
      s.left.y  = 0.1 + 0.8 * double((r >> 41) % 100) / 100.0;
      s.right.y = 0.1 + 0.8 * double((r >> 51) % 100) / 100.0;
      Move mv = a.act(view(s), Side::Left);
      h = (h ^ (uint64_t)static_cast<uint8_t>(mv)) * prime;
      if (t < 20) CHECK(mv == (stack == 1 ? kArgmax1[t] : kArgmax6[t]));
    }
  }
  CHECK(h == 0x9433f1283ac31fffull);
}
