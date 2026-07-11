#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "match_runner.hpp"
#include "rally_index.hpp"
#include <fstream>
#include <utility>
using namespace pong;

TEST_CASE("controller registry: builtin specs construct fresh agents; labels are stable"){
  for (const auto& s : builtinSpecs()){
    CHECK_FALSE(isHumanSpec(s));
    auto a = makeAgent(s);
    CHECK(a != nullptr);
  }
  CHECK(isHumanSpec("human"));
  CHECK(makeAgent("human") == nullptr);
  CHECK(makeAgent("nonsense") == nullptr);
  CHECK(makeAgent("mlp:/nonexistent/file.txt") == nullptr);
  // Labels are recorded in match metadata — changing one is a data-compat decision.
  CHECK(agentLabel("human") == "Human (keyboard)");
  CHECK(agentLabel("p:classic") == "P-controller (classic)");
  CHECK(agentLabel("interceptor:laggy") == "Interceptor (laggy)");
  CHECK(agentLabel("mlp:models/random.txt") == "Model: random.txt");
}

TEST_CASE("held-out anchor variants: parse, label, and differ from training variants"){
  auto p = makeAgent("p:heldout");
  auto i = makeAgent("interceptor:heldout");
  REQUIRE(p != nullptr);
  REQUIRE(i != nullptr);
  CHECK(agentLabel("p:heldout") == "P-controller (held-out)");
  CHECK(agentLabel("interceptor:heldout") == "Interceptor (held-out)");
  CHECK(makeAgent("p:wrong") == nullptr);
  // behavioral distinctness: offset 0.03 sits inside heldout's 0.05 deadband but outside
  // classic's 0.02 — the two must disagree on the same state
  State s = initialState(3);
  s.phase = Phase::Playing;
  s.right.y = 0.5;
  s.ball = {{0.6, 0.53},{+0.5, 0.0}, k::BallR};
  auto classic = makeAgent("p:classic");
  classic->act(view(s), Side::Right);                   // first call charges the duty accumulator
  p->act(view(s), Side::Right);
  CHECK(classic->act(view(s), Side::Right) == Move::Down);
  CHECK(p->act(view(s), Side::Right) == Move::None);    // 0.03 offset inside 0.05 deadband
}

TEST_CASE("MatchRunner: headless bot match reaches a decisive outcome, deterministically"){
  auto run = [](){
    MatchRunner r(11);
    r.setController(Side::Left,  "p:hard");
    r.setController(Side::Right, "p:easy");
    r.newGame(11);
    Outcome o = r.runToCompletion(300000);
    return std::make_pair(o, r.snapshot());
  };
  auto [o1, s1] = run();
  auto [o2, s2] = run();
  REQUIRE(s1.phase == Phase::GameOver);
  bool decisive = o1 == Outcome::LeftWin || o1 == Outcome::RightWin;
  CHECK(decisive);
  CHECK(o1 == o2);                                    // same seed + same specs → identical match
  CHECK(s1.score.left == s2.score.left);
  CHECK(s1.score.right == s2.score.right);
}

TEST_CASE("MatchRunner: ticks() counts stepped ticks and resets with newGame"){
  MatchRunner r(4);
  r.setController(Side::Left,  "p:classic");
  r.setController(Side::Right, "p:classic");
  r.newGame(4);
  CHECK(r.ticks() == 0);
  for (int i = 0; i < 100; ++i) r.tick();
  CHECK(r.ticks() == 100);
  r.newGame(4);
  CHECK(r.ticks() == 0);
}

TEST_CASE("MatchRunner: setRallyCap truncates interceptor stalemates"){
  MatchRunner r(6);
  r.setRallyCap(40);
  r.setController(Side::Left,  "interceptor");
  r.setController(Side::Right, "interceptor");
  r.newGame(6);
  Outcome o = r.runToCompletion(500000);
  Snapshot s = r.snapshot();
  // Two perfect predictors either rally to the cap or trade quantization misses; either
  // way the shortened cap must bound any singular rally at 40 hits.
  CHECK(s.phase == Phase::GameOver);
  CHECK(s.rallyMax == 40);
  if (o == Outcome::RallyCap) CHECK(s.rallyHits == 40);
  else CHECK(s.rallyHits <= 40);
}

TEST_CASE("MatchRunner: humanControlled reflects specs; external moves drive human sides"){
  MatchRunner r(0);
  r.setController(Side::Left,  "human");
  r.setController(Side::Right, "p:classic");
  r.newGame(0);
  CHECK(r.humanControlled(Side::Left));
  CHECK_FALSE(r.humanControlled(Side::Right));
  double y0 = r.snapshot().left.y;
  r.tick(Move::Up, Move::None);
  CHECK(r.snapshot().left.y < y0);                    // external move reached the human side
}

TEST_CASE("MatchRunner: recording through the runner auto-finalizes and indexes cleanly"){
  const char* path = "/tmp/aipong_runner_rec.jsonl";
  MatchRunner r(5);
  r.setController(Side::Left,  "p:hard");
  r.setController(Side::Right, "p:easy");
  r.newGame(5);
  REQUIRE(r.startRecording(path));
  CHECK(r.recording());
  r.runToCompletion(300000);
  REQUIRE(r.snapshot().phase == Phase::GameOver);
  CHECK_FALSE(r.recording());                         // auto-finalized (truncated=false) at game over

  std::ifstream f(path);
  auto rows = indexMatchStream(f, "runner.jsonl");
  Snapshot fin = r.snapshot();
  CHECK((int)rows.size() == fin.score.left + fin.score.right);
  REQUIRE(!rows.empty());
  CHECK(rows.front().left  == "P-controller (hard)"); // labels flow from the spec registry
  CHECK(rows.front().right == "P-controller (easy)");
}
