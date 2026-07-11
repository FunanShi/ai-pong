#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "batch_forward.hpp"
#include "coevolve.hpp"
#include "evolve.hpp"
#include "match_runner.hpp"
#include "recipes.hpp"
#include "test_util.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
using namespace pong;
using testutil::playing;

TEST_CASE("loadWeights and loadStream produce identical policies"){
  std::vector<int> sizes{6, 8, 3};
  std::vector<double> flat;
  for (int i = 0; i < 6*8 + 8 + 8*3 + 3; ++i) flat.push_back(0.01 * (i % 13) - 0.06);
  std::ostringstream txt;
  writeModelText(txt, sizes, flat);
  MlpAgent a("stream"), b("direct");
  std::istringstream in(txt.str());
  REQUIRE(a.loadStream(in));
  REQUIRE(b.loadWeights(sizes, flat));
  State s = playing();
  s.ball = {{0.4, 0.3},{+0.6, 0.4}, k::BallR};
  for (int i = 0; i < 5; ++i){
    s.ball.pos.y = 0.1 + 0.15 * i;
    CHECK(a.act(view(s), Side::Right) == b.act(view(s), Side::Right));
  }
  CHECK_FALSE(b.loadWeights(sizes, std::vector<double>(10)));   // wrong length rejected
}

TEST_CASE("evolve with pool + anchor games: thread-invariant, held-out evals reported"){
  std::filesystem::create_directories("/tmp/aipong_pool");
  std::vector<int> s1{6, 32, 32, 3};
  std::vector<double> f1;
  for (size_t i = 0; i < (size_t)(6*32 + 32 + 32*32 + 32 + 32*3 + 3); ++i)
    f1.push_back(0.01 * double((i * 5) % 9) - 0.02);
  { std::ofstream f("/tmp/aipong_pool/gen_0000.txt"); writeModelText(f, s1, f1); }
  { std::ofstream f("/tmp/aipong_pool/gen_0001.txt"); writeModelText(f, s1, f1); }

  EvolveConfig cfg;
  cfg.pop = 6; cfg.gens = 2; cfg.gamesPerEval = 4;
  cfg.poolGames = 1; cfg.anchorGames = 1;
  cfg.poolDirs = {"/tmp/aipong_pool"};
  cfg.evalEvery = 2; cfg.evalMatches = 2; cfg.hofEvery = 1;
  cfg.seed = 11; cfg.threads = 2; cfg.outDir = "";
  GenStats lastEval;
  EvolveResult a = evolve(cfg, [&](const GenStats& s){ if (s.vsPClassic > -1.5) lastEval = s; });
  cfg.threads = 4;
  EvolveResult b = evolve(cfg);
  REQUIRE(a.log.size() == b.log.size());
  for (size_t g = 0; g < a.log.size(); ++g)
    CHECK(a.log[g].best == b.log[g].best);              // mixed scalar+batched paths, exact
  CHECK(lastEval.vsHeldP >= 0.0);                       // held-out anchors evaluated
  CHECK(lastEval.vsHeldIcept >= 0.0);
}

TEST_CASE("evolve is deterministic across thread counts"){
  EvolveConfig cfg;
  cfg.pop = 8; cfg.gens = 3; cfg.gamesPerEval = 2;
  cfg.evalEvery = 0; cfg.hofEvery = 2;
  cfg.seed = 7; cfg.outDir = "";                                // no file output in tests
  cfg.threads = 1;
  EvolveResult r1 = evolve(cfg);
  cfg.threads = 4;
  EvolveResult r2 = evolve(cfg);
  REQUIRE(r1.log.size() == r2.log.size());
  for (size_t g = 0; g < r1.log.size(); ++g){
    CHECK(r1.log[g].best == doctest::Approx(r2.log[g].best));
    CHECK(r1.log[g].mean == doctest::Approx(r2.log[g].mean));
  }
  REQUIRE(r1.bestFlat.size() == r2.bestFlat.size());
  CHECK(r1.bestFlat == r2.bestFlat);                            // bit-identical genome
  for (const auto& s : r1.log){                                 // margins live in [-1,1]
    CHECK(s.best >= -1.0); CHECK(s.best <= 1.0);
    CHECK(s.mean >= -1.0); CHECK(s.mean <= 1.0);
  }
}

TEST_CASE("evolve trains a deeper net (3 hidden layers) deterministically"){
  EvolveConfig cfg;
  cfg.pop = 8; cfg.gens = 3; cfg.gamesPerEval = 2;
  cfg.evalEvery = 0; cfg.hofEvery = 2; cfg.seed = 7; cfg.outDir = "";
  cfg.sizes = {6, 16, 16, 16, 3};                               // 3 hidden layers (default is 2)
  cfg.threads = 1; EvolveResult r1 = evolve(cfg);
  cfg.threads = 4; EvolveResult r2 = evolve(cfg);
  REQUIRE(r1.bestFlat.size() == r2.bestFlat.size());
  CHECK(r1.bestFlat == r2.bestFlat);                            // deeper arch trains bit-identically across threads
  size_t expect = 6*16+16 + 16*16+16 + 16*16+16 + 16*3+3;       // flatSize({6,16,16,16,3}) = 707
  CHECK(r1.bestFlat.size() == expect);
}

TEST_CASE("loss shaping is wired end-to-end: enabling it changes training fitness"){
  EvolveConfig base;
  base.pop = 8; base.gens = 3; base.gamesPerEval = 2;
  base.evalEvery = 0; base.hofEvery = 2; base.seed = 7; base.outDir = "";
  base.winScore = 5; base.threads = 1;
  EvolveConfig shaped = base;
  shaped.lossBonus = 0.2; shaped.lossProxWeight = 0.7;
  EvolveResult r0 = evolve(base);
  EvolveResult r1 = evolve(shaped);
  REQUIRE(r0.log.size() == r1.log.size());
  bool differs = false;                                    // shaping lifts losing genomes → trajectory diverges
  for (size_t g = 0; g < r0.log.size(); ++g)
    if (r0.log[g].mean != doctest::Approx(r1.log[g].mean)) differs = true;
  CHECK(differs);
}

TEST_CASE("loss shaping stays deterministic across thread counts"){
  EvolveConfig cfg;
  cfg.pop = 8; cfg.gens = 3; cfg.gamesPerEval = 2;
  cfg.evalEvery = 0; cfg.hofEvery = 2; cfg.seed = 7; cfg.outDir = "";
  cfg.winScore = 5; cfg.lossBonus = 0.2; cfg.lossProxWeight = 0.7;   // shaping ON
  cfg.threads = 1;
  EvolveResult r1 = evolve(cfg);
  cfg.threads = 4;
  EvolveResult r2 = evolve(cfg);
  REQUIRE(r1.log.size() == r2.log.size());
  for (size_t g = 0; g < r1.log.size(); ++g){
    CHECK(r1.log[g].best == doctest::Approx(r2.log[g].best));
    CHECK(r1.log[g].mean == doctest::Approx(r2.log[g].mean));
  }
  REQUIRE(r1.bestFlat.size() == r2.bestFlat.size());
  CHECK(r1.bestFlat == r2.bestFlat);                       // bit-identical genome under shaping
}

TEST_CASE("shapeFitness: wins-only bonus preserves outcome ordering"){
  // beta=0 → identity
  CHECK(shapeFitness(+0.5, 100, 0.0, 30000) == doctest::Approx(+0.5));
  CHECK(shapeFitness(-0.5, 100, 0.0, 30000) == doctest::Approx(-0.5));
  // with only the win-speed bonus (lossBonus defaults to 0), losses are never shaped
  CHECK(shapeFitness(-1.0, 50, 0.25, 30000) == doctest::Approx(-1.0));
  CHECK(shapeFitness(-1.0/11.0, 100000, 0.25, 30000) == doctest::Approx(-1.0/11.0));
  // fast win > slow win at equal margin
  double fast = shapeFitness(1.0, 3000, 0.25, 30000);
  double slow = shapeFitness(1.0, 29000, 0.25, 30000);
  CHECK(fast > slow);
  CHECK(slow >= 1.0);                                   // bonus never negative
  CHECK(fast <= 1.25);                                  // bounded by beta
  // beyond T_ref the bonus is exactly zero
  CHECK(shapeFitness(1.0, 30000, 0.25, 30000) == doctest::Approx(1.0));
  CHECK(shapeFitness(1.0, 500000, 0.25, 30000) == doctest::Approx(1.0));
  // the narrowest win still outranks the narrowest loss for any beta
  CHECK(shapeFitness(+1.0/11.0, 500000, 0.25, 30000) > shapeFitness(-1.0/11.0, 100, 0.25, 30000));
}

TEST_CASE("shapeFitness: loss shaping is bounded and rewards slower, closer losses"){
  const int  W    = 5;
  const long Tref  = 30000;
  // off: lossBonus=0 → identity, however slow or close
  CHECK(shapeFitness(-1.0/W, 20000, 0.2, Tref, 0.0, 0.7, 0.0, W) == doctest::Approx(-1.0/W));
  // rule 2: slower loss (more ticks) scores strictly higher at equal margin & avgMiss
  double fastLoss = shapeFitness(-1.0/W, 3000,  0.2, Tref, 0.2, 0.7, 0.5, W);
  double slowLoss = shapeFitness(-1.0/W, 29000, 0.2, Tref, 0.2, 0.7, 0.5, W);
  CHECK(slowLoss > fastLoss);
  // rule 4: closer concession (smaller avgMiss) scores strictly higher at equal margin & ticks
  double farMiss   = shapeFitness(-1.0/W, 10000, 0.2, Tref, 0.2, 0.7, 0.8, W);
  double closeMiss = shapeFitness(-1.0/W, 10000, 0.2, Tref, 0.2, 0.7, 0.1, W);
  CHECK(closeMiss > farMiss);
  // proximity-weighted: an equal delta in avgMiss moves fitness more than the same delta in survive
  double dProx = shapeFitness(-0.5, 10000, 0.2, Tref, 0.2, 0.7, 0.0, W)
               - shapeFitness(-0.5, 10000, 0.2, Tref, 0.2, 0.7, 1.0, W);   // Δprox = 1.0
  double dSurv = shapeFitness(-0.5, Tref,  0.2, Tref, 0.2, 0.7, 0.5, W)
               - shapeFitness(-0.5, 0,     0.2, Tref, 0.2, 0.7, 0.5, W);   // Δsurvive = 1.0
  CHECK(dProx > dSurv);                                    // w_d=0.7 > 1-w_d=0.3
  // rule 3 (the hard guarantee): NO loss outranks the narrowest win, even with an absurd lossBonus
  double worstWin = shapeFitness(+1.0/W, 500000, 0.2, Tref);                     // narrowest, slowest win
  double bestLoss = shapeFitness(-1.0/W, Tref, 0.2, Tref, 100.0, 0.7, 0.0, W);   // huge bonus, max q
  CHECK(bestLoss < worstWin);
  // rule 3, hardened: an out-of-range lossProxWeight (a CLI typo) must NOT over-weight prox past
  // q=1 and let a loss outrank a win — lossProxWeight is clamped [0,1] inside the function
  CHECK(shapeFitness(-1.0/W, 0, 0.2, Tref, 1.0, 1.2, 0.0, W) < shapeFitness(+1.0/W, 500000, 0.2, Tref));
}

TEST_CASE("evalParams: canonical unshaped physics, but the genome's trained stride"){
  EvolveConfig cfg;
  cfg.stack = 6; cfg.stride = 6;
  cfg.speedBonus = 0.2; cfg.rallyCap = 500; cfg.speedup = 1.06;   // training knobs that must
  cfg.serveHold = 0.0; cfg.winScore = 5;                          // NOT leak into evals
  MatchParams mp = evalParams(cfg);
  CHECK(mp.speedBonus == 0.0);              // unshaped — evals count wins, not shaped margin
  CHECK(mp.rallyCap == 0);                  // full k::RallyMax
  CHECK(mp.speedup == 0.0);                 // canonical 1.03 per hit
  CHECK(mp.serveHold < 0.0);                // canonical 0.75 s serve hold
  CHECK(mp.winScore == 0);                  // canonical first-to-11
  // stride is agent architecture, not physics: a stack-6 stride-6 genome evaluated at
  // stride 1 sees a 0.10 s window of near-duplicate frames instead of its 0.52 s one
  CHECK(mp.stride == 6);
  EvolveConfig flat;                        // stack-1 default keeps stride 1
  CHECK(evalParams(flat).stride == 1);
}

TEST_CASE("playMatchVsSpec completes against a scripted anchor and stays in bounds"){
  EvolveConfig cfg;                                             // default sizes 6-32-32-3
  std::vector<double> flat;
  {   // deterministic junk genome of the right length
    int n = 6*32 + 32 + 32*32 + 32 + 32*3 + 3;
    for (int i = 0; i < n; ++i) flat.push_back(0.02 * ((i * 7) % 11) - 0.1);
  }
  double m = playMatchVsSpec(cfg.sizes, flat, "p:easy", nullptr, 42);
  CHECK(m >= -1.0);
  CHECK(m <= 1.0);
}

TEST_CASE("widenGenome: widened net plays bit-identically to its stack-1 parent"){
  std::vector<int> sizes{6, 8, 3};
  std::vector<double> flat;
  for (int i = 0; i < 6*8 + 8 + 8*3 + 3; ++i) flat.push_back(0.013 * ((i * 11) % 17) - 0.1);
  std::vector<int> ws; std::vector<double> wf;
  REQUIRE(widenGenome(sizes, flat, 6, ws, wf));
  CHECK(ws.front() == 36);
  MlpAgent parent("p"), child("c");
  REQUIRE(parent.loadWeights(sizes, flat, 1));
  REQUIRE(child.loadWeights(ws, wf, 6, 6));
  State s = testutil::playing();
  uint64_t r = 12345;
  for (int t = 0; t < 200; ++t){                        // varied states AND varied histories
    r = r * 6364136223846793005ull + 1442695040888963407ull;
    s.ball.pos = {0.05 + 0.9 * double((r >> 11) % 1000) / 1000.0,
                  0.05 + 0.9 * double((r >> 21) % 1000) / 1000.0};
    s.ball.vel = {((r >> 31) % 2 ? 1.0 : -1.0) * 0.8, (double((r >> 33) % 200) / 100.0) - 1.0};
    s.left.y  = 0.1 + 0.8 * double((r >> 41) % 100) / 100.0;
    s.right.y = 0.1 + 0.8 * double((r >> 51) % 100) / 100.0;
    CHECK(parent.act(view(s), Side::Right) == child.act(view(s), Side::Right));
  }
}

TEST_CASE("batched stacked matches: widened genomes reproduce their stack-1 parents' games"){
  // The end-to-end compatibility property evo4 rests on: a widened (stack-6, stride-6)
  // genome ignores history by construction, so its batched stacked match must produce
  // the exact result of the stack-1 original. Pins the ring-buffer assembly's newest slot
  // and the whole widen→batched pipeline in one shot.
  std::vector<int> s1{6, 32, 32, 3};
  std::vector<double> a, b;
  for (int i = 0; i < 6*32 + 32 + 32*32 + 32 + 32*3 + 3; ++i){
    a.push_back(0.02 * ((i * 7) % 11) - 0.1);
    b.push_back(0.1 - 0.02 * ((i * 5) % 13));
  }
  std::vector<GenomeMatch> flat1(6);
  for (int m = 0; m < 6; ++m){
    flat1[(size_t)m].self = &a; flat1[(size_t)m].opp = &b;
    flat1[(size_t)m].seed = (unsigned)(700 + m);
  }
  playMatchesBatched(s1, flat1, MatchParams{}, 2, 4);

  std::vector<int> s6; std::vector<double> wa, wb;
  REQUIRE(widenGenome(s1, a, 6, s6, wa));
  { std::vector<int> tmp; REQUIRE(widenGenome(s1, b, 6, tmp, wb)); }
  MatchParams mp6; mp6.stride = 6;
  std::vector<GenomeMatch> stacked(6);
  for (int m = 0; m < 6; ++m){
    stacked[(size_t)m].self = &wa; stacked[(size_t)m].opp = &wb;
    stacked[(size_t)m].seed = (unsigned)(700 + m);
  }
  playMatchesBatched(s6, stacked, mp6, 2, 4);
  for (size_t m = 0; m < 6; ++m){
    CHECK(flat1[m].result == stacked[m].result);
    CHECK(flat1[m].ticks == stacked[m].ticks);
  }
}

TEST_CASE("evolve --init-from widens stack-1 founders into a stacked run"){
  std::filesystem::create_directories("/tmp/aipong_stack_founders");
  std::vector<int> s1{6, 32, 32, 3};
  std::vector<double> f1;
  for (size_t i = 0; i < (size_t)(6*32+32 + 32*32+32 + 32*3+3); ++i)
    f1.push_back(0.01 * double((i*3) % 7) - 0.03);
  { std::ofstream f("/tmp/aipong_stack_founders/gen_0000.txt"); writeModelText(f, s1, f1); }

  EvolveConfig cfg;
  cfg.pop = 6; cfg.gens = 1; cfg.gamesPerEval = 2; cfg.evalEvery = 1; cfg.evalMatches = 2;
  cfg.hofEvery = 1; cfg.seed = 4; cfg.threads = 2;
  cfg.stack = 6; cfg.stride = 6;
  cfg.initFrom = "/tmp/aipong_stack_founders";
  cfg.outDir = "/tmp/aipong_stack_out";
  std::filesystem::remove_all(cfg.outDir);
  EvolveResult res = evolve(cfg);
  CHECK(res.sizes.front() == 36);
  MlpAgent chk("chk");
  REQUIRE(chk.loadFile("/tmp/aipong_stack_out/best.txt"));
  CHECK(chk.stack() == 6);
  CHECK(chk.stride() == 6);
}

TEST_CASE("BatchedMlp matches the scalar agent on well-separated decisions"){
  std::vector<int> sizes{6, 8, 3};
  std::vector<double> flat;
  for (int i = 0; i < 6*8 + 8 + 8*3 + 3; ++i) flat.push_back(0.11 * ((i * 13) % 9) - 0.4);
  MlpAgent scalar("s");
  REQUIRE(scalar.loadWeights(sizes, flat));
  BatchedMlp batched(sizes, 4);
  for (int l = 0; l < 4; ++l) batched.setLane(l, flat);
  State s = testutil::playing();
  uint64_t r = 777;
  int checked = 0;
  for (int t = 0; t < 200; ++t){
    r = r * 6364136223846793005ull + 1442695040888963407ull;
    s.ball.pos = {0.05 + 0.9 * double((r >> 11) % 1000) / 1000.0,
                  0.05 + 0.9 * double((r >> 21) % 1000) / 1000.0};
    s.ball.vel = {((r >> 31) % 2 ? 1.0 : -1.0) * 0.8, (double((r >> 33) % 200) / 100.0) - 1.0};
    s.left.y = 0.1 + 0.8 * double((r >> 41) % 100) / 100.0;
    s.right.y = 0.1 + 0.8 * double((r >> 51) % 100) / 100.0;
    auto obs = observation(view(s), Side::Left);
    float o[4 * 6];
    for (int lane = 0; lane < 4; ++lane)
      for (int j = 0; j < 6; ++j) o[lane * 6 + j] = (float)obs[(size_t)j];
    Move mv[4];
    batched.forward(o, mv);
    Move ref = scalar.act(view(s), Side::Left);
    for (int lane = 0; lane < 4; ++lane) CHECK(mv[lane] == ref);
    checked++;
  }
  CHECK(checked == 200);
}

TEST_CASE("playMatchesBatched: identical results for any thread count and lane width"){
  std::vector<int> sizes{6, 32, 32, 3};
  std::vector<double> a, b;
  for (int i = 0; i < 6*32 + 32 + 32*32 + 32 + 32*3 + 3; ++i){
    a.push_back(0.02 * ((i * 7) % 11) - 0.1);
    b.push_back(0.1 - 0.02 * ((i * 5) % 13));
  }
  auto make = [&](){
    std::vector<GenomeMatch> gms(12);
    for (int m = 0; m < 12; ++m){
      gms[(size_t)m].self = (m % 2) ? &a : &b;
      gms[(size_t)m].opp  = (m % 2) ? &b : &a;
      gms[(size_t)m].seed = (unsigned)(900 + m);
    }
    return gms;
  };
  MatchParams mp; mp.rallyCap = 300; mp.winScore = 5;   // exercise the knobs too
  auto g1 = make(); playMatchesBatched(sizes, g1, mp, 1, 4);
  auto g2 = make(); playMatchesBatched(sizes, g2, mp, 4, 16);
  for (size_t m = 0; m < g1.size(); ++m){
    CHECK(g1[m].result == g2[m].result);               // bitwise — lane/thread independent
    CHECK(g1[m].ticks == g2[m].ticks);
  }
}

TEST_CASE("population files round-trip exactly; checkpoints parse back into founders"){
  std::vector<int> sizes{6, 8, 3};
  std::vector<std::vector<double>> pop;
  for (int g = 0; g < 3; ++g){
    std::vector<double> v;
    for (int i = 0; i < 6*8 + 8 + 8*3 + 3; ++i) v.push_back(0.001 * (i * (g + 3) % 97) - 0.02);
    pop.push_back(std::move(v));
  }
  std::ostringstream out;
  writePopulation(out, sizes, pop);
  std::istringstream in(out.str());
  std::vector<int> sizes2; std::vector<std::vector<double>> pop2;
  REQUIRE(readPopulation(in, sizes2, pop2));
  CHECK(sizes2 == sizes);
  REQUIRE(pop2.size() == pop.size());
  for (size_t g = 0; g < pop.size(); ++g) CHECK(pop2[g] == pop[g]);   // bit-exact (%.17g)

  const char* mp = "/tmp/aipong_founder_test.txt";
  { std::ofstream f(mp); writeModelText(f, sizes, pop[1]); }
  std::vector<int> ms; std::vector<double> mf;
  REQUIRE(readModelFlat(mp, ms, mf));
  CHECK(ms == sizes);
  REQUIRE(mf.size() == pop[1].size());
  for (size_t i = 0; i < mf.size(); ++i) CHECK(mf[i] == doctest::Approx(pop[1][i]));  // %.9g file
}

TEST_CASE("evolve --resume continues from a saved population deterministically"){
  EvolveConfig a;
  a.pop = 6; a.gens = 2; a.gamesPerEval = 2; a.hofEvery = 1; a.evalEvery = 2;  // eval on last gen → saves population
  a.seed = 5; a.threads = 2; a.evalMatches = 2;
  a.outDir = "/tmp/aipong_resume_test";
  std::filesystem::remove_all(a.outDir);
  EvolveResult r1 = evolve(a);
  REQUIRE(std::filesystem::exists(a.outDir + "/population.txt"));
  EvolveConfig b = a;
  b.resumeDir = a.outDir; b.outDir = ""; b.gens = 1; b.evalEvery = 0;
  EvolveResult r2 = evolve(b);
  REQUIRE_FALSE(r2.bestFlat.empty());
  // resumed run starts from the saved population, not random: its gen-0 best should not
  // collapse back to a random-population fitness profile — check mechanically instead:
  std::ifstream pf(a.outDir + "/population.txt");
  std::vector<int> ps; std::vector<std::vector<double>> saved;
  REQUIRE(readPopulation(pf, ps, saved));
  CHECK(saved.size() == (size_t)a.pop);
  bool bestIsFromSaved = false;                        // elite survives unchanged in a 1-gen run
  for (const auto& g : saved) if (g == r2.bestFlat) bestIsFromSaved = true;
  CHECK(bestIsFromSaved);
}

TEST_CASE("readControl: token-file semantics, absent or junk means Run"){
  const std::string dir = "/tmp/aipong_ctl_read";
  std::filesystem::create_directories(dir);
  std::filesystem::remove(dir + "/control");
  CHECK(readControl(dir) == Control::Run);
  auto put = [&](const char* s){ std::ofstream f(dir + "/control"); f << s; };
  put("pause");     CHECK(readControl(dir) == Control::Pause);
  put("stop\n");    CHECK(readControl(dir) == Control::Stop);    // trailing whitespace ok
  put("  eval ");   CHECK(readControl(dir) == Control::Eval);
  put("run");       CHECK(readControl(dir) == Control::Run);
  put("gibberish"); CHECK(readControl(dir) == Control::Run);
}

TEST_CASE("control: stop exits after the current generation with continuation state saved"){
  EvolveConfig cfg;
  cfg.pop = 4; cfg.gens = 5; cfg.gamesPerEval = 1; cfg.hofEvery = 1; cfg.evalEvery = 0;
  cfg.seed = 21; cfg.threads = 2; cfg.outDir = "/tmp/aipong_ctl_stop";
  std::filesystem::remove_all(cfg.outDir);
  EvolveResult r = evolve(cfg, [&](const GenStats& s){
    if (s.gen == 1){ std::ofstream f(cfg.outDir + "/control"); f << "stop"; }
  });
  CHECK(r.log.size() == 2);                            // gens 0,1 ran; gen-2 top saw stop
  CHECK(std::filesystem::exists(cfg.outDir + "/population.txt"));  // lossless --resume state
  CHECK(std::filesystem::exists(cfg.outDir + "/hof.txt"));
  std::ifstream st(cfg.outDir + "/status.json");
  const std::string s((std::istreambuf_iterator<char>(st)), std::istreambuf_iterator<char>());
  CHECK(s.find("\"state\":\"stopped\"") != std::string::npos);
  CHECK(r.stopped);
}

TEST_CASE("control: a stale control file is removed at startup"){
  EvolveConfig cfg;
  cfg.pop = 4; cfg.gens = 2; cfg.gamesPerEval = 1; cfg.hofEvery = 1; cfg.evalEvery = 0;
  cfg.seed = 22; cfg.threads = 2; cfg.outDir = "/tmp/aipong_ctl_stale";
  std::filesystem::remove_all(cfg.outDir);
  std::filesystem::create_directories(cfg.outDir);
  { std::ofstream f(cfg.outDir + "/control"); f << "stop"; }        // leftover from a past run
  EvolveResult r = evolve(cfg);
  CHECK(r.log.size() == 2);                            // ran to completion, not stopped
  CHECK_FALSE(std::filesystem::exists(cfg.outDir + "/control"));
}

TEST_CASE("control: eval-now forces one anchor eval + checkpoint, then self-clears"){
  EvolveConfig cfg;
  cfg.pop = 4; cfg.gens = 3; cfg.gamesPerEval = 1; cfg.hofEvery = 1;
  cfg.evalEvery = 0; cfg.evalMatches = 1;              // would never eval on its own
  cfg.seed = 23; cfg.threads = 2; cfg.outDir = "/tmp/aipong_ctl_eval";
  std::filesystem::remove_all(cfg.outDir);
  std::vector<double> vs;
  evolve(cfg, [&](const GenStats& s){
    if (s.gen == 0){ std::ofstream f(cfg.outDir + "/control"); f << "eval"; }
    vs.push_back(s.vsPClassic);
  });
  REQUIRE(vs.size() == 3);
  CHECK(vs[1] >= 0.0);                                 // gen 1 was force-evaluated
  CHECK(std::filesystem::exists(cfg.outDir + "/gen_0001.txt"));
  CHECK(readControl(cfg.outDir) != Control::Eval);     // self-cleared
  CHECK(vs[2] < -1.5);                                 // one-shot: gen 2 back to no evals
}

TEST_CASE("control: pause parks the run, resumes bitwise-identical, no RNG drawn while parked"){
  auto run = [&](const std::string& out, bool paused){
    EvolveConfig cfg;
    cfg.pop = 4; cfg.gens = 3; cfg.gamesPerEval = 1; cfg.hofEvery = 1; cfg.evalEvery = 0;
    cfg.seed = 24; cfg.threads = 2; cfg.outDir = out;
    std::filesystem::remove_all(out);
    std::thread resumer;
    const auto t0 = std::chrono::steady_clock::now();
    EvolveResult r = evolve(cfg, [&](const GenStats& s){
      if (paused && s.gen == 0){
        { std::ofstream f(out + "/control"); f << "pause"; }
        resumer = std::thread([out]{
          std::this_thread::sleep_for(std::chrono::milliseconds(400));
          std::ofstream f(out + "/control"); f << "run";
        });
      }
    });
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (resumer.joinable()) resumer.join();
    return std::make_pair(r, sec);
  };
  auto [base, baseSec] = run("/tmp/aipong_ctl_base", false);
  auto [pausd, pausedSec] = run("/tmp/aipong_ctl_pause", true);
  REQUIRE(base.log.size() == pausd.log.size());
  for (size_t g = 0; g < base.log.size(); ++g){
    CHECK(base.log[g].best == pausd.log[g].best);      // pause changes wall time, nothing else
    CHECK(base.log[g].mean == pausd.log[g].mean);
  }
  CHECK(base.bestFlat == pausd.bestFlat);
  CHECK(pausedSec >= baseSec + 0.3);                   // it genuinely parked (~400 ms hold)
}

TEST_CASE("telemetry: per-class fitness means and champion age are logged"){
  std::filesystem::create_directories("/tmp/aipong_pool2");
  std::vector<int> s1{6, 32, 32, 3};
  std::vector<double> f1;
  for (size_t i = 0; i < (size_t)(6*32 + 32 + 32*32 + 32 + 32*3 + 3); ++i)
    f1.push_back(0.01 * double((i * 5) % 9) - 0.02);
  { std::ofstream f("/tmp/aipong_pool2/gen_0000.txt"); writeModelText(f, s1, f1); }

  EvolveConfig cfg;
  cfg.pop = 6; cfg.gens = 2; cfg.gamesPerEval = 4;
  cfg.poolGames = 1; cfg.anchorGames = 1;
  cfg.poolDirs = {"/tmp/aipong_pool2"};
  cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 31; cfg.threads = 2; cfg.outDir = "";
  EvolveResult r = evolve(cfg);
  for (const auto& s : r.log){
    CHECK(s.fitPeer   >= -1.0); CHECK(s.fitPeer   <= 1.0);   // all three classes played
    CHECK(s.fitPool   >= -1.0); CHECK(s.fitPool   <= 1.0);
    CHECK(s.fitAnchor >= -1.0); CHECK(s.fitAnchor <= 1.0);
    CHECK(s.champAge  >= 0);    CHECK(s.champAge  <= s.gen);
  }
  EvolveConfig plain;                                   // peers only → sentinels for the rest
  plain.pop = 4; plain.gens = 1; plain.gamesPerEval = 1; plain.hofEvery = 1;
  plain.evalEvery = 0; plain.seed = 32; plain.threads = 1; plain.outDir = "";
  EvolveResult p = evolve(plain);
  REQUIRE(p.log.size() == 1);
  CHECK(p.log[0].fitPeer >= -1.0);
  CHECK(p.log[0].fitPool < -1.5);
  CHECK(p.log[0].fitAnchor < -1.5);
}

TEST_CASE("exported best genome round-trips through the file format and plays in a MatchRunner"){
  EvolveConfig cfg;
  cfg.pop = 6; cfg.gens = 2; cfg.gamesPerEval = 2;
  cfg.evalEvery = 0; cfg.hofEvery = 1; cfg.seed = 3; cfg.outDir = ""; cfg.threads = 2;
  EvolveResult res = evolve(cfg);
  REQUIRE_FALSE(res.bestFlat.empty());
  const char* path = "/tmp/aipong_evo_best_test.txt";
  { std::ofstream f(path); writeModelText(f, res.sizes, res.bestFlat); }

  MatchRunner r(9);
  r.setController(Side::Left, std::string("mlp:") + path);
  r.setController(Side::Right, "p:easy");
  CHECK_FALSE(r.humanControlled(Side::Left));                   // file loaded successfully
  r.newGame(9);
  r.runToCompletion(2'000'000);
  CHECK(r.snapshot().phase == Phase::GameOver);                 // a real, terminating match
}

TEST_CASE("recipes: line format parses; unknown keys ignored; bad trainer invalid"){
  const std::string txt =
      "# comment\n"
      "desc: GA, all bells and whistles\n"
      "trainer: evolve\n"
      "out: models/evomem\n"
      "future_key: ignored\n"
      "args: --stack 6 --stride 6\n"
      "args: --pop 128 --games 8\n";
  Recipe r = parseRecipeText("evomem", txt);
  CHECK(r.valid);
  CHECK(r.name == "evomem");
  CHECK(r.desc == "GA, all bells and whistles");
  CHECK(r.trainer == "evolve");
  CHECK(r.out == "models/evomem");
  REQUIRE(r.args.size() == 8);
  CHECK(r.args[0] == "--stack");
  CHECK(r.args[7] == "8");                       // args: lines concatenate in order
  CHECK_FALSE(parseRecipeText("x", "trainer: vim\nargs: --gens 1\n").valid);
  CHECK_FALSE(parseRecipeText("x", "desc: no trainer line\n").valid);
  // The bash executor's launch queue refuses any request with an empty --out (pong:22's
  // load_recipe gate); Recipe::valid must agree, or a hand-authored .recipe missing `out:`
  // shows up as a normal, launchable entry in the GUI only to be refused at launch time.
  CHECK_FALSE(parseRecipeText("x", "trainer: evolve\nargs: --gens 1\n").valid);
}

TEST_CASE("recipes: discoverRecipes scans a dir, sorts, skips invalid"){
  const std::string dir = "/tmp/aipong_recipes";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  { std::ofstream f(dir + "/b.recipe");  f << "trainer: ppo\nout: models/b\nargs: --smoke\n"; }
  { std::ofstream f(dir + "/a.recipe");  f << "desc: basic\ntrainer: evolve\nout: models/a\nargs: --gens 300\n"; }
  { std::ofstream f(dir + "/broken.recipe"); f << "desc: no trainer\n"; }
  { std::ofstream f(dir + "/notes.txt"); f << "not a recipe\n"; }
  auto rs = discoverRecipes(dir);
  REQUIRE(rs.size() == 2);
  CHECK(rs[0].name == "a");
  CHECK(rs[1].name == "b");
  CHECK(discoverRecipes("/tmp/aipong_no_such_dir").empty());
}

TEST_CASE("writeRunConfig: provenance file lands in the run dir"){
  const std::string dir = "/tmp/aipong_cfgprov";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  writeRunConfig(dir, "evolve", "--gens 300 --seed 1 --out " + dir);
  std::ifstream f(dir + "/config.json");
  const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  CHECK(s.find("\"trainer\":\"evolve\"") != std::string::npos);
  CHECK(s.find("--gens 300 --seed 1") != std::string::npos);
  CHECK(s.find("started_unix") != std::string::npos);
}

TEST_CASE("evolve --resume: a failed population load refuses loudly instead of random-restarting"){
  EvolveConfig cfg;
  cfg.pop = 4; cfg.gens = 2; cfg.gamesPerEval = 1; cfg.hofEvery = 1; cfg.evalEvery = 0;
  cfg.seed = 41; cfg.threads = 1; cfg.outDir = "";
  cfg.resumeDir = "/tmp/aipong_no_such_resume";        // nothing there
  EvolveResult r = evolve(cfg);
  CHECK(r.resumeFailed);                               // the Decision-20 follow-up: fail LOUDLY
  CHECK(r.log.empty());                                // no silent random-founders training
}

TEST_CASE("evolve --resume: a CORRUPTED population file sets resumeFailed, never bad_allocs"){
  namespace fs = std::filesystem;
  const std::string dir = "/tmp/aipong_corrupt_resume";
  fs::remove_all(dir);
  fs::create_directories(dir);
  { std::ofstream f(dir + "/population.txt");            // absurd sizes line
    f << "aipong-pop 1\n4\n6 2000000000 3\n0.1 0.2\n"; }
  EvolveConfig cfg;
  cfg.pop = 4; cfg.gens = 1; cfg.gamesPerEval = 1; cfg.hofEvery = 1; cfg.evalEvery = 0;
  cfg.seed = 43; cfg.threads = 1; cfg.outDir = ""; cfg.resumeDir = dir;
  EvolveResult r = evolve(cfg);
  CHECK(r.resumeFailed);
  CHECK(r.log.empty());
  { std::ofstream f(dir + "/population.txt");            // absurd genome COUNT header
    f << "aipong-pop 1\n999999999\n6 8 3\n0.1\n"; }
  EvolveResult r2 = evolve(cfg);
  CHECK(r2.resumeFailed);
  fs::remove_all(dir);
}

TEST_CASE("evolve --init-from: a malformed checkpoint among good ones is skipped, run proceeds"){
  namespace fs = std::filesystem;
  const std::string dir = "/tmp/aipong_initfrom_mixed";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::vector<int> s1{6, 8, 3};
  std::vector<double> f1;
  for (int i = 0; i < 6*8 + 8 + 8*3 + 3; ++i) f1.push_back(0.01 * (i % 7) - 0.03);
  { std::ofstream f(dir + "/gen_0000.txt"); writeModelText(f, s1, f1); }     // good founder
  { std::ofstream f(dir + "/gen_0001.txt");                                   // malformed
    f << "aipong-mlp 1\n6 2000000000 3\n0.5\n"; }
  EvolveConfig cfg;
  cfg.sizes = s1;
  cfg.pop = 4; cfg.gens = 1; cfg.gamesPerEval = 1; cfg.hofEvery = 1; cfg.evalEvery = 0;
  cfg.seed = 44; cfg.threads = 1; cfg.outDir = ""; cfg.initFrom = dir;
  EvolveResult r = evolve(cfg);
  REQUIRE(r.log.size() == 1);                            // ran (didn't crash on the bad file)
  CHECK_FALSE(r.resumeFailed);
  fs::remove_all(dir);
}

TEST_CASE("recipes: trailing whitespace on scalar fields is trimmed"){
  Recipe r = parseRecipeText("x", "trainer: evolve  \nout: models/x \nargs: --gens 1\n");
  CHECK(r.valid);                                      // trailing spaces must not break validity
  CHECK(r.trainer == "evolve");
  CHECK(r.out == "models/x");
}

TEST_CASE("writeRunConfig: quotes and backslashes in argv are JSON-escaped"){
  const std::string dir = "/tmp/aipong_cfg_esc";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  writeRunConfig(dir, "evolve", "--note \"q\" back\\slash");
  std::ifstream f(dir + "/config.json");
  const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  CHECK(s.find("\\\"q\\\"") != std::string::npos);     // escaped quote survives
  CHECK(s.find("back\\\\slash") != std::string::npos); // escaped backslash survives
}

TEST_CASE("playMatchVsGenome: cross-architecture match runs, in bounds, deterministic"){
  std::vector<int> sA{6, 32, 32, 3};          // base arch
  std::vector<int> sB{6, 64, 64, 64, 3};      // deeper+wider arch
  std::vector<double> gA = randomGenome(sA, 101);
  std::vector<double> gB = randomGenome(sB, 202);
  REQUIRE(gA.size() == flatSize(sA));
  REQUIRE(gB.size() == flatSize(sB));
  double m1 = playMatchVsGenome(sA, gA, sB, gB, 321, {});
  double m2 = playMatchVsGenome(sA, gA, sB, gB, 321, {});
  CHECK(m1 == m2);                             // pure function of (sizes, genomes, seed, params)
  CHECK(m1 >= -1.0); CHECK(m1 <= 1.0);
  double n1 = playMatchVsGenome(sB, gB, sA, gA, 321, {});   // swapped sides also valid
  CHECK(n1 >= -1.0); CHECK(n1 <= 1.0);
}

TEST_CASE("flatSize and randomGenome are exposed and sized correctly"){
  CHECK(flatSize({6, 32, 32, 3}) == (size_t)(6*32 + 32 + 32*32 + 32 + 32*3 + 3));
  CHECK(flatSize({36, 64, 64, 64, 3}) == (size_t)(36*64 + 64 + 64*64 + 64 + 64*64 + 64 + 64*3 + 3));
  auto g = randomGenome({6, 8, 3}, 42);
  CHECK(g.size() == flatSize({6, 8, 3}));
  CHECK(randomGenome({6, 8, 3}, 42) == g);    // deterministic for a given key
  CHECK(randomGenome({6, 8, 3}, 43) != g);    // different key → different genome
}

TEST_CASE("archLabel: hidden widths only, dash-joined"){
  CHECK(archLabel({6, 32, 32, 3}) == "h32-32");
  CHECK(archLabel({36, 64, 64, 64, 3}) == "h64-64-64");
  CHECK(archLabel({6, 16, 3}) == "h16");
}

TEST_CASE("coEvolve is deterministic across thread counts"){
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};    // two distinct small archs
  cfg.pop = 6; cfg.gens = 3;
  cfg.selfGames = 2; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 7; cfg.outDir = "";
  cfg.threads = 1; CoEvolveResult r1 = coEvolve(cfg);
  cfg.threads = 4; CoEvolveResult r2 = coEvolve(cfg);
  REQUIRE(r1.log.size() == r2.log.size());
  for (size_t g = 0; g < r1.log.size(); ++g){
    REQUIRE(r1.log[g].islands.size() == r2.log[g].islands.size());
    for (size_t I = 0; I < r1.log[g].islands.size(); ++I){
      CHECK(r1.log[g].islands[I].best == r2.log[g].islands[I].best);   // bit-identical, mixed paths
      CHECK(r1.log[g].islands[I].mean == r2.log[g].islands[I].mean);
    }
  }
  REQUIRE(r1.bestFlat.size() == r2.bestFlat.size());
  for (size_t I = 0; I < r1.bestFlat.size(); ++I)
    CHECK(r1.bestFlat[I] == r2.bestFlat[I]);        // bit-identical champion genomes
}

TEST_CASE("coEvolve preserves every island's lineage (no crowding-out)"){
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}, {6, 6, 3}};
  cfg.pop = 6; cfg.gens = 4;
  cfg.selfGames = 2; cfg.crossGames = 2; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 9; cfg.outDir = "";
  CoEvolveResult r = coEvolve(cfg);
  REQUIRE(r.bestFlat.size() == 3);
  for (size_t I = 0; I < 3; ++I){
    CHECK(r.bestFlat[I].size() == flatSize(r.sizes[I]));   // each arch still has a champion of ITS shape
    CHECK(r.labels[I] == archLabel(cfg.islandSizes[I]));
  }
  for (const auto& gs : r.log)
    for (const auto& is : gs.islands){                     // shaped fitness stays in range
      CHECK(is.best >= -1.0); CHECK(is.best <= 1.0);
      CHECK(is.mean >= -1.0); CHECK(is.mean <= 1.0);
    }
}

TEST_CASE("coEvolve: cross-island play changes the trajectory"){
  auto run = [](int crossGames){
    CoEvolveConfig cfg;
    cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
    cfg.pop = 6; cfg.gens = 3;
    cfg.selfGames = 2; cfg.crossGames = crossGames; cfg.anchorGames = 1;
    cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 7; cfg.outDir = "";
    return coEvolve(cfg);
  };
  CoEvolveResult withCross = run(2);
  CoEvolveResult without   = run(0);
  bool differs = false;                                    // live cross-opponents shift fitness
  for (size_t g = 0; g < withCross.log.size(); ++g)
    for (size_t I = 0; I < withCross.log[g].islands.size(); ++I)
      if (withCross.log[g].islands[I].mean != without.log[g].islands[I].mean) differs = true;
  CHECK(differs);
}

TEST_CASE("coEvolve: eval gens populate per-island anchor win-rates and a cross-island matrix"){
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  cfg.pop = 6; cfg.gens = 2;
  cfg.selfGames = 2; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 2; cfg.evalMatches = 2;   // eval on the last gen
  cfg.seed = 7; cfg.threads = 2; cfg.outDir = "";
  CoGenStats lastEval;
  coEvolve(cfg, [&](const CoGenStats& s){ if (!s.islands.empty() && s.islands[0].vsPClassic > -1.5) lastEval = s; });
  REQUIRE(lastEval.islands.size() == 2);
  for (const auto& is : lastEval.islands){
    CHECK(is.vsPClassic    >= 0.0); CHECK(is.vsPClassic    <= 1.0);
    CHECK(is.vsLaggy       >= 0.0); CHECK(is.vsLaggy       <= 1.0);
    CHECK(is.vsInterceptor >= 0.0); CHECK(is.vsInterceptor <= 1.0);
    CHECK(is.fitSelf   >= -1.0); CHECK(is.fitSelf   <= 1.0);   // per-class telemetry present
    CHECK(is.fitCross  >= -1.0); CHECK(is.fitCross  <= 1.0);
    CHECK(is.fitAnchor >= -1.0); CHECK(is.fitAnchor <= 1.0);
  }
  REQUIRE(lastEval.crossWin.size() == 2);                      // N×N matrix, diagonal 0
  REQUIRE(lastEval.crossWin[0].size() == 2);
  CHECK(lastEval.crossWin[0][0] == 0.0);
  CHECK(lastEval.crossWin[1][1] == 0.0);
  CHECK(lastEval.crossWin[0][1] >= -1.0); CHECK(lastEval.crossWin[0][1] <= 1.0);
}

TEST_CASE("coEvolve: writes per-island run dirs the dashboard and ladder can read"){
  namespace fs = std::filesystem;
  const std::string base = "/tmp/aipong_coevo_io";
  fs::remove_all(base);
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  cfg.pop = 6; cfg.gens = 2;
  cfg.selfGames = 2; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 2; cfg.evalMatches = 2;
  cfg.seed = 7; cfg.threads = 2; cfg.outDir = base;
  coEvolve(cfg);
  for (const std::string label : {"h8-8", "h12"}){
    const std::string d = base + "/" + label;
    CHECK(fs::exists(d + "/best.txt"));
    CHECK(fs::exists(d + "/population.txt"));
    CHECK(fs::exists(d + "/hof.txt"));
    CHECK(fs::exists(d + "/train_log.jsonl"));
    std::ifstream lf(d + "/train_log.jsonl");
    const std::string log((std::istreambuf_iterator<char>(lf)), std::istreambuf_iterator<char>());
    CHECK(log.find("\"vs_p_classic\"") != std::string::npos);   // evolve() log schema → trainview works
    CHECK(log.find("\"fit_pool\"")     != std::string::npos);
    MlpAgent chk("chk");
    REQUIRE(chk.loadFile(d + "/best.txt"));                      // champion round-trips as a model file
  }
  CHECK(fs::exists(base + "/crosswin.jsonl"));                   // cross-island matrix artifact
  fs::remove_all(base);
}

TEST_CASE("coEvolve --resume: warm-starts every island, refuses loudly on a missing island"){
  namespace fs = std::filesystem;
  const std::string base = "/tmp/aipong_coevo_resume";
  fs::remove_all(base);
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  cfg.pop = 6; cfg.gens = 2;
  cfg.selfGames = 2; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 2; cfg.evalMatches = 2;
  cfg.seed = 7; cfg.threads = 2; cfg.outDir = base;
  coEvolve(cfg);                                             // writes base/h8-8, base/h12 with population.txt

  CoEvolveConfig rc = cfg;                                   // resume: continue from the saved populations
  rc.outDir = ""; rc.resumeDir = base; rc.gens = 1; rc.threads = 1;
  CoEvolveResult r1 = coEvolve(rc);
  CHECK_FALSE(r1.resumeFailed);
  REQUIRE(r1.bestFlat.size() == 2);
  CHECK(r1.bestFlat[0].size() == flatSize(r1.sizes[0]));     // resumed champion has the island's shape
  rc.threads = 4;
  CoEvolveResult r2 = coEvolve(rc);
  CHECK(r2.bestFlat == r1.bestFlat);                         // resume is deterministic across threads

  CoEvolveConfig miss = cfg;                                 // one island's dir absent → refuse loudly
  miss.outDir = ""; miss.resumeDir = base;
  miss.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}, {6, 6, 3}};  // "h6" was never saved
  CoEvolveResult r3 = coEvolve(miss);
  CHECK(r3.resumeFailed);
  CHECK(r3.log.empty());
  fs::remove_all(base);
}

TEST_CASE("coEvolve resume grow-pop pads with DISTINCT genomes, not identical clones"){
  namespace fs = std::filesystem;
  const std::string a = "/tmp/aipong_coevo_grow_a", b = "/tmp/aipong_coevo_grow_b";
  fs::remove_all(a); fs::remove_all(b);
  CoEvolveConfig save;
  save.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  save.pop = 4; save.gens = 1;
  save.selfGames = 2; save.crossGames = 1; save.anchorGames = 1;
  save.hofEvery = 1; save.evalEvery = 1; save.evalMatches = 2;
  save.seed = 7; save.threads = 2; save.outDir = a;
  coEvolve(save);                                        // writes a/h8-8/population.txt (4 genomes)

  CoEvolveConfig grow = save;                            // resume with a LARGER pop → 4 loaded + 4 padded
  grow.pop = 8; grow.resumeDir = a; grow.outDir = b;
  CoEvolveResult r = coEvolve(grow);
  CHECK_FALSE(r.resumeFailed);

  std::ifstream pf(b + "/h8-8/population.txt");
  std::vector<int> ps; std::vector<std::vector<double>> pop;
  REQUIRE(readPopulation(pf, ps, pop));
  REQUIRE(pop.size() == 8);                              // 4 loaded + 4 padded
  CHECK(pop[4] != pop[5]);                               // padded clones are DISTINCT (old bug: identical)
  CHECK(pop[5] != pop[6]);
  CHECK(pop[6] != pop[7]);
  fs::remove_all(a); fs::remove_all(b);
}

TEST_CASE("coEvolve logs marg_* — avg unshaped margin vs bots at eval gens"){
  namespace fs = std::filesystem;
  const std::string base = "/tmp/aipong_coevo_marg";
  fs::remove_all(base);
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  cfg.pop = 6; cfg.gens = 2;
  cfg.selfGames = 2; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 2; cfg.evalMatches = 4;   // eval on the last gen
  cfg.seed = 7; cfg.threads = 2; cfg.outDir = base;
  CoGenStats lastEval;
  coEvolve(cfg, [&](const CoGenStats& s){ if (!s.islands.empty() && s.islands[0].vsInterceptor > -1.5) lastEval = s; });
  REQUIRE(lastEval.islands.size() == 2);
  for (const auto& is : lastEval.islands){
    CHECK(is.margPClassic    >= -1.0); CHECK(is.margPClassic    <= 1.0);
    CHECK(is.margLaggy       >= -1.0); CHECK(is.margLaggy       <= 1.0);
    CHECK(is.margInterceptor >= -1.0); CHECK(is.margInterceptor <= 1.0);
  }
  std::ifstream lf(base + "/h8-8/train_log.jsonl");
  const std::string log((std::istreambuf_iterator<char>(lf)), std::istreambuf_iterator<char>());
  CHECK(log.find("\"marg_p_classic\"")   != std::string::npos);
  CHECK(log.find("\"marg_interceptor\"") != std::string::npos);
  fs::remove_all(base);
}

TEST_CASE("coEvolve control: stop saves every island's checkpoint and sets stopped"){
  namespace fs = std::filesystem;
  const std::string base = "/tmp/aipong_coevo_stop";
  fs::remove_all(base);
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  cfg.pop = 6; cfg.gens = 5;
  cfg.selfGames = 1; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 7; cfg.threads = 2; cfg.outDir = base;
  CoEvolveResult r = coEvolve(cfg, [&](const CoGenStats& s){
    if (s.gen == 1){ std::ofstream f(base + "/control"); f << "stop"; }
  });
  CHECK(r.stopped);
  CHECK(r.log.size() == 2);                             // gens 0,1 ran; gen-2 top saw stop
  CHECK(fs::exists(base + "/h8-8/population.txt"));      // every island's --resume state saved
  CHECK(fs::exists(base + "/h12/hof.txt"));
  CHECK_FALSE(fs::exists(base + "/h8-8/population.txt.tmp"));   // atomic write renamed the temp away
  CHECK_FALSE(fs::exists(base + "/h12/hof.txt.tmp"));
  std::ifstream st(base + "/status.json");
  const std::string s((std::istreambuf_iterator<char>(st)), std::istreambuf_iterator<char>());
  CHECK(s.find("\"state\":\"stopped\"") != std::string::npos);
  fs::remove_all(base);
}

TEST_CASE("coEvolve control: a stale control file is removed at startup"){
  namespace fs = std::filesystem;
  const std::string base = "/tmp/aipong_coevo_stale";
  fs::remove_all(base); fs::create_directories(base);
  { std::ofstream f(base + "/control"); f << "stop"; }   // leftover from a past run
  CoEvolveConfig cfg;
  cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
  cfg.pop = 6; cfg.gens = 2;
  cfg.selfGames = 1; cfg.crossGames = 1; cfg.anchorGames = 1;
  cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 7; cfg.threads = 2; cfg.outDir = base;
  CoEvolveResult r = coEvolve(cfg);
  CHECK_FALSE(r.stopped);                                // ran to completion, not stopped
  CHECK(r.log.size() == 2);
  CHECK_FALSE(fs::exists(base + "/control"));
  fs::remove_all(base);
}

TEST_CASE("coEvolve control: pause parks the run, resumes bit-identically"){
  namespace fs = std::filesystem;
  auto run = [&](const std::string& base, bool paused){
    fs::remove_all(base);
    CoEvolveConfig cfg;
    cfg.islandSizes = {{6, 8, 8, 3}, {6, 12, 3}};
    cfg.pop = 6; cfg.gens = 3;
    cfg.selfGames = 1; cfg.crossGames = 1; cfg.anchorGames = 1;
    cfg.hofEvery = 1; cfg.evalEvery = 0; cfg.seed = 7; cfg.threads = 2; cfg.outDir = base;
    std::thread resumer;
    CoEvolveResult r = coEvolve(cfg, [&](const CoGenStats& s){
      if (paused && s.gen == 0){
        { std::ofstream f(base + "/control"); f << "pause"; }
        resumer = std::thread([base]{
          std::this_thread::sleep_for(std::chrono::milliseconds(400));
          std::ofstream f(base + "/control"); f << "run";
        });
      }
    });
    if (resumer.joinable()) resumer.join();
    return r;
  };
  CoEvolveResult a = run("/tmp/aipong_coevo_p0", false);
  CoEvolveResult b = run("/tmp/aipong_coevo_p1", true);
  REQUIRE(a.log.size() == b.log.size());
  for (size_t g = 0; g < a.log.size(); ++g)
    for (size_t I = 0; I < a.log[g].islands.size(); ++I)
      CHECK(a.log[g].islands[I].best == b.log[g].islands[I].best);   // pause changes wall time, nothing else
  for (size_t I = 0; I < a.bestFlat.size(); ++I) CHECK(a.bestFlat[I] == b.bestFlat[I]);
  fs::remove_all("/tmp/aipong_coevo_p0"); fs::remove_all("/tmp/aipong_coevo_p1");
}
