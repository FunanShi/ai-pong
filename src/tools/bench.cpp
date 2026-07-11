// bench — single-threaded sim-throughput microbenchmark for perf A/B (DECISIONS.md #11).
// Workloads chosen to bracket the trainer's cost profile:
//   interceptor vs interceptor  — rally-cap-bound, lives at the speed ceiling: worst-case
//                                 substep load (this is where generations go to die)
//   p:classic  vs interceptor   — mixed-speed, realistic competitive match
//   p:classic  vs p:classic     — low-speed baseline (agent-forward dominated)
#include "match_runner.hpp"
#include "evolve.hpp"
#include "batch_forward.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
using namespace pong;

static void run(const char* name, const char* l, const char* r, int matches){
  auto t0 = std::chrono::steady_clock::now();
  long ticks = 0;
  for (int m = 0; m < matches; ++m){
    MatchRunner mr((unsigned)(100 + m));
    mr.setController(Side::Left, l);
    mr.setController(Side::Right, r);
    mr.newGame((unsigned)(100 + m));
    mr.runToCompletion();
    ticks += mr.ticks();
  }
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%-28s %3d matches %10ld ticks %7.2fs %9.0f ticks/s\n",
              name, matches, ticks, sec, ticks / sec);
}

// The trainer's actual hot path: MLP genomes on both paddles (allocation-heavy today),
// run single- and multi-threaded to expose allocator contention. `stack` widens the input
// to 6·stack and loads with stride == stack (mirrors evo4's stack-6/stride-6 shape and how
// tests build stacked agents: loadWeights(sizes, flat, N, N)) — default 1 = today's 6-32-32-3.
static void runMlp(const char* name, int matches, int threads, int stack){
  const std::vector<int> sizes{6 * stack, 32, 32, 3};
  std::vector<double> flatA, flatB;
  for (int i = 0; i < sizes[0]*32 + 32 + 32*32 + 32 + 32*3 + 3; ++i){
    flatA.push_back(0.02 * ((i * 7) % 11) - 0.1);
    flatB.push_back(0.1 - 0.02 * ((i * 5) % 13));
  }
  std::atomic<int> next{0};
  std::atomic<long> ticks{0};
  auto t0 = std::chrono::steady_clock::now();
  auto worker = [&](){
    for (int m; (m = next.fetch_add(1)) < matches; ){
      MatchRunner mr((unsigned)(500 + m));
      auto a = std::make_unique<MlpAgent>("a"); a->loadWeights(sizes, flatA, stack, stack);
      auto b = std::make_unique<MlpAgent>("b"); b->loadWeights(sizes, flatB, stack, stack);
      mr.setController(Side::Left,  std::move(a), "mem:a");
      mr.setController(Side::Right, std::move(b), "mem:b");
      mr.newGame((unsigned)(500 + m));
      mr.runToCompletion();
      ticks += mr.ticks();
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("%-28s %3d matches %10ld ticks %7.2fs %9.0f ticks/s\n",
              name, matches, (long)ticks, sec, ticks / sec);
}

// The batched fitness path (Decision 13): same genome matches through SoA lanes. `stack`
// widens the input the same way as runMlp above and rides along as MatchParams.stride (the
// batched lane rings sample identically to the scalar agent's history window — Decision 13).
static void runMlpBatched(const char* name, int matches, int threads, int lanes, int stack){
  const std::vector<int> sizes{6 * stack, 32, 32, 3};
  const int nParams = sizes[0]*32 + 32 + 32*32 + 32 + 32*3 + 3;
  // DISTINCT genomes per match (a 128-strong pool), NOT two shared vectors: real training
  // streams a whole population's weights, so lanes load unique weights that miss cache and
  // pressure shared L3/DRAM. Reusing two genomes kept them cache-hot and made wider lanes look
  // artificially good (Decision 25) — this pool makes the batched numbers reflect real training.
  const int POP = 128;
  std::vector<std::vector<double>> pool((size_t)POP, std::vector<double>((size_t)nParams));
  for (int g = 0; g < POP; ++g)
    for (int i = 0; i < nParams; ++i)   // + 1e-5*g so all 128 buffers differ in content, not just address
      pool[(size_t)g][(size_t)i] = 0.02 * (((i * 7) + (g * 13)) % 11) - 0.1 + 1e-5 * g;
  std::vector<GenomeMatch> gms((size_t)matches);
  for (int m = 0; m < matches; ++m){
    gms[(size_t)m].self = &pool[(size_t)(m % POP)];
    gms[(size_t)m].opp  = &pool[(size_t)((m + 1 + m / POP) % POP)];   // distinct opponent
    gms[(size_t)m].seed = (unsigned)(500 + m);
  }
  MatchParams mp; mp.stride = stack;
  auto t0 = std::chrono::steady_clock::now();
  playMatchesBatched(sizes, gms, mp, threads, lanes);
  double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  long ticks = 0;
  for (const auto& g : gms) ticks += g.ticks;
  std::printf("%-28s %3d matches %10ld ticks %7.2fs %9.0f ticks/s\n",
              name, matches, ticks, sec, ticks / sec);
}

// --granular: sub-forward breakdown of the batched SoA kernel (DECISIONS.md #24/#25). Times
// layer-prefix variants of the real forward loop (in batch_forward.cpp, -ffast-math) over many
// iterations; differences give MEASURED per-layer MAC / tanh / overhead without the per-call
// clock overhead that would swamp a ~µs region inline. Then sweeps lane width to place the MAC
// on the roofline — the SoA layout gives each lane its own weights (no reuse), so the question
// is whether the MAC is FMA-throughput-bound (→ lane-width/ILP helps) or weight-bandwidth-bound
// (→ only cutting weight traffic, i.e. int8, helps).
static void runGranular(int stack){
  const int obsDim = 6 * stack, B = 16;
  const long IT = 500'000;
  auto ns = [&](double s){ return s / IT * 1e9; };
  const double t0 = ns(benchForwardKernel(obsDim, B, IT, 0, true,  false));  // transpose+argmax
  const double t1 = ns(benchForwardKernel(obsDim, B, IT, 1, true,  false));  // +L0 MAC
  const double t2 = ns(benchForwardKernel(obsDim, B, IT, 2, true,  false));  // +L1 MAC
  const double t3 = ns(benchForwardKernel(obsDim, B, IT, 3, true,  false));  // +L2 (all MAC, no tanh)
  const double tB = ns(benchForwardKernel(obsDim, B, IT, 3, false, false));  // bias-init only (no MAC)
  const double tF = ns(benchForwardKernel(obsDim, B, IT, 3, true,  true ));  // full (+tanh)
  const double L0 = t1 - t0, L1 = t2 - t1, L2 = t3 - t2, tanh = tF - t3, over = t0;
  const double macNs = L0 + L1 + L2, biasNs = tB - t0;   // raw multiply vs bias-init (no alloc in forward)
  const double macs = obsDim * 32.0 + 32 * 32.0 + 32 * 3.0;            // per lane, per forward
  std::printf("\ngranular forward breakdown — batched SoA kernel, -ffast-math, %d lanes, %d-32-32-3\n",
              B, obsDim);
  std::printf("  full forward: %.0f ns / call (%d lanes)\n", tF, B);
  auto glops = [&](double macCount, double nsv){ return nsv > 0 ? 2.0 * macCount * B / nsv : 0.0; };  // FLOP/ns = GFLOP/s
  auto row = [&](const char* n, double v, double macCount){
    if (macCount > 0)
      std::printf("    %-18s %8.1f ns  %5.1f%%   %6.1f GFLOP/s\n", n, v, 100 * v / tF, glops(macCount, v));
    else
      std::printf("    %-18s %8.1f ns  %5.1f%%\n", n, v, 100 * v / tF); };
  row("L0 MAC (measured)", L0, obsDim * 32.0);
  row("L1 MAC (measured)", L1, 32 * 32.0);
  row("L2 MAC (measured)", L2, 32 * 3.0);
  row("tanh (L0+L1)", tanh, 0);
  row("transpose+argmax", over, 0);
  const double gflops = glops(macs, macNs), gbs = 4.0 * macs * B / macNs;   // weight bytes/ns = GB/s
  std::printf("  MAC total: %.0f ns (%.0f%%)   %.1f GFLOP/s | weight stream %.1f GB/s | arith. intensity 0.50 FLOP/byte\n",
              macNs, 100 * macNs / tF, gflops, gbs);
  std::printf("  within a layer: raw multiply %.0f ns vs bias-init %.0f ns (~%.0fx) — the multiply dominates; no per-forward alloc\n",
              macNs - biasNs, biasNs, biasNs > 0 ? (macNs - biasNs) / biasNs : 0.0);
  std::printf("  ref: 1 P-core AVX2+FMA peak ~169 GFLOP/s (this i5-14600K) — far below it; the lane sweep says why:\n");
  std::printf("  lane-width roofline (SINGLE-thread, isolated — does NOT predict training; #25 settled it end-to-end):\n");
  const long SW = 200'000;
  for (int Bi : {4, 8, 16, 32, 64}){
    const double nf = benchForwardKernel(obsDim, Bi, SW, 3, true, true) / SW * 1e9;
    std::printf("    B=%-3d  %8.0f ns/fwd  %6.1f ns/lane  %6.1f GFLOP/s  %6.1f GB/s\n",
                Bi, nf, nf / Bi, 2.0 * macs * Bi / nf, 4.0 * macs * Bi / nf);
  }
  // Is a more efficient MULTIPLY possible? It's bandwidth-bound (0.5 FLOP/byte), so the lever is
  // fewer weight bytes, not better FMA codegen. Same forward, weights at 4 / 2 / 1 bytes:
  const double w4 = ns(benchForwardWeight(obsDim, B, IT, 4));
  const double w2 = ns(benchForwardWeight(obsDim, B, IT, 2));
  const double w1 = ns(benchForwardWeight(obsDim, B, IT, 1));
  const double x2 = w2 > 0 ? w4 / w2 : 0.0, x1 = w1 > 0 ? w4 / w1 : 0.0;   // speedup vs fp32
  std::printf("  weight-precision (same MACs, fewer bytes streamed — the bandwidth lever):\n");
  std::printf("    fp32 (4B) %.0f ns/fwd | 16-bit (2B) %.0f ns (%.2fx) | 8-bit (1B) %.0f ns (%.2fx)\n",
              w4, w2, x2, w1, x1);
  std::printf("    -> narrower storage ~%.2fx/%.2fx: the int->float convert eats the bandwidth saved single-threaded;\n"
              "       a real int8 win needs native AVX-VNNI + full quantization (quality-affecting)\n", x2, x1);
  // Cheaper tanh, and at what accuracy? (tanh is ~12-20% of fwd.)
  const long TN = 40'000;
  double e0 = 0, e1 = 0, e2 = 0;
  const double n0 = benchTanh(0, TN, &e0);   // libm (exact)  — ns per element
  const double n1 = benchTanh(1, TN, &e1);   // Padé[3/2]
  const double n2 = benchTanh(2, TN, &e2);   // Padé[7/6]
  std::printf("  cheaper tanh (speed / accuracy tradeoff):\n");
  std::printf("    libm std::tanh   %.2f ns/elem   (exact)\n", n0);
  std::printf("    Pade[3/2]        %.2f ns/elem   (%.1fx faster, max err %.1e)\n", n1, n1 > 0 ? n0 / n1 : 0.0, e1);
  std::printf("    Pade[7/6]        %.2f ns/elem   (%.1fx faster, max err %.1e)\n", n2, n2 > 0 ? n0 / n2 : 0.0, e2);
  std::printf("    -> Pade[7/6]: ~%.0fx lower error than Pade[3/2], nearly as fast; still changes the activation,\n"
              "       so any swap needs an anchor-skill training A/B (#25's discipline), not just this check\n",
              e2 > 0 ? e1 / e2 : 0.0);
}

int main(int argc, char** argv){
  int stack = 1;                                    // --stack N: MLP input width 6·N (evo4 shape: N=6)
  bool granular = false;
  for (int i = 1; i < argc; ++i){
    if (std::strcmp(argv[i], "--stack") == 0 && i + 1 < argc) stack = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--granular") == 0) granular = true;
  }
  std::printf("sim throughput (ticks include full agent+physics cost)\n");
  std::printf("mlp shape: %d-32-32-3 (stack %d, stride %d)\n", 6 * stack, stack, stack);
  run("interceptor vs interceptor", "interceptor", "interceptor", 6);
  run("p:classic vs interceptor",   "p:classic",   "interceptor", 20);
  run("p:classic vs p:classic",     "p:classic",   "p:classic",   20);
  runMlp("mlp vs mlp (1 thread)",  60, 1, stack);
  runMlp("mlp vs mlp (4 threads)", 240, 4, stack);
  runMlpBatched("mlp BATCHED (1t, 16 lanes)", 240, 1, 16, stack);
  runMlpBatched("mlp BATCHED (4t, 16 lanes)", 960, 4, 16, stack);
  if (granular){
    runGranular(stack);
    // NB: the single-thread lane sweep above shows an isolated ILP ceiling (wider looks better),
    // but that does NOT predict end-to-end training — an isolated-kernel A/B overstates the gain
    // because it omits the full generation's setLane/refill/construction and 128-genome access
    // pattern. Settled by an END-TO-END training A/B (Decision 25): 16 lanes beat 32 by ~3% at
    // 18 threads (the MAC is weight-bandwidth-bound under real contention). Lever is cutting
    // weight traffic (int8), not lane width. Always A/B lane changes via `./pong train`, not here.
    std::printf("\n(lane width settled end-to-end at 16 — DECISIONS.md #25; isolated kernel A/Bs mislead)\n");
  }
  return 0;
}
