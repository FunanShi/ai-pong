#pragma once
#include "evolve.hpp"
#include <functional>
#include <string>
#include <vector>

namespace pong {

// Island (deme) co-evolution: N single-architecture GA populations in one process. Each is a
// normal truncation-selection GA, but every generation genomes ALSO play cross-island matches
// against a frozen snapshot of the other islands' elites + hall-of-fame — an arms race between
// architectures. Selection happens WITHIN each island, so a fast-converging architecture can
// never crowd a slower one out (ratio preservation by construction). Deterministic and
// thread-invariant exactly as evolve(): seeds keyed by (masterSeed, tag, gen, slot); cross-island
// opponents read from a per-generation frozen snapshot. Full design rationale: DECISIONS.md #28.
struct CoEvolveConfig {
  std::vector<std::vector<int>> islandSizes;   // one MLP arch per island (front = 6·stack); must be distinct
  int      pop          = 96;      // population PER ISLAND (equal across islands)
  int      gens         = 200;
  int      selfGames    = 3;       // self-play matches per genome (own island, batched SoA path)
  int      crossGames   = 3;       // cross-island matches per genome (rotates over the other islands)
  int      anchorGames  = 2;       // scripted-bot matches per genome (scalar; {p:hard, laggy, instant})
  double   mutEps       = 0.1;
  double   eliteFrac    = 0.25;
  int      hofEvery     = 10;
  int      hofMax       = 20;
  int      evalEvery    = 25;      // anchor eval + checkpoint cadence (0 = never)
  int      evalMatches  = 24;
  unsigned seed         = 1;
  int      threads      = 0;       // 0 = hardware_concurrency
  int      stack        = 1;       // observation frames (input width = 6·stack), shared by all islands
  int      stride       = 1;
  double   speedBonus   = 0.0;     // win-speed shaping β (see shapeFitness)
  long     speedRefTicks = 30000;
  double   lossBonus     = 0.0;    // loss shaping (defaults off for a clean architecture comparison)
  double   lossProxWeight = 0.7;
  int      rallyCap     = 0;
  double   speedup      = 0.0;
  double   serveHold    = -1.0;
  int      winScore     = 0;
  std::string outDir    = "models/coevo";   // base; each island writes <outDir>/<label>/
  std::string resumeDir;   // warm-start each island from <resumeDir>/<label>/population.txt (+ hof.txt);
                           // a missing/corrupt/mismatched island refuses loudly (no random restart).
};

// Per-island, per-generation stats (index-aligned with CoEvolveConfig::islandSizes).
struct IslandStat {
  std::string label;
  double best = 0, mean = 0;                    // shaped fitness this generation
  double fitSelf = -2, fitCross = -2, fitAnchor = -2;   // pop-mean shaped margin per opponent class
  double vsPClassic = -2, vsLaggy = -2, vsInterceptor = -2;  // champion anchor win-rates (eval gens)
  double vsHeldP = -2, vsHeldIcept = -2;         // champion held-out win-rates (eval gens; never in fitness)
  double margPClassic = -2, margLaggy = -2, margInterceptor = -2;  // avg UNSHAPED margin vs each bot (eval gens)
  int    champAge = 0;                           // generations since this island's champion last changed
};

struct CoGenStats {
  int gen = 0;
  std::vector<IslandStat> islands;
  std::vector<std::vector<double>> crossWin;     // crossWin[i][j] = island i champ mean margin vs j champ
                                                 // (eval gens; empty otherwise). Diagonal = 0.
  double sec = 0;                                // wall-clock for this generation [s]
};

struct CoEvolveResult {
  std::vector<CoGenStats>            log;
  std::vector<std::string>          labels;      // island labels, index-aligned
  std::vector<std::vector<int>>     sizes;       // island archs, index-aligned
  std::vector<std::vector<double>>  bestFlat;    // each island's final champion genome
  bool stopped = false;                          // operator control-stop ended the run
  bool resumeFailed = false;                     // --resume load failed
};

// Deterministic arch label from a layer-size list: hidden widths joined by '-', e.g.
// {6,32,32,3} → "h32-32", {36,64,64,64,3} → "h64-64-64". Independent of input/output width.
std::string archLabel(const std::vector<int>& sizes);

CoEvolveResult coEvolve(const CoEvolveConfig&, const std::function<void(const CoGenStats&)>& onGen = {});

} // namespace pong
