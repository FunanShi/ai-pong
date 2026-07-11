// coevolve — island (deme) co-evolution CLI. N single-architecture GA populations co-evolving
// via cross-island matches; per-island selection preserves each architecture's lineage.
// Design rationale: DECISIONS.md #28.
#include "coevolve.hpp"
#include "evolve.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace pong;

// Parse "32,32;64,64;32,32,32" → {{6,32,32,3},{6,64,64,3},{6,32,32,32,3}} (front rescaled at runtime).
static std::vector<std::vector<int>> parseIslands(const std::string& spec){
  std::vector<std::vector<int>> out;
  std::stringstream groups(spec); std::string grp;
  while (std::getline(groups, grp, ';')){
    if (grp.empty()) continue;
    std::vector<int> sizes{6};                       // input placeholder; rescaled to 6·stack in coEvolve
    std::stringstream widths(grp); std::string w;
    while (std::getline(widths, w, ',')){ int v = std::atoi(w.c_str()); if (v > 0) sizes.push_back(v); }
    sizes.push_back(3);                              // 3 output logits (up/down/none)
    if (sizes.size() >= 3) out.push_back(std::move(sizes));   // needs ≥1 hidden layer
  }
  return out;
}

int main(int argc, char** argv){
  CoEvolveConfig cfg;
  std::string islandsSpec = "32,32;64,64;32,32,32;64,64,64";   // spec defaults (base/wide/deep/widedeep)
  bool crossSet = false;
  for (int i = 1; i < argc; ++i){
    std::string a = argv[i];
    auto next = [&](){ if (i + 1 >= argc){ std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); } return argv[++i]; };
    if      (a == "--islands")      islandsSpec = next();
    else if (a == "--pop")          cfg.pop = std::atoi(next());
    else if (a == "--gens")         cfg.gens = std::atoi(next());
    else if (a == "--self-games")   cfg.selfGames = std::atoi(next());
    else if (a == "--cross-games"){ cfg.crossGames = std::atoi(next()); crossSet = true; }
    else if (a == "--anchor-games") cfg.anchorGames = std::atoi(next());
    else if (a == "--eps")          cfg.mutEps = std::atof(next());
    else if (a == "--elite")        cfg.eliteFrac = std::atof(next());
    else if (a == "--seed")         cfg.seed = (unsigned)std::strtoul(next(), nullptr, 10);
    else if (a == "--threads")      cfg.threads = std::atoi(next());
    else if (a == "--eval-every")   cfg.evalEvery = std::atoi(next());
    else if (a == "--eval-matches") cfg.evalMatches = std::atoi(next());
    else if (a == "--hof-every")    cfg.hofEvery = std::atoi(next());
    else if (a == "--speed-bonus")  cfg.speedBonus = std::atof(next());
    else if (a == "--speed-ref")    cfg.speedRefTicks = std::atol(next());
    else if (a == "--loss-bonus")   cfg.lossBonus = std::atof(next());
    else if (a == "--loss-prox-weight") cfg.lossProxWeight = std::atof(next());
    else if (a == "--rally-cap")    cfg.rallyCap = std::atoi(next());
    else if (a == "--speedup")      cfg.speedup = std::atof(next());
    else if (a == "--serve-hold")   cfg.serveHold = std::atof(next());
    else if (a == "--win-score")    cfg.winScore = std::atoi(next());
    else if (a == "--stack")        cfg.stack = std::atoi(next());
    else if (a == "--stride")       cfg.stride = std::atoi(next());
    else if (a == "--resume")       cfg.resumeDir = next();
    else if (a == "--out")          cfg.outDir = next();
    else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
  }
  cfg.islandSizes = parseIslands(islandsSpec);
  const int N = (int)cfg.islandSizes.size();
  if (N < 2){ std::fprintf(stderr, "need >= 2 islands (--islands \"32,32;64,64\"); got %d\n", N); return 1; }
  { // reject duplicate architectures — labels/run-dirs must be distinct
    std::vector<std::string> seen;
    for (const auto& s : cfg.islandSizes){
      std::string l = archLabel(s);
      for (const auto& p : seen) if (p == l){ std::fprintf(stderr, "duplicate island arch '%s' — archs must be distinct\n", l.c_str()); return 1; }
      seen.push_back(l);
    }
  }
  if (!crossSet) cfg.crossGames = N - 1;             // default: one cross-match per other island
  if (cfg.pop < 4 || cfg.gens < 1){ std::fprintf(stderr, "need --pop >= 4, --gens >= 1\n"); return 1; }
  if (cfg.selfGames < 0 || cfg.crossGames < 0 || cfg.anchorGames < 0 ||
      cfg.selfGames + cfg.crossGames + cfg.anchorGames < 1){
    std::fprintf(stderr, "need self+cross+anchor games >= 1\n"); return 1;
  }

  if (!cfg.outDir.empty()){
    std::string argvLine;
    for (int i = 1; i < argc; ++i){ if (i > 1) argvLine += ' '; argvLine += argv[i]; }
    std::filesystem::create_directories(cfg.outDir);
    writeRunConfig(cfg.outDir, "coevolve", argvLine);
  }

  std::printf("coevolve: %d islands, pop=%d/island, gens=%d, games=%d self + %d cross + %d anchor, "
              "seed=%u threads=%d out=%s\n", N, cfg.pop, cfg.gens, cfg.selfGames, cfg.crossGames,
              cfg.anchorGames, cfg.seed, cfg.threads, cfg.outDir.c_str());
  for (const auto& s : cfg.islandSizes){
    std::string arch = std::to_string(cfg.stack > 1 ? 6 * cfg.stack : s.front());
    for (size_t k = 1; k < s.size(); ++k) arch += "-" + std::to_string(s[k]);
    std::printf("  island %-12s  %s\n", archLabel(s).c_str(), arch.c_str());
  }
  if (cfg.stack > 1)
    std::printf("observation history: stack %d stride %d\n", cfg.stack, cfg.stride);
  if (!cfg.resumeDir.empty())
    std::printf("resuming every island from %s/<label>/population.txt\n", cfg.resumeDir.c_str());

  CoEvolveResult res = coEvolve(cfg, [&](const CoGenStats& s){
    std::printf("gen %4d |", s.gen);
    for (const auto& is : s.islands){
      if (is.vsInterceptor > -1.5)
        std::printf("  %s best %+.3f icept %3.0f%%", is.label.c_str(), is.best, 100.0 * is.vsInterceptor);
      else
        std::printf("  %s best %+.3f", is.label.c_str(), is.best);
    }
    std::printf("  | %.2fs\n", s.sec);
    std::fflush(stdout);
  });

  if (res.resumeFailed){
    std::fprintf(stderr, "resume: an island's %s/<label>/population.txt was missing, corrupted, or "
                         "arch-mismatched — refusing to silently restart from random\n", cfg.resumeDir.c_str());
    return 1;
  }
  if (res.stopped){
    std::printf("stopped by operator — %d islands checkpointed in %s; resume with: "
                "./pong coevolve --resume %s --out <new-dir> ...\n",
                (int)res.labels.size(), cfg.outDir.c_str(), cfg.outDir.c_str());
    return 0;
  }
  std::printf("done: %d islands trained — champions in %s/<label>/best.txt "
              "(compare with: ./pong ladder", (int)res.labels.size(), cfg.outDir.c_str());
  for (const auto& l : res.labels) std::printf(" --dir %s/%s", cfg.outDir.c_str(), l.c_str());
  std::printf(")\n");
  return 0;
}
