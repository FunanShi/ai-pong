// evolve — GA self-play trainer CLI (recipe + rationale: DECISIONS.md #10).
//   flags: the arg parser below is the full, authoritative list; recipes/*.recipe hold the
//   tuned configurations (e.g. evomem.recipe for the memory GA).
#include "evolve.hpp"
#include "pong_core.hpp"
#include "prof.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace pong;

int main(int argc, char** argv){
  EvolveConfig cfg;
  for (int i = 1; i < argc; ++i){
    std::string a = argv[i];
    auto next = [&](){ if (i + 1 >= argc){ std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); } return argv[++i]; };
    if      (a == "--pop")          cfg.pop = std::atoi(next());
    else if (a == "--gens")         cfg.gens = std::atoi(next());
    else if (a == "--games")        cfg.gamesPerEval = std::atoi(next());
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
    else if (a == "--pool-dir")     cfg.poolDirs.push_back(next());
    else if (a == "--pool-games")   cfg.poolGames = std::atoi(next());
    else if (a == "--anchor-games") cfg.anchorGames = std::atoi(next());
    else if (a == "--resume")       cfg.resumeDir = next();
    else if (a == "--init-from")    cfg.initFrom = next();
    else if (a == "--out")          cfg.outDir = next();
    else if (a == "--hidden") {     // hidden-layer widths, e.g. "64,64" (wider) or "32,32,32" (deeper)
      cfg.sizes.assign(1, 6);       // input placeholder; evolve() rescales front to 6*stack at runtime
      std::stringstream ss(next()); std::string tok;
      while (std::getline(ss, tok, ',')){ int w = std::atoi(tok.c_str()); if (w > 0) cfg.sizes.push_back(w); }
      cfg.sizes.push_back(3);       // 3 output logits (up/down/none)
    }
    else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
  }
  if (cfg.pop < 4 || cfg.gens < 1){ std::fprintf(stderr, "need --pop >= 4, --gens >= 1\n"); return 1; }
  if (cfg.gamesPerEval - cfg.poolGames - cfg.anchorGames < 1){
    std::fprintf(stderr, "--pool-games + --anchor-games must leave >= 1 peer game of --games %d\n",
                 cfg.gamesPerEval);
    return 1;
  }

  if (!cfg.outDir.empty()){
    std::string argvLine;
    for (int i = 1; i < argc; ++i){ if (i > 1) argvLine += ' '; argvLine += argv[i]; }
    std::filesystem::create_directories(cfg.outDir);
    writeRunConfig(cfg.outDir, "evolve", argvLine);
  }

  std::printf("evolve: pop=%d gens=%d games=%d eps=%.3f elite=%.2f seed=%u threads=%d out=%s\n",
              cfg.pop, cfg.gens, cfg.gamesPerEval, cfg.mutEps, cfg.eliteFrac,
              cfg.seed, cfg.threads, cfg.outDir.c_str());
  std::printf("fitness = mean score margin in [-1,1] over self-play (peers + hall of fame); "
              "anchors are reporting-only\n");
  if (cfg.speedBonus > 0)
    std::printf("win-speed shaping ON: +%.2f max bonus on wins faster than %ld ticks\n",
                cfg.speedBonus, cfg.speedRefTicks);
  if (cfg.lossBonus > 0)
    std::printf("loss shaping ON: up to +%.2f on losses, concession-proximity weight %.2f "
                "vs survival-time (slower/closer losses score higher; bounded so any win still "
                "beats any loss; anchor evals unshaped)\n", cfg.lossBonus, cfg.lossProxWeight);
  if (cfg.rallyCap > 0)
    std::printf("training rally cap: %d hits (anchor evals keep the full %d)\n",
                cfg.rallyCap, k::RallyMax);
  if (cfg.speedup > 0)
    std::printf("training speedup: x%.3f per hit (anchor evals keep the canonical x%.3f)\n",
                cfg.speedup, k::Speedup);
  if (cfg.serveHold >= 0)
    std::printf("training serve hold: %.2fs (canonical %.2fs kept for anchor evals)\n",
                cfg.serveHold, k::ServeHold);
  if (cfg.winScore > 0)
    std::printf("training win score: first to %d (canonical %d kept for anchor evals)\n",
                cfg.winScore, k::WinScore);
  if (cfg.stack > 1)
    std::printf("observation history: stack %d stride %d -> window %.2f s of game time\n",
                cfg.stack, cfg.stride, ((cfg.stack - 1) * cfg.stride + 1) * k::Dt);
  { std::string arch = std::to_string(cfg.stack > 1 ? 6 * cfg.stack : cfg.sizes.front());
    for (size_t i = 1; i < cfg.sizes.size(); ++i) arch += "-" + std::to_string(cfg.sizes[i]);
    std::printf("network: %s (%zu hidden layers)\n", arch.c_str(), cfg.sizes.size() - 2); }
  if (cfg.poolGames > 0 || cfg.anchorGames > 0)
    std::printf("opponent mix per genome: %d peers/hof + %d pool + %d bots "
                "(pool dirs: %zu; held-out anchors stay out of fitness)\n",
                cfg.gamesPerEval - cfg.poolGames - cfg.anchorGames,
                cfg.poolGames, cfg.anchorGames, cfg.poolDirs.size());
  if (!cfg.resumeDir.empty())
    std::printf("resuming population from %s\n", cfg.resumeDir.c_str());
  else if (!cfg.initFrom.empty())
    std::printf("seeding founders from checkpoints in %s\n", cfg.initFrom.c_str());

  // Formatted analysis block, printed at every anchor-eval checkpoint: anchor trends,
  // per-anchor records, pace and ETA. The per-gen ticker stays for continuity.
  struct EvalRec { int gen; double laggy, icept; };
  std::vector<EvalRec> evalHist;
  std::deque<double> recentSec;
  double elapsedSec = 0.0, recentSecSum = 0.0;
  double bestLaggy = -1.0, bestIcept = -1.0;
  int bestLaggyGen = 0, bestIceptGen = 0;

  EvolveResult res = evolve(cfg, [&](const GenStats& s){
    elapsedSec += s.sec;
    recentSec.push_back(s.sec); recentSecSum += s.sec;
    if (recentSec.size() > 50){ recentSecSum -= recentSec.front(); recentSec.pop_front(); }
    if (s.vsPClassic > -1.5){
      std::printf("gen %4d  best %+.3f  mean %+.3f  | vs p:classic %3.0f%%  laggy %3.0f%%  interceptor %3.0f%%  | %.2fs\n",
                  s.gen, s.best, s.mean, 100.0 * s.vsPClassic, 100.0 * s.vsLaggy,
                  100.0 * s.vsInterceptor, s.sec);
      evalHist.push_back({s.gen, s.vsLaggy, s.vsInterceptor});
      if (s.vsLaggy > bestLaggy){ bestLaggy = s.vsLaggy; bestLaggyGen = s.gen; }
      if (s.vsInterceptor > bestIcept){ bestIcept = s.vsInterceptor; bestIceptGen = s.gen; }
      const double perGen = recentSecSum / (double)recentSec.size();
      const double etaMin = perGen * (double)(cfg.gens - 1 - s.gen) / 60.0;
      auto last5 = [&](double EvalRec::*f){
        std::string t;
        size_t n = evalHist.size(), from = n > 5 ? n - 5 : 0;
        for (size_t i = from; i < n; ++i){
          char b[8];
          std::snprintf(b, sizeof b, " %3.0f", 100.0 * (evalHist[i].*f));
          t += b;
        }
        return t;
      };
      std::printf("+---- %s | gen %d/%d | elapsed %.1fm | ETA %.1fm\n",
                  cfg.outDir.c_str(), s.gen, cfg.gens, elapsedSec / 60.0, etaMin);
      std::printf("| fitness  best %+.3f  mean %+.3f                 (shaped self-play)\n",
                  s.best, s.mean);
      std::printf("| anchors  p:classic %3.0f%%   laggy %3.0f%%   instant %3.0f%%     (canonical)\n",
                  100.0 * s.vsPClassic, 100.0 * s.vsLaggy, 100.0 * s.vsInterceptor);
      std::printf("| laggy    last:%s      best %3.0f%% @g%d\n",
                  last5(&EvalRec::laggy).c_str(), 100.0 * bestLaggy, bestLaggyGen);
      std::printf("| instant  last:%s      best %3.0f%% @g%d\n",
                  last5(&EvalRec::icept).c_str(), 100.0 * bestIcept, bestIceptGen);
      std::printf("| heldout  p %3.0f%%  icept %3.0f%%                    (never in fitness)\n",
                  100.0 * s.vsHeldP, 100.0 * s.vsHeldIcept);
      std::printf("| pace     %.2f s/gen (last %zu)  |  evals every %d gens\n",
                  perGen, recentSec.size(), cfg.evalEvery);
      std::printf("+---------------------------------------------------------------------\n");
    } else {
      std::printf("gen %4d  best %+.3f  mean %+.3f  | %.2fs\n", s.gen, s.best, s.mean, s.sec);
    }
    std::fflush(stdout);
  });

  if (res.resumeFailed){
    std::fprintf(stderr, "resume: could not load %s/population.txt (missing, corrupted, or "
                         "architecture mismatch) — refusing to silently restart from random\n",
                 cfg.resumeDir.c_str());
    return 1;
  }
  prof::dump();   // no-op unless AIPONG_PROFILE=1 — phase breakdown to stderr (#24)
  if (res.stopped)
    std::printf("stopped by operator (control file) — population saved in %s for --resume\n",
                cfg.outDir.c_str());
  else
    std::printf("done: final best fitness %+.3f — checkpoints in %s (load in the GUI with "
                "AIPONG_MODEL=%s/best.txt ./pong)\n",
                res.bestFitness, cfg.outDir.c_str(), cfg.outDir.c_str());
  return 0;
}
