#include "coevolve.hpp"
#include "evolve.hpp"
#include "rng.hpp"
#include "match_runner.hpp"   // writeModelText, MlpAgent, k::WinScore
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace pong {
namespace {

// Crash-safe checkpoint write: serialize into "<path>.tmp", then rename() it over <path>.
// rename(2) is atomic within a filesystem and the stream is flushed/closed before the rename,
// so a crash or power loss leaves a reader either the previous complete file or the new complete
// one — never a half-written checkpoint. On rename failure the temp is dropped, prior file kept.
template <class WriteFn>
void writeAtomic(const std::string& path, WriteFn&& write){
  const std::string tmp = path + ".tmp";
  { std::ofstream f(tmp); write(f); }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) std::filesystem::remove(tmp, ec);
}

// rng stream tags — distinct from evolve()'s (1..5) for hygiene. Keyed by (seed, tag, gen, slot).
// TagCoEval keeps eval/cross-win match seeds off the training-match stream (mirrors evolve()'s TagEval).
constexpr uint64_t TagCoInit = 11, TagCoOpp = 12, TagCoMatch = 13, TagCoMutate = 14, TagCoEval = 15;
static const char* kBots[3] = {"p:hard", "interceptor:laggy", "interceptor"};

// A scalar match owned by (island, owner genome): either a live cross-island genome opponent
// (oppGenome set) or a scripted anchor (oppSpec set). Result written by the worker into its own
// slot — fitness is accumulated single-threaded afterward, so results are thread-count-invariant.
struct CoJob {
  int island = 0, owner = 0;
  const std::vector<int>*    selfSizes  = nullptr;
  const std::vector<double>* selfGenome = nullptr;
  const std::vector<int>*    oppSizes   = nullptr;   // set iff oppGenome set (cross-island); runCoJobs branches on oppGenome
  const std::vector<double>* oppGenome  = nullptr;
  std::string oppSpec;
  unsigned seed = 0;
  double result = 0;
  int cls = 1;                                        // 1 = cross-island, 2 = anchor (telemetry)
};

void runCoJobs(std::vector<CoJob>& jobs, int threads, const MatchParams& mp){
  std::atomic<size_t> next{0};
  auto worker = [&](){
    for (size_t j; (j = next.fetch_add(1)) < jobs.size(); ){
      CoJob& c = jobs[j];
      if (c.oppGenome)
        c.result = playMatchVsGenome(*c.selfSizes, *c.selfGenome, *c.oppSizes, *c.oppGenome, c.seed, mp);
      else
        c.result = playMatchVsSpec(*c.selfSizes, *c.selfGenome, c.oppSpec, nullptr, c.seed, mp);
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
}

// FNV-1a over a genome's raw bytes → champion identity for age telemetry (mirrors evolve()).
uint64_t genomeHash(const std::vector<double>& g){
  uint64_t h = 1469598103934665603ull;
  for (double w : g)
    for (size_t b = 0; b < sizeof(double); ++b){
      h ^= reinterpret_cast<const unsigned char*>(&w)[b];
      h *= 1099511628211ull;
    }
  return h;
}

struct Island {
  std::vector<int> sizes;
  std::string label;
  std::vector<std::vector<double>> pop, hof;
  uint64_t champHash = 0;
  int champAge = 0;
};

} // namespace

std::string archLabel(const std::vector<int>& sizes){
  std::string s = "h";
  for (size_t i = 1; i + 1 < sizes.size(); ++i) s += (i > 1 ? "-" : "") + std::to_string(sizes[i]);
  return s;
}

CoEvolveResult coEvolve(const CoEvolveConfig& cfg, const std::function<void(const CoGenStats&)>& onGen){
  const int N = (int)cfg.islandSizes.size();
  int T = cfg.threads > 0 ? cfg.threads : (int)std::thread::hardware_concurrency();
  if (T < 1) T = 1;
  const int nElite   = std::max(1, (int)(cfg.eliteFrac * cfg.pop));
  const int gamesPer = cfg.selfGames + cfg.crossGames + cfg.anchorGames;

  // ---- islands: resume from saved per-island populations, else seeded random init ----
  std::vector<Island> isl((size_t)N);
  for (int I = 0; I < N; ++I){
    isl[(size_t)I].sizes = cfg.islandSizes[(size_t)I];
    if (cfg.stack > 1) isl[(size_t)I].sizes.front() = 6 * cfg.stack;
    isl[(size_t)I].label = archLabel(isl[(size_t)I].sizes);
  }
  if (!cfg.resumeDir.empty()){
    for (int I = 0; I < N; ++I){
      const std::string d = cfg.resumeDir + "/" + isl[(size_t)I].label;
      std::ifstream pf(d + "/population.txt");
      std::vector<int> fsizes; std::vector<std::vector<double>> founders;
      if (!pf || !readPopulation(pf, fsizes, founders) || fsizes != isl[(size_t)I].sizes){
        CoEvolveResult fail;                                   // refuse loudly (Decision 22 contract)
        for (int J = 0; J < N; ++J){ fail.labels.push_back(isl[(size_t)J].label); fail.sizes.push_back(isl[(size_t)J].sizes); }
        fail.resumeFailed = true;
        return fail;
      }
      isl[(size_t)I].pop = std::move(founders);
      // Pad a grown population (resumed --pop larger than saved) the way evolve() does: round-robin
      // founder clones, each a DISTINCT per-slot mutation. NOT resize(n,value) — that clones ONE
      // genome into every new slot, collapsing initial diversity. readPopulation guarantees >=1 loaded.
      const size_t loaded = isl[(size_t)I].pop.size();
      for (size_t i = loaded; i < (size_t)cfg.pop; ++i){
        std::vector<double> clone = isl[(size_t)I].pop[i % loaded];
        uint64_t ms = rng::key(cfg.seed, TagCoInit, (uint64_t)I, (uint64_t)i);
        for (auto& w : clone) w += cfg.mutEps * rng::gaussian(ms);
        isl[(size_t)I].pop.push_back(std::move(clone));
      }
      if (isl[(size_t)I].pop.size() > (size_t)cfg.pop) isl[(size_t)I].pop.resize((size_t)cfg.pop);  // shrink truncates
      std::ifstream hf(d + "/hof.txt");
      std::vector<int> hsizes; std::vector<std::vector<double>> h;
      if (hf && readPopulation(hf, hsizes, h) && hsizes == isl[(size_t)I].sizes) isl[(size_t)I].hof = std::move(h);
    }
  } else {
    for (int I = 0; I < N; ++I){
      isl[(size_t)I].pop.resize((size_t)cfg.pop);
      for (int i = 0; i < cfg.pop; ++i)
        isl[(size_t)I].pop[(size_t)i] =
            randomGenome(isl[(size_t)I].sizes, rng::key(cfg.seed, TagCoInit, (uint64_t)I, (uint64_t)i));
    }
  }

  const MatchParams trainMp{cfg.speedBonus, cfg.speedRefTicks, cfg.rallyCap, cfg.speedup,
                            cfg.serveHold, cfg.winScore, cfg.stride, cfg.lossBonus, cfg.lossProxWeight};

  CoEvolveResult res;
  for (int I = 0; I < N; ++I){ res.labels.push_back(isl[(size_t)I].label); res.sizes.push_back(isl[(size_t)I].sizes); }
  res.bestFlat.resize((size_t)N);

  bool stopped = false;
  if (!cfg.outDir.empty()){
    std::filesystem::create_directories(cfg.outDir);
    std::error_code ec; std::filesystem::remove(cfg.outDir + "/control", ec);   // never start paused
    writeStatus(cfg.outDir, "running", 0, cfg.gens);
  }

  for (int gen = 0; gen < cfg.gens; ++gen){
    // ---- operator control (base <outDir>/control), between generations only — draws no RNG,
    // reorders nothing: results are pause/stop-pattern-invariant (determinism preserved). ----
    bool forceEval = false;
    if (!cfg.outDir.empty()){
      Control c = readControl(cfg.outDir);
      if (c == Control::Pause){
        writeStatus(cfg.outDir, "paused", gen, cfg.gens);
        while (readControl(cfg.outDir) == Control::Pause)
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        c = readControl(cfg.outDir);
        writeStatus(cfg.outDir, "running", gen, cfg.gens);
      }
      if (c == Control::Stop){
        for (int I = 0; I < N; ++I){
          const std::string d = cfg.outDir + "/" + isl[(size_t)I].label;
          std::filesystem::create_directories(d);
          writeAtomic(d + "/population.txt", [&](std::ofstream& f){ writePopulation(f, isl[(size_t)I].sizes, isl[(size_t)I].pop); });
          writeAtomic(d + "/hof.txt",        [&](std::ofstream& f){ writePopulation(f, isl[(size_t)I].sizes, isl[(size_t)I].hof); });
        }
        writeStatus(cfg.outDir, "stopped", gen, cfg.gens);
        stopped = true;
        break;
      }
      if (c == Control::Eval){
        forceEval = true;
        std::ofstream cf(cfg.outDir + "/control", std::ios::trunc); cf << "run";
      }
    }

    auto t0 = std::chrono::steady_clock::now();

    // ---- 1. FROZEN SNAPSHOT of cross-island opponents (reigning elites + HoF), pre-mutation.
    // pop[0..nElite) are last generation's carried-over elites (fitness-ordered); copying makes
    // the determinism argument airtight regardless of scheduling. ----
    struct Snap { std::vector<std::vector<double>> elites, hof; };
    std::vector<Snap> snap((size_t)N);
    for (int I = 0; I < N; ++I){
      int e = std::min(nElite, (int)isl[(size_t)I].pop.size());
      snap[(size_t)I].elites.assign(isl[(size_t)I].pop.begin(), isl[(size_t)I].pop.begin() + e);
      snap[(size_t)I].hof = isl[(size_t)I].hof;
    }

    // ---- 2. Build matches. Self-play → per-island batched lists (uniform arch). Cross + anchor
    // → one merged scalar CoJob list. All building is single-threaded → fixed, seed-keyed order. ----
    std::vector<std::vector<GenomeMatch>> selfM((size_t)N);
    std::vector<CoJob> cojobs;
    for (int I = 0; I < N; ++I){
      selfM[(size_t)I].reserve((size_t)cfg.pop * (size_t)cfg.selfGames);
      for (int i = 0; i < cfg.pop; ++i){
        uint64_t os   = rng::key(cfg.seed, TagCoOpp, (uint64_t)gen, (uint64_t)(I * cfg.pop + i));
        uint64_t base = (uint64_t)(I * cfg.pop + i) * 64u;   // per-genome match-seed block (gamesPer < 64)
        int g = 0;
        // self-play (batched): peers from the LIVE population; last slot vs an own-HoF ancestor.
        for (int s = 0; s < cfg.selfGames; ++s, ++g){
          GenomeMatch gm; gm.self = &isl[(size_t)I].pop[(size_t)i];
          bool useHof = !isl[(size_t)I].hof.empty() && s == cfg.selfGames - 1;
          if (useHof){
            size_t h = (size_t)(rng::uniform01(os) * (double)isl[(size_t)I].hof.size());
            if (h >= isl[(size_t)I].hof.size()) h = isl[(size_t)I].hof.size() - 1;
            gm.opp = &isl[(size_t)I].hof[h];
          } else {
            size_t p = (size_t)(rng::uniform01(os) * (double)(cfg.pop - 1));
            if ((int)p >= i) p++;                             // peer ≠ self
            gm.opp = &isl[(size_t)I].pop[p];
          }
          gm.seed = (unsigned)rng::key(cfg.seed, TagCoMatch, (uint64_t)gen, base + (uint64_t)g);
          selfM[(size_t)I].push_back(gm);
        }
        // cross-island (scalar): rotate over the OTHER islands; last slot vs that island's HoF.
        for (int c = 0; c < cfg.crossGames; ++c, ++g){
          int J = N > 1 ? (I + 1 + (c % (N - 1))) % N : I;
          CoJob cj; cj.island = I; cj.owner = i; cj.cls = 1;
          cj.selfSizes = &isl[(size_t)I].sizes; cj.selfGenome = &isl[(size_t)I].pop[(size_t)i];
          cj.oppSizes  = &isl[(size_t)J].sizes;
          bool useHof = !snap[(size_t)J].hof.empty() && c == cfg.crossGames - 1;
          if (useHof){
            size_t h = (size_t)(rng::uniform01(os) * (double)snap[(size_t)J].hof.size());
            if (h >= snap[(size_t)J].hof.size()) h = snap[(size_t)J].hof.size() - 1;
            cj.oppGenome = &snap[(size_t)J].hof[h];
          } else {
            size_t k = (size_t)(rng::uniform01(os) * (double)snap[(size_t)J].elites.size());
            if (k >= snap[(size_t)J].elites.size()) k = snap[(size_t)J].elites.size() - 1;
            cj.oppGenome = &snap[(size_t)J].elites[k];
          }
          cj.seed = (unsigned)rng::key(cfg.seed, TagCoMatch, (uint64_t)gen, base + (uint64_t)g);
          cojobs.push_back(std::move(cj));
        }
        // anchor (scalar): rotating scripted bot.
        for (int a = 0; a < cfg.anchorGames; ++a, ++g){
          CoJob cj; cj.island = I; cj.owner = i; cj.cls = 2;
          cj.selfSizes = &isl[(size_t)I].sizes; cj.selfGenome = &isl[(size_t)I].pop[(size_t)i];
          cj.oppSpec = kBots[(size_t)(rng::uniform01(os) * 3.0) % 3];
          cj.seed = (unsigned)rng::key(cfg.seed, TagCoMatch, (uint64_t)gen, base + (uint64_t)g);
          cojobs.push_back(std::move(cj));
        }
      }
    }

    // ---- 3. Run. Self-play batched (per island); cross + anchor scalar (merged). ----
    for (int I = 0; I < N; ++I)
      if (!selfM[(size_t)I].empty()) playMatchesBatched(isl[(size_t)I].sizes, selfM[(size_t)I], trainMp, T);
    if (!cojobs.empty()) runCoJobs(cojobs, T, trainMp);

    // ---- 4. Fitness — single-threaded accumulation in fixed order → thread-invariant.
    // Self match m of island I belongs to genome m / selfGames (matches built in genome order). ----
    std::vector<std::vector<double>> fit((size_t)N);
    std::vector<std::array<double, 3>> clsSum((size_t)N, {0.0, 0.0, 0.0});   // [self, cross, anchor]
    std::vector<std::array<long, 3>>   clsN((size_t)N, {0, 0, 0});
    for (int I = 0; I < N; ++I) fit[(size_t)I].assign((size_t)cfg.pop, 0.0);
    for (int I = 0; I < N; ++I)
      for (size_t m = 0; m < selfM[(size_t)I].size(); ++m){
        int own = (int)(m / (size_t)cfg.selfGames);           // cfg.selfGames > 0 whenever selfM non-empty
        fit[(size_t)I][(size_t)own] += selfM[(size_t)I][m].result;
        clsSum[(size_t)I][0] += selfM[(size_t)I][m].result; clsN[(size_t)I][0]++;
      }
    for (const auto& c : cojobs){
      int cl = (c.cls == 2) ? 2 : 1;
      fit[(size_t)c.island][(size_t)c.owner] += c.result;
      clsSum[(size_t)c.island][(size_t)cl] += c.result; clsN[(size_t)c.island][(size_t)cl]++;
    }
    for (int I = 0; I < N; ++I)
      for (auto& f : fit[(size_t)I]) f /= (double)gamesPer;

    // ---- 5. Per-island selection + HoF + stats. ----
    CoGenStats gs; gs.gen = gen;
    std::vector<std::vector<int>> order((size_t)N);
    for (int I = 0; I < N; ++I){
      order[(size_t)I].resize((size_t)cfg.pop);
      std::iota(order[(size_t)I].begin(), order[(size_t)I].end(), 0);
      std::stable_sort(order[(size_t)I].begin(), order[(size_t)I].end(),
                       [&](int a, int b){ return fit[(size_t)I][(size_t)a] > fit[(size_t)I][(size_t)b]; });
      IslandStat isstat; isstat.label = isl[(size_t)I].label;
      isstat.best = fit[(size_t)I][(size_t)order[(size_t)I][0]];
      isstat.mean = std::accumulate(fit[(size_t)I].begin(), fit[(size_t)I].end(), 0.0) / (double)cfg.pop;
      if (clsN[(size_t)I][0]) isstat.fitSelf   = clsSum[(size_t)I][0] / (double)clsN[(size_t)I][0];
      if (clsN[(size_t)I][1]) isstat.fitCross  = clsSum[(size_t)I][1] / (double)clsN[(size_t)I][1];
      if (clsN[(size_t)I][2]) isstat.fitAnchor = clsSum[(size_t)I][2] / (double)clsN[(size_t)I][2];
      uint64_t h = genomeHash(isl[(size_t)I].pop[(size_t)order[(size_t)I][0]]);
      isl[(size_t)I].champAge = (gen > 0 && h == isl[(size_t)I].champHash) ? isl[(size_t)I].champAge + 1 : 0;
      isl[(size_t)I].champHash = h;
      isstat.champAge = isl[(size_t)I].champAge;
      gs.islands.push_back(std::move(isstat));
      if (cfg.hofEvery > 0 && gen % cfg.hofEvery == 0){
        isl[(size_t)I].hof.push_back(isl[(size_t)I].pop[(size_t)order[(size_t)I][0]]);
        if ((int)isl[(size_t)I].hof.size() > cfg.hofMax) isl[(size_t)I].hof.erase(isl[(size_t)I].hof.begin());
      }
    }

    // ---- Anchor evaluation (reporting only) + cross-island win matrix, on eval gens. Anchor
    // evals use canonical unshaped physics with the genome's stride (evalParams contract). ----
    const bool evalNow = forceEval || (cfg.evalEvery > 0 &&
        (gen % cfg.evalEvery == cfg.evalEvery - 1 || gen == cfg.gens - 1));
    if (evalNow){
      MatchParams evalMp{}; evalMp.stride = cfg.stride > 0 ? cfg.stride : 1;   // canonical, unshaped
      // Runs cfg.evalMatches unshaped matches vs `spec`; returns the win-rate and (out) the
      // average margin — e.result IS the raw margin under evalMp, so both come from one pass.
      auto evalVsBot = [&](int I, const std::string& spec, uint64_t tag, double& avgMargin){
        std::vector<CoJob> ev((size_t)cfg.evalMatches);
        for (int m = 0; m < cfg.evalMatches; ++m){
          ev[(size_t)m].selfSizes  = &isl[(size_t)I].sizes;
          ev[(size_t)m].selfGenome = &isl[(size_t)I].pop[(size_t)order[(size_t)I][0]];
          ev[(size_t)m].oppSpec = spec;
          ev[(size_t)m].seed = (unsigned)rng::key(cfg.seed, TagCoEval,
                                                  (uint64_t)(gen * 8 + (int)tag), (uint64_t)(I * 4096 + m));
        }
        runCoJobs(ev, T, evalMp);
        int wins = 0; double msum = 0;
        for (const auto& e : ev){ if (e.result > 0) wins++; msum += e.result; }
        avgMargin = msum / (double)cfg.evalMatches;
        return (double)wins / (double)cfg.evalMatches;
      };
      for (int I = 0; I < N; ++I){
        double mC, mL, mIc, mHp, mHi;
        gs.islands[(size_t)I].vsPClassic    = evalVsBot(I, "p:classic", 0, mC);
        gs.islands[(size_t)I].vsLaggy       = evalVsBot(I, "interceptor:laggy", 1, mL);
        gs.islands[(size_t)I].vsInterceptor = evalVsBot(I, "interceptor", 2, mIc);
        gs.islands[(size_t)I].vsHeldP       = evalVsBot(I, "p:heldout", 3, mHp);
        gs.islands[(size_t)I].vsHeldIcept   = evalVsBot(I, "interceptor:heldout", 4, mHi);
        gs.islands[(size_t)I].margPClassic    = mC;
        gs.islands[(size_t)I].margLaggy       = mL;
        gs.islands[(size_t)I].margInterceptor = mIc;
      }
      // cross-island win matrix: island i champion mean shaped margin vs island j champion (i≠j).
      gs.crossWin.assign((size_t)N, std::vector<double>((size_t)N, 0.0));
      std::vector<CoJob> xw;
      for (int I = 0; I < N; ++I)
        for (int J = 0; J < N; ++J) if (I != J){
          CoJob cj; cj.island = I; cj.owner = J;                 // owner reused as column index j
          cj.selfSizes = &isl[(size_t)I].sizes; cj.selfGenome = &isl[(size_t)I].pop[(size_t)order[(size_t)I][0]];
          cj.oppSizes  = &isl[(size_t)J].sizes; cj.oppGenome  = &isl[(size_t)J].pop[(size_t)order[(size_t)J][0]];
          cj.seed = (unsigned)rng::key(cfg.seed, TagCoEval, (uint64_t)(gen * 8 + 5), (uint64_t)(I * 64 + J));
          xw.push_back(std::move(cj));
        }
      if (!xw.empty()) runCoJobs(xw, T, evalMp);
      for (const auto& c : xw) gs.crossWin[(size_t)c.island][(size_t)c.owner] = c.result;
    }
    gs.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- File output: each island gets a standard run dir the existing trainview + ladder read;
    // the base dir gets the cross-island matrix. Wall-clock IO only — never read back mid-run. ----
    if (!cfg.outDir.empty()){
      for (int I = 0; I < N; ++I){
        const std::string d = cfg.outDir + "/" + isl[(size_t)I].label;
        std::filesystem::create_directories(d);
        const IslandStat& is = gs.islands[(size_t)I];
        std::ofstream lg(d + "/train_log.jsonl", std::ios::app);
        lg << "{\"gen\":" << gen << ",\"best\":" << is.best << ",\"mean\":" << is.mean
           << ",\"vs_p_classic\":" << is.vsPClassic << ",\"vs_laggy\":" << is.vsLaggy
           << ",\"vs_interceptor\":" << is.vsInterceptor
           << ",\"vs_held_p\":" << is.vsHeldP << ",\"vs_held_icept\":" << is.vsHeldIcept
           << ",\"marg_p_classic\":" << is.margPClassic << ",\"marg_laggy\":" << is.margLaggy
           << ",\"marg_interceptor\":" << is.margInterceptor
           << ",\"fit_peer\":" << is.fitSelf << ",\"fit_pool\":" << is.fitCross
           << ",\"fit_anchor\":" << is.fitAnchor << ",\"champ_age\":" << is.champAge
           << ",\"sec\":" << gs.sec << "}\n";
        writeStatus(d, gen + 1 < cfg.gens ? "running" : "finished", gen, cfg.gens);
        if (evalNow){
          char name[64]; std::snprintf(name, sizeof name, "/gen_%04d.txt", gen);
          const auto& champ = isl[(size_t)I].pop[(size_t)order[(size_t)I][0]];
          writeAtomic(d + name,              [&](std::ofstream& f){ writeModelText(f, isl[(size_t)I].sizes, champ, cfg.stack, cfg.stride); });
          writeAtomic(d + "/best.txt",       [&](std::ofstream& f){ writeModelText(f, isl[(size_t)I].sizes, champ, cfg.stack, cfg.stride); });
          writeAtomic(d + "/population.txt", [&](std::ofstream& f){ writePopulation(f, isl[(size_t)I].sizes, isl[(size_t)I].pop); });
          writeAtomic(d + "/hof.txt",        [&](std::ofstream& f){ writePopulation(f, isl[(size_t)I].sizes, isl[(size_t)I].hof); });
        }
      }
      if (evalNow && !gs.crossWin.empty()){
        std::ofstream xf(cfg.outDir + "/crosswin.jsonl", std::ios::app);
        xf << "{\"gen\":" << gen << ",\"labels\":[";
        for (int I = 0; I < N; ++I) xf << (I ? "," : "") << "\"" << isl[(size_t)I].label << "\"";
        xf << "],\"matrix\":[";
        for (int I = 0; I < N; ++I){
          xf << (I ? ",[" : "[");
          for (int J = 0; J < N; ++J) xf << (J ? "," : "") << gs.crossWin[(size_t)I][(size_t)J];
          xf << "]";
        }
        xf << "]}\n";
      }
      writeStatus(cfg.outDir, "running", gen, cfg.gens);   // base aggregate heartbeat for coview
    }

    res.log.push_back(gs);
    if (onGen) onGen(gs);

    // ---- 6. Next generation per island: elites survive unchanged, rest are mutated elite clones. ----
    if (gen + 1 < cfg.gens){
      for (int I = 0; I < N; ++I){
        std::vector<std::vector<double>> nxt((size_t)cfg.pop);
        for (int e = 0; e < nElite; ++e) nxt[(size_t)e] = isl[(size_t)I].pop[(size_t)order[(size_t)I][(size_t)e]];
        for (int c = nElite; c < cfg.pop; ++c){
          uint64_t ms = rng::key(cfg.seed, TagCoMutate, (uint64_t)gen, (uint64_t)(I * cfg.pop + c));
          size_t pIdx = (size_t)(rng::uniform01(ms) * (double)nElite);
          if (pIdx >= (size_t)nElite) pIdx = (size_t)nElite - 1;
          std::vector<double> child = isl[(size_t)I].pop[(size_t)order[(size_t)I][pIdx]];
          for (auto& w : child) w += cfg.mutEps * rng::gaussian(ms);
          nxt[(size_t)c] = std::move(child);
        }
        isl[(size_t)I].pop = std::move(nxt);
      }
    } else {
      for (int I = 0; I < N; ++I) res.bestFlat[(size_t)I] = isl[(size_t)I].pop[(size_t)order[(size_t)I][0]];
    }
  }
  res.stopped = stopped;
  if (!cfg.outDir.empty() && !stopped) writeStatus(cfg.outDir, "finished", cfg.gens - 1, cfg.gens);
  return res;
}

} // namespace pong
