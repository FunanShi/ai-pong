#include "evolve.hpp"
#include "batch_forward.hpp"
#include "match_runner.hpp"
#include "prof.hpp"
#include "rng.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <thread>

namespace pong {

size_t flatSize(const std::vector<int>& sizes){
  size_t n = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l)
    n += (size_t)sizes[l] * (size_t)sizes[l + 1] + (size_t)sizes[l + 1];
  return n;
}

std::vector<double> randomGenome(const std::vector<int>& sizes, uint64_t k){
  std::vector<double> g; g.reserve(flatSize(sizes));
  uint64_t s = k;
  for (size_t l = 0; l + 1 < sizes.size(); ++l){
    double scale = 1.0 / double(sizes[l]);              // keep pre-activations tame
    for (int i = 0; i < sizes[l] * sizes[l + 1]; ++i) g.push_back(rng::uniform(s, -scale, scale));
    for (int i = 0; i < sizes[l + 1]; ++i)            g.push_back(rng::uniform(s, -0.05, 0.05));
  }
  return g;
}

namespace {

// rng stream tags — every random decision draws from a stream keyed by
// (masterSeed, TAG, generation, slot), so nothing depends on thread scheduling.
constexpr uint64_t TagInit = 1, TagOpp = 2, TagMatch = 3, TagMutate = 4, TagEval = 5;
// NOTE: src/tools/reeval.cpp mirrors TagEval and the gen*8+tag eval-seed formula — change both together.

// (flatSize and randomGenome were here; now defined above at pong-namespace scope.)

struct MatchJob {
  int owner = 0;                            // genome index whose fitness this match feeds
  const std::vector<double>* oppFlat = nullptr;   // peer/HoF weights, or nullptr for scripted spec
  std::string oppSpec;                      // scripted spec, or "mem:peer"/"mem:hof"
  unsigned seed = 0;
  double result = 0;                        // fitness contribution ∈ [-1,1]
  int cls = 0;                              // 0 peer/HoF, 1 pool, 2 anchor (telemetry, #19)
};

void runJobs(std::vector<MatchJob>& jobs, const std::vector<int>& sizes,
             const std::vector<std::vector<double>>& genomes, int threads,
             const MatchParams& mp){
  std::atomic<size_t> nextJob{0};
  auto worker = [&](){
    for (size_t j; (j = nextJob.fetch_add(1)) < jobs.size(); ){
      MatchJob& mj = jobs[j];
      mj.result = playMatchVsSpec(sizes, genomes[(size_t)mj.owner], mj.oppSpec, mj.oppFlat,
                                  mj.seed, mp);
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
}

} // namespace

Control readControl(const std::string& outDir){
  std::ifstream f(outDir + "/control");
  if (!f) return Control::Run;
  std::string tok;
  f >> tok;                                   // first token; whitespace-tolerant
  if (tok == "pause") return Control::Pause;
  if (tok == "stop")  return Control::Stop;
  if (tok == "eval")  return Control::Eval;
  return Control::Run;
}

void writeStatus(const std::string& outDir, const char* state, int gen, int gensTotal){
  if (outDir.empty()) return;
  std::ofstream f(outDir + "/status.json", std::ios::trunc);
  f << "{\"state\":\"" << state << "\",\"gen\":" << gen
    << ",\"gens_total\":" << gensTotal
    << ",\"ts_unix\":" << (long long)std::time(nullptr) << "}\n";
}

void writeRunConfig(const std::string& outDir, const std::string& trainer,
                    const std::string& argvLine){
  if (outDir.empty()) return;
  std::string esc;                                   // minimal JSON string escaping
  esc.reserve(argvLine.size());
  for (char c : argvLine){
    if (c == '"' || c == '\\') esc.push_back('\\');
    esc.push_back(c);
  }
  std::ofstream f(outDir + "/config.json", std::ios::trunc);
  f << "{\"trainer\":\"" << trainer << "\",\"argv\":\"" << esc
    << "\",\"started_unix\":" << (long long)std::time(nullptr) << "}\n";
}

MatchParams evalParams(const EvolveConfig& cfg){
  MatchParams mp{};                         // canonical physics, unshaped, full rally cap
  mp.stride = cfg.stride > 0 ? cfg.stride : 1;   // agent architecture rides along (#18)
  return mp;
}

double shapeFitness(double margin, long ticks, double speedBonus, long speedRefTicks,
                    double lossBonus, double lossProxWeight, double avgMiss, int winScore){
  if (margin > 0.0){                                       // win: wins-only speed bonus (#11)
    if (speedBonus <= 0.0) return margin;
    double frac = 1.0 - (double)ticks / (double)speedRefTicks;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return margin + speedBonus * frac;
  }
  if (margin < 0.0 && lossBonus > 0.0){                    // loss: survival + concession-proximity
    int W = winScore > 0 ? winScore : k::WinScore;
    double survive = (double)ticks / (double)speedRefTicks; // longer survival → less-bad loss
    if (survive < 0.0) survive = 0.0;
    if (survive > 1.0) survive = 1.0;
    double prox = 1.0 - avgMiss;                            // closer concession → less-bad loss
    if (prox < 0.0) prox = 0.0;
    if (prox > 1.0) prox = 1.0;
    double w = lossProxWeight;                              // clamp weight: keeps q ∈ [0,1]
    if (w < 0.0) w = 0.0;
    if (w > 1.0) w = 1.0;
    double q = (1.0 - w) * survive + w * prox;             // ∈ [0,1]
    double effLoss = lossBonus;                             // clamp: guarantees any win > any loss
    double cap = 2.0 / (double)W - 1e-9;
    if (effLoss > cap) effLoss = cap;
    return margin + effLoss * q;
  }
  return margin;                                            // RallyCap/draw/unshaped loss: unchanged
}

void writePopulation(std::ostream& out, const std::vector<int>& sizes,
                     const std::vector<std::vector<double>>& genomes){
  out << "aipong-pop 1\n" << genomes.size() << "\n";
  for (size_t i = 0; i < sizes.size(); ++i) out << sizes[i] << (i + 1 < sizes.size() ? ' ' : '\n');
  out.precision(17);                                  // exact double round-trip
  for (const auto& g : genomes){
    for (size_t i = 0; i < g.size(); ++i) out << g[i] << (i + 1 < g.size() ? ' ' : '\n');
  }
}

bool readPopulation(std::istream& in, std::vector<int>& sizesOut,
                    std::vector<std::vector<double>>& genomesOut){
  std::string magic; int version = 0; size_t count = 0;
  if (!(in >> magic >> version >> count) || magic != "aipong-pop" || version != 1) return false;
  std::string line;
  std::getline(in, line);
  if (!std::getline(in, line)) return false;
  std::vector<int> sizes;
  { std::istringstream ls(line); int v; while (ls >> v) sizes.push_back(v); }
  if (!sizesInBounds(sizes)) return false;
  size_t n = flatSize(sizes);
  if (count == 0 || count > 10000 || count * n > 16'000'000)   // genome grid bounded too
    return false;
  std::vector<std::vector<double>> genomes(count, std::vector<double>(n));
  for (auto& g : genomes)
    for (double& x : g) if (!(in >> x)) return false;
  sizesOut = sizes;
  genomesOut = std::move(genomes);
  return true;
}

bool readModelFlat(const std::string& path, std::vector<int>& sizesOut, std::vector<double>& flatOut){
  std::ifstream f(path);
  if (!f) return false;
  std::string magic; int version = 0;
  if (!(f >> magic >> version) || magic != "aipong-mlp" || (version != 1 && version != 2)) return false;
  std::string line;
  std::getline(f, line);
  if (version == 2 && !std::getline(f, line)) return false;   // skip "obs stack K"
  if (!std::getline(f, line)) return false;
  std::vector<int> sizes;
  { std::istringstream ls(line); int v; while (ls >> v) sizes.push_back(v); }
  if (!sizesInBounds(sizes)) return false;
  std::vector<double> flat(flatSize(sizes));
  for (double& x : flat) if (!(f >> x)) return false;
  sizesOut = sizes;
  flatOut = std::move(flat);
  return true;
}

bool widenGenome(const std::vector<int>& oldSizes, const std::vector<double>& oldFlat,
                 int newStack, std::vector<int>& newSizesOut, std::vector<double>& newFlatOut){
  if (oldSizes.size() < 2 || oldSizes.front() != 6 || newStack < 1) return false;
  if (oldFlat.size() != flatSize(oldSizes)) return false;
  std::vector<int> ns = oldSizes;
  ns.front() = 6 * newStack;
  std::vector<double> nf(flatSize(ns), 0.0);
  const int h = oldSizes[1];
  for (int o = 0; o < h; ++o)                              // newest-frame slot gets the old weights
    for (int j = 0; j < 6; ++j)
      nf[(size_t)o * (size_t)ns.front() + (size_t)(6 * (newStack - 1) + j)]
          = oldFlat[(size_t)o * 6 + (size_t)j];
  const size_t oldOff = (size_t)h * 6, newOff = (size_t)h * (size_t)ns.front();
  std::copy(oldFlat.begin() + (long)oldOff, oldFlat.end(), nf.begin() + (long)newOff);
  newSizesOut = std::move(ns);
  newFlatOut  = std::move(nf);
  return true;
}

double playMatchVsSpec(const std::vector<int>& sizes, const std::vector<double>& genome,
                       const std::string& oppSpec, const std::vector<double>* oppFlat,
                       unsigned matchSeed, const MatchParams& mp){
  MatchRunner r(matchSeed);
  if (mp.rallyCap > 0) r.setRallyCap(mp.rallyCap);
  if (mp.speedup > 0.0) r.setSpeedup(mp.speedup);
  if (mp.serveHold >= 0.0) r.setServeHold(mp.serveHold);
  if (mp.winScore > 0) r.setWinScore(mp.winScore);
  const int stk = sizes.front() / 6;                  // stack is implied by the input width
  auto self = std::make_unique<MlpAgent>("genome");
  if (!self->loadWeights(sizes, genome, stk, mp.stride)) return -1.0;
  r.setController(Side::Left, std::move(self), "mem:genome");
  if (oppFlat){
    auto opp = std::make_unique<MlpAgent>("opp");
    if (!opp->loadWeights(sizes, *oppFlat, stk, mp.stride)) return -1.0;
    r.setController(Side::Right, std::move(opp), oppSpec);
  } else {
    r.setController(Side::Right, oppSpec);            // scripted anchor via the registry
  }
  r.newGame(matchSeed);
  Outcome o = r.runToCompletion();
  Snapshot s = r.snapshot();
  if (o == Outcome::RallyCap) return -1.0;            // mutual loss (Decision 4) as anti-stall pressure
  const double ws = mp.winScore > 0 ? mp.winScore : k::WinScore;
  double margin = double(s.score.left - s.score.right) / ws;
  double avgMiss = s.missN[0] > 0 ? s.missSum[0] / s.missN[0] : 0.0;   // policy is left ([0])
  return shapeFitness(margin, r.ticks(), mp.speedBonus, mp.speedRefTicks,
                      mp.lossBonus, mp.lossProxWeight, avgMiss, mp.winScore);
}

double playMatchVsGenome(const std::vector<int>& selfSizes, const std::vector<double>& selfGenome,
                         const std::vector<int>& oppSizes,  const std::vector<double>& oppGenome,
                         unsigned matchSeed, const MatchParams& mp){
  MatchRunner r(matchSeed);
  if (mp.rallyCap > 0) r.setRallyCap(mp.rallyCap);
  if (mp.speedup > 0.0) r.setSpeedup(mp.speedup);
  if (mp.serveHold >= 0.0) r.setServeHold(mp.serveHold);
  if (mp.winScore > 0) r.setWinScore(mp.winScore);
  auto self = std::make_unique<MlpAgent>("self");
  if (!self->loadWeights(selfSizes, selfGenome, selfSizes.front() / 6, mp.stride)) return -1.0;
  r.setController(Side::Left, std::move(self), "mem:self");
  auto opp = std::make_unique<MlpAgent>("opp");
  if (!opp->loadWeights(oppSizes, oppGenome, oppSizes.front() / 6, mp.stride)) return -1.0;
  r.setController(Side::Right, std::move(opp), "mem:opp");
  r.newGame(matchSeed);
  Outcome o = r.runToCompletion();
  Snapshot s = r.snapshot();
  if (o == Outcome::RallyCap) return -1.0;            // mutual loss (Decision 4)
  const double ws = mp.winScore > 0 ? mp.winScore : k::WinScore;
  double margin = double(s.score.left - s.score.right) / ws;
  double avgMiss = s.missN[0] > 0 ? s.missSum[0] / s.missN[0] : 0.0;   // policy is left ([0])
  return shapeFitness(margin, r.ticks(), mp.speedBonus, mp.speedRefTicks,
                      mp.lossBonus, mp.lossProxWeight, avgMiss, mp.winScore);
}

void playMatchesBatched(const std::vector<int>& sizes, std::vector<GenomeMatch>& matches,
                        const MatchParams& mp, int threads, int lanes){
  int T = threads > 0 ? threads : (int)std::thread::hardware_concurrency();
  if (T < 1) T = 1;
  if (lanes < 1) lanes = 1;
  const double ws = mp.winScore > 0 ? mp.winScore : k::WinScore;
  std::atomic<size_t> next{0};

  const int stack  = sizes.front() / 6;
  const int stride = mp.stride > 0 ? mp.stride : 1;
  const int window = (stack - 1) * stride + 1;        // per-lane history depth [ticks]
  const int obsDim = 6 * stack;

  auto worker = [&](){
    BatchedMlp netL(sizes, lanes), netR(sizes, lanes);
    struct Lane { State st; int job = -1; long ticks = 0; long seen = 0; };
    std::vector<Lane> lane((size_t)lanes);
    std::vector<float> obsL((size_t)lanes * (size_t)obsDim, 0.0f);
    std::vector<float> obsR((size_t)lanes * (size_t)obsDim, 0.0f);
    // history rings (stack > 1): [lane][window][6] per side, indexed by tick count
    std::vector<float> ringL((size_t)lanes * (size_t)window * 6, 0.0f);
    std::vector<float> ringR((size_t)lanes * (size_t)window * 6, 0.0f);
    std::vector<Move> mvL((size_t)lanes), mvR((size_t)lanes);
    int active = 0;

    auto refill = [&](int li){
      Lane& L = lane[(size_t)li];
      for (;;){
        size_t j = next.fetch_add(1);
        if (j >= matches.size()){ L.job = -1; return; }
        GenomeMatch& gm = matches[j];
        if (!gm.self || !gm.opp){ gm.result = -1.0; continue; }   // malformed: score and skip
        L.st = initialState(gm.seed);
        if (mp.rallyCap > 0)    L.st.rallyMax = mp.rallyCap;
        if (mp.speedup > 0.0)   L.st.speedup  = mp.speedup;
        if (mp.winScore > 0)    L.st.winScore = mp.winScore;
        if (mp.serveHold >= 0.0){ L.st.serveHold = mp.serveHold; L.st.serveTimer = mp.serveHold; }
        L.ticks = 0;
        L.seen = 0;                                    // fresh history for the new episode
        L.job = (int)j;
        netL.setLane(li, *gm.self);
        netR.setLane(li, *gm.opp);
        active++;
        return;
      }
    };
    for (int li = 0; li < lanes; ++li) refill(li);

    const bool P = prof::on();                         // AIPONG_PROFILE inference split (#24)
    uint64_t pObs = 0, pFwd = 0, pStep = 0;
    while (active > 0){
      uint64_t pt = P ? prof::ns() : 0;
      for (int li = 0; li < lanes; ++li){
        float* oL = obsL.data() + (size_t)li * (size_t)obsDim;
        float* oR = obsR.data() + (size_t)li * (size_t)obsDim;
        Lane& L = lane[(size_t)li];
        if (L.job < 0){                                // idle lane: zeros keep the math tame
          for (int j = 0; j < obsDim; ++j){ oL[j] = 0.0f; oR[j] = 0.0f; }
          continue;
        }
        auto a = observation(L.st, Side::Left);
        auto b = observation(L.st, Side::Right);
        if (stack == 1){
          for (int j = 0; j < 6; ++j){ oL[j] = (float)a[(size_t)j]; oR[j] = (float)b[(size_t)j]; }
        } else {
          // push into the ring at slot (seen % window), then sample slots
          // t − S·(K−1−f), clamped to the oldest frame this episode has produced —
          // the exact semantics of MlpAgent's deque path.
          size_t slot = (size_t)(L.seen % window);
          float* rL = ringL.data() + ((size_t)li * (size_t)window + slot) * 6;
          float* rR = ringR.data() + ((size_t)li * (size_t)window + slot) * 6;
          for (int j = 0; j < 6; ++j){ rL[j] = (float)a[(size_t)j]; rR[j] = (float)b[(size_t)j]; }
          L.seen++;
          const long newest = L.seen - 1;
          const long oldest = L.seen > window ? L.seen - window : 0;
          for (int f = 0; f < stack; ++f){
            long logical = newest - (long)(stack - 1 - f) * stride;
            if (logical < oldest) logical = oldest;
            size_t src = ((size_t)li * (size_t)window + (size_t)(logical % window)) * 6;
            for (int j = 0; j < 6; ++j){
              oL[f * 6 + j] = ringL[src + (size_t)j];
              oR[f * 6 + j] = ringR[src + (size_t)j];
            }
          }
        }
      }
      if (P){ uint64_t n = prof::ns(); pObs += n - pt; pt = n; }
      netL.forward(obsL.data(), mvL.data());
      netR.forward(obsR.data(), mvR.data());
      if (P){ uint64_t n = prof::ns(); pFwd += n - pt; pt = n; }
      for (int li = 0; li < lanes; ++li){
        Lane& L = lane[(size_t)li];
        if (L.job < 0) continue;
        stepOnce(L.st, Input{ mvL[(size_t)li], mvR[(size_t)li] });
        L.ticks++;
        if (L.st.phase == Phase::GameOver){
          GenomeMatch& gm = matches[(size_t)L.job];
          gm.ticks = L.ticks;
          if (L.st.outcome == Outcome::RallyCap) gm.result = -1.0;
          else {
            double margin = double(L.st.score.left - L.st.score.right) / ws;
            double avgMiss = L.st.missN[0] > 0 ? L.st.missSum[0] / L.st.missN[0] : 0.0;  // policy is left ([0])
            gm.result = shapeFitness(margin, L.ticks, mp.speedBonus, mp.speedRefTicks,
                                     mp.lossBonus, mp.lossProxWeight, avgMiss, mp.winScore);
          }
          active--;
          refill(li);
        }
      }
      if (P) pStep += prof::ns() - pt;
    }
    if (P){ prof::C().obs += pObs; prof::C().fwd += pFwd; prof::C().step += pStep; }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < T; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
}

EvolveResult evolve(const EvolveConfig& cfg, const std::function<void(const GenStats&)>& onGen){
  std::vector<int> sizes = cfg.sizes;
  if (cfg.stack > 1) sizes.front() = 6 * cfg.stack;   // input width follows the stack config
  int T = cfg.threads > 0 ? cfg.threads : (int)std::thread::hardware_concurrency();
  if (T < 1) T = 1;
  const int nElite = std::max(1, (int)(cfg.eliteFrac * cfg.pop));

  // ---- initial population: resume > founders-from-checkpoints > random ----
  std::vector<std::vector<double>> pop((size_t)cfg.pop);
  std::vector<std::vector<double>> hof;
  std::vector<std::vector<double>> founders;
  if (!cfg.resumeDir.empty()){
    std::ifstream pf(cfg.resumeDir + "/population.txt");
    std::vector<int> fsizes;
    if (pf && readPopulation(pf, fsizes, founders) && fsizes == sizes){
      std::ifstream hf(cfg.resumeDir + "/hof.txt");
      std::vector<int> hsizes; std::vector<std::vector<double>> h;
      // Deliberate asymmetry vs. population.txt above: a missing/corrupted/mismatched
      // hof.txt leaves `hof` silently empty here instead of setting resumeFailed — see
      // Decision 22's watch-item.
      if (hf && readPopulation(hf, hsizes, h) && hsizes == sizes) hof = std::move(h);
    } else {
      EvolveResult res; res.sizes = sizes; res.resumeFailed = true;
      return res;                 // refusing to silently restart from random (#22)
    }
  } else if (!cfg.initFrom.empty()){
    std::vector<std::filesystem::path> paths;
    for (const auto& e : std::filesystem::directory_iterator(cfg.initFrom))
      if (e.is_regular_file() && e.path().extension() == ".txt") paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths){
      std::vector<int> fsizes; std::vector<double> flat;
      if (!readModelFlat(p.string(), fsizes, flat)) continue;
      if (fsizes == sizes){
        founders.push_back(std::move(flat));
      } else if (cfg.stack > 1 && fsizes.front() == 6){
        // stack-1 checkpoint into a stacked run: function-preserving widening (evo3 → evo4)
        std::vector<int> ws; std::vector<double> wf;
        std::vector<int> target = fsizes; target.front() = sizes.front();
        if (target == sizes && widenGenome(fsizes, flat, cfg.stack, ws, wf) && ws == sizes)
          founders.push_back(std::move(wf));
      }
    }
    for (size_t i = 0; i < founders.size() && (int)hof.size() < cfg.hofMax; i += 4)
      hof.push_back(founders[i]);                     // spread of era champions as the initial HoF
  }
  for (int i = 0; i < cfg.pop; ++i){
    if (!founders.empty()){
      pop[(size_t)i] = founders[(size_t)i % founders.size()];
      if ((size_t)i >= founders.size()){              // fill with mutated founder clones
        uint64_t ms = rng::key(cfg.seed, TagInit, (uint64_t)i);
        for (auto& w : pop[(size_t)i]) w += cfg.mutEps * rng::gaussian(ms);
      }
    } else {
      pop[(size_t)i] = randomGenome(sizes, rng::key(cfg.seed, TagInit, (uint64_t)i));
    }
  }

  // ---- opponent pool roster (Decision 14): frozen checkpoints from any lineage/arch ----
  std::vector<std::string> poolSpecs;
  for (const auto& dir : cfg.poolDirs){
    std::vector<std::filesystem::path> paths;
    for (const auto& e : std::filesystem::directory_iterator(dir))
      if (e.is_regular_file() && e.path().extension() == ".txt"
          && e.path().filename() != "population.txt" && e.path().filename() != "hof.txt")
        paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths){
      std::string spec = "mlp:" + p.string();
      if (makeAgent(spec)) poolSpecs.push_back(std::move(spec));   // probe: skip unloadables
    }
  }
  int poolGames   = poolSpecs.empty() ? 0 : cfg.poolGames;
  int anchorGames = cfg.anchorGames;
  if (cfg.gamesPerEval - poolGames - anchorGames < 1){  // defensive: peers must keep ≥1 game
    poolGames = 0; anchorGames = 0;
  }
  const int peerGames = cfg.gamesPerEval - poolGames - anchorGames;
  static const char* kBots[3] = {"p:hard", "interceptor:laggy", "interceptor"};

  EvolveResult res; res.sizes = sizes;
  const bool files = !cfg.outDir.empty();
  std::ofstream logf;
  if (files){
    std::filesystem::create_directories(cfg.outDir);
    std::error_code ec;
    std::filesystem::remove(cfg.outDir + "/control", ec);  // stale-state guard: never start paused
    logf.open(cfg.outDir + "/train_log.jsonl", std::ios::app);
    writeStatus(cfg.outDir, "running", 0, cfg.gens);
  }
  bool stopped = false;
  uint64_t champHash = 0; int champAge = 0;              // champion identity across gens (#19)

  for (int gen = 0; gen < cfg.gens; ++gen){
    // ---- operator control (DECISIONS.md #19) — checked between generations only, so a
    // pause draws no RNG and reorders nothing: results are pause-pattern-invariant ----
    bool forceEval = false;
    if (files){
      Control c = readControl(cfg.outDir);
      if (c == Control::Pause){                          // park: 0.5 s polls, cores released
        writeStatus(cfg.outDir, "paused", gen, cfg.gens);
        while (readControl(cfg.outDir) == Control::Pause)
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
        c = readControl(cfg.outDir);
        writeStatus(cfg.outDir, "running", gen, cfg.gens);
      }
      if (c == Control::Stop){                           // graceful: save --resume state, leave
        std::ofstream pf(cfg.outDir + "/population.txt");
        writePopulation(pf, sizes, pop);
        std::ofstream hf(cfg.outDir + "/hof.txt");
        writePopulation(hf, sizes, hof);
        writeStatus(cfg.outDir, "stopped", gen, cfg.gens);
        stopped = true;
        break;
      }
      if (c == Control::Eval){                           // one-shot: force an eval, self-clear
        forceEval = true;
        std::ofstream cf(cfg.outDir + "/control", std::ios::trunc);
        cf << "run";
      }
    }
    auto t0 = std::chrono::steady_clock::now();
    if (prof::on()) prof::C().gens.fetch_add(1, std::memory_order_relaxed);
    uint64_t _pjb = prof::tic();                       // generation-phase timers (#24)

    // ---- fitness: peers/HoF (batched) + pool checkpoints & bots (scalar; Decision 14) ----
    std::vector<MatchJob> jobs;  jobs.reserve((size_t)cfg.pop * (size_t)peerGames);
    std::vector<MatchJob> sjobs; sjobs.reserve((size_t)cfg.pop * (size_t)(poolGames + anchorGames));
    for (int i = 0; i < cfg.pop; ++i){
      uint64_t os = rng::key(cfg.seed, TagOpp, (uint64_t)gen, (uint64_t)i);
      for (int g = 0; g < cfg.gamesPerEval; ++g){
        MatchJob mj; mj.owner = i;
        mj.seed = (unsigned)rng::key(cfg.seed, TagMatch, (uint64_t)gen, (uint64_t)(i * 1024 + g));
        if (g < peerGames){
          bool useHof = !hof.empty() && g == peerGames - 1;        // last peer slot vs an ancestor
          if (useHof){
            size_t h = (size_t)(rng::uniform01(os) * (double)hof.size());
            if (h >= hof.size()) h = hof.size() - 1;
            mj.oppFlat = &hof[h]; mj.oppSpec = "mem:hof";
          } else {
            size_t p = (size_t)(rng::uniform01(os) * (double)(cfg.pop - 1));
            if ((int)p >= i) p++;                                  // peer ≠ self
            mj.oppFlat = &pop[p]; mj.oppSpec = "mem:peer";
          }
          jobs.push_back(std::move(mj));
        } else if (g < peerGames + poolGames){
          size_t k = (size_t)(rng::uniform01(os) * (double)poolSpecs.size());
          if (k >= poolSpecs.size()) k = poolSpecs.size() - 1;
          mj.oppSpec = poolSpecs[k];                               // any architecture — scalar path
          mj.cls = 1;
          sjobs.push_back(std::move(mj));
        } else {
          mj.oppSpec = kBots[(size_t)(rng::uniform01(os) * 3.0) % 3];
          mj.cls = 2;
          sjobs.push_back(std::move(mj));
        }
      }
    }
    const MatchParams trainMp{cfg.speedBonus, cfg.speedRefTicks, cfg.rallyCap, cfg.speedup,
                              cfg.serveHold, cfg.winScore, cfg.stride,
                              cfg.lossBonus, cfg.lossProxWeight};
    // Genome-vs-genome matches take the batched SoA path (Decision 13).
    std::vector<GenomeMatch> gms(jobs.size());
    for (size_t j = 0; j < jobs.size(); ++j){
      gms[j].self = &pop[(size_t)jobs[j].owner];
      gms[j].opp  = jobs[j].oppFlat;
      gms[j].seed = jobs[j].seed;
    }
    prof::toc(prof::C().jobbuild, _pjb);
    uint64_t _pb = prof::tic();
    playMatchesBatched(sizes, gms, trainMp, T);
    for (size_t j = 0; j < jobs.size(); ++j) jobs[j].result = gms[j].result;
    prof::toc(prof::C().batched, _pb);
    uint64_t _ps = prof::tic();
    if (!sjobs.empty()) runJobs(sjobs, sizes, pop, T, trainMp);    // pool + bot games
    prof::toc(prof::C().scalar, _ps);
    uint64_t _psel = prof::tic();

    std::vector<double> fit((size_t)cfg.pop, 0.0);
    double clsSum[3] = {0, 0, 0}; long clsN[3] = {0, 0, 0};   // population totals per class
    for (const auto& mj : jobs){  fit[(size_t)mj.owner] += mj.result;
                                  clsSum[0] += mj.result; clsN[0]++; }
    for (const auto& mj : sjobs){ fit[(size_t)mj.owner] += mj.result;
                                  clsSum[mj.cls] += mj.result; clsN[mj.cls]++; }
    for (auto& f : fit) f /= (double)cfg.gamesPerEval;

    std::vector<int> order((size_t)cfg.pop);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return fit[(size_t)a] > fit[(size_t)b]; });

    GenStats st; st.gen = gen;
    st.best = fit[(size_t)order[0]];
    st.mean = std::accumulate(fit.begin(), fit.end(), 0.0) / (double)cfg.pop;
    if (clsN[0]) st.fitPeer   = clsSum[0] / (double)clsN[0];  // population-mean margin per class
    if (clsN[1]) st.fitPool   = clsSum[1] / (double)clsN[1];
    if (clsN[2]) st.fitAnchor = clsSum[2] / (double)clsN[2];
    {   // champion identity: FNV-1a over the best genome's bytes → age in generations
      uint64_t h = 1469598103934665603ull;
      for (double w : pop[(size_t)order[0]])
        for (size_t b = 0; b < sizeof(double); ++b){
          h ^= reinterpret_cast<const unsigned char*>(&w)[b];
          h *= 1099511628211ull;
        }
      champAge = (gen > 0 && h == champHash) ? champAge + 1 : 0;
      champHash = h;
      st.champAge = champAge;
    }

    // ---- hall of fame (copies — safe across generation replacement) ----
    if (cfg.hofEvery > 0 && gen % cfg.hofEvery == 0){
      hof.push_back(pop[(size_t)order[0]]);
      if ((int)hof.size() > cfg.hofMax) hof.erase(hof.begin());
    }

    prof::toc(prof::C().select, _psel);
    uint64_t _pe = prof::tic();

    // ---- anchor evaluation + checkpoint (reporting only, never selection) ----
    bool evalNow = forceEval ||
        (cfg.evalEvery > 0 && (gen % cfg.evalEvery == cfg.evalEvery - 1 || gen == cfg.gens - 1));
    if (evalNow){
      auto winRate = [&](const std::string& spec, uint64_t tag){
        std::vector<MatchJob> ev((size_t)cfg.evalMatches);
        for (int m = 0; m < cfg.evalMatches; ++m){
          ev[(size_t)m].owner = order[0];
          ev[(size_t)m].oppSpec = spec;
          ev[(size_t)m].seed = (unsigned)rng::key(cfg.seed, TagEval, (uint64_t)(gen * 8 + (int)tag), (uint64_t)m);
        }
        runJobs(ev, sizes, pop, T, evalParams(cfg)); // canonical physics + the genome's stride
        int wins = 0;
        for (const auto& e : ev) if (e.result > 0) wins++;
        return (double)wins / (double)cfg.evalMatches;
      };
      st.vsPClassic    = winRate("p:classic", 0);
      st.vsLaggy       = winRate("interceptor:laggy", 1);
      st.vsInterceptor = winRate("interceptor", 2);
      st.vsHeldP       = winRate("p:heldout", 3);           // never in fitness — generalization
      st.vsHeldIcept   = winRate("interceptor:heldout", 4); // gauge (Decision 14)
      if (files){
        char name[64];
        std::snprintf(name, sizeof name, "/gen_%04d.txt", gen);
        std::ofstream ck(cfg.outDir + name);
        writeModelText(ck, sizes, pop[(size_t)order[0]], cfg.stack, cfg.stride);
        std::ofstream best(cfg.outDir + "/best.txt");
        writeModelText(best, sizes, pop[(size_t)order[0]], cfg.stack, cfg.stride);
        std::ofstream pf(cfg.outDir + "/population.txt");   // continuation checkpoint (--resume)
        writePopulation(pf, sizes, pop);
        std::ofstream hf(cfg.outDir + "/hof.txt");
        writePopulation(hf, sizes, hof);
      }
    }

    prof::toc(prof::C().eval, _pe);
    uint64_t _prep = prof::tic();

    st.sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (files && logf){
      logf << "{\"gen\":" << gen << ",\"best\":" << st.best << ",\"mean\":" << st.mean
           << ",\"vs_p_classic\":" << st.vsPClassic << ",\"vs_laggy\":" << st.vsLaggy
           << ",\"vs_interceptor\":" << st.vsInterceptor
           << ",\"vs_held_p\":" << st.vsHeldP << ",\"vs_held_icept\":" << st.vsHeldIcept
           << ",\"fit_peer\":" << st.fitPeer << ",\"fit_pool\":" << st.fitPool
           << ",\"fit_anchor\":" << st.fitAnchor << ",\"champ_age\":" << st.champAge
           << ",\"sec\":" << st.sec << "}\n";
      logf.flush();
    }
    if (files) writeStatus(cfg.outDir, "running", gen, cfg.gens);   // heartbeat for dashboards
    res.log.push_back(st);
    if (onGen) onGen(st);
    prof::toc(prof::C().report, _prep);

    // ---- next generation: elites survive, rest are mutated elite clones ----
    uint64_t _pbr = prof::tic();
    if (gen + 1 < cfg.gens){
      std::vector<std::vector<double>> next((size_t)cfg.pop);
      for (int e = 0; e < nElite; ++e) next[(size_t)e] = pop[(size_t)order[(size_t)e]];
      for (int c = nElite; c < cfg.pop; ++c){
        uint64_t ms = rng::key(cfg.seed, TagMutate, (uint64_t)gen, (uint64_t)c);
        size_t pIdx = (size_t)(rng::uniform01(ms) * (double)nElite);
        if (pIdx >= (size_t)nElite) pIdx = (size_t)nElite - 1;
        std::vector<double> child = pop[(size_t)order[pIdx]];
        for (auto& w : child) w += cfg.mutEps * rng::gaussian(ms);
        next[(size_t)c] = std::move(child);
      }
      pop = std::move(next);
    } else {
      res.bestFlat = pop[(size_t)order[0]];
      res.bestFitness = st.best;
    }
    prof::toc(prof::C().select, _pbr);
  }
  res.stopped = stopped;
  if (files && !stopped)
    writeStatus(cfg.outDir, "finished", cfg.gens - 1, cfg.gens);
  return res;
}

} // namespace pong
