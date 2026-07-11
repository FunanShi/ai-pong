#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pong {

// Genetic-algorithm self-play trainer (slimevolleygym-derived recipe — DECISIONS.md #10):
// truncation selection + Gaussian mutation over flat MLP weight vectors. Fitness is PURE
// self-play (population peers + hall-of-fame) — scripted anchors are used for progress
// reporting only, never for selection, so the population can't overfit a fixed opponent.
// Deterministic by construction: every match seed and mutation stream is keyed by
// (masterSeed, generation, slot), so results are identical for any thread count.
struct EvolveConfig {
  std::vector<int> sizes{6, 32, 32, 3};  // MLP layer widths (input 6·stack, output 3)
  int      pop          = 64;
  int      gens         = 200;
  int      gamesPerEval = 6;      // self-play matches per individual per generation
  double   mutEps       = 0.1;    // Gaussian mutation std-dev on every weight (slimevolley value)
  double   eliteFrac    = 0.25;   // top fraction survives unchanged; rest = mutated elite clones
  int      hofEvery     = 10;     // push best-of-generation to the hall of fame every N gens
  int      hofMax       = 20;     // hall-of-fame ring size
  int      evalEvery    = 10;     // anchor evaluation + checkpoint cadence (0 = never)
  int      evalMatches  = 16;     // matches per scripted anchor at evaluation time
  unsigned seed         = 1;      // master seed
  int      threads      = 0;      // 0 = hardware_concurrency
  std::string outDir    = "models/evo";   // checkpoints + train_log.jsonl ("" = no file output)
  // Win-speed shaping (0 = off, the comparable default). Additive bonus on WINS only:
  // fitness = margin + speedBonus·max(0, 1 − ticks/speedRefTicks). Ordering guarantees:
  // any win (margin ≥ +1/11) still beats any loss (≤ −1/11) for every β, and the bonus
  // never applies to losses, so defensive play in losing games is not punished.
  double   speedBonus   = 0.0;    // β; suggested 0.2 when enabled
  long     speedRefTicks = 30000; // T_ref [ticks]; ~8 sim-minutes — matches faster than this earn bonus
  double   lossBonus     = 0.0;    // loss-shaping magnitude; 0 = off (byte-identical to today). See shapeFitness.
  double   lossProxWeight = 0.7;   // w_d ∈ [0,1]: concession-proximity weight vs survival-time in loss shaping
  int      rallyCap     = 0;      // 0 = default (k::RallyMax); lower to truncate stalemate marathons
  double   speedup      = 0.0;    // 0 = default (k::Speedup 1.03); e.g. 1.06 ≈ 2× fewer rally ticks.
                                  // TRAINING dynamics only — anchor evals always run canonical physics
  double   serveHold    = -1.0;   // <0 = canonical 0.75 s; 0 skips the dead serve wait in training
  int      winScore     = 0;      // 0 = canonical 11; e.g. 5 halves match length (training only)
  int      stack        = 1;      // observation frames (input width = 6·stack); evo4: 6
  int      stride       = 1;      // ticks between sampled frames; evo4: 6 (≈0.5 s window)
  // Opponent diversity (DECISIONS.md #14): carve fitness games out of gamesPerEval so part
  // of every genome's fitness comes from OUT-OF-META opponents — frozen cross-run
  // checkpoints and scripted bots — making motion-feature overfitting unprofitable.
  std::vector<std::string> poolDirs;  // dirs of checkpoint .txt files (any architecture)
  int      poolGames    = 0;      // games per genome vs seeded picks from the pool roster
  int      anchorGames  = 0;      // games per genome vs rotating {p:hard, laggy, instant}
  // Continuation (see DECISIONS.md #11): resumeDir loads <dir>/population.txt (+hof.txt) —
  // exact population continuation; initFrom seeds founders from every aipong-mlp *.txt in a
  // directory (era champions as founders), filling the rest with mutated clones. resume wins.
  std::string resumeDir;
  std::string initFrom;
};

struct GenStats {
  int    gen = 0;
  double best = 0, mean = 0;      // fitness ∈ [-1,1]: mean score margin, RallyCap counted as -1
  double vsPClassic = -2, vsLaggy = -2, vsInterceptor = -2;  // anchor win rates ∈ [0,1]; -2 = not evaluated
  double vsHeldP = -2, vsHeldIcept = -2;  // held-out variants — NEVER in fitness (DECISIONS.md #14)
  // Curriculum telemetry (DECISIONS.md #19): population-mean shaped margin per opponent
  // class this generation — the live "is the curriculum giving gradient" signal. -2 = the
  // class had no games (mirrors the vs_* sentinel convention).
  double fitPeer = -2, fitPool = -2, fitAnchor = -2;
  int    champAge = 0;             // generations since the best genome last changed (hash)
  double sec = 0;                  // wall-clock for this generation [s]
};

struct EvolveResult {
  std::vector<GenStats>   log;
  std::vector<double>     bestFlat;   // best genome of the final generation
  std::vector<int>        sizes;
  double                  bestFitness = 0;
  bool stopped = false;       // operator control-stop ended the run (population saved)
  bool resumeFailed = false;  // --resume load failed; run refused (no silent random restart, #22)
};

// Runs the full GA. onGen (optional) fires after every generation — the CLI uses it to
// print progress; tests leave it empty.
EvolveResult evolve(const EvolveConfig&, const std::function<void(const GenStats&)>& onGen = {});

// Operator control channel (DECISIONS.md #19): a dashboard or shell writes one token to
// <outDir>/control; evolve() reads it at each generation top. pause parks the main thread
// (0.5 s polls — no worker threads exist between generations, so cores are released);
// stop writes population.txt/hof.txt (lossless --resume state) and exits cleanly; eval
// forces an anchor eval + checkpoint this generation, then self-clears to run. Absent or
// unrecognized = Run. A stale control file is removed at startup — runs never begin paused.
enum class Control { Run, Pause, Stop, Eval };
Control readControl(const std::string& outDir);

// Status heartbeat for dashboards: rewrites <outDir>/status.json with
// {"state":"running|paused|stopped|finished","gen":G,"gens_total":T,"ts_unix":S}.
// Wall-clock IO only — never read back by the trainer, no effect on determinism.
void writeStatus(const std::string& outDir, const char* state, int gen, int gensTotal);

// Provenance (DECISIONS.md #20): how was this run made — written once at startup by the trainer CLI, displayed by the dashboard.
void writeRunConfig(const std::string& outDir, const std::string& trainer, const std::string& argvLine);

// Fitness shaping (see EvolveConfig). Pure; unit-tested. Wins get the #11 wins-only speed
// bonus. Losses (lossBonus>0) get a bounded survival+proximity bonus: slower and closer-
// conceding losses score higher, but eff_loss=min(lossBonus, 2/winScore − 1e-9) keeps ANY
// win STRICTLY above ANY loss. survive, prox, and lossProxWeight are all clamped to [0,1], so
// the guarantee holds for any argument. lossBonus=0 → identity on losses (byte-identical to #11).
double shapeFitness(double margin, long ticks, double speedBonus, long speedRefTicks,
                    double lossBonus = 0.0, double lossProxWeight = 0.7,
                    double avgMiss = 0.0, int winScore = 0);

// Population files ("aipong-pop 1"): header, then sizes line, then one flat genome per line.
void writePopulation(std::ostream&, const std::vector<int>& sizes,
                     const std::vector<std::vector<double>>& genomes);
bool readPopulation(std::istream&, std::vector<int>& sizesOut,
                    std::vector<std::vector<double>>& genomesOut);
// Parse an aipong-mlp v1/v2 weight file back into (sizes, flat) — checkpoint → founder.
bool readModelFlat(const std::string& path, std::vector<int>& sizesOut,
                   std::vector<double>& flatOut);

// Net2Net-style function-preserving widening (Decision 13 / evo4): map a stack-1 genome
// into a stack-K input layer — old input weights land in the NEWEST-frame columns
// [6·(K−1), 6·K), all other first-layer weights are 0, deeper layers copy verbatim.
// The widened net plays bit-identically to its parent (history has zero weight), so
// evo3 founders enter evo4 with no skill loss and evolution grows into the new inputs.
bool widenGenome(const std::vector<int>& oldSizes, const std::vector<double>& oldFlat,
                 int newStack, std::vector<int>& newSizesOut, std::vector<double>& newFlatOut);

// Per-match training knobs (defaults = canonical game + unshaped fitness — what anchor
// evaluations always use).
struct MatchParams {
  double speedBonus    = 0.0;
  long   speedRefTicks = 30000;
  int    rallyCap      = 0;     // 0 = k::RallyMax
  double speedup       = 0.0;   // 0 = k::Speedup
  double serveHold     = -1.0;  // s; <0 = k::ServeHold (training may zero the dead serve wait)
  int    winScore      = 0;     // 0 = k::WinScore (training may shorten games; margin renormalizes)
  int    stride        = 1;     // frame-stack sampling stride for genome agents (stack = input/6)
  double lossBonus      = 0.0;   // loss-shaping magnitude (training only; evals leave it 0)
  double lossProxWeight = 0.7;   // w_d: concession-proximity weight in loss shaping
};

// Params for anchor/held-out evaluations (exposed for tests). Contract: canonical physics,
// UNSHAPED fitness, full rally cap (Decision 11/13 — eval curves stay comparable across runs
// and training knobs), BUT the genome's trained stride rides along — stride is the agent's
// architecture, not a physics knob. Evaluating a stack-K stride-S policy at stride 1
// collapses its (K−1)·S+1-tick history window and misreports it (DECISIONS.md #18).
MatchParams evalParams(const EvolveConfig&);

// One match: genome (left) vs an opponent agent built from `oppSpec` ("mem:" specs are
// resolved against `oppFlat`). Returns shaped fitness contribution. Exposed for tests.
double playMatchVsSpec(const std::vector<int>& sizes, const std::vector<double>& genome,
                       const std::string& oppSpec, const std::vector<double>* oppFlat,
                       unsigned matchSeed, const MatchParams& mp = {});

// Flat MLP weight-vector length for a layer-size list (input·h1 + h1 biases + …).
size_t flatSize(const std::vector<int>& sizes);

// Seeded random genome for `sizes` (weights ∈ [-1/fan_in, 1/fan_in], biases ∈ [-0.05, 0.05]).
// Single source of truth for initial-population layout — every trainer builds byte-identical
// founders from the same key. Exposed for co-evolution (src/train/coevolve.cpp).
std::vector<double> randomGenome(const std::vector<int>& sizes, uint64_t key);

// One CROSS-ARCHITECTURE match: `self` (left, selfSizes) vs `opp` (right, oppSizes), both
// in-memory genomes of possibly different architectures. Mirrors playMatchVsSpec's genome
// branch but loads each side with its OWN sizes — the primitive co-evolution needs for live
// cross-island play. Returns self's shaped fitness ∈ [-1,1]; deterministic in (sizes, genomes,
// seed, params). Both sides ride mp.stride (their shared frame-stack architecture).
double playMatchVsGenome(const std::vector<int>& selfSizes, const std::vector<double>& selfGenome,
                         const std::vector<int>& oppSizes,  const std::vector<double>& oppGenome,
                         unsigned matchSeed, const MatchParams& mp = {});

// The batched fitness path (Decision 13): genome-vs-genome matches stepped in lockstep
// lanes with SoA float32 forwards. Each match's result depends only on (genomes, seed,
// params) — bitwise identical for any thread count or lane width. This path defines
// canonical fitness results; the scalar path remains for scripted opponents (anchor evals).
struct GenomeMatch {
  const std::vector<double>* self = nullptr;   // plays left
  const std::vector<double>* opp  = nullptr;   // plays right
  unsigned seed = 0;
  double result = 0;                           // shaped fitness (from self's perspective)
  long ticks = 0;
};
// lanes = 16. A single-thread bench makes 32 look ~13% faster (more AVX2 accumulators hide
// FMA latency), but the END-TO-END training A/B reversed it: 32 is ~3% slower at 18 threads.
// Real training streams 128 distinct genomes' weights; the wider working set thrashes shared
// L3/DRAM and the MAC is weight-bandwidth-bound (~0.5 FLOP/byte), so the extra ILP doesn't pay.
// Measure end-to-end, not in isolation, before changing this (Decision 25). Lane width is
// bit-identical (thread/lane-invariance doctest), so this is purely a throughput knob.
void playMatchesBatched(const std::vector<int>& sizes, std::vector<GenomeMatch>& matches,
                        const MatchParams& mp, int threads, int lanes = 16);

} // namespace pong
