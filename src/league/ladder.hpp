#pragma once
#include <string>
#include <vector>

namespace pong {

// The Elo ladder (DECISIONS.md #12): rates a roster of controller specs — scripted
// anchors + mlp: checkpoints from any number of runs — by round-robin match play on
// CANONICAL physics (no training knobs, full rally cap). This is the deployment-champion
// selector: self-play fitness picks who trains, the ladder picks who ships.
struct LadderConfig {
  std::vector<std::string> roster;   // controller specs; unloadable entries are skipped
  int      perPair = 2;              // matches per unordered pair (sides alternate)
  unsigned seed    = 1;
  int      threads = 0;              // 0 = hardware_concurrency
  std::string outDir = "results/ladder"; // ratings.jsonl + matches.jsonl ("" = no files)
};

struct MatchRecord {
  int a = 0, b = 0;                  // roster indices
  unsigned seed = 0;
  int winner = -1;                   // 0 = a, 1 = b, -1 = rally cap (no winner)
  int scoreA = 0, scoreB = 0;
  long ticks = 0;
};

struct RatingRow {
  std::string spec, label;
  double elo = 0;                    // -1e9 sentinel when no rated games
  int games = 0, wins = 0, caps = 0; // rated games / wins / cap (excluded-from-rating) matches
};

// Bradley–Terry strengths via the MM algorithm — order-free and deterministic, unlike
// sequential Elo updates. +0.5/+0.5 smoothing per pair with rated games (keeps undefeated
// specs finite). RallyCap matches are EXCLUDED from rating (Decision 4: no winner).
// Scale: elo = 1000 + 400·log10(strength / strength[pinIndex]); pinIndex -1 pins the
// geometric mean to 1000 instead. Exposed for unit tests.
std::vector<double> bradleyTerryElo(int n, const std::vector<MatchRecord>& results,
                                    int pinIndex);

struct LadderResult {
  std::vector<RatingRow>   rows;     // sorted by elo, descending
  std::vector<MatchRecord> matches;
};

LadderResult runLadder(const LadderConfig&);

} // namespace pong
