#include "ladder.hpp"
#include "match_runner.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <thread>

namespace pong {
namespace {
uint64_t mix(uint64_t& s){
  s += 0x9e3779b97f4a7c15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
uint64_t key(uint64_t a, uint64_t b, uint64_t c){
  uint64_t s = a; (void)mix(s); s ^= b; (void)mix(s); s ^= c; return mix(s);
}
} // namespace

std::vector<double> bradleyTerryElo(int n, const std::vector<MatchRecord>& results, int pinIndex){
  std::vector<std::vector<double>> W((size_t)n, std::vector<double>((size_t)n, 0.0));
  for (const auto& m : results){
    if (m.winner == 0) W[(size_t)m.a][(size_t)m.b] += 1.0;
    else if (m.winner == 1) W[(size_t)m.b][(size_t)m.a] += 1.0;
    // winner == -1 (rally cap): excluded from rating entirely
  }
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (W[(size_t)i][(size_t)j] + W[(size_t)j][(size_t)i] > 0.0){
        W[(size_t)i][(size_t)j] += 0.5;                 // smoothing keeps strengths finite
        W[(size_t)j][(size_t)i] += 0.5;
      }
  std::vector<double> g((size_t)n, 1.0), gNew((size_t)n, 1.0);
  for (int iter = 0; iter < 1000; ++iter){
    double maxRel = 0.0;
    for (int i = 0; i < n; ++i){
      double wi = 0.0, denom = 0.0;
      for (int j = 0; j < n; ++j){
        if (j == i) continue;
        double nij = W[(size_t)i][(size_t)j] + W[(size_t)j][(size_t)i];
        if (nij <= 0.0) continue;
        wi    += W[(size_t)i][(size_t)j];
        denom += nij / (g[(size_t)i] + g[(size_t)j]);
      }
      gNew[(size_t)i] = (denom > 0.0) ? wi / denom : g[(size_t)i];
      if (gNew[(size_t)i] <= 0.0) gNew[(size_t)i] = 1e-12;
    }
    // normalize by geometric mean for numeric stability + a fixed gauge
    double logSum = 0.0;
    for (double v : gNew) logSum += std::log(v);
    double gm = std::exp(logSum / n);
    for (int i = 0; i < n; ++i){
      gNew[(size_t)i] /= gm;
      double rel = std::fabs(gNew[(size_t)i] - g[(size_t)i]) / g[(size_t)i];
      if (rel > maxRel) maxRel = rel;
      g[(size_t)i] = gNew[(size_t)i];
    }
    if (maxRel < 1e-12) break;
  }
  double base = (pinIndex >= 0 && pinIndex < n) ? g[(size_t)pinIndex] : 1.0;
  std::vector<double> elo((size_t)n, 0.0);
  for (int i = 0; i < n; ++i) elo[(size_t)i] = 1000.0 + 400.0 * std::log10(g[(size_t)i] / base);
  return elo;
}

LadderResult runLadder(const LadderConfig& cfg){
  // probe-filter the roster (unloadable mlp files would silently play as frozen paddles)
  std::vector<std::string> roster;
  for (const auto& s : cfg.roster)
    if (!isHumanSpec(s) && makeAgent(s)) roster.push_back(s);
  const int n = (int)roster.size();

  std::vector<MatchRecord> jobs;
  int pairIdx = 0;
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j, ++pairIdx)
      for (int m = 0; m < cfg.perPair; ++m){
        MatchRecord r;
        r.a = i; r.b = j;
        r.seed = (unsigned)key(cfg.seed, (uint64_t)pairIdx, (uint64_t)m);
        r.winner = -2;                                  // -2 marks "not yet played"
        r.scoreA = m;                                   // stash side assignment: even m → a plays left
        jobs.push_back(r);
      }

  int T = cfg.threads > 0 ? cfg.threads : (int)std::thread::hardware_concurrency();
  if (T < 1) T = 1;
  std::atomic<size_t> next{0};
  auto worker = [&](){
    for (size_t k; (k = next.fetch_add(1)) < jobs.size(); ){
      MatchRecord& r = jobs[k];
      bool aLeft = (r.scoreA % 2 == 0);                 // alternate sides across the pair
      MatchRunner mr(r.seed);
      mr.setController(Side::Left,  aLeft ? roster[(size_t)r.a] : roster[(size_t)r.b]);
      mr.setController(Side::Right, aLeft ? roster[(size_t)r.b] : roster[(size_t)r.a]);
      mr.newGame(r.seed);
      Outcome o = mr.runToCompletion();
      Snapshot s = mr.snapshot();
      r.ticks = mr.ticks();
      int lScore = s.score.left, rScore = s.score.right;
      r.scoreA = aLeft ? lScore : rScore;
      r.scoreB = aLeft ? rScore : lScore;
      if (o == Outcome::RallyCap) r.winner = -1;
      else if ((o == Outcome::LeftWin) == aLeft) r.winner = 0;
      else r.winner = 1;
    }
  };
  {
    std::vector<std::thread> pool;
    for (int t = 1; t < T; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
  }

  int pin = -1;
  for (int i = 0; i < n; ++i) if (roster[(size_t)i] == "p:classic") pin = i;
  std::vector<double> elo = bradleyTerryElo(n, jobs, pin);

  LadderResult res;
  res.matches = jobs;
  std::vector<int> games((size_t)n, 0), wins((size_t)n, 0), caps((size_t)n, 0);
  for (const auto& m : jobs){
    if (m.winner == -1){ caps[(size_t)m.a]++; caps[(size_t)m.b]++; continue; }
    games[(size_t)m.a]++; games[(size_t)m.b]++;
    if (m.winner == 0) wins[(size_t)m.a]++; else wins[(size_t)m.b]++;
  }
  for (int i = 0; i < n; ++i){
    RatingRow row;
    row.spec = roster[(size_t)i];
    row.label = agentLabel(roster[(size_t)i]);
    row.games = games[(size_t)i]; row.wins = wins[(size_t)i]; row.caps = caps[(size_t)i];
    row.elo = (row.games > 0) ? elo[(size_t)i] : -1e9;  // unrated sentinel
    res.rows.push_back(std::move(row));
  }
  std::stable_sort(res.rows.begin(), res.rows.end(),
                   [](const RatingRow& x, const RatingRow& y){ return x.elo > y.elo; });

  if (!cfg.outDir.empty()){
    std::filesystem::create_directories(cfg.outDir);
    { std::ofstream rf(cfg.outDir + "/ratings.jsonl.tmp");
      rf.setf(std::ios::fixed); rf.precision(1);
      rf << "{\"aipong_ladder\":1,\"specs\":" << n << ",\"per_pair\":" << cfg.perPair
         << ",\"seed\":" << cfg.seed << "}\n";
      for (const auto& r : res.rows)
        rf << "{\"spec\":\"" << r.spec << "\",\"elo\":" << r.elo << ",\"games\":" << r.games
           << ",\"wins\":" << r.wins << ",\"caps\":" << r.caps << "}\n";
    }
    std::error_code wec;
    std::filesystem::rename(cfg.outDir + "/ratings.jsonl.tmp",
                            cfg.outDir + "/ratings.jsonl", wec);   // readers never see a torn file
    { std::ofstream mf(cfg.outDir + "/matches.jsonl.tmp");
      for (const auto& m : res.matches)
        mf << "{\"a\":\"" << roster[(size_t)m.a] << "\",\"b\":\"" << roster[(size_t)m.b]
           << "\",\"seed\":" << m.seed << ",\"winner\":" << m.winner
           << ",\"score\":[" << m.scoreA << ',' << m.scoreB << "],\"ticks\":" << m.ticks << "}\n";
    }
    std::error_code mec;
    std::filesystem::rename(cfg.outDir + "/matches.jsonl.tmp",
                            cfg.outDir + "/matches.jsonl", mec);   // readers never see a torn file
  }
  return res;
}

} // namespace pong
