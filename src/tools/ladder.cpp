// ladder — the Elo ladder CLI (DECISIONS.md #12). Rates checkpoints + scripted anchors by
// round-robin on CANONICAL physics and names the deployment champion.
//   ladder [--dir models/evoN]... [--no-anchors] [--per-pair N] [--seed S]
//          [--threads T] [--out DIR] [--top K]
#include "ladder.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace pong;
namespace fs = std::filesystem;

int main(int argc, char** argv){
  LadderConfig cfg;
  bool anchors = true;
  long top = 40;
  std::vector<std::string> dirs;
  for (int i = 1; i < argc; ++i){
    std::string a = argv[i];
    auto next = [&](){ if (i + 1 >= argc){ std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(1); } return argv[++i]; };
    if      (a == "--dir")        dirs.push_back(next());
    else if (a == "--no-anchors") anchors = false;
    else if (a == "--per-pair")   cfg.perPair = std::atoi(next());
    else if (a == "--seed")       cfg.seed = (unsigned)std::strtoul(next(), nullptr, 10);
    else if (a == "--threads")    cfg.threads = std::atoi(next());
    else if (a == "--out")        cfg.outDir = next();
    else if (a == "--top")        top = std::atol(next());
    else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return 1; }
  }
  if (anchors)
    cfg.roster = {"p:easy", "p:classic", "p:hard", "interceptor:laggy", "interceptor"};
  for (const auto& d : dirs){
    std::error_code ec;
    if (!fs::is_directory(d, ec)){
      std::fprintf(stderr, "--dir %s: no such directory\n", d.c_str());
      return 1;
    }
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(d, ec))
      if (e.is_regular_file() && e.path().extension() == ".txt"
          && e.path().filename() != "population.txt" && e.path().filename() != "hof.txt")
        paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths) cfg.roster.push_back("mlp:" + p.string());
  }
  if (cfg.roster.size() < 2){ std::fprintf(stderr, "roster needs >= 2 specs (use --dir)\n"); return 1; }

  size_t pairs = cfg.roster.size() * (cfg.roster.size() - 1) / 2;
  std::printf("ladder: %zu specs, %zu pairs x %d matches = %zu matches (canonical physics)\n",
              cfg.roster.size(), pairs, cfg.perPair, pairs * (size_t)cfg.perPair);

  LadderResult res = runLadder(cfg);

  std::printf("%4s %-44s %7s %6s %5s %5s\n", "#", "controller", "elo", "games", "win%", "caps");
  long shown = 0;
  for (const auto& r : res.rows){
    if (shown++ >= top) break;
    if (r.elo < -1e8)
      std::printf("%4ld %-44.44s %7s %6d %5s %5d\n", shown, r.label.c_str(), "n/a", r.games, "-", r.caps);
    else
      std::printf("%4ld %-44.44s %7.0f %6d %4.0f%% %5d\n", shown, r.label.c_str(), r.elo,
                  r.games, 100.0 * r.wins / r.games, r.caps);
  }
  for (const auto& r : res.rows)
    if (r.spec.rfind("mlp:", 0) == 0 && r.elo > -1e8){
      std::printf("deployment champion: %s  (elo %.0f)  -> AIPONG_MODEL=%s ./pong\n",
                  r.label.c_str(), r.elo, r.spec.substr(4).c_str());
      break;
    }
  if (!cfg.outDir.empty())
    std::printf("written: %s/ratings.jsonl, %s/matches.jsonl\n", cfg.outDir.c_str(), cfg.outDir.c_str());
  return 0;
}
