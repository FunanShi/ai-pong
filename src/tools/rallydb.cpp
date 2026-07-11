// rallydb — the rally database CLI (see docs/formats.md for formats).
//   rallydb build [dir]                      scan dir (default datasets) -> dir/rallies.jsonl
//   rallydb list  [dir] [--sort ticks|hits|vmax|dur] [--asc] [--winner left|right|none]
//                 [--end left_missed|right_missed|rally_cap|truncated] [--player SUBSTR] [--top N]
//   rallydb extract <match_file> <rally_idx> [dir]   print that rally's tick lines (JSONL)
#include "rally_index.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace pong;
namespace fs = std::filesystem;

static int usage(){
  std::fprintf(stderr,
    "usage: rallydb build   [dir]\n"
    "       rallydb list    [dir] [--sort ticks|hits|vmax|dur] [--asc] [--winner W]\n"
    "                       [--end E] [--player SUBSTR] [--top N]\n"
    "       rallydb extract <match_file> <rally_idx> [dir]\n");
  return 1;
}

static std::vector<RallyRow> loadIndex(const std::string& dir){
  std::ifstream f(dir + "/rallies.jsonl");
  if (!f){ std::fprintf(stderr, "no %s/rallies.jsonl — run: rallydb build %s\n", dir.c_str(), dir.c_str()); return {}; }
  return readIndex(f);
}

static int cmdBuild(const std::string& dir){
  std::vector<RallyRow> all;
  int files = 0;
  std::vector<fs::path> paths;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.is_regular_file() && e.path().extension() == ".jsonl" && e.path().filename() != "rallies.jsonl")
      paths.push_back(e.path());
  std::sort(paths.begin(), paths.end());          // deterministic index order
  for (const auto& p : paths){
    std::ifstream f(p);
    if (!f) continue;
    auto rows = indexMatchStream(f, p.filename().string());
    all.insert(all.end(), rows.begin(), rows.end());
    files++;
  }
  std::ofstream out(dir + "/rallies.jsonl");
  if (!out){ std::fprintf(stderr, "cannot write %s/rallies.jsonl\n", dir.c_str()); return 1; }
  writeIndex(out, all);
  std::printf("indexed %zu rallies from %d match files -> %s/rallies.jsonl\n", all.size(), files, dir.c_str());
  return 0;
}

static int cmdList(int argc, char** argv, std::string dir){
  std::string sortKey, winner, end, player;
  bool asc = false;
  long top = -1;
  for (int i = 0; i < argc; ++i){
    std::string a = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc){ std::fprintf(stderr, "%s needs a value\n", what); std::exit(1); }
      return argv[++i];
    };
    if      (a == "--sort")   sortKey = next("--sort");
    else if (a == "--asc")    asc = true;
    else if (a == "--winner") winner = next("--winner");
    else if (a == "--end")    end = next("--end");
    else if (a == "--player") player = next("--player");
    else if (a == "--top")    top = std::strtol(next("--top"), nullptr, 10);
    else if (a.rfind("--", 0) != 0) dir = a;
    else { std::fprintf(stderr, "unknown flag %s\n", a.c_str()); return usage(); }
  }
  auto rows = loadIndex(dir);
  if (rows.empty()) return 1;

  std::vector<RallyRow> v;
  for (const auto& r : rows){
    if (!winner.empty() && r.winner != winner) continue;
    if (!end.empty() && r.end != end) continue;
    if (!player.empty() && r.left.find(player) == std::string::npos
                        && r.right.find(player) == std::string::npos) continue;
    v.push_back(r);
  }
  if (!sortKey.empty()){
    auto key = [&](const RallyRow& r) -> double {
      if (sortKey == "hits") return r.hits;
      if (sortKey == "vmax") return r.vmax;
      if (sortKey == "dur")  return r.dur_s;
      return r.ticks;
    };
    std::stable_sort(v.begin(), v.end(),
      [&](const RallyRow& a, const RallyRow& b){ return asc ? key(a) < key(b) : key(a) > key(b); });
  }
  if (top >= 0 && (size_t)top < v.size()) v.resize((size_t)top);

  std::printf("%-34s %-3s %-18s %-18s %-6s %-13s %5s %6s %8s %8s %5s\n",
              "match", "#", "left", "right", "win", "end", "hits", "ticks", "dur[s]", "vmax", "srv");
  for (const auto& r : v)
    std::printf("%-34.34s %-3d %-18.18s %-18.18s %-6s %-13s %5d %6d %8.2f %8.2f %5s\n",
                r.match.c_str(), r.rallyIdx, r.left.c_str(), r.right.c_str(),
                r.winner.c_str(), r.end.c_str(), r.hits, r.ticks, r.dur_s, r.vmax,
                r.servedTo.c_str());
  std::printf("(%zu rallies%s)\n", v.size(), sortKey.empty() ? "" : (", sorted by " + sortKey).c_str());
  return 0;
}

static int cmdExtract(const std::string& match, int rallyIdx, const std::string& dir){
  auto rows = loadIndex(dir);
  for (const auto& r : rows){
    if (r.match != match || r.rallyIdx != rallyIdx) continue;
    std::ifstream f(dir + "/" + match);
    if (!f){ std::fprintf(stderr, "cannot open %s/%s\n", dir.c_str(), match.c_str()); return 1; }
    std::printf("%s\n", toJson(r).c_str());        // rally header, then the raw tick lines
    std::string line;
    for (int n = 1; std::getline(f, line); ++n){
      if (n < r.lineStart) continue;
      if (n > r.lineEnd) break;
      std::printf("%s\n", line.c_str());
    }
    return 0;
  }
  std::fprintf(stderr, "rally %s:%d not in index (rebuild?)\n", match.c_str(), rallyIdx);
  return 1;
}

int main(int argc, char** argv){
  if (argc < 2) return usage();
  std::string cmd = argv[1];
  if (cmd == "build")   return cmdBuild(argc > 2 ? argv[2] : "datasets");
  if (cmd == "list")    return cmdList(argc - 2, argv + 2, "datasets");
  if (cmd == "extract"){
    if (argc < 4) return usage();
    return cmdExtract(argv[2], (int)std::strtol(argv[3], nullptr, 10), argc > 4 ? argv[4] : "datasets");
  }
  return usage();
}
