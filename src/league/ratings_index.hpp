#pragma once
// Ratings index for the Play GUI's model picker (DECISIONS.md #21). Elo is POOL-RELATIVE
// (Decisions 12/16: the same checkpoint rates 2014 in the champions sweep and 1576 in a
// 355-spec sweep), so one number per model is a sweep choice: the sweep dir named
// "champions" is canonical when it rates a model; otherwise the newest ratings.jsonl
// (mtime) wins. Callers must display the source sweep next to the number.
// Pre-reorg sweeps say models/_leaderboard/ — normalized to models/leaderboard/ (#17).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace pong {

struct Rating { double elo = 0; int games = 0; std::string sweep; };
struct PlayModel { std::string spec, name, sweep; double elo = -1; };  // elo < 0 = unrated

inline std::string normalizeRatingSpec(std::string spec){
  static const std::string pre = "mlp:models/_leaderboard/";
  if (spec.rfind(pre, 0) == 0)
    spec = "mlp:models/leaderboard/" + spec.substr(pre.size());
  return spec;
}

// One ratings.jsonl row -> (spec, elo, games); false for the header row / junk lines.
inline bool parseRatingRow(const std::string& line, std::string& spec, double& elo, int& games){
  const size_t k = line.find("\"spec\":\"");
  if (k == std::string::npos) return false;
  const size_t e = line.find('"', k + 8);
  if (e == std::string::npos) return false;
  spec = line.substr(k + 8, e - (k + 8));
  const size_t p = line.find("\"elo\":");
  if (p == std::string::npos) return false;
  elo = std::strtod(line.c_str() + p + 6, nullptr);
  const size_t g = line.find("\"games\":");
  games = g == std::string::npos ? 0 : std::atoi(line.c_str() + g + 8);
  return true;
}

inline std::map<std::string, Rating> loadRatings(const std::string& ladderDir){
  namespace fs = std::filesystem;
  struct Sweep { std::string name; fs::path file; fs::file_time_type mtime; };
  std::vector<Sweep> sweeps;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(ladderDir, ec)){
    if (!e.is_directory()) continue;
    const fs::path f = e.path() / "ratings.jsonl";
    if (fs::exists(f, ec))
      sweeps.push_back({e.path().filename().string(), f, fs::last_write_time(f, ec)});
  }
  std::sort(sweeps.begin(), sweeps.end(),
            [](const Sweep& a, const Sweep& b){ return a.mtime < b.mtime; });
  std::map<std::string, Rating> out;
  auto ingest = [&](const Sweep& s){
    std::ifstream f(s.file);
    std::string line, spec; double elo = 0; int games = 0;
    while (std::getline(f, line))
      if (parseRatingRow(line, spec, elo, games)){
        if (games <= 0) continue;   // -1e9 unrated sentinels (all-rally-cap specs) are not ratings
        out[normalizeRatingSpec(spec)] = Rating{elo, games, s.name};
      }
  };
  for (const auto& s : sweeps) if (s.name != "champions") ingest(s);  // oldest→newest overwrite
  // canonical: champions overwrites a spec's rating only among rows it actually RATES —
  // two exceptions to a flat "always wins" rule: a spec absent from champions' ratings.jsonl,
  // or present with a sentinel games==0 row (skipped above), both leave the newest
  // non-champions sweep's number untouched.
  for (const auto& s : sweeps) if (s.name == "champions") ingest(s);
  return out;
}

inline std::vector<PlayModel> curatedModels(const std::string& modelsDir,
                                            const std::map<std::string, Rating>& ratings){
  namespace fs = std::filesystem;
  std::vector<fs::path> files;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(modelsDir + "/leaderboard", ec))
    if (e.is_regular_file() && e.path().extension() == ".txt") files.push_back(e.path());
  for (const auto& e : fs::directory_iterator(modelsDir, ec)){
    if (!e.is_directory()) continue;
    for (const char* b : {"best.txt", "latest.txt"}){
      const fs::path f = e.path() / b;
      if (fs::exists(f, ec)) files.push_back(f);
    }
  }
  std::sort(files.begin(), files.end());
  files.erase(std::unique(files.begin(), files.end()), files.end());
  std::vector<PlayModel> out;
  for (const auto& f : files){
    PlayModel m;
    m.spec = "mlp:" + f.generic_string();
    m.name = f.parent_path().filename().string() + "/" + f.stem().string();
    // ratings.jsonl specs are always repo-relative "mlp:models/..." (the ladder runs from
    // the repo root), so the lookup key uses that literal prefix + the file's path inside
    // modelsDir — independent of whether modelsDir itself is absolute or relative.
    const std::string rel = fs::relative(f, modelsDir, ec).generic_string();
    const auto it = ratings.find("mlp:models/" + rel);
    if (it != ratings.end()){ m.elo = it->second.elo; m.sweep = it->second.sweep; }
    out.push_back(std::move(m));
  }
  std::stable_sort(out.begin(), out.end(), [](const PlayModel& a, const PlayModel& b){
    if ((a.elo < 0) != (b.elo < 0)) return b.elo < 0;               // rated block first
    return a.elo > b.elo;
  });
  return out;
}

} // namespace pong
