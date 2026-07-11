#pragma once
// Live reader for the trainers' train_log.jsonl (evolve + ppo). Schema-agnostic: parses
// each flat JSON row into key→number, tails the file incrementally (the trainer is
// appending while this reads), and pulls named series for plotting. Header-only + pure
// stdlib so the GUI and the doctest share it. See renderers/imgui/train_view.cpp.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace pong {

// Parse a flat {"key": number, ...} row. Handles negatives, decimals, exponents, and the
// NaN/inf that early ppo episode returns emit; skips array values (ppo's pool_ema) and any
// string values. Tolerant: a malformed or half-written fragment yields whatever it can.
inline void parseLogRow(const std::string& s, std::map<std::string, double>& out){
  const size_t n = s.size();
  size_t i = 0;
  while (i < n){
    if (s[i] != '"'){ ++i; continue; }               // scan to a key's opening quote
    const size_t ke = s.find('"', i + 1);
    if (ke == std::string::npos) break;
    std::string key = s.substr(i + 1, ke - i - 1);
    i = ke + 1;
    while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == ':')) ++i;
    if (i >= n) break;
    if (s[i] == '['){ const size_t e = s.find(']', i); i = (e == std::string::npos) ? n : e + 1; continue; }
    if (s[i] == '"'){ const size_t e = s.find('"', i + 1); i = (e == std::string::npos) ? n : e + 1; continue; }
    const char* start = s.c_str() + i;
    char* end = nullptr;
    const double v = std::strtod(start, &end);
    if (end == start){ ++i; continue; }               // not a number after the colon
    out[key] = v;
    i += (size_t)(end - start);
  }
}

// Incremental tail of one train_log.jsonl. poll() consumes only newline-terminated lines
// (a half-written trailing line waits for the next poll), returns the count of new rows,
// and copes with a file that does not exist yet or was truncated/replaced (re-reads).
struct LogTail {
  std::string path;
  std::uintmax_t offset = 0;
  std::vector<std::map<std::string, double>> rows;
  std::string schema;                                 // "evolution" | "ppo" | ""

  void reset(const std::string& p){ path = p; offset = 0; rows.clear(); schema.clear(); }

  int poll(){
    std::error_code ec;
    if (path.empty() || !std::filesystem::exists(path, ec)) return 0;
    const std::uintmax_t sz = std::filesystem::file_size(path, ec);
    if (ec) return 0;
    if (sz < offset){ offset = 0; rows.clear(); schema.clear(); }   // truncated / replaced
    if (sz <= offset) return 0;
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;
    f.seekg((std::streamoff)offset);
    const std::string chunk((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t base = 0, nl, consumed = 0;
    int added = 0;
    while ((nl = chunk.find('\n', base)) != std::string::npos){
      const std::string line = chunk.substr(base, nl - base);
      base = nl + 1; consumed = base;
      if (line.find('{') == std::string::npos) continue;
      std::map<std::string, double> row;
      parseLogRow(line, row);
      if (row.empty()) continue;
      if (schema.empty())
        schema = row.count("step") ? "ppo" : (row.count("gen") ? "evolution" : "");
      rows.push_back(std::move(row));
      ++added;
    }
    offset += consumed;
    return added;
  }
};

// Trainer heartbeat (<run>/status.json, DECISIONS.md #19): state string + numeric fields.
// Tolerant like parseLogRow — a missing/garbled file yields valid = false.
struct RunStatus {
  std::string state;                                  // "running" | "paused" | "stopped" | "finished"
  double gen = -1, gensTotal = -1, tsUnix = -1;
  bool valid = false;
};
inline RunStatus parseStatus(const std::string& text){
  RunStatus st;
  static const std::string kState = "\"state\":\"";
  const size_t k = text.find(kState);
  if (k == std::string::npos) return st;
  const size_t e = text.find('"', k + kState.size());
  if (e == std::string::npos) return st;
  st.state = text.substr(k + kState.size(), e - (k + kState.size()));
  std::map<std::string, double> nums;
  parseLogRow(text, nums);
  const auto pick = [&](const char* key, double& out){
    const auto it = nums.find(key);
    if (it != nums.end()) out = it->second;
  };
  pick("gen", st.gen);
  pick("gens_total", st.gensTotal);
  pick("ts_unix", st.tsUnix);
  st.valid = !st.state.empty();
  return st;
}

// EMA smoothing for noisy eval curves (alpha = weight of the new sample); [] stays [].
inline std::vector<float> emaSmooth(const std::vector<float>& ys, float alpha){
  std::vector<float> out;
  out.reserve(ys.size());
  float acc = 0.0f;
  for (size_t i = 0; i < ys.size(); ++i){
    acc = (i == 0) ? ys[0] : alpha * ys[i] + (1.0f - alpha) * acc;
    out.push_back(acc);
  }
  return out;
}

// Remaining wall-clock [s] from the last work unit done, the run's total, and the pace
// per unit [s]; -1 when unknowable (no total, no pace yet). Clamped at 0 past the end.
// Units are generic: (gen, gens_total, sec/gen) for evolve, (step, total_steps, 1/sps)
// for ppo — both trainers report progress through the same status keys.
inline double etaSeconds(double done, double total, double secPerUnit){
  if (total <= 0 || secPerUnit <= 0) return -1;
  double remain = total - done;
  if (remain < 0) remain = 0;
  return remain * secPerUnit;
}

// Pull (x=xkey, y=ykey) pairs for plotting; drops rows missing either key, non-finite y
// (early ppo NaN return), and — when dropSentinel — the evolve "-2" anchor-not-evaluated
// marker (real win rates are ≥ 0, so the −1.5 cut is safe).
inline void extractSeries(const std::vector<std::map<std::string, double>>& rows,
                          const std::string& xkey, const std::string& ykey, bool dropSentinel,
                          std::vector<float>& xs, std::vector<float>& ys){
  xs.clear(); ys.clear();
  for (const auto& r : rows){
    const auto xi = r.find(xkey), yi = r.find(ykey);
    if (xi == r.end() || yi == r.end()) continue;
    const double y = yi->second;
    if (!std::isfinite(y)) continue;
    if (dropSentinel && y <= -1.5) continue;
    xs.push_back((float)xi->second);
    ys.push_back((float)y);
  }
}

} // namespace pong
