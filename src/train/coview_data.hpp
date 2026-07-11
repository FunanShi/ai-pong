#pragma once
// Co-evolution dashboard data layer (sibling to train_log_view.hpp): parse the cross-island
// win matrix (crosswin.jsonl, whose array values parseLogRow skips) and build the per-species
// "avg margin vs bots" series (with a win-rate fallback for runs predating marg_* logging).
#include "train_log_view.hpp"
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace pong {

struct CrossWin { int gen = -1; std::vector<std::string> labels; std::vector<std::vector<double>> matrix; };

// Parse {"gen":G,"labels":["a",...],"matrix":[[..],...]}. Tolerant of whitespace; false if the
// labels or matrix are missing or their counts disagree.
inline bool parseCrossWinLine(const std::string& line, CrossWin& out){
  out = CrossWin{};
  const size_t g = line.find("\"gen\":");
  if (g != std::string::npos) out.gen = (int)std::strtod(line.c_str() + g + 6, nullptr);
  const size_t lb = line.find("\"labels\":[");
  if (lb == std::string::npos) return false;
  size_t i = lb + 10;
  while (i < line.size() && line[i] != ']'){
    if (line[i] == '"'){
      const size_t e = line.find('"', i + 1);
      if (e == std::string::npos) return false;
      out.labels.push_back(line.substr(i + 1, e - i - 1));
      i = e + 1;
    } else ++i;
  }
  const size_t mb = line.find("\"matrix\":[");
  if (mb == std::string::npos) return false;
  i = mb + 10;                                       // past the outer '['; now rows [num,...],...
  while (i < line.size() && line[i] != ']'){         // ']' here = end of the outer matrix array
    if (line[i] != '['){ ++i; continue; }
    ++i;                                             // enter a row
    std::vector<double> row;
    while (i < line.size() && line[i] != ']'){
      if (line[i] == ',' || line[i] == ' '){ ++i; continue; }
      char* end = nullptr;
      const double v = std::strtod(line.c_str() + i, &end);
      if (end == line.c_str() + i){ ++i; continue; }
      row.push_back(v); i = (size_t)(end - line.c_str());
    }
    if (i < line.size()) ++i;                        // past the row's ']'
    out.matrix.push_back(std::move(row));
  }
  if (out.labels.empty() || out.matrix.size() != out.labels.size()) return false;
  for (const auto& row : out.matrix) if (row.size() != out.labels.size()) return false;   // reject ragged/truncated rows
  return true;
}

// Parse the last non-empty line of a crosswin.jsonl (the latest eval).
inline bool readLastCrossWin(const std::string& path, CrossWin& out){
  std::ifstream f(path);
  if (!f) return false;
  std::string line, last;
  while (std::getline(f, line)) if (line.find('{') != std::string::npos) last = line;
  return !last.empty() && parseCrossWinLine(last, out);
}

// Per eval row build the "avg vs bots" point. The mode is decided ONCE for the whole series:
// if every eval-contributing row carries a real marg_* triple, plot margin (usedMargin=true,
// y ∈ [−1,1]); otherwise fall back to win-% for all rows (mean vs_* · 100). Both marg_* and vs_*
// use the −2 "not-evaluated" sentinel — values ≤ −1.5 are dropped (real readings are ≥ −1), so
// non-eval rows (which carry the sentinel on both) contribute to neither pass. x = gen.
inline void avgMarginSeries(const std::vector<std::map<std::string,double>>& rows,
                            std::vector<float>& xs, std::vector<float>& ys, bool& usedMargin){
  xs.clear(); ys.clear();
  static const char* MK[3] = {"marg_p_classic", "marg_laggy", "marg_interceptor"};
  static const char* VK[3] = {"vs_p_classic", "vs_laggy", "vs_interceptor"};
  auto realTriple = [](const std::map<std::string,double>& r, const char* const k[3], double& mean){
    double sum = 0;
    for (int i = 0; i < 3; ++i){ const auto it = r.find(k[i]); if (it == r.end() || it->second <= -1.5) return false; sum += it->second; }
    mean = sum / 3.0; return true;
  };
  // Pass 1 — pick the mode once: margin only if EVERY contributing row has a real marg_* triple.
  bool anyContrib = false, allMargin = true; double tmp;
  for (const auto& r : rows){
    if (r.find("gen") == r.end()) continue;
    const bool hasM = realTriple(r, MK, tmp), hasV = realTriple(r, VK, tmp);
    if (!hasM && !hasV) continue;                    // non-eval row → contributes to neither
    anyContrib = true;
    if (!hasM) allMargin = false;
  }
  usedMargin = anyContrib && allMargin;
  // Pass 2 — emit in the chosen scale.
  for (const auto& r : rows){
    const auto gi = r.find("gen");
    if (gi == r.end()) continue;
    double mean;
    if (usedMargin){
      if (realTriple(r, MK, mean)){ xs.push_back((float)gi->second); ys.push_back((float)mean); }
    } else {
      if (realTriple(r, VK, mean)){ xs.push_back((float)gi->second); ys.push_back((float)(mean * 100.0)); }
    }
  }
}

} // namespace pong
