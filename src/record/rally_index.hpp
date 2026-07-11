#pragma once
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace pong {

// The rally database: one row per rally, derived by scanning aipong_match v1 files
// (docs/formats.md). Raw match files remain the single source of truth — rows carry
// the exact line span so training pipelines extract tick data without copying it.
// A rally = serve → point / rally-cap / end-of-recording.
struct RallyRow {
  std::string match;         // source file name (as passed at index-build time)
  int         rallyIdx = 0;  // 0-based within the match
  std::string left, right;   // controller labels (from match metadata)
  unsigned    seed = 0;
  std::string winner;        // "left" | "right" | "none" (rally_cap and truncated have no winner)
  std::string end;           // "left_missed" | "right_missed" | "rally_cap" | "truncated"
  int    lineStart = 0;      // 1-based line numbers in the source file, inclusive span
  int    lineEnd   = 0;
  int    ticks = 0;          // tick lines in the rally (includes the serve hold)
  double dur_s = 0;          // ticks × k::Dt  [s of sim time]
  int    hits  = 0;          // paddle hits (max pre-step rally counter observed)
  double vmax  = 0;          // max |ball velocity| observed  [cu/s]
  std::string servedTo;      // "left" | "right" — who received the serve
};

// Parse one match stream into rally rows. The parser is deliberately coupled to the
// MatchRecorder v1 writer (targeted key scanning, no general JSON) — pinned by tests.
std::vector<RallyRow> indexMatchStream(std::istream&, const std::string& matchName);

std::string toJson(const RallyRow&);
void writeIndex(std::ostream&, const std::vector<RallyRow>&);   // header line + one row per line
std::vector<RallyRow> readIndex(std::istream&);

} // namespace pong
