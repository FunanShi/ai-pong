#pragma once
#include "pong_core.hpp"
#include <fstream>
#include <string>

namespace pong {

// Writes one match as JSONL for training pipelines (format spec: docs/formats.md).
// Line 1 = metadata; then one line per stepped tick with the PRE-step snapshot and the
// actions applied that tick — the (s_t, a_t) alignment BC/RL training wants — plus event
// flags derived from the post-step result (point scored, rally cap); final line = terminal
// summary. Units: cu / cu/s, court frame C (y down); t = tick index at k::Dt per tick.
// Lives outside core/ on purpose: the sim stays IO-free. This is an operator-side tool —
// nothing in the agent interface can start, stop, or observe recording.
class MatchRecorder {
public:
  // Controller names are embedded verbatim in the metadata line (ASCII labels only —
  // no JSON escaping is performed).
  bool begin(const std::string& path, unsigned seed,
             const std::string& leftName, const std::string& rightName);
  void step(const Snapshot& preStep, const Input& applied, const Snapshot& postStep);
  void end(const Snapshot& final, bool truncated);            // safe to call when inactive
  bool active() const { return out_.is_open(); }
  int  ticks() const { return ticks_; }
  const std::string& path() const { return path_; }
private:
  std::ofstream out_;
  std::string path_;
  int ticks_ = 0;
  Score lastScore_{};
};

} // namespace pong
