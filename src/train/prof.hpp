#pragma once
// Opt-in wall-clock profiler for the GA training hot path (DECISIONS.md #24). Enabled by
// the env var AIPONG_PROFILE=1; otherwise every hook is a single predictable branch, so
// normal runs pay nothing. Timing only — reads steady_clock, touches no RNG, state, or
// control flow — so results stay bitwise identical whether it's on or off (the
// thread-invariance doctests pass either way). Accumulates nanoseconds into named phase
// counters and the CLI prints a breakdown at run end (tools/evolve.cpp → prof::dump()).
//
// Two layers:
//   generation phases  — sequential main-thread spans: job build / batched games /
//                        scalar games / selection+breed / anchor eval. These sum to the
//                        per-generation wall clock (`sec` in train_log.jsonl).
//   inference split    — per-tick regions inside the batched kernel, summed across all
//                        worker threads (so it is CPU-time, not wall-clock): observation
//                        assembly / MLP forward / physics step. The RATIO is the point —
//                        which part of a forward-tick dominates.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace pong::prof {

// Non-null AND non-empty: `-e AIPONG_PROFILE=` (empty) stays disabled. Cached once.
inline bool on(){
  static const bool e = [](){ const char* v = std::getenv("AIPONG_PROFILE"); return v && *v; }();
  return e;
}

struct Counters {
  // generation phases — these SUM to the per-generation wall clock (`report` = the log/status
  // write + onGen callback span, so nothing between t0 and gen end is left unaccounted).
  std::atomic<uint64_t> jobbuild{0}, batched{0}, scalar{0}, select{0}, eval{0}, report{0};
  std::atomic<uint64_t> obs{0}, fwd{0}, step{0};                                  // inference split
  std::atomic<uint64_t> gens{0};
};
inline Counters& C(){ static Counters c; return c; }  // one instance across TUs (inline-local static)

using clk = std::chrono::steady_clock;
inline uint64_t ns(){
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clk::now().time_since_epoch()).count();
}

// Paired region timing for the sequential generation phases. tic() returns 0 (no clock
// read) when disabled; toc() adds nothing when disabled — both are one branch off.
inline uint64_t tic(){ return on() ? ns() : 0; }
inline void toc(std::atomic<uint64_t>& bucket, uint64_t t0){
  if (on()) bucket.fetch_add(ns() - t0, std::memory_order_relaxed);
}

inline void dump(){
  if (!on()) return;
  Counters& c = C();
  auto ms = [](uint64_t n){ return n / 1e6; };
  const uint64_t g = c.jobbuild + c.batched + c.scalar + c.select + c.eval + c.report;
  const uint64_t i = c.obs + c.fwd + c.step;
  const uint64_t ng = c.gens.load();
  std::fprintf(stderr, "\n=== AIPONG_PROFILE — %llu generations ===\n", (unsigned long long)ng);
  auto row = [&](const char* n, uint64_t v, uint64_t tot){
    std::fprintf(stderr, "  %-16s %10.1f ms  %5.1f%%  %8.3f ms/gen\n",
                 n, ms(v), tot ? 100.0 * (double)v / (double)tot : 0.0, ng ? ms(v) / (double)ng : 0.0);
  };
  std::fprintf(stderr, "generation phases (wall-clock, main thread):\n");
  row("job build",    c.jobbuild, g);
  row("batched games", c.batched, g);
  row("scalar games",  c.scalar,  g);
  row("selection+breed", c.select, g);
  row("anchor eval",   c.eval,    g);
  row("log+status+cb", c.report,  g);
  row("TOTAL",         g,         g);
  std::fprintf(stderr, "batched inference split (CPU-time across worker threads — ratio is the signal):\n");
  row("obs assembly",  c.obs,  i);
  row("mlp forward",   c.fwd,  i);
  row("physics step",  c.step, i);
}

} // namespace pong::prof
