#pragma once
#include <cstdint>
#include <cmath>

namespace pong::rng {

// splitmix64 — the project-standard deterministic stream (same finalizer the core's
// serve hash uses). Streams are keyed, never shared: seed each from (masterSeed, gen,
// slot, ...) so results are independent of thread scheduling.
inline uint64_t next(uint64_t& s){
  s += 0x9e3779b97f4a7c15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
inline uint64_t key(uint64_t a, uint64_t b, uint64_t c = 0, uint64_t d = 0){
  uint64_t s = a;
  s = next(s) ^ b; s = next(s) ^ c; s = next(s) ^ d;
  return next(s);
}
inline double uniform01(uint64_t& s){ return double(next(s) >> 11) * (1.0 / 9007199254740992.0); }
inline double uniform(uint64_t& s, double lo, double hi){ return lo + (hi - lo) * uniform01(s); }
// Standard normal via Box–Muller (fresh pair each call; fine at trainer scale).
inline double gaussian(uint64_t& s){
  double u1 = uniform01(s), u2 = uniform01(s);
  if (u1 < 1e-300) u1 = 1e-300;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

} // namespace pong::rng
