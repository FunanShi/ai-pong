#pragma once
#include "pong_core.hpp"
#include <vector>

namespace pong {

// SoA float32 batched MLP forward (Decision 13): B agent-lanes with INDEPENDENT weights
// evaluated in lockstep. Weights are stored lane-contiguous ([nOut][nIn][B]) so the inner
// loop vectorizes across lanes (AVX via the -ffast-math property on this file). Each
// lane's arithmetic touches only its own lane slots, so results are bitwise independent
// of lane composition and thread count — the property the trainer's determinism rests on.
class BatchedMlp {
public:
  BatchedMlp(const std::vector<int>& sizes, int lanes);
  int  lanes() const { return B_; }
  void setLane(int lane, const std::vector<double>& flat);   // repack one genome into SoA
  // obs layout: [lane][obsDim] floats, all lanes every call (idle lanes: zeros are safe).
  // movesOut: one Move per lane (argmax over 3 logits, ties to the lower index).
  void forward(const float* obs, Move* movesOut);
private:
  std::vector<int> sizes_;
  int B_ = 0;
  std::vector<std::vector<float>> w_;   // per layer: [nOut][nIn][B]
  std::vector<std::vector<float>> b_;   // per layer: [nOut][B]
  std::vector<float> actA_, actB_;      // activations: [width][B]
};

// Granular timing of the batched forward for `bench --granular` (DECISIONS.md #24/#25): runs a
// faithful copy of forward()'s inner loop (same TU, same -ffast-math) `iters` times, returning
// seconds. nLayers 0..3 runs the first N layers then argmax (0 = transpose+argmax baseline);
// doMAC=false runs bias-init with the FMA loop skipped (isolates the raw multiply vs setup),
// honoured only at nLayers 3 — the prefix modes always run the MAC; doTanh adds the hidden tanh.
// Prefix / doMAC / doTanh differences isolate per-layer MAC, bias-init, and tanh; varying `lanes`
// probes the compute-vs-bandwidth roofline — non-perturbing (clock read only at the loop boundary).
double benchForwardKernel(int obsDim, int lanes, long iters, int nLayers, bool doMAC, bool doTanh);

// Full forward timed with weights stored at `weightBytes` per element (4=float, 2=int16, 1=int8;
// anything else → float), returning seconds. Same FLOPs, fewer weight bytes streamed — measures the
// ceiling of the weight-traffic lever for the bandwidth-bound multiply (convert cost included). #25.
double benchForwardWeight(int obsDim, int lanes, long iters, int weightBytes);

// Cheaper-tanh study for `bench --granular` (#25 follow-up): times `iters` sweeps of a fixed
// [-6,6] range through method 0 = libm std::tanh, 1 = crude Padé[3/2], or 2 = Padé[7/6] (far
// lower error), returning **ns per element** (sweep size owned internally so callers can't desync
// it) and, via maxErrOut, the method's max abs error vs true tanh.
double benchTanh(int method, long iters, double* maxErrOut);

} // namespace pong
