#include "batch_forward.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pong {

namespace {
// Granular timing kernel for `bench --granular` (DECISIONS.md #24/#25). A faithful copy of
// BatchedMlp::forward's inner loop, in THIS translation unit so it gets the same
// -ffast-math codegen (libmvec vectorized tanh, reassociated reductions) as the real
// kernel — a copy in bench.cpp would be built without fast-math and mismeasure tanh.
// NL (number of layers to run, 0..3) and DoTanh are compile-time, so each variant's inner
// loop has no runtime branch; timing is amortized over `iters` (clock read only at the
// boundary), the only non-perturbing way to resolve sub-microsecond regions. Running layer
// prefixes (NL = 0/1/2/3) and differencing gives MEASURED per-layer MAC cost; sweeping B
// (lanes) probes whether the MAC is FMA-throughput- or weight-bandwidth-bound (the SoA
// layout gives each lane its own weights, so there is no weight reuse across lanes).
// WT = weight storage type (float / int16_t / int8_t): the MAC converts to float, so shrinking
// WT shrinks the weight bytes streamed — the lever for a bandwidth-bound multiply (#25 follow-up).
template <int NL, bool DoMAC, bool DoTanh, typename WT = float>
double timedKernel(int obsDim, int B, long iters){
  const int sizes[4] = {obsDim, 32, 32, 3};
  std::vector<std::vector<WT>> w(3);
  std::vector<std::vector<float>> b(3);
  int maxW = obsDim > 32 ? obsDim : 32;
  for (int l = 0; l < 3; ++l){
    w[l].assign((size_t)sizes[l] * (size_t)sizes[l + 1] * (size_t)B, (WT)0);
    b[l].assign((size_t)sizes[l + 1] * (size_t)B, 0.0f);
    for (size_t k = 0; k < w[l].size(); ++k) w[l][k] = (WT)((int)((k * 7) % 11) - 5);  // non-degenerate for int WT too
    for (size_t k = 0; k < b[l].size(); ++k) b[l][k] = 0.01f * (float)((k * 5) % 7) - 0.03f;
  }
  std::vector<float> obs((size_t)B * (size_t)obsDim), actA((size_t)maxW * (size_t)B),
                     actB((size_t)maxW * (size_t)B);
  for (size_t k = 0; k < obs.size(); ++k) obs[k] = 0.01f * (float)((k * 3) % 13) - 0.05f;
  volatile int sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (long it = 0; it < iters; ++it){
    for (int i = 0; i < obsDim; ++i)                            // transpose [lane][dim]->[dim][lane]
      for (int lane = 0; lane < B; ++lane)
        actA[(size_t)i * (size_t)B + (size_t)lane] = obs[(size_t)lane * (size_t)obsDim + (size_t)i];
    float* x = actA.data(); float* y = actB.data();
    for (int l = 0; l < NL; ++l){
      const int nIn = sizes[l], nOut = sizes[l + 1];
      const bool hidden = (l + 2 < 4);
      const WT* W = w[l].data(); const float* Bv = b[l].data();
      for (int o = 0; o < nOut; ++o){
        float* out = y + (size_t)o * (size_t)B;
        const float* bo = Bv + (size_t)o * (size_t)B;
        for (int lane = 0; lane < B; ++lane) out[lane] = bo[lane];   // bias-init
        if constexpr (DoMAC)                                          // the raw matrix multiply
          for (int i = 0; i < nIn; ++i){
            const WT* wr = W + ((size_t)o * (size_t)nIn + (size_t)i) * (size_t)B;
            const float* xi = x + (size_t)i * (size_t)B;
            for (int lane = 0; lane < B; ++lane) out[lane] += (float)wr[lane] * xi[lane];
          }
        if (DoTanh && hidden)
          for (int lane = 0; lane < B; ++lane) out[lane] = std::tanh(out[lane]);
      }
      float* t = x; x = y; y = t;
    }
    for (int lane = 0; lane < B; ++lane){                       // argmax over first 3 rows
      int best = 0; float bv = x[(size_t)lane];
      for (int o = 1; o < 3; ++o){ float v = x[(size_t)o * (size_t)B + (size_t)lane]; if (v > bv){ bv = v; best = o; } }
      sink += best;
    }
    obs[(size_t)(it % (long)obs.size())] += 1e-9f;              // defeat loop-invariant hoisting
  }
  (void)sink;
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace

// nLayers 0..3 = run the first N layers then argmax (0 = transpose+argmax only); doMAC gates the
// raw multiply (false = bias-init only, isolating MAC vs setup); doTanh adds the hidden tanh.
// Only the combinations runGranular uses are instantiated. Layer-prefix / doMAC / doTanh
// differences isolate per-layer MAC, bias-init, and tanh; sweeping `lanes` gives the roofline.
double benchForwardKernel(int obsDim, int lanes, long iters, int nLayers, bool doMAC, bool doTanh){
  if (nLayers <= 0) return timedKernel<0, true,  false>(obsDim, lanes, iters);   // transpose+argmax
  if (nLayers == 1) return timedKernel<1, true,  false>(obsDim, lanes, iters);
  if (nLayers == 2) return timedKernel<2, true,  false>(obsDim, lanes, iters);
  if (!doMAC)       return timedKernel<3, false, false>(obsDim, lanes, iters);   // bias-init, no MAC
  return doTanh ? timedKernel<3, true, true >(obsDim, lanes, iters)              // full
                : timedKernel<3, true, false>(obsDim, lanes, iters);             // all MAC, no tanh
}

// Full forward with weights stored at `weightBytes` per element (4=float, 2=int16, 1=int8) —
// same FLOPs, fewer bytes streamed. Quantifies the ceiling of the weight-traffic lever for the
// bandwidth-bound multiply (#25 follow-up: the convert cost is included, so it's honest).
double benchForwardWeight(int obsDim, int lanes, long iters, int weightBytes){
  if (weightBytes == 1) return timedKernel<3, true, true, std::int8_t >(obsDim, lanes, iters);
  if (weightBytes == 2) return timedKernel<3, true, true, std::int16_t>(obsDim, lanes, iters);
  return                        timedKernel<3, true, true, float       >(obsDim, lanes, iters);
}

namespace {
// tanh approximations (method 1 = crude clamped Padé[3/2]; 2 = Padé[7/6], far lower error, a few
// more ops). One definition shared by benchTanh's timing loop and error loop — no drift.
inline float approxTanh(float x, int method){
  if (method == 1){
    float t = x > 3.0f ? 3.0f : x < -3.0f ? -3.0f : x;          // overshoots past ±3
    return t * (27.0f + t * t) / (27.0f + 9.0f * t * t);
  }
  const float x2 = x * x;                                       // method 2: Padé[7/6], output-clamped
  const float num = x * (945.0f + x2 * (105.0f + x2));
  const float den = 945.0f + x2 * (420.0f + x2 * 15.0f);
  const float r = num / den;
  return r > 1.0f ? 1.0f : r < -1.0f ? -1.0f : r;
}
} // namespace

// Cheaper-tanh study (Q: a faster tanh than libm's vectorized tanhf, and at what accuracy?). Times
// per element over the pre-activation range and returns max abs error vs true tanh. method 0 = libm
// std::tanh; 1 = Padé[3/2]; 2 = Padé[7/6]. In THIS -ffast-math TU so all vectorize like the kernel.
double benchTanh(int method, long iters, double* maxErrOut){
  const int N = 512;
  std::vector<float> in((size_t)N), out((size_t)N);
  for (int i = 0; i < N; ++i) in[(size_t)i] = -6.0f + 12.0f * (float)i / (float)(N - 1);
  volatile float sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (long it = 0; it < iters; ++it){
    if (method == 0)
      for (int i = 0; i < N; ++i) out[(size_t)i] = std::tanh(in[(size_t)i]);
    else
      for (int i = 0; i < N; ++i) out[(size_t)i] = approxTanh(in[(size_t)i], method);
    sink += out[(size_t)(it % N)];
    in[(size_t)(it % N)] += 1e-9f;                               // defeat hoisting
  }
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  double me = 0.0;
  for (int i = 0; i < N; ++i){
    const float x = in[(size_t)i], ref = std::tanh(x);
    const float ap = method == 0 ? ref : approxTanh(x, method);
    const double e = (double)(ap > ref ? ap - ref : ref - ap);
    if (e > me) me = e;
  }
  *maxErrOut = me; (void)sink;
  return sec / (double)iters / (double)N * 1e9;   // ns per element (N owned here, no caller duplication)
}

BatchedMlp::BatchedMlp(const std::vector<int>& sizes, int lanes)
    : sizes_(sizes), B_(lanes){
  w_.resize(sizes.size() - 1);
  b_.resize(sizes.size() - 1);
  int maxW = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l){
    w_[l].assign((size_t)sizes[l] * (size_t)sizes[l + 1] * (size_t)B_, 0.0f);
    b_[l].assign((size_t)sizes[l + 1] * (size_t)B_, 0.0f);
    if (sizes[l] > maxW) maxW = sizes[l];
    if (sizes[l + 1] > maxW) maxW = sizes[l + 1];
  }
  actA_.assign((size_t)maxW * (size_t)B_, 0.0f);
  actB_.assign((size_t)maxW * (size_t)B_, 0.0f);
}

void BatchedMlp::setLane(int lane, const std::vector<double>& flat){
  size_t off = 0;
  for (size_t l = 0; l + 1 < sizes_.size(); ++l){
    const int nIn = sizes_[l], nOut = sizes_[l + 1];
    for (int o = 0; o < nOut; ++o)
      for (int i = 0; i < nIn; ++i)
        w_[l][((size_t)o * (size_t)nIn + (size_t)i) * (size_t)B_ + (size_t)lane]
            = (float)flat[off + (size_t)o * (size_t)nIn + (size_t)i];
    off += (size_t)nIn * (size_t)nOut;
    for (int o = 0; o < nOut; ++o)
      b_[l][(size_t)o * (size_t)B_ + (size_t)lane] = (float)flat[off + (size_t)o];
    off += (size_t)nOut;
  }
}

void BatchedMlp::forward(const float* obs, Move* movesOut){
  const int obsDim = sizes_.front();
  // transpose obs [lane][dim] -> activations [dim][lane]
  for (int i = 0; i < obsDim; ++i)
    for (int lane = 0; lane < B_; ++lane)
      actA_[(size_t)i * (size_t)B_ + (size_t)lane] = obs[(size_t)lane * (size_t)obsDim + (size_t)i];

  float* x = actA_.data();
  float* y = actB_.data();
  for (size_t l = 0; l + 1 < sizes_.size(); ++l){
    const int nIn = sizes_[l], nOut = sizes_[l + 1];
    const bool hidden = (l + 2 < sizes_.size());
    const float* W = w_[l].data();
    const float* Bv = b_[l].data();
    for (int o = 0; o < nOut; ++o){
      float* out = y + (size_t)o * (size_t)B_;
      const float* bo = Bv + (size_t)o * (size_t)B_;
      for (int lane = 0; lane < B_; ++lane) out[lane] = bo[lane];
      for (int i = 0; i < nIn; ++i){
        const float* wrow = W + ((size_t)o * (size_t)nIn + (size_t)i) * (size_t)B_;
        const float* xi   = x + (size_t)i * (size_t)B_;
        for (int lane = 0; lane < B_; ++lane)        // the SIMD dimension
          out[lane] += wrow[lane] * xi[lane];
      }
      if (hidden)
        for (int lane = 0; lane < B_; ++lane) out[lane] = std::tanh(out[lane]);
    }
    float* t = x; x = y; y = t;
  }
  // argmax over the 3 output rows, ties to the lower index (matches MlpAgent)
  for (int lane = 0; lane < B_; ++lane){
    int best = 0;
    float bv = x[(size_t)0 * (size_t)B_ + (size_t)lane];
    for (int o = 1; o < 3; ++o){
      float v = x[(size_t)o * (size_t)B_ + (size_t)lane];
      if (v > bv){ bv = v; best = o; }
    }
    movesOut[lane] = best == 1 ? Move::Up : best == 2 ? Move::Down : Move::None;
  }
}

} // namespace pong
