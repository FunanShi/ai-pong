#pragma once
#include "pong_core.hpp"
#include <array>
#include <deque>
#include <istream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// Opponents live OUTSIDE the core: the sim knows only Input{left,right}. Every agent
// sees a Snapshot and its Side and returns a discrete Move — identical actuation to a
// human (PaddleSpeed cap enforced by the core), so agent-vs-agent and human-vs-agent
// matches are symmetric by construction. Deterministic: no RNG anywhere in this module.

namespace pong {

class Agent {
public:
  virtual ~Agent() = default;
  virtual Move act(const Snapshot&, Side) = 0;
  virtual std::string name() const = 0;
  virtual void reset() {}                    // clear per-match internal state
};

// Ball-following proportional controller (port of the original ui_pong "computer").
// Tracks the ball while it approaches, recenters otherwise; deadband prevents chatter.
// `duty` ∈ (0,1] emulates a slower paddle with the discrete action set: moves are
// emitted on a deterministic Bresenham schedule, so effective speed ≈ duty·PaddleSpeed
// (duty 0.75 ≈ the original AiVmax 0.9 cu/s).
class PAgent : public Agent {
public:
  explicit PAgent(std::string label, double deadband = 0.02, double duty = 1.0)
    : label_(std::move(label)), deadband_(deadband), duty_(duty) {}
  Move act(const Snapshot&, Side) override;
  std::string name() const override { return label_; }
  void reset() override { acc_ = 0.0; }
private:
  std::string label_;
  double deadband_;  // cu
  double duty_;      // fraction of ticks allowed to move
  double acc_ = 0.0; // Bresenham accumulator
};

// Predicts the ball's intercept height at its own paddle plane by unrolling the
// straight-line flight with wall reflections (triangle-wave fold), then drives there.
// `reactionTicks` re-computes the prediction only every N ticks (latency knob).
class InterceptorAgent : public Agent {
public:
  explicit InterceptorAgent(std::string label, double deadband = 0.015, int reactionTicks = 1)
    : label_(std::move(label)), deadband_(deadband), reactionTicks_(reactionTicks) {}
  Move act(const Snapshot&, Side) override;
  std::string name() const override { return label_; }
  void reset() override { tick_ = 0; target_ = 0.5; }
private:
  std::string label_;
  double deadband_;      // cu
  int    reactionTicks_; // ticks between prediction updates (≥1)
  int    tick_   = 0;
  double target_ = 0.5;  // cu, cached predicted intercept y
};

// y-coordinate (cu, court frame) where a ball at `b` crosses the vertical plane
// x = targetX, folding wall bounces. Requires the ball to be moving toward targetX
// (returns b.pos.y unchanged if not). Exposed for unit tests.
double predictInterceptY(const Ball& b, double targetX);

// Tiny MLP policy over observation(snapshot, side) — the "trained AI" slot.
// File formats (whitespace-separated text):
//   v1:  aipong-mlp 1                          v2:  aipong-mlp 2
//        6 32 32 3                                  obs stack K          (1 ≤ K ≤ 32)
//        <weights/biases...>                        6·K 32 32 3
//                                                   <weights/biases...>
// Per layer: weights row-major [n_out × n_in], then biases [n_out].
// Hidden activation tanh, linear output, argmax → {0:None, 1:Up, 2:Down}.
// v2 frame stacking: the network input is the last K observations concatenated
// OLDEST→NEWEST (slots [0,6) = t−K+1 … slots [6(K−1),6K) = t); a fresh episode pads
// by repeating the earliest available frame. v1 ≡ v2 with K=1. History is cleared by
// reset() — call it between matches, and keep one MlpAgent instance per side (the
// buffer stores side-mirrored observations). Future recurrent policies follow the
// same seam: per-episode state owned by the agent, cleared by reset(), format v3.
class MlpAgent : public Agent {
public:
  explicit MlpAgent(std::string label) : label_(std::move(label)) {}
  bool loadFile(const std::string& path);     // false + intact state on failure
  bool loadStream(std::istream&);
  // Direct in-memory load (the trainer's hot path — no text round-trip). `flat` holds
  // per layer: weights row-major [n_out × n_in], then biases [n_out], concatenated.
  bool loadWeights(const std::vector<int>& sizes, const std::vector<double>& flat,
                   int stack = 1, int stride = 1);
  Move act(const Snapshot&, Side) override;
  std::string name() const override { return label_; }
  void reset() override { history_.clear(); }
  bool loaded() const { return !layers_.empty(); }
  int  stack() const { return stack_; }
  int  stride() const { return stride_; }
  // Memory-use probe (Decision 16 follow-up): when set, every history slot is filled with
  // the CURRENT frame, so the policy runs with no temporal information. Comparing play with
  // this on vs off measures whether the memory window is functionally used, not just wired.
  void setAblateHistory(bool b){ ablateHistory_ = b; }
private:
  // Inference arithmetic is float32 (Decision 13): the on-disk format stays double text,
  // weights are converted at load. Decisions can drift vs the historical double path at
  // ~1e-7 logit level — within-binary determinism is exact.
  // w is stored TRANSPOSED [nIn][nOut] (row i = input i's weight to every output — Decision
  // 23), NOT the wire format's [nOut][nIn]: loadWeights transposes once at load so act()'s
  // hot loop can walk w contiguously in the o dimension (flat, no transcendental inside the
  // reduction) and let the compiler vectorize both the MAC loop and the tanh pass. The
  // on-disk/flat wire contract (row-major [n_out x n_in], DECISIONS.md #6) is unchanged.
  struct Layer { int nIn = 0, nOut = 0; std::vector<float> w, b; };
  std::string label_;
  std::vector<Layer> layers_;
  int stack_ = 1;                                     // frames of observation history
  int stride_ = 1;                                    // ticks between sampled frames
  bool ablateHistory_ = false;                        // memory-use probe (see setAblateHistory)
  std::deque<std::array<double, 6>> history_;         // window (K−1)·S+1, oldest front (stack>1 only)
  std::vector<float> bufA_, bufB_;                    // preallocated forward-pass scratch (act() is
                                                      // the trainer's hot path — no per-call allocs)
  std::vector<float> acc_;                            // preallocated per-layer accumulator (Decision 23)
};

// Sanity bounds for a parsed aipong-mlp/aipong-pop sizes line, checked BEFORE any
// allocation (a malformed or truncated file must fail cleanly, never bad_alloc):
// each layer width in [1, 100000], total parameter count <= 8,000,000 (~64 MB doubles).
bool sizesInBounds(const std::vector<int>& sizes);

// Serialize weights to the aipong-mlp v2 text format (the single writer — trainers and
// tools emit through this, so the on-disk grammar has exactly one owner). Grammar:
// "obs stack K" (stride 1, byte-identical to pre-stride files) or "obs stack K stride S";
// caps K ≤ 32, S ≤ 60, window (K−1)·S+1 ≤ 512 ticks. Slot f of the input holds the frame
// from t − S·(K−1−f): slot K−1 = now, slot 0 = the oldest sample.
void writeModelText(std::ostream&, const std::vector<int>& sizes,
                    const std::vector<double>& flat, int stack = 1, int stride = 1);

} // namespace pong
