#include "agents.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

namespace pong {

bool sizesInBounds(const std::vector<int>& sizes){
  // Bounds: layer widths in [1, 100000], total params <= 8,000,000 (~64 MB doubles).
  if (sizes.size() < 2) return false;
  size_t total = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l){
    if (sizes[l] <= 0 || sizes[l] > 100000 || sizes[l+1] <= 0 || sizes[l+1] > 100000)
      return false;
    total += (size_t)sizes[l] * (size_t)sizes[l+1] + (size_t)sizes[l+1];
    if (total > 8'000'000) return false;
  }
  return true;
}

namespace {
// Own paddle, its front-face x, and whether the ball approaches this side.
struct SideView { const Paddle& own; double frontX; bool incoming; };
SideView sideView(const Snapshot& s, Side side){
  if (side == Side::Left)
    return { s.left,  s.left.x  + s.left.w  * 0.5, s.ball.vel.x < 0 };
  return   { s.right, s.right.x - s.right.w * 0.5, s.ball.vel.x > 0 };
}
Move toward(double targetY, double ownY, double deadband){
  double e = targetY - ownY;
  if (std::abs(e) < deadband) return Move::None;
  return (e < 0) ? Move::Up : Move::Down;         // Up = y decreases (court frame C)
}
} // namespace

Move PAgent::act(const Snapshot& s, Side side){
  acc_ += duty_;                                   // deterministic duty-cycle gate
  if (acc_ < 1.0) return Move::None;
  acc_ -= 1.0;
  SideView v = sideView(s, side);
  double target = v.incoming ? s.ball.pos.y : 0.5; // track when incoming, else center
  return toward(target, v.own.y, deadband_);
}

double predictInterceptY(const Ball& b, double targetX){
  double dx = targetX - b.pos.x;
  if (b.vel.x == 0.0 || dx * b.vel.x <= 0.0) return b.pos.y;  // not moving toward plane
  double t    = dx / b.vel.x;                    // s (>0)
  double yRaw = b.pos.y + b.vel.y * t;           // unfolded
  double L    = 1.0 - 2.0 * b.r;                 // reachable span of the ball center
  double u    = std::fmod(yRaw - b.r, 2.0 * L);
  if (u < 0) u += 2.0 * L;
  return b.r + (u <= L ? u : 2.0 * L - u);       // triangle-wave fold into [r, 1-r]
}

Move InterceptorAgent::act(const Snapshot& s, Side side){
  SideView v = sideView(s, side);
  if (tick_++ % std::max(1, reactionTicks_) == 0)
    target_ = v.incoming ? predictInterceptY(s.ball, v.frontX) : 0.5;
  return toward(target_, v.own.y, deadband_);
}

bool MlpAgent::loadStream(std::istream& in){
  std::string magic; int version = 0;
  if (!(in >> magic >> version) || magic != "aipong-mlp" || (version != 1 && version != 2)) return false;
  std::string line;
  std::getline(in, line);                         // rest of version line
  int stack = 1, stride = 1;
  if (version == 2){                              // v2: "obs stack K" or "obs stack K stride S"
    if (!std::getline(in, line)) return false;
    std::istringstream os(line);
    std::string w1, w2;
    if (!(os >> w1 >> w2 >> stack) || w1 != "obs" || w2 != "stack") return false;
    std::string w3;
    if (os >> w3){
      if (w3 != "stride" || !(os >> stride)) return false;
    }
  }
  std::vector<int> sizes;
  if (!std::getline(in, line)) return false;
  {
    std::istringstream ls(line);
    int v;
    while (ls >> v) sizes.push_back(v);
  }
  if (!sizesInBounds(sizes)) return false;
  size_t need = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l)
    need += (size_t)sizes[l] * (size_t)sizes[l + 1] + (size_t)sizes[l + 1];
  std::vector<double> flat(need);
  for (double& x : flat) if (!(in >> x)) return false;
  return loadWeights(sizes, flat, stack, stride);
}

// Trust boundary: text-parsing entry points (loadStream above, and train/evolve.cpp's
// readPopulation/readModelFlat) run every untrusted `sizes` line through sizesInBounds()
// before it ever reaches here. loadWeights is the in-memory path — trainer hot loop, bench,
// tests — whose sizes are always already-vetted or program-constructed, so it re-checks only
// its own narrower shape contract (input == 6·stack, output == 3), not the general
// sizesInBounds bounds; it deliberately does NOT call sizesInBounds itself.
bool MlpAgent::loadWeights(const std::vector<int>& sizes, const std::vector<double>& flat,
                           int stack, int stride){
  if (stack < 1 || stack > 32 || stride < 1 || stride > 60) return false;
  if ((stack - 1) * stride + 1 > 512) return false;   // history window sanity cap [ticks]
  if (sizes.size() < 2 || sizes.front() != 6 * stack || sizes.back() != 3) return false;
  size_t need = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l)
    need += (size_t)sizes[l] * (size_t)sizes[l + 1] + (size_t)sizes[l + 1];
  if (flat.size() != need) return false;
  std::vector<Layer> layers(sizes.size() - 1);
  size_t off = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l){
    Layer& L = layers[l];
    L.nIn = sizes[l]; L.nOut = sizes[l + 1];
    L.w.resize((size_t)L.nIn * (size_t)L.nOut);
    // flat is row-major [nOut][nIn] (the wire contract); transpose to [nIn][nOut] here — a
    // one-time O(nIn*nOut) cost at load, not per act() call (Decision 23).
    for (int o = 0; o < L.nOut; ++o)
      for (int i = 0; i < L.nIn; ++i)
        L.w[(size_t)i * (size_t)L.nOut + (size_t)o] = (float)flat[off + (size_t)o * (size_t)L.nIn + (size_t)i];
    off += L.w.size();
    L.b.resize((size_t)L.nOut);
    for (size_t i = 0; i < L.b.size(); ++i) L.b[i] = (float)flat[off + i];
    off += L.b.size();
  }
  layers_ = std::move(layers);
  stack_ = stack;
  stride_ = stride;
  history_.clear();
  return true;
}

void writeModelText(std::ostream& out, const std::vector<int>& sizes,
                    const std::vector<double>& flat, int stack, int stride){
  out << "aipong-mlp 2\n" << "obs stack " << stack;
  if (stride > 1) out << " stride " << stride;
  out << "\n";
  for (size_t i = 0; i < sizes.size(); ++i) out << sizes[i] << (i + 1 < sizes.size() ? ' ' : '\n');
  out.precision(9);
  size_t off = 0;
  for (size_t l = 0; l + 1 < sizes.size(); ++l){
    size_t nw = (size_t)sizes[l] * (size_t)sizes[l + 1], nb = (size_t)sizes[l + 1];
    for (size_t i = 0; i < nw; ++i) out << flat[off + i] << ' ';
    out << '\n';
    off += nw;
    for (size_t i = 0; i < nb; ++i) out << flat[off + i] << ' ';
    out << '\n';
    off += nb;
  }
}

bool MlpAgent::loadFile(const std::string& path){
  std::ifstream f(path);
  if (!f) return false;
  return loadStream(f);
}

Move MlpAgent::act(const Snapshot& s, Side side){
  if (layers_.empty()) return Move::None;
  auto obs = observation(s, side);
  // Assemble the network input into preallocated scratch (resize() reuses capacity after
  // the first call — this function runs millions of times per training generation).
  bufA_.resize((size_t)(6 * stack_));
  if (stack_ == 1){                                   // fast path: no history machinery at all
    for (int j = 0; j < 6; ++j) bufA_[(size_t)j] = (float)obs[(size_t)j];
  } else if (ablateHistory_){                         // memory-use probe: every slot = now (no motion)
    for (int f = 0; f < stack_; ++f)
      for (int j = 0; j < 6; ++j) bufA_[(size_t)(f * 6 + j)] = (float)obs[(size_t)j];
  } else {
    const int window = (stack_ - 1) * stride_ + 1;
    history_.push_back(obs);
    if ((int)history_.size() > window) history_.pop_front();
    const int newest = (int)history_.size() - 1;
    for (int f = 0; f < stack_; ++f){
      // slot f = frame t − S·(K−1−f); fresh episodes pad by repeating the earliest frame
      int idx = newest - (stack_ - 1 - f) * stride_;
      const auto& frame = history_[(size_t)std::max(0, idx)];
      for (int j = 0; j < 6; ++j) bufA_[(size_t)(f * 6 + j)] = (float)frame[(size_t)j];
    }
  }
  std::vector<float>* x = &bufA_;
  std::vector<float>* y = &bufB_;
  for (size_t l = 0; l < layers_.size(); ++l){
    const Layer& L = layers_[l];
    y->resize((size_t)L.nOut);
    acc_.resize((size_t)L.nOut);
    for (int o = 0; o < L.nOut; ++o) acc_[(size_t)o] = L.b[(size_t)o];    // bias-init
    const float* xp = x->data();
    const float* w = L.w.data();                       // transposed [nIn][nOut] (agents.hpp)
    for (int i = 0; i < L.nIn; ++i){
      const float xi = xp[i];
      const float* wRow = w + (size_t)i * (size_t)L.nOut;  // input i's weight to every output
      for (int o = 0; o < L.nOut; ++o) acc_[(size_t)o] += wRow[o] * xi;   // flat, contiguous -> vectorizes
    }
    const bool hidden = (l + 1 < layers_.size());
    float* yp = y->data();
    for (int o = 0; o < L.nOut; ++o) yp[o] = hidden ? std::tanh(acc_[(size_t)o]) : acc_[(size_t)o];  // separate
    std::swap(x, y);                                    // flat pass -> libmvec vectorizes tanh (Decision 23)
  }
  const std::vector<float>& out = *x;
  int best = 0;
  for (int i = 1; i < 3; ++i) if (out[(size_t)i] > out[(size_t)best]) best = i;
  return best == 1 ? Move::Up : best == 2 ? Move::Down : Move::None;
}

} // namespace pong
