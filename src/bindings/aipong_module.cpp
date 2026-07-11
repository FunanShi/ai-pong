// Python bindings: a batched vector env over MatchRunner for the PPO track.
// The policy always plays LEFT (external Moves through the "human" slot; observations are
// already left-mirrored by contract). Opponents are controller specs resolved by the
// match/ registry — scripted bots or "mlp:<path>" checkpoints, so the opponent pool spans
// both training tracks. Rewards: +1/-1 per point from the policy's perspective, extra -1
// on RallyCap (Decision 4 as training pressure). Episodes auto-reset on game over; the
// observation returned for a done step belongs to the FRESH episode (bootstrap with 0 on
// done, CleanRL-style). Deterministic per (seed, env index): episode seeds come from a
// per-env splitmix stream.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "match_runner.hpp"
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace pong;

namespace {
uint64_t mix(uint64_t& s){
  s += 0x9e3779b97f4a7c15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
} // namespace

class VecPong {
  struct Slot {
    MatchRunner runner{0};
    std::string opp = "p:classic";
    std::string pending;            // applied at the next episode reset
    uint64_t stream = 0;            // per-env episode-seed stream
    int prevL = 0, prevR = 0;
  };
public:
  VecPong(int n, unsigned seed, const std::string& opponent)
      : slots_((size_t)std::max(1, n)) {
    for (size_t i = 0; i < slots_.size(); ++i){
      Slot& e = slots_[i];
      e.stream = (uint64_t)seed;
      (void)mix(e.stream);
      e.stream ^= 0x9e3779b97f4a7c15ull * (uint64_t)(i + 1);
      e.opp = opponent;
      resetSlot(i);
    }
  }
  int  size() const { return (int)slots_.size(); }
  void set_opponent(int i, const std::string& spec){ slots_[(size_t)i].pending = spec; }
  std::string opponent(int i) const { return slots_[(size_t)i].opp; }
  bool opponent_loaded(int i) const { return !slots_[(size_t)i].runner.humanControlled(Side::Right); }
  void reset_all(){ for (size_t i = 0; i < slots_.size(); ++i) resetSlot(i); }

  py::array_t<float> observations() const {
    py::array_t<float> obs({(py::ssize_t)slots_.size(), (py::ssize_t)6});
    auto o = obs.mutable_unchecked<2>();
    for (size_t i = 0; i < slots_.size(); ++i){
      auto v = observation(slots_[i].runner.snapshot(), Side::Left);
      for (int j = 0; j < 6; ++j) o((py::ssize_t)i, j) = (float)v[(size_t)j];
    }
    return obs;
  }

  // actions: int32 [n] in {0:None, 1:Up, 2:Down} → (obs f32 [n,6], reward f32 [n], done bool [n])
  py::tuple step(py::array_t<int, py::array::c_style | py::array::forcecast> actions){
    auto a = actions.unchecked<1>();
    if ((size_t)a.shape(0) != slots_.size()) throw std::runtime_error("actions length != n_envs");
    std::vector<float>   rew(slots_.size(), 0.f);
    std::vector<uint8_t> don(slots_.size(), 0);
    {
      py::gil_scoped_release nogil;
      for (size_t i = 0; i < slots_.size(); ++i){
        Slot& e = slots_[i];
        int ai = a[(py::ssize_t)i];
        Move m = ai == 1 ? Move::Up : ai == 2 ? Move::Down : Move::None;
        e.runner.tick(m, Move::None);
        Snapshot post = e.runner.snapshot();
        float r = (float)((post.score.left - e.prevL) - (post.score.right - e.prevR));
        e.prevL = post.score.left; e.prevR = post.score.right;
        if (post.phase == Phase::GameOver){
          if (post.outcome == Outcome::RallyCap) r -= 1.0f;
          don[i] = 1;
        }
        rew[i] = r;
        if (don[i]) resetSlot(i);     // auto-reset: returned obs = fresh episode
      }
    }
    py::array_t<float> rewA((py::ssize_t)slots_.size());
    py::array_t<bool>  donA((py::ssize_t)slots_.size());
    auto rA = rewA.mutable_unchecked<1>();
    auto dA = donA.mutable_unchecked<1>();
    for (size_t i = 0; i < slots_.size(); ++i){
      rA[(py::ssize_t)i] = rew[i];
      dA[(py::ssize_t)i] = don[i] != 0;
    }
    return py::make_tuple(observations(), rewA, donA);
  }

private:
  void resetSlot(size_t i){
    Slot& e = slots_[i];
    if (!e.pending.empty()){ e.opp = e.pending; e.pending.clear(); }
    unsigned s = (unsigned)mix(e.stream);
    e.runner.setController(Side::Left, "human");   // policy side — driven by step() actions
    e.runner.setController(Side::Right, e.opp);
    e.runner.newGame(s);
    e.prevL = 0; e.prevR = 0;
  }
  std::deque<Slot> slots_;
};

PYBIND11_MODULE(aipong, m){
  m.doc() = "AI Pong vectorized env (policy plays LEFT; obs = observation(snapshot, Left))";
  m.attr("OBS_DIM")   = 6;
  m.attr("N_ACTIONS") = 3;
  m.attr("DT_S")      = k::Dt;
  m.attr("WIN_SCORE") = k::WinScore;
  py::class_<VecPong>(m, "VecPong")
    .def(py::init<int, unsigned, const std::string&>(),
         py::arg("n"), py::arg("seed"), py::arg("opponent") = "p:classic")
    .def("step", &VecPong::step)
    .def("observations", &VecPong::observations)
    .def("set_opponent", &VecPong::set_opponent)
    .def("opponent", &VecPong::opponent)
    .def("opponent_loaded", &VecPong::opponent_loaded)
    .def("reset_all", &VecPong::reset_all)
    .def("size", &VecPong::size);
}
