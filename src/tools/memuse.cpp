// memuse — is the memory window actually being used?  Three probes for a stack-K model:
//   (A) structural  — |history-input weights| vs |newest-frame weights| in layer 1.
//   (B) functional  — win rate vs the instant Interceptor with true history vs history ablated
//                     (every slot forced to the current frame). A drop = memory has value.
//   (C) behavioral  — fraction of decisions that change when history is ablated.
// The evo4-6 "memory champions" score zero on all three (they are widened memoryless nets);
// a genome that genuinely uses its 0.5 s window scores nonzero.  usage: memuse <model> [matches]
#include "match_runner.hpp"
#include "agents.hpp"
#include "pong_core.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>
using namespace pong;

namespace {
int winsVs(const std::string& opp, int N, const std::function<std::unique_ptr<Agent>()>& buildLeft){
  int w = 0;
  for (int m = 0; m < N; ++m){
    MatchRunner r((unsigned)(3000 + m));
    r.setController(Side::Left, buildLeft(), "left");
    r.setController(Side::Right, opp);
    r.newGame((unsigned)(3000 + m));
    if (r.runToCompletion() == Outcome::LeftWin) w++;
  }
  return w;
}
}

int main(int argc, char** argv){
  if (argc < 2){ std::fprintf(stderr, "usage: memuse <model.txt> [matches]\n"); return 1; }
  const std::string path = argv[1];
  const std::string spec = "mlp:" + path;
  const int N = argc > 2 ? std::atoi(argv[2]) : 60;
  const std::string opp = argc > 3 ? argv[3] : "interceptor";   // (B)/(C) probe opponent
  { MlpAgent probe("p"); if (!probe.loadFile(path)){ std::fprintf(stderr, "cannot load %s\n", path.c_str()); return 2; } }

  // ---- (A) structural: parse layer-1 weights, split history vs newest-frame column mass ----
  std::ifstream f(path); std::vector<std::string> t; std::string tok;
  while (f >> tok) t.push_back(tok);
  size_t k = 2; int stack = 1, stride = 1;
  if (t[1] == "2"){ stack = std::stoi(t[4]); k = 5; if (t[k] == "stride"){ stride = std::stoi(t[k + 1]); k += 2; } }
  const int nin = std::stoi(t[k]), nout = std::stoi(t[k + 1]); k += 4;
  double hist = 0, newest = 0;
  for (int o = 0; o < nout; ++o)
    for (int i = 0; i < nin; ++i){
      const double w = std::fabs(std::stod(t[k + (size_t)o * nin + (size_t)i]));
      if (i < nin - 6) hist += w; else newest += w;
    }
  std::printf("model: %s   stack %d stride %d (%.2f s window)\n",
              path.c_str(), stack, stride, ((stack - 1) * stride + 1) * k::Dt);
  if (stack <= 1){ std::printf("  memoryless (stack 1) — nothing to probe\n"); return 0; }
  std::printf("(A) structural: |history weights| %.3f vs |newest-frame| %.3f   -> history is %.1f%% of newest\n",
              hist, newest, newest > 0 ? 100.0 * hist / newest : 0.0);

  // ---- (B) functional: win rate true vs ablated ----
  const int wTrue = winsVs(opp, N, [&]{ auto a = std::make_unique<MlpAgent>("t"); a->loadFile(path); return a; });
  const int wFlat = winsVs(opp, N, [&]{ auto a = std::make_unique<MlpAgent>("f"); a->loadFile(path); a->setAblateHistory(true); return a; });
  std::printf("(B) functional: vs %s, %d matches — true history %d%%  vs  ablated %d%%   (drop = memory's value)\n",
              opp.c_str(), N, 100 * wTrue / N, 100 * wFlat / N);

  // ---- (C) behavioral: decision divergence when history is ablated ----
  MlpAgent norm("n"), abl("a"); norm.loadFile(path); abl.loadFile(path); abl.setAblateHistory(true);
  long diff = 0, tot = 0; const int M = std::max(3, N / 6);
  for (int m = 0; m < M; ++m){
    MatchRunner r((unsigned)(7000 + m));
    r.setController(Side::Left, spec); r.setController(Side::Right, opp); r.newGame((unsigned)(7000 + m));
    norm.reset(); abl.reset(); long g = 0;
    while (r.snapshot().phase != Phase::GameOver && g++ < 20'000'000){
      const Snapshot s = r.snapshot();
      const Move a = norm.act(s, Side::Left), b = abl.act(s, Side::Left);
      if (a != b) diff++;
      tot++;
      r.tick(Move::None, Move::None);
    }
  }
  std::printf("(C) behavioral: %.1f%% of decisions change when history is ablated (%ld/%ld)\n",
              tot ? 100.0 * diff / tot : 0.0, diff, tot);

  const bool wired = hist > 0.05 * newest;
  const bool functional = (wTrue - wFlat) > N / 20 || (tot && 100.0 * diff / tot > 2.0);
  std::printf("VERDICT: memory %s\n",
              (wired && functional) ? "IS being used." :
              wired ? "is WIRED but has little functional effect." : "is NOT used (history weights ~0).");
  return 0;
}
