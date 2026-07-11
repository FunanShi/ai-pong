// reeval — recover TRUE anchor curves for a finished run from its stored checkpoints.
// Train logs written before the evalParams fix (DECISIONS.md #18) under-report every
// stack>1 genome: in-training evals loaded genomes at stride 1, collapsing the history
// window. Checkpoint FILES always bake in the correct stride, so re-playing them
// reconstructs the run's real anchor history.
//   usage: reeval <run-dir> [matches-per-anchor=24] [out.jsonl]
// Each gen_*.txt plays the three reporting anchors + both held-outs on canonical physics
// (full rally cap, unshaped — deployment semantics), the same columns as train_log.jsonl.
// Seeds mirror the trainer's eval stream (TagEval keyed by gen) for SEED-1 runs only — this
// tool hard-codes master seed 1 below, so numbers are directly comparable to what a fixed
// seed-1 trainer would have logged; other seeds' curves are unbiased but not row-identical.
// Default out: results/reeval/<run-name>.jsonl; a table always prints.
#include "match_runner.hpp"
#include "agents.hpp"
#include "rng.hpp"
#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
using namespace pong;
namespace fs = std::filesystem;

namespace {
constexpr uint64_t TagEval = 5;                        // matches evolve.cpp's eval stream
struct Anchor { const char* spec; const char* key; };
constexpr Anchor kAnchors[] = {
  {"p:classic",            "vs_p_classic"},
  {"interceptor:laggy",    "vs_laggy"},
  {"interceptor",          "vs_interceptor"},
  {"p:heldout",            "vs_held_p"},
  {"interceptor:heldout",  "vs_held_icept"},
};
constexpr int kNA = (int)(sizeof kAnchors / sizeof kAnchors[0]);
}

int main(int argc, char** argv){
  if (argc < 2){ std::fprintf(stderr, "usage: reeval <run-dir> [matches-per-anchor] [out.jsonl]\n"); return 1; }
  const std::string dir = argv[1];
  const int N = argc > 2 ? std::atoi(argv[2]) : 24;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)){ std::fprintf(stderr, "reeval: %s: no such directory\n", dir.c_str()); return 1; }

  struct Ckpt { int gen; std::string path; };
  std::vector<Ckpt> ckpts;
  for (const auto& e : fs::directory_iterator(dir, ec)){
    const std::string name = e.path().filename().string();
    if (e.is_regular_file() && name.rfind("gen_", 0) == 0 && e.path().extension() == ".txt")
      ckpts.push_back({std::atoi(name.c_str() + 4), e.path().string()});
  }
  std::sort(ckpts.begin(), ckpts.end(), [](const Ckpt& a, const Ckpt& b){ return a.gen < b.gen; });
  if (ckpts.empty()){ std::fprintf(stderr, "reeval: no gen_*.txt in %s\n", dir.c_str()); return 1; }
  { MlpAgent probe("p");                                  // fail fast on an unloadable run
    if (!probe.loadFile(ckpts.front().path)){
      std::fprintf(stderr, "reeval: cannot load %s\n", ckpts.front().path.c_str()); return 2; } }

  std::string out = argc > 3 ? argv[3]
      : "results/reeval/" + fs::path(dir).filename().string() + ".jsonl";

  // one job per (checkpoint, anchor, match) — canonical physics, deterministic seeds
  struct Job { int c, a, m; bool win = false; };
  std::vector<Job> jobs;
  jobs.reserve(ckpts.size() * (size_t)kNA * (size_t)N);
  for (int c = 0; c < (int)ckpts.size(); ++c)
    for (int a = 0; a < kNA; ++a)
      for (int m = 0; m < N; ++m) jobs.push_back({c, a, m});
  std::atomic<size_t> next{0};
  auto worker = [&](){
    for (size_t j; (j = next.fetch_add(1)) < jobs.size(); ){
      Job& job = jobs[j];
      auto agent = std::make_unique<MlpAgent>("re");
      if (!agent->loadFile(ckpts[(size_t)job.c].path)) continue;   // unloadable = loss
      const unsigned seed = (unsigned)rng::key(1u, TagEval,
          (uint64_t)(ckpts[(size_t)job.c].gen * 8 + job.a), (uint64_t)job.m);
      MatchRunner r(seed);
      r.setController(Side::Left, std::move(agent), "reeval");
      r.setController(Side::Right, kAnchors[job.a].spec);
      r.newGame(seed);
      job.win = r.runToCompletion() == Outcome::LeftWin;
    }
  };
  int T = (int)std::thread::hardware_concurrency(); if (T < 1) T = 1;
  std::vector<std::thread> pool;
  for (int t = 1; t < T; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();

  std::vector<std::vector<int>> wins(ckpts.size(), std::vector<int>(kNA, 0));
  for (const auto& j : jobs) if (j.win) wins[(size_t)j.c][(size_t)j.a]++;

  fs::create_directories(fs::path(out).parent_path(), ec);
  std::ofstream of(out);
  std::printf("reeval: %zu checkpoints x %d anchors x %d matches (canonical physics)\n",
              ckpts.size(), kNA, N);
  std::printf(" gen   ");
  for (const auto& a : kAnchors) std::printf(" %14s", a.key);
  std::printf("\n");
  for (size_t c = 0; c < ckpts.size(); ++c){
    std::printf("%5d  ", ckpts[c].gen);
    of << "{\"gen\":" << ckpts[c].gen;
    for (int a = 0; a < kNA; ++a){
      const double r = (double)wins[c][(size_t)a] / (double)N;
      std::printf(" %13.0f%%", 100.0 * r);
      char buf[32]; std::snprintf(buf, sizeof buf, "%.4f", r);
      of << ",\"" << kAnchors[a].key << "\":" << buf;
    }
    of << ",\"matches\":" << N << "}\n";
    std::printf("\n");
  }
  std::printf("written: %s\n", out.c_str());
  return 0;
}
