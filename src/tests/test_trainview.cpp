#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "train_log_view.hpp"
#include "coview_data.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
using namespace pong;

TEST_CASE("parseLogRow: evolve + ppo schemas, arrays skipped, NaN and exponents handled"){
  std::map<std::string, double> e;
  parseLogRow("{\"gen\":124,\"best\":0.981375,\"mean\":-0.0152,\"vs_p_classic\":1,"
              "\"vs_laggy\":0,\"vs_interceptor\":-2,\"sec\":1.10843e0}", e);
  CHECK(e["gen"] == 124);
  CHECK(e["best"] == doctest::Approx(0.981375));
  CHECK(e["mean"] == doctest::Approx(-0.0152));
  CHECK(e["vs_p_classic"] == 1);
  CHECK(e["vs_interceptor"] == -2);               // the "not evaluated" sentinel survives parsing
  CHECK(e["sec"] == doctest::Approx(1.10843));

  std::map<std::string, double> p;
  parseLogRow("{\"update\":1,\"step\":8192,\"ep_return_mean\":NaN,\"sps\":42183,"
              "\"pool_ema\":[0.5,0.4,0.6]}", p);
  CHECK(p["step"] == 8192);
  CHECK(p["sps"] == 42183);
  REQUIRE(p.count("ep_return_mean") == 1);
  CHECK(std::isnan(p["ep_return_mean"]));
  CHECK(p.count("pool_ema") == 0);                // array value skipped, not mis-parsed as a number
}

TEST_CASE("LogTail: incremental, complete-lines-only, schema detection"){
  const auto path = (std::filesystem::temp_directory_path() / "aipong_tail_test.jsonl").string();
  std::filesystem::remove(path);
  { std::ofstream f(path);
    f << "{\"gen\":0,\"best\":0.5,\"mean\":0.0,\"sec\":0.1}\n"
      << "{\"gen\":1,\"best\":0.6,\"mean\":0.1,\"sec\":0.1}\n"; }
  LogTail t; t.reset(path);
  CHECK(t.poll() == 2);
  CHECK(t.schema == "evolution");
  CHECK(t.rows.size() == 2);
  CHECK(t.poll() == 0);                            // nothing new

  { std::ofstream f(path, std::ios::app);
    f << "{\"gen\":2,\"best\":0.7,\"mean\":0.2,\"sec\":0.1}\n"
      << "{\"gen\":3,\"best\":0.8";  }             // trailing partial line, no newline
  CHECK(t.poll() == 1);                            // only the complete gen-2 line consumed
  CHECK(t.rows.size() == 3);
  { std::ofstream f(path, std::ios::app); f << ",\"mean\":0.3,\"sec\":0.1}\n"; }  // finish it
  CHECK(t.poll() == 1);
  CHECK(t.rows.back().at("gen") == 3);

  std::vector<float> xs, ys;
  extractSeries(t.rows, "gen", "best", false, xs, ys);
  CHECK(xs.size() == 4);
  CHECK(ys.back() == doctest::Approx(0.8));
  std::filesystem::remove(path);
}

TEST_CASE("parseStatus: state string + numeric fields; garbage is invalid"){
  RunStatus st = parseStatus(
      "{\"state\":\"paused\",\"gen\":42,\"gens_total\":3000,\"ts_unix\":1783100000}");
  CHECK(st.valid);
  CHECK(st.state == "paused");
  CHECK(st.gen == 42);
  CHECK(st.gensTotal == 3000);
  CHECK(st.tsUnix == doctest::Approx(1.7831e9));
  CHECK_FALSE(parseStatus("").valid);
  CHECK_FALSE(parseStatus("not json at all").valid);
}

TEST_CASE("emaSmooth: exponential smoothing follows the data"){
  const std::vector<float> raw{0, 0, 0, 100, 100, 100};
  std::vector<float> sm = emaSmooth(raw, 0.5f);
  REQUIRE(sm.size() == raw.size());
  CHECK(sm[0] == doctest::Approx(0));
  CHECK(sm[3] == doctest::Approx(50));                 // 0.5·100 + 0.5·0
  CHECK(sm[4] == doctest::Approx(75));
  CHECK(sm[5] > 80);                                   // converging toward 100
  CHECK(sm[5] < 100);
  CHECK(emaSmooth({}, 0.5f).empty());
}

TEST_CASE("etaSeconds: remaining work times pace; -1 when unknowable"){
  CHECK(etaSeconds(1500, 3000, 0.9) == doctest::Approx(1350.0));
  CHECK(etaSeconds(3000, 3000, 0.9) == doctest::Approx(0.0));
  CHECK(etaSeconds(3100, 3000, 0.9) == doctest::Approx(0.0));      // past the end → clamp 0
  CHECK(etaSeconds(10, -1, 0.9) < 0);                              // no known total
  CHECK(etaSeconds(10, 3000, 0.0) < 0);                            // no pace yet
}

TEST_CASE("extractSeries: drops the -2 anchor sentinel but keeps real win rates"){
  const std::vector<std::map<std::string, double>> rows = {
    {{"gen", 0},  {"vs_laggy", -2}},
    {{"gen", 25}, {"vs_laggy", 0.58}},
    {{"gen", 26}, {"vs_laggy", -2}},
    {{"gen", 50}, {"vs_laggy", 0.42}},
  };
  std::vector<float> xs, ys;
  extractSeries(rows, "gen", "vs_laggy", true, xs, ys);
  REQUIRE(xs.size() == 2);
  CHECK(xs[0] == 25);
  CHECK(ys[1] == doctest::Approx(0.42));
}

TEST_CASE("parseCrossWinLine: labels + N×N matrix round-trip"){
  const std::string line =
    "{\"gen\":24,\"labels\":[\"h32-32\",\"h64-64\",\"h32-32-32\"],"
    "\"matrix\":[[0,-0.636364,-1],[0.0909091,0,0.181818],[-1,-0.636364,0]]}";
  CrossWin cw;
  REQUIRE(parseCrossWinLine(line, cw));
  CHECK(cw.gen == 24);
  REQUIRE(cw.labels.size() == 3);
  CHECK(cw.labels[0] == "h32-32");
  CHECK(cw.labels[2] == "h32-32-32");
  REQUIRE(cw.matrix.size() == 3);
  REQUIRE(cw.matrix[1].size() == 3);
  CHECK(cw.matrix[0][0] == doctest::Approx(0.0));       // diagonal
  CHECK(cw.matrix[0][1] == doctest::Approx(-0.636364)); // negative margin
  CHECK(cw.matrix[1][0] == doctest::Approx(0.0909091));
  CrossWin bad;
  CHECK_FALSE(parseCrossWinLine("{\"gen\":1}", bad));    // no labels/matrix
  CrossWin ragged;                                        // truncated/ragged final row → reject
  CHECK_FALSE(parseCrossWinLine(
    "{\"gen\":3,\"labels\":[\"a\",\"b\"],\"matrix\":[[0,0.5],[0.3]]}", ragged));
}

TEST_CASE("avgMarginSeries: prefers marg_*, falls back to vs_* win-rate"){
  std::vector<std::map<std::string,double>> rows = {
    {{"gen",0},{"vs_p_classic",-2},{"vs_laggy",-2},{"vs_interceptor",-2}},   // non-eval sentinel row
    {{"gen",24},{"marg_p_classic",0.6},{"marg_laggy",0.0},{"marg_interceptor",-0.3},
                {"vs_p_classic",1.0},{"vs_laggy",0.5},{"vs_interceptor",0.0}},
  };
  std::vector<float> xs, ys; bool usedMargin = false;
  avgMarginSeries(rows, xs, ys, usedMargin);
  REQUIRE(xs.size() == 1);                               // only the eval row contributes
  CHECK(xs[0] == doctest::Approx(24.0f));
  CHECK(usedMargin);
  CHECK(ys[0] == doctest::Approx((0.6f + 0.0f - 0.3f) / 3.0f));   // mean of marg_*

  std::vector<std::map<std::string,double>> winOnly = {   // a run predating marg_* (like the live 10k)
    {{"gen",0},{"vs_p_classic",-2},{"vs_laggy",-2},{"vs_interceptor",-2}},
    {{"gen",25},{"vs_p_classic",1.0},{"vs_laggy",0.5},{"vs_interceptor",0.25}},
  };
  std::vector<float> xs2, ys2; bool um2 = true;
  avgMarginSeries(winOnly, xs2, ys2, um2);
  REQUIRE(xs2.size() == 1);
  CHECK_FALSE(um2);                                       // fell back to win-rate
  CHECK(ys2[0] == doctest::Approx((1.0f + 0.5f + 0.25f) / 3.0f * 100.0f));   // win-% scale
}

TEST_CASE("avgMarginSeries: series-wide mode + drops marg_* -2 sentinel"){
  // Mixed log: one row has real marg_*, a later row has only vs_* — series-wide rule falls the
  // WHOLE series back to win-% (uniform axis), not a per-row mix.
  std::vector<std::map<std::string,double>> mixed = {
    {{"gen",10},{"marg_p_classic",0.4},{"marg_laggy",0.2},{"marg_interceptor",0.0},
                {"vs_p_classic",0.8},{"vs_laggy",0.6},{"vs_interceptor",0.5}},
    {{"gen",20},{"vs_p_classic",0.3},{"vs_laggy",0.3},{"vs_interceptor",0.3}},   // no marg_*
  };
  std::vector<float> xs, ys; bool um = true;
  avgMarginSeries(mixed, xs, ys, um);
  CHECK_FALSE(um);                                        // any marg-less contributing row → win-% for all
  REQUIRE(xs.size() == 2);
  CHECK(ys[0] == doctest::Approx((0.8f + 0.6f + 0.5f) / 3.0f * 100.0f));   // marg row plotted via ITS vs_*
  CHECK(ys[1] == doctest::Approx((0.3f + 0.3f + 0.3f) / 3.0f * 100.0f));

  // Realistic post-Task-3 log: non-eval rows carry marg_* = -2 (sentinel). They must be dropped,
  // not plotted as margin -2.
  std::vector<std::map<std::string,double>> withSentinel = {
    {{"gen",0}, {"marg_p_classic",-2},{"marg_laggy",-2},{"marg_interceptor",-2},
                {"vs_p_classic",-2},{"vs_laggy",-2},{"vs_interceptor",-2}},
    {{"gen",24},{"marg_p_classic",0.5},{"marg_laggy",0.5},{"marg_interceptor",0.5},
                {"vs_p_classic",0.9},{"vs_laggy",0.9},{"vs_interceptor",0.9}},
  };
  std::vector<float> xs2, ys2; bool um2 = false;
  avgMarginSeries(withSentinel, xs2, ys2, um2);
  CHECK(um2);                                             // the one real eval row is margin-capable
  REQUIRE(xs2.size() == 1);                               // gen-0 sentinel row dropped
  CHECK(xs2[0] == doctest::Approx(24.0f));
  CHECK(ys2[0] == doctest::Approx(0.5f));
}
