#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "ladder.hpp"
#include "ratings_index.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
using namespace pong;

static MatchRecord rec(int a, int b, int winner){
  MatchRecord m; m.a = a; m.b = b; m.winner = winner; return m;
}

TEST_CASE("Bradley-Terry: recovers ordering and sane gaps from win matrices"){
  // A beats B 3-1; B beats C 3-1; A beats C 4-0
  std::vector<MatchRecord> res;
  for (int i = 0; i < 3; ++i) res.push_back(rec(0, 1, 0));
  res.push_back(rec(0, 1, 1));
  for (int i = 0; i < 3; ++i) res.push_back(rec(1, 2, 0));
  res.push_back(rec(1, 2, 1));
  for (int i = 0; i < 4; ++i) res.push_back(rec(0, 2, 0));
  auto elo = bradleyTerryElo(3, res, -1);
  CHECK(elo[0] > elo[1]);
  CHECK(elo[1] > elo[2]);
  double gapAB = elo[0] - elo[1], gapBC = elo[1] - elo[2];
  CHECK(gapAB > 60);  CHECK(gapAB < 300);              // ~75% winrate ≈ 191 elo, smoothed
  CHECK(gapBC > 60);  CHECK(gapBC < 300);
  auto elo2 = bradleyTerryElo(3, res, -1);
  CHECK(elo == elo2);                                   // deterministic
}

TEST_CASE("Bradley-Terry: rally-cap results are excluded from rating"){
  std::vector<MatchRecord> base;
  for (int i = 0; i < 3; ++i) base.push_back(rec(0, 1, 0));
  base.push_back(rec(0, 1, 1));
  auto eloBase = bradleyTerryElo(2, base, -1);
  std::vector<MatchRecord> withCaps = base;
  for (int i = 0; i < 10; ++i) withCaps.push_back(rec(0, 1, -1));   // stalemates
  auto eloCaps = bradleyTerryElo(2, withCaps, -1);
  CHECK(eloBase[0] == doctest::Approx(eloCaps[0]));
  CHECK(eloBase[1] == doctest::Approx(eloCaps[1]));
}

TEST_CASE("Bradley-Terry: pinning fixes the scale at 1000"){
  std::vector<MatchRecord> res;
  for (int i = 0; i < 3; ++i) res.push_back(rec(0, 1, 0));
  res.push_back(rec(0, 1, 1));
  auto elo = bradleyTerryElo(2, res, 1);
  CHECK(elo[1] == doctest::Approx(1000.0));
  CHECK(elo[0] > 1000.0);
}

TEST_CASE("runLadder: scripted mini-roster produces a sane, deterministic table"){
  LadderConfig cfg;
  cfg.roster = {"p:easy", "p:classic", "p:hard"};
  cfg.perPair = 2;
  cfg.threads = 2;
  cfg.seed = 3;
  cfg.outDir = "";
  LadderResult r1 = runLadder(cfg);
  REQUIRE(r1.rows.size() == 3);
  REQUIRE(r1.matches.size() == 6);                      // 3 pairs x 2
  for (const auto& row : r1.rows) CHECK(row.games + row.caps == 4);
  auto eloOf = [&](const std::string& spec){
    for (const auto& row : r1.rows) if (row.spec == spec) return row.elo;
    return -1.0;
  };
  CHECK(eloOf("p:hard") > eloOf("p:easy"));             // full-duty tracker beats easy
  CHECK(eloOf("p:classic") == doctest::Approx(1000.0)); // pinned anchor
  LadderResult r2 = runLadder(cfg);                     // deterministic end-to-end
  for (size_t i = 0; i < r1.rows.size(); ++i){
    CHECK(r1.rows[i].spec == r2.rows[i].spec);
    CHECK(r1.rows[i].elo == doctest::Approx(r2.rows[i].elo));
  }
}

TEST_CASE("ratings index: champions canonical, newest otherwise, alias, headers, sort"){
  namespace fs = std::filesystem;
  const std::string root = "/tmp/aipong_ridx";
  const std::string lad = root + "/ladder", mod = root + "/models";
  fs::remove_all(root);
  fs::create_directories(lad + "/old");
  fs::create_directories(lad + "/champions");
  fs::create_directories(lad + "/newer");
  fs::create_directories(mod + "/leaderboard");
  fs::create_directories(mod + "/run1");
  fs::create_directories(mod + "/run2");
  { std::ofstream f(mod + "/leaderboard/champ.txt"); f << "x"; }   // content irrelevant to the
  { std::ofstream f(mod + "/run1/best.txt");         f << "x"; }   // index; loadability is the
  { std::ofstream f(mod + "/run2/latest.txt");       f << "x"; }   // GUI's probe, not ours
  { std::ofstream f(mod + "/run1/notes.txt");        f << "x"; }   // not best/latest: ignored
  { std::ofstream f(lad + "/old/ratings.jsonl");
    f << "{\"aipong_ladder\":1,\"specs\":3,\"per_pair\":2,\"seed\":1}\n"      // header: skipped
      << "{\"spec\":\"mlp:models/run1/best.txt\",\"elo\":1300.0,\"games\":10,\"wins\":5,\"caps\":0}\n"
      << "{\"spec\":\"mlp:models/_leaderboard/champ.txt\",\"elo\":1500.0,\"games\":10,\"wins\":9,\"caps\":0}\n"; }
  { std::ofstream f(lad + "/champions/ratings.jsonl");
    f << "{\"spec\":\"mlp:models/_leaderboard/champ.txt\",\"elo\":2014.0,\"games\":80,\"wins\":79,\"caps\":0}\n"
      << "{\"spec\":\"p:classic\",\"elo\":1000.0,\"games\":80,\"wins\":40,\"caps\":0}\n"; }
  { std::ofstream f(lad + "/newer/ratings.jsonl");
    f << "{\"spec\":\"mlp:models/run1/best.txt\",\"elo\":1350.5,\"games\":20,\"wins\":12,\"caps\":0}\n"; }
  fs::last_write_time(lad + "/newer/ratings.jsonl", fs::file_time_type::clock::now());
  // champions is NOT the newest file, so a 2014 win below proves canonical-rule, not recency

  auto r = loadRatings(lad);
  CHECK(r.at("mlp:models/leaderboard/champ.txt").elo == doctest::Approx(2014.0));
  CHECK(r.at("mlp:models/leaderboard/champ.txt").sweep == "champions");
  CHECK(r.count("mlp:models/_leaderboard/champ.txt") == 0);        // alias normalized, not duplicated
  CHECK(r.at("mlp:models/run1/best.txt").elo == doctest::Approx(1350.5));   // newest sweep wins
  CHECK(r.at("mlp:models/run1/best.txt").sweep == "newer");
  CHECK(r.at("p:classic").elo == doctest::Approx(1000.0));         // scripted bots carried too
  CHECK(loadRatings(root + "/no_such_dir").empty());

  auto ms = curatedModels(mod, r);
  REQUIRE(ms.size() == 3);                                          // champ + run1/best + run2/latest
  CHECK(ms[0].name == "leaderboard/champ");                         // rated, Elo-descending
  CHECK(ms[0].elo == doctest::Approx(2014.0));
  CHECK(ms[0].sweep == "champions");
  CHECK(ms[0].spec == "mlp:" + mod + "/leaderboard/champ.txt");
  CHECK(ms[1].name == "run1/best");
  CHECK(ms[1].elo == doctest::Approx(1350.5));
  CHECK(ms[2].name == "run2/latest");                               // unrated sorts to the tail
  CHECK(ms[2].elo < 0);
  fs::remove_all(root);
}

TEST_CASE("ratings index: lookup works for a bare relative modelsDir (the GUI's call shape)"){
  namespace fs = std::filesystem;
  fs::remove_all("ridx_rel_models");
  fs::create_directories("ridx_rel_models/leaderboard");
  fs::create_directories("ridx_rel_models/run1");
  { std::ofstream f("ridx_rel_models/leaderboard/champ.txt"); f << "x"; }
  { std::ofstream f("ridx_rel_models/run1/best.txt"); f << "x"; }
  std::map<std::string, Rating> r;
  r["mlp:models/leaderboard/champ.txt"] = Rating{2014.0, 80, "champions"};  // repo-relative, as real sweeps write
  r["mlp:models/run1/best.txt"]         = Rating{1350.5, 20, "newer"};
  auto ms = curatedModels("ridx_rel_models", r);
  REQUIRE(ms.size() == 2);
  CHECK(ms[0].elo == doctest::Approx(2014.0));   // rated via the repo-relative key
  CHECK(ms[1].elo == doctest::Approx(1350.5));
  fs::remove_all("ridx_rel_models");
}

TEST_CASE("ratings round-trip: what runLadder writes, loadRatings reads (writer-parser contract)"){
  namespace fs = std::filesystem;
  const std::string root = "/tmp/aipong_roundtrip";
  fs::remove_all(root);
  fs::create_directories(root + "/sweepA");
  LadderConfig cfg;
  cfg.roster = {"p:easy", "p:classic"};
  cfg.perPair = 1; cfg.threads = 2; cfg.seed = 3;
  cfg.outDir = root + "/sweepA";
  runLadder(cfg);
  auto r = loadRatings(root);                          // scans root/*/ratings.jsonl
  REQUIRE(r.count("p:classic") == 1);                  // parser reads the writer's format
  REQUIRE(r.count("p:easy") == 1);
  CHECK(r.at("p:classic").elo == doctest::Approx(1000.0));   // pinned anchor round-trips
  CHECK(r.at("p:easy").games > 0);
  CHECK(r.at("p:easy").sweep == "sweepA");
  CHECK(fs::exists(root + "/sweepA/ratings.jsonl"));   // final names exist,
  CHECK_FALSE(fs::exists(root + "/sweepA/ratings.jsonl.tmp"));  // tmp names don't linger
  fs::remove_all(root);
}

TEST_CASE("loadRatings: games==0 sentinel rows are not ratings and never overwrite one"){
  namespace fs = std::filesystem;
  const std::string root = "/tmp/aipong_sentinel";
  fs::remove_all(root);
  fs::create_directories(root + "/old");
  fs::create_directories(root + "/newer");
  { std::ofstream f(root + "/old/ratings.jsonl");
    f << "{\"spec\":\"mlp:models/x/best.txt\",\"elo\":1400.0,\"games\":12,\"wins\":8,\"caps\":0}\n"; }
  { std::ofstream f(root + "/newer/ratings.jsonl");    // all-rally-cap spec: unrated sentinel
    f << "{\"spec\":\"mlp:models/x/best.txt\",\"elo\":-1e+09,\"games\":0,\"wins\":0,\"caps\":4}\n"; }
  fs::last_write_time(root + "/newer/ratings.jsonl", fs::file_time_type::clock::now());
  auto r = loadRatings(root);
  REQUIRE(r.count("mlp:models/x/best.txt") == 1);
  CHECK(r.at("mlp:models/x/best.txt").elo == doctest::Approx(1400.0));  // old real rating survives
  CHECK(r.at("mlp:models/x/best.txt").sweep == "old");
  fs::remove_all(root);
}
