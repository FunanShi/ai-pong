#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "agents.hpp"
#include "recorder.hpp"
#include "rally_index.hpp"
#include "test_util.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace pong;
using testutil::playing;

// ---------------- match recorder ----------------

TEST_CASE("MatchRecorder writes metadata, aligned tick lines, and a terminal summary"){
  const char* path = "/tmp/aipong_rec_test.jsonl";
  MatchRecorder rec;
  PongCore core(3);
  CHECK(rec.begin(path, 3, "botA", "botB"));
  CHECK(rec.active());
  for (int i=0;i<10;i++){
    Snapshot pre = core.snapshot();
    Input in{Move::Up, Move::Down};
    core.step(in);
    rec.step(pre, in, core.snapshot());
  }
  rec.end(core.snapshot(), /*truncated=*/true);
  CHECK_FALSE(rec.active());
  CHECK(rec.ticks() == 10);

  std::ifstream f(path);
  std::vector<std::string> lines; std::string l;
  while (std::getline(f, l)) lines.push_back(l);
  REQUIRE(lines.size() == 12);                            // meta + 10 ticks + summary
  CHECK(lines.front().find("\"aipong_match\":1") != std::string::npos);
  CHECK(lines.front().find("\"seed\":3") != std::string::npos);
  CHECK(lines.front().find("\"left\":\"botA\"") != std::string::npos);
  CHECK(lines[1].find("\"t\":0") != std::string::npos);
  CHECK(lines[1].find("\"al\":1") != std::string::npos);  // Up = 1
  CHECK(lines[1].find("\"ar\":2") != std::string::npos);  // Down = 2
  CHECK(lines[1].find("\"ph\":\"S\"") != std::string::npos); // recording starts at the serve
  CHECK(lines.back().find("\"end\":true") != std::string::npos);
  CHECK(lines.back().find("\"ticks\":10") != std::string::npos);
  CHECK(lines.back().find("\"truncated\":true") != std::string::npos);
}
TEST_CASE("MatchRecorder flags point events on the tick that caused them"){
  const char* path = "/tmp/aipong_rec_event_test.jsonl";
  MatchRecorder rec;
  CHECK(rec.begin(path, 0, "a", "b"));
  State pre = playing();                                   // score 0-0
  State post = playing(); post.score.left = 1;             // left scored this tick
  rec.step(view(pre), Input{}, view(post));
  rec.end(view(post), false);
  std::ifstream f(path);
  std::vector<std::string> lines; std::string l;
  while (std::getline(f, l)) lines.push_back(l);
  REQUIRE(lines.size() == 3);
  CHECK(lines[1].find("\"ev\":\"pointL\"") != std::string::npos);
  CHECK(lines.back().find("\"truncated\":false") != std::string::npos);
}

// ---------------- rally database ----------------

TEST_CASE("rally index: a full bot match segments into rallies matching the final score"){
  const char* path = "/tmp/aipong_rdb_match.jsonl";
  PongCore core(11);
  PAgent l("hard", 0.02, 1.0), r("easy", 0.02, 0.55);
  MatchRecorder rec;
  REQUIRE(rec.begin(path, 11, "P hard", "P easy"));
  int guard = 0;
  while (core.snapshot().phase != Phase::GameOver && guard++ < 200000){
    Snapshot pre = core.snapshot();
    Input in{ l.act(pre, Side::Left), r.act(pre, Side::Right) };
    core.step(in);
    rec.step(pre, in, core.snapshot());
  }
  Snapshot fin = core.snapshot();
  REQUIRE(fin.phase == Phase::GameOver);              // hard-vs-easy must terminate by skill
  rec.end(fin, false);

  std::ifstream f(path);
  auto rows = indexMatchStream(f, "m.jsonl");
  CHECK((int)rows.size() == fin.score.left + fin.score.right);
  int lw = 0, rw = 0, totalTicks = 0;
  for (const auto& row : rows){
    if (row.winner == "left")  lw++;
    if (row.winner == "right") rw++;
    bool endOk = row.end == "left_missed" || row.end == "right_missed";
    CHECK(endOk);
    CHECK(row.left == "P hard");
    CHECK(row.right == "P easy");
    CHECK(row.ticks > 0);
    CHECK(row.dur_s == doctest::Approx(row.ticks * k::Dt));
    CHECK(row.vmax >= k::ServeSpeed - 1e-9);
    totalTicks += row.ticks;
  }
  CHECK(lw == fin.score.left);                        // winners tally == the scoreboard
  CHECK(rw == fin.score.right);
  CHECK(totalTicks == rec.ticks());                   // every recorded tick belongs to exactly one rally
}
TEST_CASE("rally index: a truncated recording yields a no-winner truncated row"){
  const char* path = "/tmp/aipong_rdb_trunc.jsonl";
  PongCore core(2);
  MatchRecorder rec;
  REQUIRE(rec.begin(path, 2, "a", "b"));
  for (int i = 0; i < 50; i++){
    Snapshot pre = core.snapshot();
    Input in{};
    core.step(in);
    rec.step(pre, in, core.snapshot());
  }
  rec.end(core.snapshot(), true);
  std::ifstream f(path);
  auto rows = indexMatchStream(f, "t.jsonl");
  REQUIRE(rows.size() == 1);
  CHECK(rows[0].winner == "none");
  CHECK(rows[0].end == "truncated");
  CHECK(rows[0].ticks == 50);
  CHECK(rows[0].lineStart == 2);                      // line 1 is the metadata
  CHECK(rows[0].lineEnd == 51);
}
TEST_CASE("rally index: rally-cap row has no winner; index round-trips"){
  std::string m =
    "{\"aipong_match\":1,\"seed\":9,\"left\":\"A\",\"right\":\"B\",\"dt_s\":0.01667}\n"
    "{\"t\":0,\"b\":[0.50000,0.50000,0.70000,0.20000],\"ly\":0.5,\"ry\":0.5,\"al\":0,\"ar\":0,\"ph\":\"P\",\"rally\":998}\n"
    "{\"t\":1,\"b\":[0.60000,0.50000,-0.70000,0.20000],\"ly\":0.5,\"ry\":0.5,\"al\":0,\"ar\":0,\"ph\":\"P\",\"rally\":999,\"ev\":\"rallycap\"}\n"
    "{\"end\":true,\"outcome\":\"RallyCap\",\"score\":[0,0],\"ticks\":2,\"truncated\":false}\n";
  std::istringstream in(m);
  auto rows = indexMatchStream(in, "cap.jsonl");
  REQUIRE(rows.size() == 1);
  CHECK(rows[0].winner == "none");
  CHECK(rows[0].end == "rally_cap");
  CHECK(rows[0].hits == k::RallyMax);
  CHECK(rows[0].servedTo == "right");                 // first tick vx > 0
  CHECK(rows[0].ticks == 2);

  std::ostringstream idx;
  writeIndex(idx, rows);
  std::istringstream idxIn(idx.str());
  auto back = readIndex(idxIn);
  REQUIRE(back.size() == 1);
  CHECK(back[0].match == "cap.jsonl");
  CHECK(back[0].winner == "none");
  CHECK(back[0].end == "rally_cap");
  CHECK(back[0].hits == rows[0].hits);
  CHECK(back[0].ticks == rows[0].ticks);
  CHECK(back[0].lineStart == rows[0].lineStart);
  CHECK(back[0].lineEnd == rows[0].lineEnd);
  CHECK(back[0].vmax == doctest::Approx(rows[0].vmax));
  CHECK(back[0].servedTo == "right");
}
