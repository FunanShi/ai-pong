#include "recorder.hpp"
#include <ctime>

namespace pong {
namespace {
int moveInt(Move m){ return m == Move::Up ? 1 : m == Move::Down ? 2 : 0; }
char phaseChar(Phase p){ return p == Phase::Serving ? 'S' : p == Phase::Playing ? 'P' : 'O'; }
const char* outcomeStr(Outcome o){
  return o == Outcome::LeftWin  ? "LeftWin"
       : o == Outcome::RightWin ? "RightWin"
       : o == Outcome::RallyCap ? "RallyCap" : "None";
}
} // namespace

bool MatchRecorder::begin(const std::string& path, unsigned seed,
                          const std::string& leftName, const std::string& rightName){
  if (out_.is_open()) return false;
  out_.open(path);
  if (!out_) return false;
  path_ = path;
  ticks_ = 0;
  lastScore_ = {};
  out_.setf(std::ios::fixed);
  out_.precision(5);
  out_ << "{\"aipong_match\":1,\"seed\":" << seed
       << ",\"left\":\"" << leftName << "\",\"right\":\"" << rightName << "\""
       << ",\"dt_s\":" << k::Dt << ",\"win_score\":" << k::WinScore
       << ",\"rally_max\":" << k::RallyMax << ",\"speedup\":" << k::Speedup
       << ",\"ball_speed_max\":" << k::BallSpeedMax
       << ",\"started_unix_s\":" << (long long)std::time(nullptr) << "}\n";
  return true;
}

void MatchRecorder::step(const Snapshot& pre, const Input& in, const Snapshot& post){
  if (!out_.is_open()) return;
  out_ << "{\"t\":" << ticks_
       << ",\"b\":[" << pre.ball.pos.x << ',' << pre.ball.pos.y << ','
                     << pre.ball.vel.x << ',' << pre.ball.vel.y << ']'
       << ",\"ly\":" << pre.left.y << ",\"ry\":" << pre.right.y
       << ",\"al\":" << moveInt(in.left) << ",\"ar\":" << moveInt(in.right)
       << ",\"ph\":\"" << phaseChar(pre.phase) << "\",\"rally\":" << pre.rallyHits;
  if (post.score.left  > lastScore_.left)  out_ << ",\"ev\":\"pointL\"";
  if (post.score.right > lastScore_.right) out_ << ",\"ev\":\"pointR\"";
  if (post.outcome == Outcome::RallyCap && pre.outcome == Outcome::None)
    out_ << ",\"ev\":\"rallycap\"";
  out_ << "}\n";
  lastScore_ = post.score;
  ticks_++;
}

void MatchRecorder::end(const Snapshot& fin, bool truncated){
  if (!out_.is_open()) return;
  out_ << "{\"end\":true,\"outcome\":\"" << outcomeStr(fin.outcome) << "\""
       << ",\"score\":[" << fin.score.left << ',' << fin.score.right << ']'
       << ",\"ticks\":" << ticks_
       << ",\"truncated\":" << (truncated ? "true" : "false") << "}\n";
  out_.close();
}

} // namespace pong
