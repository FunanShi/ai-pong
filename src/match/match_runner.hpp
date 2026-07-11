#pragma once
#include "controllers.hpp"
#include "recorder.hpp"
#include <string>

namespace pong {

// Owns the one true step loop: two controllers → Input → core tick → optional recording.
// The GUI drives tick() at 60 Hz (passing keyboard Moves for human sides); the league
// drives runToCompletion() flat-out. Recording auto-finalizes (truncated=false) when the
// game ends; newGame() and destruction finalize a live recording as truncated.
class MatchRunner {
public:
  explicit MatchRunner(unsigned seed = 0) : seed_(seed), core_(seed) {}
  ~MatchRunner(){ if (rec_.active()) rec_.end(core_.snapshot(), /*truncated=*/true); }

  // Recreates that side's agent from the spec (nullptr agent = externally driven /
  // human). Does not reset the game — call newGame() after, as the GUI does.
  void setController(Side s, const std::string& spec){
    (s == Side::Left ? leftSpec_ : rightSpec_) = spec;
    (s == Side::Left ? left_ : right_) = makeAgent(spec);
  }
  // Direct injection for trainers (in-memory genomes have no file spec). The label
  // stands in for the spec in match metadata if recording is used.
  void setController(Side s, std::unique_ptr<Agent> agent, const std::string& label){
    (s == Side::Left ? leftSpec_ : rightSpec_) = label;
    (s == Side::Left ? left_ : right_) = std::move(agent);
  }
  const std::string& controllerSpec(Side s) const { return s == Side::Left ? leftSpec_ : rightSpec_; }
  bool humanControlled(Side s) const { return s == Side::Left ? !left_ : !right_; }

  void newGame(unsigned seed){
    if (rec_.active()) rec_.end(core_.snapshot(), /*truncated=*/true);
    seed_ = seed;
    ticks_ = 0;
    core_.reset(seed);
    if (left_)  left_->reset();
    if (right_) right_->reset();
  }

  long ticks() const { return ticks_; }   // ticks actually stepped since newGame (match length)

  // Training-config knobs (persist across newGame). Ladder/league play uses the defaults
  // (k::RallyMax, k::Speedup) for comparability; trainers may shorten marathons and
  // accelerate the per-hit speedup to compress rally time.
  void setRallyCap(int cap){ core_.setRallyCap(cap); }
  void setSpeedup(double g){ core_.setSpeedup(g); }
  void setServeHold(double s){ core_.setServeHold(s); }
  void setWinScore(int w){ core_.setWinScore(w); }

  bool startRecording(const std::string& path){
    return rec_.begin(path, seed_, agentLabel(leftSpec_), agentLabel(rightSpec_));
  }
  void stopRecording(bool truncated){ rec_.end(core_.snapshot(), truncated); }
  bool recording() const { return rec_.active(); }
  int  recordedTicks() const { return rec_.ticks(); }
  const std::string& recordPath() const { return rec_.path(); }

  // One core tick. extLeft/extRight drive sides whose agent is nullptr and are ignored
  // otherwise. No-op once the game is over (after finalizing any live recording).
  void tick(Move extLeft = Move::None, Move extRight = Move::None){
    Snapshot pre = core_.snapshot();
    if (pre.phase == Phase::GameOver){
      if (rec_.active()) rec_.end(pre, /*truncated=*/false);
      return;
    }
    Input in{ left_  ? left_->act(pre, Side::Left)   : extLeft,
              right_ ? right_->act(pre, Side::Right) : extRight };
    core_.step(in);
    ticks_++;
    if (rec_.active()){
      Snapshot post = core_.snapshot();
      rec_.step(pre, in, post);
      if (post.phase == Phase::GameOver) rec_.end(post, /*truncated=*/false);
    }
  }

  // Headless driver (league path): both sides should be agents. maxTicks is a safety
  // bound; games terminate on their own via WinScore or the rally cap.
  Outcome runToCompletion(long maxTicks = 10'000'000){
    long n = 0;
    while (core_.snapshot().phase != Phase::GameOver && n++ < maxTicks) tick();
    return core_.snapshot().outcome;
  }

  Snapshot snapshot() const { return core_.snapshot(); }

private:
  unsigned seed_ = 0;
  long ticks_ = 0;
  PongCore core_;
  MatchRecorder rec_;
  std::unique_ptr<Agent> left_, right_;
  std::string leftSpec_{"human"}, rightSpec_{"human"};
};

} // namespace pong
