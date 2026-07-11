// aiming — characterize HOW a model beats the perfect defender. Plays <model> (left) vs the
// instant Interceptor (right) and, at every champion return (ball.vx flips −→+ in the left
// half), measures: contact offset (which sets the shot angle), the opponent's position, the
// predicted arrival-y, and whether that arrival lies OUTSIDE the defender's reachable set
// (|arrival − opp_y| > PaddleSpeed·t_arrival + paddle_half) — i.e. an unreturnable-by-geometry
// shot. Aggregates aiming correlations, reachability-exploitation %, and the speed/rally-depth
// regime of the winning shots.  usage: aiming <model.txt> [matches]
#include "match_runner.hpp"
#include "agents.hpp"
#include "pong_core.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace pong;

namespace {
struct Contact { double offset, oppY, speed, arrivalY, rallyHits; bool unreturnable; };

double pearson(const std::vector<Contact>& v, double Contact::* a, double Contact::* b){
  const size_t n = v.size(); if (n < 2) return 0.0;
  double sa = 0, sb = 0; for (const auto& c : v){ sa += c.*a; sb += c.*b; }
  const double ma = sa / n, mb = sb / n; double saa = 0, sbb = 0, sab = 0;
  for (const auto& c : v){ const double da = c.*a - ma, db = c.*b - mb; saa += da*da; sbb += db*db; sab += da*db; }
  if (saa <= 0 || sbb <= 0) return 0.0;
  return sab / std::sqrt(saa * sbb);
}
double frac(const std::vector<Contact>& v, bool Contact::* f){
  if (v.empty()) return 0.0; int k = 0; for (const auto& c : v) if (c.*f) k++; return (double)k / v.size();
}
double mean(const std::vector<Contact>& v, double Contact::* f){
  if (v.empty()) return 0.0; double s = 0; for (const auto& c : v) s += c.*f; return s / v.size();
}
double fastFrac(const std::vector<Contact>& v, double thr){
  if (v.empty()) return 0.0; int k = 0; for (const auto& c : v) if (c.speed > thr) k++; return (double)k / v.size();
}
}

int main(int argc, char** argv){
  if (argc < 2){ std::fprintf(stderr, "usage: aiming <model.txt> [matches]\n"); return 1; }
  const std::string model = std::string("mlp:") + argv[1];
  const int N = argc > 2 ? std::atoi(argv[2]) : 200;
  if (!makeAgent(model)){ std::fprintf(stderr, "cannot load %s\n", argv[1]); return 2; }

  std::vector<Contact> all, won;
  int leftPts = 0, rightPts = 0, caps = 0;

  for (int m = 0; m < N; ++m){
    MatchRunner r((unsigned)(1000 + m));
    r.setController(Side::Left, model);
    r.setController(Side::Right, "interceptor");
    r.newGame((unsigned)(1000 + m));

    double prevVx = 0; int prevL = 0, prevR = 0;
    std::vector<Contact> rally;
    long guard = 0;
    while (r.snapshot().phase != Phase::GameOver && guard++ < 20'000'000){
      r.tick(Move::None, Move::None);
      const Snapshot s = r.snapshot();
      if (prevVx < 0 && s.ball.vel.x > 0 && s.ball.pos.x < 0.5){       // a champion return
        Contact c;
        const double halfH = s.left.h * 0.5;
        c.offset = (s.ball.pos.y - s.left.y) / halfH;
        if (c.offset > 1) c.offset = 1; else if (c.offset < -1) c.offset = -1;
        c.oppY = s.right.y;
        c.speed = std::sqrt(s.ball.vel.x * s.ball.vel.x + s.ball.vel.y * s.ball.vel.y);
        const double planeX = s.right.x - s.right.w * 0.5;
        c.arrivalY = predictInterceptY(s.ball, planeX);
        const double t = (planeX - s.ball.pos.x) / s.ball.vel.x;        // s
        const double reach = k::PaddleSpeed * t + s.right.h * 0.5;
        c.unreturnable = std::fabs(c.arrivalY - s.right.y) > reach;
        c.rallyHits = s.rallyHits;
        rally.push_back(c); all.push_back(c);
      }
      if (s.score.left > prevL){ if (!rally.empty()) won.push_back(rally.back()); rally.clear(); }
      else if (s.score.right > prevR){ rally.clear(); }
      prevL = s.score.left; prevR = s.score.right; prevVx = s.ball.vel.x;
    }
    const Outcome o = r.snapshot().outcome;
    if (o == Outcome::LeftWin) leftPts++; else if (o == Outcome::RightWin) rightPts++; else caps++;
  }

  std::printf("model: %s   matches: %d   vs instant Interceptor\n", argv[1], N);
  std::printf("  record: L %d  R %d  cap %d  (%.0f%% match win)\n",
              leftPts, rightPts, caps, 100.0 * leftPts / (leftPts + rightPts + caps));
  std::printf("  returns logged: %zu   winning shots: %zu\n", all.size(), won.size());
  std::printf("AIM depends on opponent?\n");
  std::printf("  r(contact_offset, opp_y) = %+.3f\n", pearson(all, &Contact::offset, &Contact::oppY));
  std::printf("  r(arrival_y,      opp_y) = %+.3f   (negative = aims AWAY from the opponent)\n",
              pearson(all, &Contact::arrivalY, &Contact::oppY));
  std::printf("REACHABILITY exploited?\n");
  std::printf("  all returns outside defender's reachable set:  %.1f%%\n", 100.0 * frac(all, &Contact::unreturnable));
  std::printf("  winning shots outside reachable set:           %.1f%%\n", 100.0 * frac(won, &Contact::unreturnable));
  int quick = 0; for (const auto& c : won) if (c.rallyHits <= 2) quick++;
  const double quickFrac = won.empty() ? 0.0 : 100.0 * quick / won.size();
  std::printf("REGIME of the winning shot:\n");
  std::printf("  mean ball speed %.2f cu/s   (%.0f%% > 7 cu/s = one-shot edge regime)\n",
              mean(won, &Contact::speed), 100.0 * fastFrac(won, 7.0));
  std::printf("  mean rally depth at the kill: %.1f hits   (%.0f%% on <= 2 hits = quick, else setup)\n",
              mean(won, &Contact::rallyHits), quickFrac);
  return 0;
}
