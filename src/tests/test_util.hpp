#pragma once
#include "pong_core.hpp"
#include <cmath>

namespace testutil {

inline double spd(const pong::Vec2& v){ return std::sqrt(v.x*v.x + v.y*v.y); }

// Court with both paddles centered, mid-rally — the standard physics-test fixture.
inline pong::State playing(){
  pong::State s; s.phase = pong::Phase::Playing;
  s.left  = {pong::k::LeftX,  0.5, pong::k::PaddleH, pong::k::PaddleW};
  s.right = {pong::k::RightX, 0.5, pong::k::PaddleH, pong::k::PaddleW};
  return s;
}

} // namespace testutil
