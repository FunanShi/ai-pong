#pragma once
// Shared multi-series line plot for the training dashboards (trainview + coview).
// Extracted verbatim from train_view.cpp — behavior-preserving.
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace pong {

struct Series { const char* name; ImU32 col; std::vector<float> xs, ys; };

// A titled multi-series line plot drawn at the current cursor. `percent` fixes the y-axis
// to 0..100 (win-rate panel); otherwise it auto-scales to the data (fitness / return).
inline void drawPlot(const char* title, float height, std::vector<Series>& series, bool percent){
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float w = ImGui::GetContentRegionAvail().x;
  const ImVec2 size(w, height);
  ImGui::Dummy(size);
  const bool hovered = ImGui::IsItemHovered();
  const ImVec2 mouse = ImGui::GetIO().MousePos;

  const float padL = 46, padR = 14, padT = 24, padB = 20;
  const ImVec2 a(p0.x + padL, p0.y + padT);
  const ImVec2 b(p0.x + size.x - padR, p0.y + size.y - padB);
  const ImU32 cGrid = IM_COL32(255, 255, 255, 20);
  const ImU32 cAxis = IM_COL32(255, 255, 255, 70);
  const ImU32 cText = IM_COL32(188, 187, 178, 255);

  dl->AddRectFilled(p0, ImVec2(p0.x + size.x, p0.y + size.y), IM_COL32(255, 255, 255, 8), 5.0f);
  dl->AddText(ImVec2(p0.x + padL, p0.y + 5), IM_COL32(232, 231, 226, 255), title);

  float x0 = 1e30f, x1 = -1e30f;
  bool any = false;
  for (const auto& s : series) for (float x : s.xs){ x0 = std::min(x0, x); x1 = std::max(x1, x); any = true; }
  if (!any){ dl->AddText(ImVec2(a.x, (a.y + b.y) * 0.5f), cText, "waiting for data..."); return; }
  if (x1 <= x0) x1 = x0 + 1;

  float y0, y1;
  if (percent){ y0 = 0; y1 = 100; }
  else {
    y0 = 1e30f; y1 = -1e30f;
    for (const auto& s : series) for (float y : s.ys){ y0 = std::min(y0, y); y1 = std::max(y1, y); }
    if (y1 <= y0){ y0 -= 1; y1 += 1; }
    const float m = (y1 - y0) * 0.08f; y0 -= m; y1 += m;
  }
  auto X = [&](float x){ return a.x + (x - x0) / (x1 - x0) * (b.x - a.x); };
  auto Y = [&](float y){ return b.y - (y - y0) / (y1 - y0) * (b.y - a.y); };

  for (int i = 0; i <= 4; ++i){
    const float yy = y0 + (y1 - y0) * (float)i / 4.0f;
    const float py = Y(yy);
    dl->AddLine(ImVec2(a.x, py), ImVec2(b.x, py), cGrid);
    char lb[24]; std::snprintf(lb, sizeof lb, percent ? "%.0f%%" : "%.2f", yy);
    dl->AddText(ImVec2(p0.x + 4, py - 7), cText, lb);
  }
  for (int i = 0; i <= 2; ++i){
    const float xx = x0 + (x1 - x0) * (float)i / 2.0f;
    char lb[24]; std::snprintf(lb, sizeof lb, "%.0f", xx);
    const ImVec2 ts = ImGui::CalcTextSize(lb);
    dl->AddText(ImVec2(X(xx) - ts.x * 0.5f, b.y + 4), cText, lb);
  }
  dl->AddLine(ImVec2(a.x, b.y), ImVec2(b.x, b.y), cAxis);

  float lx = a.x + 6;
  const float ly = a.y + 2;
  for (const auto& s : series){
    if (s.xs.size() >= 2){
      std::vector<ImVec2> pts; pts.reserve(s.xs.size());
      for (size_t i = 0; i < s.xs.size(); ++i) pts.push_back(ImVec2(X(s.xs[i]), Y(s.ys[i])));
      dl->AddPolyline(pts.data(), (int)pts.size(), s.col, 0, 2.0f);
      dl->AddCircleFilled(pts.back(), 3.0f, s.col);
    } else if (s.xs.size() == 1){
      dl->AddCircleFilled(ImVec2(X(s.xs[0]), Y(s.ys[0])), 3.0f, s.col);
    }
    dl->AddRectFilled(ImVec2(lx, ly + 3), ImVec2(lx + 12, ly + 9), s.col, 1.0f);
    dl->AddText(ImVec2(lx + 15, ly - 1), cText, s.name);
    lx += 15 + ImGui::CalcTextSize(s.name).x + 14;
  }

  if (hovered && mouse.x >= a.x && mouse.x <= b.x){
    dl->AddLine(ImVec2(mouse.x, a.y), ImVec2(mouse.x, b.y), IM_COL32(255, 255, 255, 55));
    const float xv = x0 + (mouse.x - a.x) / (b.x - a.x) * (x1 - x0);
    ImGui::BeginTooltip();
    ImGui::Text("%s = %.0f", percent ? "gen/step" : "gen/step", xv);
    for (const auto& s : series){
      if (s.xs.empty()) continue;
      size_t bi = 0; float bd = 1e30f;
      for (size_t i = 0; i < s.xs.size(); ++i){ const float d = std::fabs(s.xs[i] - xv); if (d < bd){ bd = d; bi = i; } }
      const ImVec4 c = ImGui::ColorConvertU32ToFloat4(s.col);
      ImGui::TextColored(c, percent ? "%s: %.0f%%" : "%s: %.3f", s.name, s.ys[bi]);
    }
    ImGui::EndTooltip();
  }
}

} // namespace pong
