// AI Pong — Dear ImGui frontend: rendering, keyboard input, and the Match panel.
// Everything else (controller construction, the step loop, recording lifecycle) is
// owned by match/MatchRunner — the same engine the headless league runner drives.
#include "match_runner.hpp"
#include "ratings_index.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <map>
#include <string>
#include <vector>
using namespace pong;

namespace {

Move humanMove(GLFWwindow* w, Side side, bool bothHuman){
  // Two humans: left = W/S, right = arrows. One human: that side gets all four keys.
  bool up, down;
  if (bothHuman){
    if (side == Side::Left){ up = glfwGetKey(w, GLFW_KEY_W)==GLFW_PRESS;  down = glfwGetKey(w, GLFW_KEY_S)==GLFW_PRESS; }
    else                   { up = glfwGetKey(w, GLFW_KEY_UP)==GLFW_PRESS; down = glfwGetKey(w, GLFW_KEY_DOWN)==GLFW_PRESS; }
  } else {
    up   = glfwGetKey(w, GLFW_KEY_W)==GLFW_PRESS || glfwGetKey(w, GLFW_KEY_UP)==GLFW_PRESS;
    down = glfwGetKey(w, GLFW_KEY_S)==GLFW_PRESS || glfwGetKey(w, GLFW_KEY_DOWN)==GLFW_PRESS;
  }
  return up && !down ? Move::Up : down && !up ? Move::Down : Move::None;
}

} // namespace

int main(){
  if(!glfwInit()){ std::fprintf(stderr,"glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
  GLFWwindow* win = glfwCreateWindow(960,640,"AI Pong",nullptr,nullptr);
  if(!win){ std::fprintf(stderr,"window failed\n"); glfwTerminate(); return 2; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  IMGUI_CHECKVERSION(); ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(win,true); ImGui_ImplOpenGL3_Init("#version 130");

  // Selectable controllers: Human + scripted roster (with canonical Elo where rated,
  // DECISIONS.md #21) + the curated model list (leaderboard set + run bests) + the env
  // model. Rebuilt every time a paddle combo opens, so fresh ladder rankings show up on
  // the next click. Elo is pool-relative: the source sweep is always shown.
  struct Entry { std::string spec, label; };
  std::vector<Entry> entries;
  const char* envModel = std::getenv("AIPONG_MODEL");
  auto rebuildEntries = [&](){
    const auto ratings = loadRatings("results/ladder");
    auto rated = [&](const std::string& spec, const std::string& base){
      const auto it = ratings.find(spec);
      if (it == ratings.end()) return base;
      char b[160];
      std::snprintf(b, sizeof b, "%s — %.0f · %s", base.c_str(), it->second.elo,
                    it->second.sweep.c_str());
      return std::string(b);
    };
    entries.clear();
    entries.push_back({"human", agentLabel("human")});
    for (const auto& s : builtinSpecs()) entries.push_back({s, rated(s, agentLabel(s))});
    for (const auto& m : curatedModels("models", ratings)){
      if (!makeAgent(m.spec)) continue;               // unloadable checkpoints drop out here
      char b[160];
      if (m.elo >= 0)
        std::snprintf(b, sizeof b, "%s — %.0f · %s", m.name.c_str(), m.elo, m.sweep.c_str());
      else
        std::snprintf(b, sizeof b, "%s — (unrated)", m.name.c_str());
      entries.push_back({m.spec, b});
    }
    if (envModel){                                    // env model still injected (not auto-selected)
      const std::string spec = std::string("mlp:") + envModel;
      bool present = false;
      for (const auto& e : entries) present = present || e.spec == spec;
      if (!present && makeAgent(spec)) entries.push_back({spec, rated(spec, agentLabel(spec))});
    }
  };
  rebuildEntries();

  std::string selLeft = "human";                      // selections are SPECS, not indices —
  std::string selRight = "p:classic";                 // a rescan can never invalidate them
  int  seed = 1;                                     // 1 = varied serves; 0 = legacy fixed 20°
  int  simSpeedIdx = 0;
  const int simSpeeds[] = {1, 2, 8, 32};
  bool paused = false;
  bool pKeyPrev = false;

  MatchRunner runner((unsigned)seed);
  runner.setController(Side::Left,  selLeft);
  runner.setController(Side::Right, selRight);
  runner.newGame((unsigned)seed);

  auto prev = std::chrono::steady_clock::now();
  double acc = 0;

  while(!glfwWindowShouldClose(win)){
    glfwPollEvents();
    auto now = std::chrono::steady_clock::now();
    acc += std::chrono::duration<double>(now-prev).count(); prev = now;
    ImGuiIO& io = ImGui::GetIO();

    bool bothHuman = runner.humanControlled(Side::Left) && runner.humanControlled(Side::Right);
    if (!io.WantCaptureKeyboard &&
        glfwGetKey(win,GLFW_KEY_SPACE)==GLFW_PRESS && runner.snapshot().phase==Phase::GameOver)
      runner.newGame((unsigned)seed);
    // Pause is a UI affordance (button or P key), deliberately outside the game's action
    // space: an agent's only output channel is a paddle Move, so only a human can pause.
    bool pKeyNow = glfwGetKey(win,GLFW_KEY_P)==GLFW_PRESS;
    if (pKeyNow && !pKeyPrev && !io.WantCaptureKeyboard) paused = !paused;
    pKeyPrev = pKeyNow;

    if (paused){
      acc = 0;                       // drain: resuming must not fast-forward a burst of ticks
    } else {
      while(acc >= k::Dt){
        acc -= k::Dt;
        for (int t = 0; t < simSpeeds[simSpeedIdx]; ++t){
          Move ml = Move::None, mr = Move::None;
          if (!io.WantCaptureKeyboard){
            if (runner.humanControlled(Side::Left))  ml = humanMove(win, Side::Left, bothHuman);
            if (runner.humanControlled(Side::Right)) mr = humanMove(win, Side::Right, bothHuman);
          }
          runner.tick(ml, mr);
        }
      }
    }

    Snapshot s = runner.snapshot();
    ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();

    // --- control panel: a pinned strip BELOW the court, so play is never obscured ---
    int W, H; glfwGetFramebufferSize(win, &W, &H);
    const float panelH = 270.0f;                       // control strip height [px]
    const float courtH = std::max(200.0f, (float)H - panelH);
    ImGui::SetNextWindowPos(ImVec2(0, courtH), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2((float)W, (float)H - courtH), ImGuiCond_Always);
    ImGui::Begin("Match", nullptr,
                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
    auto combo = [&](const char* id, std::string& sel, Side side){
      auto labelOf = [&](const std::string& spec) -> std::string {
        for (const auto& e : entries) if (e.spec == spec) return e.label;
        return agentLabel(spec);                      // selection vanished from the list: still shown
      };
      // the popup must never run off-screen: cap its height, ImGui scrolls past the cap
      ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0),
          ImVec2(FLT_MAX, std::min(320.0f, courtH * 0.6f)));
      if (ImGui::BeginCombo(id, labelOf(sel).c_str())){
        static ImGuiTextFilter filter;                // shared across both combos: fine, popup-scoped
        if (ImGui::IsWindowAppearing()){ rebuildEntries(); filter.Clear(); }
        if ((int)entries.size() > 15) filter.Draw("filter", 140.0f);
        for (const auto& e : entries){
          if (!filter.PassFilter(e.label.c_str())) continue;
          if (ImGui::Selectable(e.label.c_str(), e.spec == sel) && e.spec != sel){
            sel = e.spec;
            runner.setController(side, sel);
            runner.newGame((unsigned)seed);
          }
        }
        ImGui::EndCombo();
      }
    };
    combo("Left paddle", selLeft, Side::Left);
    combo("Right paddle", selRight, Side::Right);
    ImGui::SetNextItemWidth(120);
    if (ImGui::InputInt("Serve seed", &seed)) { if (seed < 0) seed = 0; }
    ImGui::SameLine(); if (ImGui::Button("New game")) runner.newGame((unsigned)seed);
    ImGui::SetNextItemWidth(120);
    if (ImGui::BeginCombo("Sim speed", (std::to_string(simSpeeds[simSpeedIdx]) + "x").c_str())){
      for (int i = 0; i < 4; ++i)
        if (ImGui::Selectable((std::to_string(simSpeeds[i]) + "x").c_str(), i == simSpeedIdx)) simSpeedIdx = i;
      ImGui::EndCombo();
    }
    if (ImGui::Button(paused ? "Resume (P)" : "Pause (P)")) paused = !paused;
    ImGui::Separator();
    if (!runner.recording()){
      if (ImGui::Button("Record match")){          // restarts for a clean episode from the serve
        runner.newGame((unsigned)seed);
        std::filesystem::create_directories("datasets");
        runner.startRecording("datasets/match_" + std::to_string((long long)std::time(nullptr))
                              + "_seed" + std::to_string(seed) + ".jsonl");
      }
    } else {
      ImGui::TextColored(ImVec4(1.0f,0.35f,0.35f,1.0f), "REC %d ticks", runner.recordedTicks());
      ImGui::TextUnformatted(runner.recordPath().c_str());
      if (ImGui::Button("Stop recording (marks truncated)")) runner.stopRecording(true);
    }
    ImGui::Separator();
    double ballSpeed = std::sqrt(s.ball.vel.x*s.ball.vel.x + s.ball.vel.y*s.ball.vel.y);
    ImGui::Text("ball speed: %6.2f cu/s (max %.0f)", ballSpeed, k::BallSpeedMax);
    ImGui::Text("rally:      %d / %d hits", s.rallyHits, s.rallyMax);
    if (s.phase == Phase::GameOver){
      const char* msg = s.outcome==Outcome::RallyCap ? "RALLY LIMIT — no winner (loss for both)"
                       : s.outcome==Outcome::LeftWin ? "LEFT WINS" : "RIGHT WINS";
      ImGui::TextColored(ImVec4(1.0f,0.8f,0.2f,1.0f), "%s", msg);
    }
    ImGui::End();

    // --- court ---
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 col = IM_COL32(255,255,255,255);
    auto rect = [&](const Paddle& p){
      dl->AddRectFilled(ImVec2((float)((p.x-p.w/2)*W),(float)((p.y-p.h/2)*courtH)),
                        ImVec2((float)((p.x+p.w/2)*W),(float)((p.y+p.h/2)*courtH)), col); };
    rect(s.left); rect(s.right);
    dl->AddCircleFilled(ImVec2((float)(s.ball.pos.x*W),(float)(s.ball.pos.y*courtH)),(float)(s.ball.r*W), col);
    dl->AddLine(ImVec2(0, courtH), ImVec2((float)W, courtH), IM_COL32(255,255,255,70));
    char buf[32]; std::snprintf(buf,sizeof buf,"%d    %d",s.score.left,s.score.right);
    dl->AddText(nullptr, 36.0f, ImVec2(W*0.5f-40,20), col, buf);
    if(s.phase==Phase::GameOver){
      const char* over = s.outcome==Outcome::RallyCap
        ? "RALLY LIMIT — no winner. Space for a new game"
        : (s.outcome==Outcome::LeftWin ? "LEFT WINS — Space for a new game"
                                       : "RIGHT WINS — Space for a new game");
      dl->AddText(nullptr, 28.0f, ImVec2(W*0.5f-260,courtH*0.5f), col, over);
    }
    if (paused)
      dl->AddText(nullptr, 30.0f, ImVec2(W*0.5f-60,courtH*0.42f), col, "PAUSED");
    ImGui::Render();
    glViewport(0,0,W,H); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(win);
  }
  if (runner.recording()) runner.stopRecording(/*truncated=*/true);   // window closed mid-recording
  ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext();
  glfwDestroyWindow(win); glfwTerminate();
  return 0;
}
