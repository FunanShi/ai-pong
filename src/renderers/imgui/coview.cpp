// AI Pong — co-evolution dashboard + control panel. Scans a base dir's island subdirs
// (models/coevo/<label>/train_log.jsonl), overlays each species' avg-margin-vs-bots and
// self-play fitness, and renders the cross-island win matrix. Writes <base>/control
// (pause/run/stop/eval) — the channel coEvolve() honors. Parsing lives in the tested
// train/train_log_view.hpp + train/coview_data.hpp; this file is window + rendering + that write.
//
//   ./pong coview [models/coevo]        GUI
//   ./build/pong_coview DIR --dump      headless summary (no display; for verification)
#include "train_log_view.hpp"
#include "coview_data.hpp"
#include "train_plot.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
using namespace pong;
namespace fs = std::filesystem;

namespace {

const ImU32 kIslandCol[6] = {
  IM_COL32( 58, 135, 229, 255), IM_COL32(237, 161,   0, 255), IM_COL32( 26, 175, 122, 255),
  IM_COL32(227,  73,  72, 255), IM_COL32(144, 133, 233, 255), IM_COL32(196, 148,  78, 255),
};

struct Island { std::string dir, label; LogTail tail; };

std::vector<Island> discoverIslands(const std::string& base){
  std::vector<Island> isl;
  std::error_code ec;
  if (fs::exists(base, ec))
    for (const auto& e : fs::directory_iterator(base, ec))
      if (e.is_directory() && fs::exists(e.path() / "train_log.jsonl")){
        Island is; is.dir = e.path().string(); is.label = e.path().filename().string();
        isl.push_back(std::move(is));
      }
  std::sort(isl.begin(), isl.end(), [](const Island& a, const Island& b){ return a.label < b.label; });
  for (auto& is : isl) is.tail.reset(is.dir + "/train_log.jsonl");
  return isl;
}

RunStatus readBaseStatus(const std::string& base){
  std::ifstream sf(base + "/status.json");
  const std::string txt((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
  return parseStatus(txt);
}

// Diverging red→green by margin ∈ [−1,1]; 0 ≈ neutral grey.
ImU32 marginColor(double v){
  if (v > 1) v = 1;
  if (v < -1) v = -1;
  const int lo = 70, hi = 200;
  if (v >= 0) return IM_COL32(lo + (int)((1 - v) * (hi - lo)), hi, lo, 255);   // → green
  return IM_COL32(hi, lo + (int)((1 + v) * (hi - lo)), lo, 255);              // → red
}

// Compact matrix label: "h32-32-32" -> "32x3" (width x #hidden-layers); falls back to the raw
// label (minus a leading 'h') when the hidden layers are not all the same width.
std::string shortArch(const std::string& label){
  std::string s = (!label.empty() && label[0] == 'h') ? label.substr(1) : label;
  std::vector<std::string> parts; size_t p = 0;
  for (;;){
    const size_t q = s.find('-', p);
    parts.push_back(s.substr(p, q == std::string::npos ? q : q - p));
    if (q == std::string::npos) break;
    p = q + 1;
  }
  bool same = parts.size() >= 2;
  for (const auto& t : parts) if (t != parts[0]) same = false;
  return same ? parts[0] + "x" + std::to_string(parts.size()) : s;
}

// Compact cross-island win matrix: small cells (diverging red->green by margin), short species
// labels colored to match the plot legends. Sized to sit beside the plots with no scroll.
void drawHeatmap(const CrossWin& cw){
  ImDrawList* dl = ImGui::GetWindowDrawList();
  const ImVec2 p0 = ImGui::GetCursorScreenPos();
  const int n = (int)cw.labels.size();
  const float cell = 44, lblW = 44, lblH = 18, gap = 3;
  if (n == 0){ ImGui::TextUnformatted("(appears at the first eval gen)"); return; }
  std::vector<std::string> lab((size_t)n);
  for (int i = 0; i < n; ++i) lab[(size_t)i] = shortArch(cw.labels[(size_t)i]);
  for (int j = 0; j < n; ++j)                                        // column headers (short, colored)
    dl->AddText(ImVec2(p0.x + lblW + j * cell + 2, p0.y), kIslandCol[(size_t)j % 6], lab[(size_t)j].c_str());
  for (int i = 0; i < n; ++i){
    dl->AddText(ImVec2(p0.x, p0.y + lblH + i * cell + cell * 0.35f), kIslandCol[(size_t)i % 6], lab[(size_t)i].c_str());
    for (int j = 0; j < n; ++j){
      const double v = (i < (int)cw.matrix.size() && j < (int)cw.matrix[(size_t)i].size())
                       ? cw.matrix[(size_t)i][(size_t)j] : 0.0;
      const ImVec2 c0(p0.x + lblW + j * cell, p0.y + lblH + i * cell);
      const ImU32 col = (i == j) ? IM_COL32(60, 60, 60, 255) : marginColor(v);
      dl->AddRectFilled(c0, ImVec2(c0.x + cell - gap, c0.y + cell - gap), col, 3.0f);
      if (i != j){
        char t[16]; std::snprintf(t, sizeof t, "%+.2f", v);
        const ImVec2 ts = ImGui::CalcTextSize(t);
        dl->AddText(ImVec2(c0.x + (cell - gap - ts.x) * 0.5f, c0.y + (cell - gap - ts.y) * 0.5f),
                    IM_COL32(20, 20, 20, 255), t);
      }
    }
  }
  ImGui::Dummy(ImVec2(lblW + n * cell, lblH + n * cell + 4));        // reserve layout space
}

int runDump(const std::string& base){
  std::vector<Island> isl = discoverIslands(base);
  std::printf("base: %s\nislands: %zu\n", base.c_str(), isl.size());
  for (auto& is : isl){
    is.tail.poll();
    std::vector<float> xs, ys; bool marg = false;
    avgMarginSeries(is.tail.rows, xs, ys, marg);
    std::printf("  %-12s rows=%3zu  %s last=%s\n", is.label.c_str(), is.tail.rows.size(),
                marg ? "avg-margin" : "avg-winrate%",
                ys.empty() ? "(none)" : (std::to_string(ys.back())).c_str());
  }
  const RunStatus st = readBaseStatus(base);
  if (st.valid) std::printf("status: %s  at %.0f / %.0f\n", st.state.c_str(), st.gen, st.gensTotal);
  CrossWin cw;
  if (readLastCrossWin(base + "/crosswin.jsonl", cw))
    std::printf("crosswin: gen %d, %zux%zu\n", cw.gen, cw.labels.size(), cw.matrix.size());
  return 0;
}

} // namespace

int main(int argc, char** argv){
  bool dump = false;
  std::string base = "models/coevo";
  for (int i = 1; i < argc; ++i){
    const std::string a = argv[i];
    if (a == "--dump") dump = true;
    else base = a;
  }
  if (dump) return runDump(base);

  if (!glfwInit()){ std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  GLFWwindow* win = glfwCreateWindow(1000, 760, "AI Pong - Co-evolution", nullptr, nullptr);
  if (!win){ std::fprintf(stderr, "window failed\n"); glfwTerminate(); return 2; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  IMGUI_CHECKVERSION(); ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(win, true); ImGui_ImplOpenGL3_Init("#version 130");

  std::vector<Island> isl = discoverIslands(base);
  RunStatus status;
  CrossWin cross;
  double pollAcc = 0;
  auto prev = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(win)){
    glfwPollEvents();
    const auto now = std::chrono::steady_clock::now();
    pollAcc += std::chrono::duration<double>(now - prev).count(); prev = now;
    if (pollAcc > 0.4){                                    // ~2.5 Hz live refresh
      pollAcc = 0;
      // Islands appear as coEvolve writes their dirs (all N at gen-0 end); a poll landing
      // mid-creation must adopt the rest, not stick at a subset. Rescan when the set grows,
      // migrating existing LogTails by label so their read offsets survive (no full re-read).
      std::vector<Island> found = discoverIslands(base);
      if (found.size() > isl.size()){
        for (auto& f : found)
          for (auto& e : isl)
            if (e.label == f.label){ f.tail = std::move(e.tail); break; }
        isl = std::move(found);
      }
      for (auto& is : isl) is.tail.poll();
      status = readBaseStatus(base);
      CrossWin latest;                                     // keep last-good on a torn/partial read
      if (readLastCrossWin(base + "/crosswin.jsonl", latest)) cross = latest;
    }

    ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
    int W, H; glfwGetFramebufferSize(win, &W, &H);
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)W, (float)H));
    ImGui::Begin("coview", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

    // header
    char eta[48] = "";
    if (status.valid && status.gen >= 0 && status.gensTotal > 0){
      double pace = 0; int n = 0;                          // mean sec/gen from island 0's last 20 rows
      if (!isl.empty())
        for (size_t i = isl[0].tail.rows.size() > 20 ? isl[0].tail.rows.size() - 20 : 0;
             i < isl[0].tail.rows.size(); ++i){
          const auto it = isl[0].tail.rows[i].find("sec");
          if (it != isl[0].tail.rows[i].end()){ pace += it->second; ++n; }
        }
      const double e = etaSeconds(status.gen, status.gensTotal, n ? pace / n : 0);
      if (e >= 0) std::snprintf(eta, sizeof eta, "   ETA ~%.0f min", e / 60.0);
    }
    ImGui::Text("CO-EVOLUTION  %s   %d islands   gen %.0f / %.0f%s   [%s]",
                fs::path(base).filename().string().c_str(), (int)isl.size(),
                status.valid ? status.gen : -1, status.valid ? status.gensTotal : -1, eta,
                status.valid ? status.state.c_str() : "?");

    // control row — writes <base>/control (coEvolve honors it)
    auto ctl = [&](const char* tok){ std::ofstream f(base + "/control", std::ios::trunc); f << tok; };
    const bool paused = status.valid && status.state == "paused";
    if (ImGui::Button(paused ? "Resume" : "Pause")) ctl(paused ? "run" : "pause");
    ImGui::SameLine(); if (ImGui::Button("Stop")) ctl("stop");
    ImGui::SameLine(); if (ImGui::Button("Eval now")) ctl("eval");
    ImGui::SameLine(); ImGui::TextDisabled("(controls write %s/control)", base.c_str());
    ImGui::Separator();

    if (isl.empty()){
      ImGui::TextUnformatted("no island runs under this dir yet - launch with ./pong coevolve --out <dir>");
    } else {
      const ImVec2 avail = ImGui::GetContentRegionAvail();
      const float rightW = std::clamp(avail.x * 0.30f, 230.0f, 300.0f);   // win-matrix column
      const float leftW  = avail.x - rightW - 10.0f;                      // plots (narrower)

      // ---- left column: the two overlaid line plots, stacked to fill the height ----
      ImGui::BeginChild("plots", ImVec2(leftW, avail.y), false);
      {
        const float ph = (ImGui::GetContentRegionAvail().y - 6.0f) / 2.0f;

        // Panel 1 - avg margin vs bots (fallback: win-rate %), per species
        std::vector<Series> marg; bool anyMargin = false;
        for (size_t i = 0; i < isl.size(); ++i){
          Series s{isl[i].label.c_str(), kIslandCol[i % 6], {}, {}};
          bool used = false; avgMarginSeries(isl[i].tail.rows, s.xs, s.ys, used);
          // All islands in one run share a log format (coEvolve writes marg_* for every island
          // each gen), so OR here == AND - the panel y-axis mode is uniform across species.
          anyMargin = anyMargin || used;
          if (!s.xs.empty()) marg.push_back(std::move(s));
        }
        drawPlot(anyMargin ? "Avg margin vs bots per species  (higher = stronger)"
                           : "Avg win% vs bots per species  (win-rate fallback)",
                 ph, marg, !anyMargin);
        ImGui::Dummy(ImVec2(0, 6));

        // Panel 2 - self-play fitness (the Red-Queen foil)
        std::vector<Series> fitn;
        for (size_t i = 0; i < isl.size(); ++i){
          Series s{isl[i].label.c_str(), kIslandCol[i % 6], {}, {}};
          extractSeries(isl[i].tail.rows, "gen", "best", false, s.xs, s.ys);
          if (!s.xs.empty()) fitn.push_back(std::move(s));
        }
        drawPlot("Self-play best fitness per species  (relative, not a bot comparison)", ph, fitn, false);
      }
      ImGui::EndChild();

      // ---- right column: the cross-island win matrix, beside the plots (no scroll) ----
      ImGui::SameLine();
      ImGui::BeginChild("winmatrix", ImVec2(rightW, avail.y), false);
      {
        ImGui::TextUnformatted("Win matrix");
        ImGui::TextDisabled("row vs col; green = winning");
        ImGui::Dummy(ImVec2(0, 4));
        drawHeatmap(cross);
      }
      ImGui::EndChild();
    }

    ImGui::End();
    ImGui::Render();
    glViewport(0, 0, W, H); glClearColor(0.055f, 0.055f, 0.05f, 1); glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(win);
  }
  ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown(); ImGui::DestroyContext();
  glfwDestroyWindow(win); glfwTerminate();
  return 0;
}
