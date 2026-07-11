// AI Pong — live training dashboard AND operator control panel. Tails
// models/<run>/train_log.jsonl (evolve or ppo, auto-detected) and plots the two things a
// training run is judged by: the scoring metric over time, and success % against the
// deterministic bots. It is NOT read-only: it writes one-token <run>/control files
// (pause/run/stop/eval, DECISIONS.md #19) and results/queue/ launch requests for the host
// executor (DECISIONS.md #20). Parsing lives in the tested train/train_log_view.hpp and
// train/recipes.hpp; this file is window + rendering + those file writes only.
//
//   ./pong trainview [models/evoN]     GUI (default: newest run in models/)
//   ./build/pong_trainview DIR --dump  headless summary (no display; for verification)
#include "train_log_view.hpp"
#include "recipes.hpp"
#include "train_plot.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
using namespace pong;
namespace fs = std::filesystem;

namespace {

std::vector<std::string> discoverRuns(){
  std::vector<std::string> dirs;
  std::error_code ec;
  if (fs::exists("models", ec))
    for (const auto& e : fs::directory_iterator("models", ec))
      if (e.is_directory() && fs::exists(e.path() / "train_log.jsonl"))
        dirs.push_back(e.path().string());
  std::sort(dirs.begin(), dirs.end());
  return dirs;
}

// PPO "warm-start from" candidates: every models/<run>/best.txt|latest.txt plus every curated
// models/leaderboard/*.txt — the same file set Decision 21's play picker curates, scanned here
// directly (plain fs, no link dependency on src/league/ratings_index.hpp).
std::vector<std::string> discoverWarmStarts(){
  std::vector<std::string> files;
  std::error_code ec;
  if (fs::exists("models", ec))
    for (const auto& e : fs::directory_iterator("models", ec)){
      if (!e.is_directory()) continue;
      for (const char* b : {"best.txt", "latest.txt"}){
        const fs::path f = e.path() / b;
        if (fs::exists(f, ec)) files.push_back(f.string());
      }
    }
  if (fs::exists("models/leaderboard", ec))
    for (const auto& e : fs::directory_iterator("models/leaderboard", ec))
      if (e.is_regular_file() && e.path().extension() == ".txt") files.push_back(e.path().string());
  std::sort(files.begin(), files.end());
  return files;
}

// Anchor panel definition: the deterministic bots, in difficulty order. Missing keys (old
// runs predating held-out anchors; ppo, which logs only three) are simply absent.
struct AnchorDef { const char* key; const char* name; const char* reevalName; ImU32 col; };
const std::vector<AnchorDef> kAnchors = {
  {"vs_p_classic",   "P-classic",             "P-classic (reeval)",       IM_COL32( 26, 175, 122, 255)},
  {"vs_laggy",       "Interceptor (laggy)",   "laggy (reeval)",           IM_COL32(237, 161,   0, 255)},
  {"vs_interceptor", "Interceptor (instant)", "instant (reeval)",         IM_COL32(227,  73,  72, 255)},
  {"vs_held_p",      "P held-out",            "P held-out (reeval)",      IM_COL32( 58, 135, 229, 255)},
  {"vs_held_icept",  "Interceptor held-out",  "held-out icept (reeval)",  IM_COL32(144, 133, 233, 255)},
};

// Curriculum telemetry panel (DECISIONS.md #19): population-mean margin per opponent class.
struct ClassDef { const char* key; const char* name; ImU32 col; };
const std::vector<ClassDef> kClasses = {
  {"fit_peer",   "peers + HoF",       IM_COL32( 58, 135, 229, 255)},
  {"fit_pool",   "frozen pool",       IM_COL32(237, 161,   0, 255)},
  {"fit_anchor", "scripted anchors",  IM_COL32(227,  73,  72, 255)},
};

std::string reevalPathFor(const std::string& runDir){
  return "results/reeval/" + fs::path(runDir).filename().string() + ".jsonl";
}

int runDump(const std::string& wantDir){
  std::vector<std::string> runs = discoverRuns();
  std::string dir = wantDir;
  if (!dir.empty()){
    const auto pos = dir.find("train_log.jsonl");
    if (pos != std::string::npos) dir = dir.substr(0, pos);
    while (!dir.empty() && (dir.back() == '/' || dir.back() == ' ')) dir.pop_back();
  } else if (!runs.empty()) dir = runs.back();
  if (dir.empty()){ std::printf("no runs found under models/\n"); return 0; }

  LogTail t; t.reset(dir + "/train_log.jsonl");
  t.poll();
  std::printf("run: %s\nschema: %s\nrows: %zu\n", dir.c_str(), t.schema.c_str(), t.rows.size());
  const char* xk = (t.schema == "ppo") ? "step" : "gen";
  auto summ = [&](const char* yk, bool drop){
    std::vector<float> xs, ys; extractSeries(t.rows, xk, yk, drop, xs, ys);
    if (ys.empty()) return;
    float mn = ys[0], mx = ys[0];
    for (float v : ys){ mn = std::min(mn, v); mx = std::max(mx, v); }
    std::printf("  %-16s n=%3zu  range[% .3f, % .3f]  last=% .3f\n", yk, ys.size(), mn, mx, ys.back());
  };
  if (t.schema == "ppo") summ("ep_return_mean", false);
  else { summ("best", false); summ("mean", false); }
  for (const auto& d : kAnchors) summ(d.key, true);
  for (const auto& d : kClasses) summ(d.key, true);
  summ("champ_age", true);
  { std::ifstream sf(dir + "/status.json");
    const std::string txt((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
    const RunStatus st = parseStatus(txt);
    if (st.valid) std::printf("status: %s  at %.0f / %.0f\n", st.state.c_str(), st.gen, st.gensTotal); }
  { LogTail rv; rv.reset(reevalPathFor(dir)); rv.poll();
    if (!rv.rows.empty()) std::printf("reeval: %zu recovered rows\n", rv.rows.size()); }
  std::printf("recipes: %zu available\n", discoverRecipes("recipes").size());
  return 0;
}

} // namespace

int main(int argc, char** argv){
  bool dump = false;
  std::string wantDir;
  for (int i = 1; i < argc; ++i){
    const std::string a = argv[i];
    if (a == "--dump") dump = true;
    else wantDir = a;
  }
  if (dump) return runDump(wantDir);

  if (!glfwInit()){ std::fprintf(stderr, "glfwInit failed\n"); return 1; }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  GLFWwindow* win = glfwCreateWindow(1000, 720, "AI Pong — Training", nullptr, nullptr);
  if (!win){ std::fprintf(stderr, "window failed\n"); glfwTerminate(); return 2; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  IMGUI_CHECKVERSION(); ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(win, true); ImGui_ImplOpenGL3_Init("#version 130");

  std::vector<std::string> runs = discoverRuns();
  int sel = 0;
  if (!wantDir.empty()){
    std::string want = wantDir;
    const auto pos = want.find("train_log.jsonl");
    if (pos != std::string::npos) want = want.substr(0, pos);
    while (!want.empty() && (want.back() == '/' || want.back() == ' ')) want.pop_back();
    const std::string wbase = fs::path(want).filename().string();
    int found = -1;
    for (int i = 0; i < (int)runs.size(); ++i)
      if (runs[i] == want || fs::path(runs[i]).filename().string() == wbase){ found = i; break; }
    if (found < 0){ runs.push_back(want); found = (int)runs.size() - 1; }   // log not written yet
    sel = found;
  }
  LogTail tail;
  LogTail reeval;                                       // recovered history (results/reeval/)
  RunStatus status;                                     // trainer heartbeat (<run>/status.json)
  bool smooth = false;
  std::vector<Recipe> recipes = discoverRecipes("recipes");
  int recipeSel = 0;
  char nrArgs[2048] = "", nrOut[256] = "", nrSeed[32] = "", nrResume[256] = "", nrInit[256] = "";
  auto loadRecipeForm = [&](int i){
    if (i < 0 || i >= (int)recipes.size()) return;
    recipeSel = i;
    std::string joined;
    for (const auto& a : recipes[(size_t)i].args){ if (!joined.empty()) joined += ' '; joined += a; }
    std::snprintf(nrArgs, sizeof nrArgs, "%s", joined.c_str());
    std::snprintf(nrOut, sizeof nrOut, "%s", recipes[(size_t)i].out.c_str());
    nrSeed[0] = 0; nrResume[0] = 0; nrInit[0] = 0;
  };
  if (!recipes.empty()) loadRecipeForm(0);
  auto selectRun = [&](int i){
    if (i >= 0 && i < (int)runs.size()){
      sel = i;
      tail.reset(runs[i] + "/train_log.jsonl"); tail.poll();
      reeval.reset(reevalPathFor(runs[i])); reeval.poll();
      status = RunStatus{};
    }
  };
  if (!runs.empty()) selectRun(sel);

  double pollAcc = 0;
  auto prev = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(win)){
    glfwPollEvents();
    const auto now = std::chrono::steady_clock::now();
    pollAcc += std::chrono::duration<double>(now - prev).count(); prev = now;
    if (pollAcc > 0.4){                                    // ~2.5 Hz live refresh
      pollAcc = 0; tail.poll(); reeval.poll();
      if (!runs.empty()){
        std::ifstream sf(runs[sel] + "/status.json");
        const std::string txt((std::istreambuf_iterator<char>(sf)), std::istreambuf_iterator<char>());
        status = parseStatus(txt);
      }
    }

    ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
    int W, H; glfwGetFramebufferSize(win, &W, &H);
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2((float)W, (float)H));
    ImGui::Begin("train", nullptr,
                 ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoBringToFrontOnFocus);

    if (ImGui::Button("Rescan")){ runs = discoverRuns(); if (sel >= (int)runs.size()) sel = 0; if (!runs.empty()) selectRun(sel); recipes = discoverRecipes("recipes"); if (recipeSel >= (int)recipes.size()) recipeSel = 0; }
    ImGui::SameLine();

    if (runs.empty()){
      ImGui::TextUnformatted("No runs yet — launch one from \"New run\" below, or:  ./pong train <recipe>");
    } else {
      ImGui::SetNextItemWidth(300);
      if (ImGui::BeginCombo("run", fs::path(runs[sel]).filename().string().c_str())){
        for (int i = 0; i < (int)runs.size(); ++i)
          if (ImGui::Selectable(fs::path(runs[i]).filename().string().c_str(), i == sel)) selectRun(i);
        ImGui::EndCombo();
      }

      // ---- operator controls + trainer heartbeat (DECISIONS.md #19). The buttons write
      // <run>/control; the trainer acts on it at its next generation/update boundary. ----
      auto writeControl = [&](const char* tok){
        std::ofstream f(runs[sel] + "/control", std::ios::trunc); f << tok;
      };
      ImGui::SameLine();
      if (ImGui::SmallButton("Pause"))    writeControl("pause");
      ImGui::SameLine();
      if (ImGui::SmallButton("Resume"))   writeControl("run");
      ImGui::SameLine();
      if (ImGui::SmallButton("Stop"))     writeControl("stop");
      ImGui::SameLine();
      if (ImGui::SmallButton("Eval now")) writeControl("eval");
      ImGui::SameLine();
      ImGui::Checkbox("smooth", &smooth);
      ImGui::SameLine();
      if (status.valid){
        const double age = std::max(0.0, (double)std::time(nullptr) - status.tsUnix);
        const bool stalled = status.state == "running" && age > 15.0;
        const ImVec4 col = stalled                   ? ImVec4(0.89f, 0.29f, 0.28f, 1.0f)
                         : status.state == "running" ? ImVec4(0.10f, 0.69f, 0.48f, 1.0f)
                         : status.state == "paused"  ? ImVec4(0.93f, 0.63f, 0.00f, 1.0f)
                                                     : ImVec4(0.59f, 0.58f, 0.56f, 1.0f);
        ImGui::TextColored(col, "[%s  %.0fs ago]", stalled ? "stalled?" : status.state.c_str(), age);
      } else {
        ImGui::TextDisabled("[no status file]");
      }
    }

    if (ImGui::CollapsingHeader("New run")){
      if (recipes.empty()){
        ImGui::TextDisabled("no recipes found under recipes/");
      } else {
        if (ImGui::BeginCombo("recipe", recipes[(size_t)recipeSel].name.c_str())){
          for (int i = 0; i < (int)recipes.size(); ++i)
            if (ImGui::Selectable(recipes[(size_t)i].name.c_str(), i == recipeSel))
              loadRecipeForm(i);
          ImGui::EndCombo();
        }
        ImGui::TextDisabled("[%s] %s", recipes[(size_t)recipeSel].trainer.c_str(), recipes[(size_t)recipeSel].desc.c_str());
        ImGui::InputTextMultiline("args", nrArgs, sizeof nrArgs, ImVec2(-1, 54));
        ImGui::SetNextItemWidth(260); ImGui::InputText("out dir", nrOut, sizeof nrOut);
        ImGui::SameLine(); ImGui::TextDisabled("(runs under models/ appear in the dashboard's run list)");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);  ImGui::InputText("seed", nrSeed, sizeof nrSeed);
        if (recipes[(size_t)recipeSel].trainer == "evolve"){
          ImGui::SetNextItemWidth(260);
          if (ImGui::BeginCombo("resume from", nrResume[0] ? nrResume : "(fresh run)")){
            // Mirror the warm-start combo below: cache the candidate list, rescan only
            // when the popup opens (never per-frame). Always-open combo — like
            // warm-start's "(recipe default)" — rather than swapping in disabled text
            // when empty: that outer empty-check would itself need a per-frame
            // fs::exists scan to stay live (and, hidden behind its own emptiness, could
            // never re-open itself to refresh once a new resumable run appeared). The
            // in-popup hint below covers the empty case instead.
            static std::vector<std::string> resumable;         // only genuinely resumable GA runs
            if (ImGui::IsWindowAppearing()){
              resumable.clear();
              for (const auto& r : runs) if (fs::exists(r + "/population.txt")) resumable.push_back(r);
            }
            if (ImGui::Selectable("(fresh run)", nrResume[0] == 0)) nrResume[0] = 0;
            if (resumable.empty()) ImGui::TextDisabled("(no resumable GA runs)");
            for (const auto& r : resumable){
              if (ImGui::Selectable(r.c_str(), r == nrResume)){
                std::snprintf(nrResume, sizeof nrResume, "%s", r.c_str());
                if (std::string(nrOut) == r){                  // Decision 18: never same-dir resume
                  for (int suf = 2; suf <= 9; ++suf){
                    std::snprintf(nrOut, sizeof nrOut, "%s%d", r.c_str(), suf);
                    if (!fs::exists(std::string(nrOut) + "/train_log.jsonl")) break;
                  }
                }
              }
            }
            ImGui::EndCombo();
          }
        } else if (recipes[(size_t)recipeSel].trainer == "ppo"){
          ImGui::SetNextItemWidth(260);
          if (ImGui::BeginCombo("warm-start from", nrInit[0] ? nrInit : "(recipe default)")){
            // Mirror main.cpp's play-picker combo (Decision 21): cache the candidate list,
            // rescan only when the popup opens (never per-frame), filter box past 15 entries.
            static std::vector<std::string> warmStarts;
            static ImGuiTextFilter filter;
            if (ImGui::IsWindowAppearing()){ warmStarts = discoverWarmStarts(); filter.Clear(); }
            if ((int)warmStarts.size() > 15) filter.Draw("filter", 140.0f);
            if (ImGui::Selectable("(recipe default)", nrInit[0] == 0)) nrInit[0] = 0;
            for (const auto& m : warmStarts){
              if (!filter.PassFilter(m.c_str())) continue;
              if (ImGui::Selectable(m.c_str(), m == nrInit))
                std::snprintf(nrInit, sizeof nrInit, "%s", m.c_str());
            }
            ImGui::EndCombo();
          }
        }
        const bool sameDirResume = nrResume[0] && std::string(nrResume) == nrOut;
        if (sameDirResume)
          ImGui::TextColored(ImVec4(0.89f, 0.29f, 0.28f, 1.0f),
              "same-dir resume overwrites the first segment's checkpoints (Decision 18) — pick a fresh out dir");
        const bool outOccupied = fs::exists(std::string(nrOut) + "/train_log.jsonl");
        if (outOccupied && !sameDirResume)
          ImGui::TextColored(ImVec4(0.89f, 0.29f, 0.28f, 1.0f),
              "out dir already holds a run — launching there interleaves/overwrites it; pick a fresh dir");
        if (!sameDirResume && !outOccupied && ImGui::Button("Launch")){
          std::filesystem::create_directories("results/queue");
          char path[128], tmp[140];
          static int reqN = 0;
          std::snprintf(path, sizeof path, "results/queue/req_%lld_%d.launch",
                        (long long)std::time(nullptr), reqN++);
          std::snprintf(tmp, sizeof tmp, "%s.tmp", path);
          {
            std::ofstream f(tmp);
            f << "recipe: " << recipes[(size_t)recipeSel].name << "\n"
              << "out: " << nrOut << "\n"
              << "args: " << nrArgs << "\n";                 // FULL replacement (pool-dir rule)
            if (nrResume[0]) f << "resume: " << nrResume << "\n";
            if (nrSeed[0])   f << "override: --seed " << nrSeed << "\n";
            if (nrInit[0])   f << "override: --init-model " << nrInit << "\n";
          }                                                  // closed + flushed before visible
          std::error_code rec;
          fs::rename(tmp, path, rec);   // atomic: the executor's req_*.launch glob never sees a partial file
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(runs detached — closing this window never stops a trainer)");
      }
      // queue visibility: pending requests + failures
      std::error_code qec;
      for (const auto& e : fs::directory_iterator("results/queue", qec)){
        const std::string fn = e.path().filename().string();
        if (e.path().extension() == ".launch"){
          long long ts = std::atoll(fn.c_str() + 4);       // req_<unix_ts>.launch
          const long long age = (long long)std::time(nullptr) - ts;
          if (age > 5)
            ImGui::TextColored(ImVec4(0.93f, 0.63f, 0.0f, 1),
                "pending %llds: %s — launcher not running? start ./pong trainview or ./pong watch",
                age, fn.c_str());
          else ImGui::TextDisabled("queued: %s", fn.c_str());
        } else if (e.path().extension() == ".failed"){
          std::ifstream ff(e.path());
          std::string first; std::getline(ff, first);
          ImGui::TextColored(ImVec4(0.89f, 0.29f, 0.28f, 1), "failed: %s — %s", fn.c_str(), first.c_str());
          ImGui::SameLine();
          ImGui::PushID(fn.c_str());
          if (ImGui::SmallButton("dismiss")) fs::remove(e.path(), qec);
          ImGui::PopID();
        }
      }
    }

    if (!runs.empty()){
      if (status.valid && ImGui::CollapsingHeader("run config")){
        std::ifstream cf(runs[sel] + "/config.json");
        const std::string cfg((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        ImGui::TextWrapped("%s", cfg.empty() ? "(no config.json — pre-#20 run)" : cfg.c_str());
      }

      const bool ppo = tail.schema == "ppo";
      const char* xk = ppo ? "step" : "gen";
      if (tail.rows.empty()){
        ImGui::TextUnformatted("waiting for the first logged row...");
      } else {
        const auto& last = tail.rows.back();
        auto g = [&](const char* k){ const auto it = last.find(k); return it == last.end() ? 0.0 : it->second; };
        char etaTxt[48] = "";
        if (ppo){
          const double sps = g("sps");
          const double eta = etaSeconds(g("step"), status.valid ? status.gensTotal : -1,
                                        sps > 0 ? 1.0 / sps : 0);
          if (eta >= 0) std::snprintf(etaTxt, sizeof etaTxt, "   ETA ~%.0f min", eta / 60.0);
          ImGui::Text("PPO   step %.0f   update %.0f   %.0f steps/s   return %.2f%s   |   %zu rows, live ~2.5 Hz",
                      g("step"), g("update"), sps, g("ep_return_mean"), etaTxt, tail.rows.size());
        } else {
          double pace = 0; int n = 0;                    // mean sec/gen over the last 20 rows
          for (size_t i = tail.rows.size() > 20 ? tail.rows.size() - 20 : 0; i < tail.rows.size(); ++i){
            const auto it = tail.rows[i].find("sec");
            if (it != tail.rows[i].end()){ pace += it->second; ++n; }
          }
          const double eta = etaSeconds(g("gen"), status.valid ? status.gensTotal : -1,
                                        n ? pace / n : 0);
          if (eta >= 0) std::snprintf(etaTxt, sizeof etaTxt, "   ETA ~%.0f min", eta / 60.0);
          ImGui::Text("EVOLUTION   gen %.0f   best %.3f   mean %.3f   champ age %.0f   %.2f s/gen%s   |   %zu rows, live ~2.5 Hz",
                      g("gen"), g("best"), g("mean"), g("champ_age"), g("sec"), etaTxt, tail.rows.size());
        }
      }
      ImGui::Separator();

      std::vector<Series> classes;                       // curriculum panel (evolve runs only)
      if (!ppo){
        for (const auto& d : kClasses){
          Series s{d.name, d.col, {}, {}};
          extractSeries(tail.rows, xk, d.key, true, s.xs, s.ys);
          if (!s.xs.empty()) classes.push_back(std::move(s));
        }
      }
      const float panels = classes.empty() ? 2.0f : 3.0f;
      const float ph = (ImGui::GetContentRegionAvail().y - 6.0f * panels) / panels;

      std::vector<Series> scoring;
      if (ppo){
        Series r{"episode return (training pool)", IM_COL32(58, 135, 229, 255), {}, {}};
        extractSeries(tail.rows, xk, "ep_return_mean", false, r.xs, r.ys);
        scoring.push_back(std::move(r));
      } else {
        Series bst{"best (champion)", IM_COL32(58, 135, 229, 255), {}, {}};
        Series men{"mean (population)", IM_COL32(150, 148, 142, 255), {}, {}};
        extractSeries(tail.rows, xk, "best", false, bst.xs, bst.ys);
        extractSeries(tail.rows, xk, "mean", false, men.xs, men.ys);
        scoring.push_back(std::move(bst)); scoring.push_back(std::move(men));
      }
      drawPlot(ppo ? "Scoring — episode return vs step"
                   : "Scoring — fitness vs generation", ph, scoring, false);

      ImGui::Dummy(ImVec2(0, 6));

      std::vector<Series> anchors;
      for (const auto& d : kAnchors){
        Series s{d.name, d.col, {}, {}};
        extractSeries(tail.rows, xk, d.key, true, s.xs, s.ys);
        for (float& v : s.ys) v *= 100.0f;
        if (smooth) s.ys = emaSmooth(s.ys, 0.25f);
        if (!s.xs.empty()) anchors.push_back(std::move(s));
      }
      for (const auto& d : kAnchors){                    // recovered history, drawn faint
        Series s{d.reevalName, (d.col & 0x00FFFFFFu) | 0x50000000u, {}, {}};
        extractSeries(reeval.rows, "gen", d.key, true, s.xs, s.ys);
        for (float& v : s.ys) v *= 100.0f;
        if (smooth) s.ys = emaSmooth(s.ys, 0.25f);
        if (!s.xs.empty()) anchors.push_back(std::move(s));
      }
      drawPlot(reeval.rows.empty()
                   ? "Success % vs deterministic bots  (canonical physics, held-out never trained on)"
                   : "Success % vs deterministic bots  (faint = recovered history, ./pong reeval)",
               ph, anchors, true);

      if (!classes.empty()){
        ImGui::Dummy(ImVec2(0, 6));
        drawPlot("Fitness by opponent class  (population mean margin, training physics)",
                 ph, classes, false);
      }
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
