#pragma once
// Named training recipes (DECISIONS.md #20): recipes/<name>.recipe, a line format —
//   desc: <one line>          shown in menus
//   trainer: evolve | ppo     which trainer binary runs it
//   out: <dir>                default --out (overridable)
//   args: <flags...>          repeatable; concatenated in order
// Shared source of truth for the GUI (this parser), ./pong (grep/sed on the same
// format), and docs. Unknown keys are ignored (forward compatibility).
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace pong {

struct Recipe {
  std::string name, desc, trainer, out;
  std::vector<std::string> args;
  bool valid = false;
};

inline Recipe parseRecipeText(const std::string& name, const std::string& text){
  Recipe r; r.name = name;
  std::istringstream in(text);
  std::string line;
  auto valOf = [&](const char* key) -> const char* {   // "key: value" → value or nullptr
    const size_t klen = std::char_traits<char>::length(key);
    if (line.rfind(key, 0) != 0) return nullptr;
    size_t v = klen;
    while (v < line.size() && (line[v] == ' ' || line[v] == '\t')) ++v;
    return line.c_str() + v;
  };
  auto rtrim = [](std::string s){
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
  };
  while (std::getline(in, line)){
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    if (const char* v = valOf("desc:"))         r.desc = rtrim(v);
    else if (const char* v2 = valOf("trainer:")) r.trainer = rtrim(v2);
    else if (const char* v3 = valOf("out:"))     r.out = rtrim(v3);
    else if (const char* v4 = valOf("args:")){
      std::istringstream toks(v4);
      std::string t;
      while (toks >> t) r.args.push_back(t);
    }
    // unknown keys: ignored
  }
  r.valid = (r.trainer == "evolve" || r.trainer == "ppo") && !r.args.empty() && !r.out.empty();
  return r;
}

inline std::vector<Recipe> discoverRecipes(const std::string& dir){
  std::vector<Recipe> out;
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(dir, ec)){
    if (!e.is_regular_file() || e.path().extension() != ".recipe") continue;
    std::ifstream f(e.path());
    std::stringstream ss; ss << f.rdbuf();
    Recipe r = parseRecipeText(e.path().stem().string(), ss.str());
    if (r.valid) out.push_back(std::move(r));
  }
  std::sort(out.begin(), out.end(),
            [](const Recipe& a, const Recipe& b){ return a.name < b.name; });
  return out;
}

} // namespace pong
