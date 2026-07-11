#include "rally_index.hpp"
#include "pong_core.hpp"
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace pong {
namespace {
// Targeted extraction from our own machine-written JSONL. Keys are unique per line by
// construction of MatchRecorder/writeIndex; no general JSON parsing needed or wanted.
bool findRaw(const std::string& line, const std::string& key, size_t& pos){
  pos = line.find("\"" + key + "\":");
  if (pos == std::string::npos) return false;
  pos += key.size() + 3;
  return true;
}
double numAt(const std::string& line, const std::string& key, double fallback = 0){
  size_t p; if (!findRaw(line, key, p)) return fallback;
  return std::strtod(line.c_str() + p, nullptr);
}
std::string strAt(const std::string& line, const std::string& key){
  size_t p; if (!findRaw(line, key, p)) return "";
  if (line[p] != '"') return "";
  size_t q = line.find('"', p + 1);
  return line.substr(p + 1, q - p - 1);
}
// "b":[x,y,vx,vy]
bool ballAt(const std::string& line, double out[4]){
  size_t p; if (!findRaw(line, "b", p) || line[p] != '[') return false;
  const char* c = line.c_str() + p + 1;
  for (int i = 0; i < 4; ++i){
    char* nxt = nullptr;
    out[i] = std::strtod(c, &nxt);
    if (nxt == c) return false;
    c = nxt;
    while (*c == ',' || *c == ' ') ++c;
  }
  return true;
}
} // namespace

std::vector<RallyRow> indexMatchStream(std::istream& in, const std::string& matchName){
  std::vector<RallyRow> rows;
  std::string line;
  int lineNo = 0;

  // metadata line
  if (!std::getline(in, line)) return rows;
  lineNo++;
  if (line.find("\"aipong_match\":1") == std::string::npos) return rows;
  const std::string left  = strAt(line, "left");
  const std::string right = strAt(line, "right");
  const unsigned seed = (unsigned)numAt(line, "seed");

  RallyRow cur;
  bool open = false;
  int idx = 0;
  auto beginRally = [&](int startLine, const std::string& tickLine){
    cur = RallyRow{};
    cur.match = matchName; cur.rallyIdx = idx;
    cur.left = left; cur.right = right; cur.seed = seed;
    cur.lineStart = startLine;
    double b[4];
    if (ballAt(tickLine, b)) cur.servedTo = (b[2] < 0) ? "left" : "right";
    open = true;
  };
  auto closeRally = [&](int endLine, const std::string& winner, const std::string& end){
    cur.lineEnd = endLine;
    cur.winner = winner; cur.end = end;
    cur.dur_s = cur.ticks * k::Dt;
    if (end == "rally_cap") cur.hits = k::RallyMax;   // pre-step counter undershoots the capping hit
    rows.push_back(cur);
    open = false; idx++;
  };

  while (std::getline(in, line)){
    lineNo++;
    if (line.find("\"end\":true") != std::string::npos) break;      // summary line
    if (line.find("\"t\":") == std::string::npos) continue;         // not a tick line
    if (!open) beginRally(lineNo, line);
    cur.ticks++;
    cur.hits = std::max(cur.hits, (int)numAt(line, "rally"));
    double b[4];
    if (ballAt(line, b)) cur.vmax = std::max(cur.vmax, std::sqrt(b[2]*b[2] + b[3]*b[3]));
    const std::string ev = strAt(line, "ev");
    if (ev == "pointL")        closeRally(lineNo, "left",  "right_missed"); // ball out past the right edge
    else if (ev == "pointR")   closeRally(lineNo, "right", "left_missed");
    else if (ev == "rallycap") closeRally(lineNo, "none",  "rally_cap");
  }
  if (open && cur.ticks > 0) closeRally(cur.lineStart + cur.ticks - 1, "none", "truncated");
  return rows;
}

std::string toJson(const RallyRow& r){
  std::ostringstream o; o.setf(std::ios::fixed); o.precision(5);
  o << "{\"match\":\"" << r.match << "\",\"rally\":" << r.rallyIdx
    << ",\"left\":\"" << r.left << "\",\"right\":\"" << r.right << "\""
    << ",\"seed\":" << r.seed
    << ",\"winner\":\"" << r.winner << "\",\"end\":\"" << r.end << "\""
    << ",\"lines\":[" << r.lineStart << ',' << r.lineEnd << ']'
    << ",\"ticks\":" << r.ticks << ",\"dur_s\":" << r.dur_s
    << ",\"hits\":" << r.hits << ",\"vmax\":" << r.vmax
    << ",\"served_to\":\"" << r.servedTo << "\"}";
  return o.str();
}

void writeIndex(std::ostream& out, const std::vector<RallyRow>& rows){
  out << "{\"aipong_rally_index\":1,\"rows\":" << rows.size() << "}\n";
  for (const auto& r : rows) out << toJson(r) << "\n";
}

std::vector<RallyRow> readIndex(std::istream& in){
  std::vector<RallyRow> rows;
  std::string line;
  if (!std::getline(in, line)) return rows;
  if (line.find("\"aipong_rally_index\":1") == std::string::npos) return rows;
  while (std::getline(in, line)){
    if (line.find("\"match\":") == std::string::npos) continue;
    RallyRow r;
    r.match = strAt(line, "match");
    r.rallyIdx = (int)numAt(line, "rally");
    r.left = strAt(line, "left"); r.right = strAt(line, "right");
    r.seed = (unsigned)numAt(line, "seed");
    r.winner = strAt(line, "winner"); r.end = strAt(line, "end");
    size_t p;
    if (findRaw(line, "lines", p) && line[p] == '['){
      char* nxt = nullptr;
      r.lineStart = (int)std::strtol(line.c_str() + p + 1, &nxt, 10);
      while (*nxt == ',' || *nxt == ' ') ++nxt;
      r.lineEnd = (int)std::strtol(nxt, nullptr, 10);
    }
    r.ticks = (int)numAt(line, "ticks");
    r.dur_s = numAt(line, "dur_s");
    r.hits = (int)numAt(line, "hits");
    r.vmax = numAt(line, "vmax");
    r.servedTo = strAt(line, "served_to");
    rows.push_back(r);
  }
  return rows;
}

} // namespace pong
