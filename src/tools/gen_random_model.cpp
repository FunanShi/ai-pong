// Writes a randomly-initialized aipong-mlp v2 weight file (6K-32-32-3, tanh hidden,
// K = observation frames stacked). A random policy plays badly by design — this exists
// to exercise the Model slot end-to-end before any real training pipeline lands.
//   usage: gen_random_model <out_path> [seed] [stack]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static uint64_t splitmix64(uint64_t& x){
  x += 0x9e3779b97f4a7c15ull;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
static double uniform(uint64_t& s, double lo, double hi){
  double u = double(splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
  return lo + (hi - lo) * u;
}

int main(int argc, char** argv){
  if (argc < 2){ std::fprintf(stderr, "usage: gen_random_model <out_path> [seed] [stack]\n"); return 1; }
  const uint64_t seed0 = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 42;
  uint64_t seed = seed0;                              // splitmix64 mutates its state in place
  const int stack = (argc > 3) ? std::atoi(argv[3]) : 1;
  if (stack < 1 || stack > 8){ std::fprintf(stderr, "stack must be 1..8\n"); return 1; }
  const std::vector<int> sizes = {6 * stack, 32, 32, 3};

  std::ofstream out(argv[1]);
  if (!out){ std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  out << "aipong-mlp 2\n" << "obs stack " << stack << "\n";
  for (size_t i = 0; i < sizes.size(); ++i) out << sizes[i] << (i + 1 < sizes.size() ? ' ' : '\n');
  out.precision(9);
  for (size_t l = 0; l + 1 < sizes.size(); ++l){
    int nIn = sizes[l], nOut = sizes[l + 1];
    double scale = 1.0 / double(nIn);                 // keep pre-activations tame
    for (int i = 0; i < nIn * nOut; ++i) out << uniform(seed, -scale, scale) << ' ';
    out << '\n';
    for (int i = 0; i < nOut; ++i) out << uniform(seed, -0.05, 0.05) << ' ';
    out << '\n';
  }
  std::printf("wrote %s (%d-32-32-3, obs stack %d, seed %llu)\n",
              argv[1], 6 * stack, stack, (unsigned long long)seed0);
  return 0;
}
