# ai_pong — project instructions

Self-play/evolutionary pong arena. Read `DECISIONS.md` before
changing core behavior; the sim's contracts (observation vector, `aipong-mlp` v1/v2 formats
incl. the oldest→newest frame-stack convention, outcome semantics — specs in
`docs/formats.md`) are training interfaces — breaking them invalidates future checkpoints.

Rules:
1. `src/core/` stays a pure deterministic sim: no agents, no IO, no RNG objects (seeded jitter
   is hashed). `src/agents/` stays deterministic too. Anything stochastic belongs to the
   training harness (`src/train/`), parameterized by explicit seeds.
2. Physical quantities carry units in comments (cu, cu/s, s — court frame C, y down), matching
   the existing header style.
3. Every core/agent behavior change lands with a doctest case; run `./pong build && ./pong test`
   (everything builds inside docker; no host toolchain).
4. Decisions → `DECISIONS.md` in the established format (chronological — append at the end).
5. `build/`, `imgui.ini`, and the data dirs `models/`, `datasets/`, `results/` are gitignored
   (each kept by a `.gitkeep`); don't commit binaries or weights.
6. `models/<run>/` paths are an append-only namespace (Decision 17): they're baked into
   `src/train/ppo.py`'s default pool, trainview's scan, and ladder provenance — add new runs,
   never rename or move old ones.
