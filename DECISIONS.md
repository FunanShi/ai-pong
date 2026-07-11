# Project Decisions — ai_pong

**Date started:** 2026-07-06 · **Workspace:** `~/robotics_ws/ai_pong` · **Derived from:** an earlier single-player pong prototype
Companion: `README.md`.

Entries are chronological — **append new decisions at the end**, never mid-file.

## Index

1. Copy ui_pong, keep only the imgui frontend (native-feel requirement; imgui choice questioned & re-confirmed)
2. Symmetric discrete actuation; opponents live outside the core
3. Ball speed ceiling 100 cu/s (not literally uncapped) + substepped collision
4. Rally cap 1000 → game over, no winner, a loss for BOTH sides (interpretation — review welcome)
5. Deterministic seeded serves (`initialState(seed)`)
6. Training interface contracts: observation vector + `aipong-mlp v1`
7. Match recording (`aipong_match` v1 JSONL) + pause as operator-only controls
8. The rally is the training unit; the database is an index, not a copy
9. Pre-league prep: extract `match/`, split the test monolith, drop dead core code
10. Training plan: evolution first (C++ GA), PPO second (Python, staged); 6-32-32-3
11. Perf work + training continuation (measure-first vindicated twice)
12. The Elo ladder (league/): deployment champions by rating, not final fitness
13. evo4 readiness: float32 + batched SoA fitness path + economics knobs + strided memory
14. evo5: opponent diversity in fitness (the evo4 lesson)
15. Live training dashboard (`./pong trainview`)
16. The memory experiment's null result: evo4–6 "champions" are the widened memoryless evo3.5 policy
17. Repo reorganization: src/ consolidation, results/ split, one docs/ tree (goal: navigable root)
18. Eval-stride bug: in-training anchor evals misloaded stack>1 genomes (fix + `reeval` recovery)
19. Trainview becomes the control panel: pause/stop/eval channel + curriculum telemetry
20. Launcher: recipes as data, boot menu, GUI New-run via host-executed launch queue (full GUI launcher on a recipe layer; native binary rejected)
21. Play picker: curated models with champions-canonical Elo in the game GUI
22. Interview polish: reviewed-backlog burn-down, CI, fail-loud resume, sweep corrections
23. Scalar forward flattened: act() vectorizes at last (measured 2.1–2.3×)
24. Training profile: the hot path is the forward's MACs; profiler + thread choice
25. Granular MAC timing + the lane-width A/B that an isolated bench got backwards
26. Loss-side fitness shaping: bounded survival + concession-proximity (win>loss always; A/B pending)
27. evomem_10k: 10,000 gens reproduce the ~3,000-gen plateau — gens 3000 vindicated
28. Island (deme) co-evolution: N architectures cross-play under per-island selection (`coEvolve()` / `./pong coevolve`)
29. Co-evolution dashboard (`pong_coview`/`./pong coview`): per-species avg-margin-vs-bots (self-play fitness as labeled foil) + `coEvolve` pause/stop/eval control channel

## Decision 1 — Copy ui_pong, keep only the imgui frontend (native-feel requirement; imgui choice questioned & re-confirmed)

Immediate-mode fits this repo's purpose — an *experiment control panel* that will keep growing
(per-side opponent combos today; Elo dashboards, rally histograms, generation pickers later).
Qt gives nicer native chrome but costs signal/slot ceremony per widget and heavier plotting.
qt/web/webview renderers and their deps (Qt, httplib, webview) dropped from the build; the
docker GL shim and `imgui-gpu` fallback are inherited unchanged from ui_pong.
- **Requirement folded in:** controller selection for **both** paddles — any of
  Human / P-controller ×3 / Interceptor ×2 / Model per side. Bot-vs-bot spectating and
  two-human play fall out for free; a sim-speed multiplier (1/2/8/32×) added for spectating.

## Decision 2 — Symmetric discrete actuation; opponents live outside the core

`Input{left, right}`; both paddles move at the same `PaddleSpeed` (1.2 cu/s) with the same
3-action set {None, Up, Down}. The old in-core proportional "computer" became `PAgent` in
`agents/`, choosing discrete moves through the same interface a human or trained policy uses —
fair self-play by construction, and the core stays a pure sim.
- **Behavior change vs ui_pong (flagged):** the old AI moved proportionally, capped at
  0.9 cu/s; agents now move bang-bang at 1.2 cu/s gated by a deterministic **duty cycle**
  (duty 0.75 ≈ the old average speed). Difficulty presets: easy 0.55 / classic 0.75 / hard 1.0.
- **New scripted tier:** `InterceptorAgent` predicts the intercept height by unrolling the
  flight with wall reflections (triangle-wave fold; `predictInterceptY` unit-tested) — the
  "more advanced, swappable" opponent; a laggy variant re-plans every 12 ticks.

## Decision 3 — Ball speed ceiling 100 cu/s (not literally uncapped) + substepped collision

The requirement was uncapped-or-super-high, so every match eventually ends. Literal uncapping is a
hang: with ×1.03 per hit, a 1000-hit rally reaches 0.75·1.03¹⁰⁰⁰ ≈ 5×10¹² cu/s and physics
work per tick grows with speed. At **100 cu/s** the ball crosses the court in ~10 ms against
paddles doing 1.2 cu/s — unreturnable by humans (~3 cu/s territory) and by any anticipatory
bot well before the cap; reached after ~165 consecutive hits.
- **Cost accepted / physics fix:** one Euler step at 100 cu/s travels 1.67 cu — through
  everything. `stepOnce` now substeps the ball at ≤ `MaxSubDisp` 0.008 cu (< BallR 0.012 <
  PaddleW 0.02) per substep, colliding each substep (≤ ~209 substeps/tick at the cap; a
  mid-tick bounce adds ×1.03, keeping worst-case travel under BallR). Determinism preserved;
  a 99 cu/s ball provably bounces off both paddles within a single tick in the test suite.

## Decision 4 — Rally cap 1000 → game over, no winner, a loss for BOTH sides (interpretation — review welcome)

"Counted as a loss": implemented as `Outcome::RallyCap` — the game ends immediately, no point
awarded, and league scoring will treat it as a loss for both players (anti-stalling pressure;
no Elo transfer between the pair). Alternatives considered: void-point-and-reserve (rewards
stalling), award-to-fewer-hits (arbitrary). The cap matters in practice for perfect-bot
matches (e.g., Interceptor vs Interceptor can sustain arbitrarily long rallies at the speed
ceiling); score-driven games rarely reach it.

## Decision 5 — Deterministic seeded serves (`initialState(seed)`)

Serve angle jitters in [10°, 35°] via a splitmix64 hash of (seed, pointIndex) — no RNG state,
`stepOnce` stays pure, a seed fully determines a match. Seed 0 reproduces the legacy fixed
20° serve, keeping ui_pong's determinism tests meaningful. Without this, two deterministic
policies replay one identical rally forever — useless for training and boring to spectate.

## Decision 6 — Training interface contracts: observation vector + `aipong-mlp v1`

`observation(snapshot, side)` → 6 doubles `{ball x, ball y, vx, vy, own y, opp y}` (cu, cu/s,
court frame C, y down), x-mirrored for the right side so **every policy plays left** —
symmetry halves the learning problem. Model weights: text format `aipong-mlp v1`
(`models/README.md`), input width 6, output 3 logits {None, Up, Down}, tanh hidden — chosen
so both future tracks (C++ GA population, Python PPO export) can emit it trivially.
`tools/gen_random_model` exercises the slot before training exists.

## Decision 7 — Match recording (`aipong_match` v1 JSONL) + pause as operator-only controls

Recording lives in `record/MatchRecorder` (new module — `core/` stays IO-free per project
rules) and writes one JSONL file per match to `datasets/`: metadata line (controllers, seed,
embedded sim constants), one line per tick with the **pre-step state + actions applied that
tick** — the (s_t, a_t) alignment behavior cloning (BC) consumes — plus `ev` point/rally-cap markers derived
from the post-step state (the reward hooks for future RL), and a terminal summary with a
`truncated` flag (operator stop / controller change / window close vs. natural game end).
Raw state is stored, not per-side observations — those derive via `observation()`'s mirror,
so one file feeds both sides' training views. Pressing **Record** restarts the game so every
recording is a complete episode from the serve.

**Pause** (button or P key) is a frontend affordance with an accumulator drain (resume never
fast-forwards). Both controls are structurally human-only: the agent interface is a paddle
`Move` and nothing else — no policy can pause, stall the clock, or touch recording. The same
`MatchRecorder` is the intended logging path for the future headless league runner.

## Decision 8 — The rally is the training unit; the database is an index, not a copy

**Decision:** chunk data per **rally** with winner/loser, players, duration, and end state.
Adopted with two refinements: (a) `winner` gains a third value **none** — rally-cap rallies
have no winner by Decision 4, and operator-truncated rallies are neither won nor lost;
end states are `left_missed` / `right_missed` / `rally_cap` / `truncated`. (b) The
"database" is a derived **index over the raw match files** (`rallydb build` →
`datasets/rallies.jsonl`), one row per rally carrying players, winner, end, ticks, dur_s,
hits, vmax, served_to, and the exact **line span** in the source file — training extracts
tick data via the index instead of duplicating it (LeRobot-style metadata-over-raw-files;
SQLite is the upgrade path if this ever reaches millions of rows, not before).
`rallydb list` sorts/filters (--sort ticks|hits|vmax|dur, --winner, --end, --player, --top);
`rallydb extract` emits one rally's tick lines for pipelines. The index parser is
deliberately format-coupled to the MatchRecorder v1 writer and pinned by tests (incl. a
full bot-vs-bot recorded match whose rally rows must tally exactly with the final score).

## Decision 9 — Pre-league prep: extract `match/`, split the test monolith, drop dead core code

Done before the league runner exists, so the tournament is built *on* the architecture
rather than around it:
- **`match/` module** now owns what had accreted in the imgui frontend: the controller
  **spec registry** (`"human" | "p:easy|classic|hard" | "interceptor[:laggy]" | "mlp:<path>"`
  → `makeAgent`/`agentLabel`/`builtinSpecs`) and **`MatchRunner`** (header-only), the one
  true step loop: two controllers → Input → tick → optional recording, with auto-finalize
  at game over and truncation on newGame/destruction. The GUI drives `tick()` at 60 Hz
  passing keyboard Moves for human sides; the league will drive `runToCompletion()`
  flat-out. Specs are the league's serializable controller identity (Elo keys, metadata);
  labels are recorded in match files, so **changing a label is a data-compat decision**.
- **Tests split** by module: `test_core / test_agents / test_record / test_match`
  (four executables, shared fixture in `tests/test_util.hpp`) — replaces the misnamed
  454-line `core_test.cpp` monolith. New match tests pin registry semantics, headless
  determinism (same seed+specs ⇒ identical match), and recording-through-runner.
- **Dead code removed:** `toJson(Snapshot)` (ui_pong web-renderer legacy; zero callers) —
  core is again strictly sim + observation.
- *Named watch-items, deliberately NOT done:* `k::WinScore`/`k::RallyMax` stay compile-time
  until the league needs short-game fitness evals (then: runtime MatchConfig); hand-rolled
  JSON budget is spent — the next format buys a real JSON library.

## Decision 10 — Training plan: evolution first (C++ GA), PPO second (Python, staged); 6-32-32-3

**Decision:** train both, one then the other. Order rationale: the GA is all-C++ (zero new
deps, trains in ~minutes on 20 threads) and exercises the full export→load→ladder pipeline;
PPO needs a pybind11 boundary + torch image — its own infra pass.

**GA recipe (train/evolve.cpp), hyperparameters sourced from slimevolleygym's TRAINING.md**
(the closest published analog: state-vector self-play, tiny nets, GA beat its baseline in a
few hours on ONE 2015 CPU): truncation selection (elite 25%), Gaussian mutation ε=0.1 on
every weight, mutated-elite-clone refill. Network 6-32-32-3 = 1,379 params (~10× the
120-param net that plays competent slime volleyball; inside CMA-ES range if upgraded).
- **Fitness = mean score margin ∈ [-1,1] over pure self-play** (peers + one hall-of-fame
  ancestor per eval); **RallyCap counts as −1** (Decision 4 as training pressure).
  **Scripted anchors are reporting-only, never selection** — slimevolley's key finding:
  training against a fixed opponent overfits it; self-play wins the meta-comparison.
- Hall of fame: best-of-generation every 10 gens, ring of 20 — the anti-cycling mechanism.
- **Deterministic by construction:** every match seed and mutation stream is keyed
  (masterSeed, tag, generation, slot) via splitmix64 — results are identical for any
  `--threads`; pinned by test.
- Full matches to 11 as the fitness unit (measured cheap enough; the WinScore→MatchConfig
  watch-item stays parked until proven otherwise).
- Checkpoints `models/evo/gen_*.txt` + `best.txt` (aipong-mlp v2) + `train_log.jsonl`.

**PPO track (built 2026-07-06, second per sequencing):** pybind11 `VecPong`
(bindings/aipong_module.cpp — policy plays LEFT through the "human" slot, opponents are
controller specs, rewards ±1/point with −1 RallyCap, auto-reset with bootstrap-0-on-done,
deterministic per (seed, env)), CleanRL-style single-file PPO (train/ppo.py) with the actor
pinned to the deployment shape 6-32-32-3 (critic 6-64-64-1 amputated at export), opponent
pool = scripted bots + any `models/evo*/best.txt` resampled per episode, greedy-argmax
anchor evals (deployment semantics), exports via the same text format. Docker image grew
python3 + CPU torch (a 5k-param net needs no GPU).

## Decision 11 — Perf work + training continuation (measure-first vindicated twice)

`tools/bench` (single-threaded sim-throughput A/B) killed two hypotheses before code shipped:
substep cost was real but minor (scripted workloads were already at 15–27M ticks/s), and the
suspected malloc contention was noise (scratch buffers moved nothing). The true hot path:
**the MLP forward itself** — ~2.6k MACs in a scalar loop (gcc can't vectorize FP reductions
without reassociation) + ~128 libm tanh calls ≈ 2.5 µs/tick, exactly the measured 403k
ticks/s. Landed, with measured results (baseline → final, same contended machine):
- **Conservative advancement** in `stepOnce` (fine ≤MaxSubDisp steps only inside 2·MaxSubDisp
  bands around walls/paddle slabs/score planes; leaps elsewhere). Same contact resolution;
  trajectories differ only by FP summation order (cross-version bit-compat broken, within-
  version determinism exact — pinned by tests). Scripted workloads +40–44%.
- **`-ffast-math` scoped to agents.cpp only** (physics keeps strict FP): unlocks vectorized
  reduction + libmvec SIMD tanh. **MLP hot path 403k → 701k ticks/s (~1.75×)** — the one that
  sets generation time. Scratch buffers kept (harmless, allocation-free is still right).
- **`-march=native`** behind `AIPONG_MARCH_NATIVE` (ON; binaries pinned to this CPU).
- **Runtime rally cap**: `State.rallyMax` (+ Snapshot), `PongCore/MatchRunner::setRallyCap`,
  `evolve --rally-cap` — training may truncate stalemate marathons; **anchor evals and ladder
  play always use the full k::RallyMax** for comparability.
- **Continuation (answers "can the next run pick up where this one leaves off?" — yes, two
  ways):** `--resume DIR` reloads `population.txt` + `hof.txt` (written at every checkpoint
  by new binaries — also crash insurance); `--init-from DIR` seeds founders from every
  aipong-mlp checkpoint in a directory (works on evo2's output today: ~120 era champions as
  founders, rest of the population = mutated clones, HoF pre-seeded from a founder spread).
  Determinism preserved; resume-continuation pinned by test.

- **Update (2026-07-06) — training speedup knob:** `--speedup` (e.g. 1.06
  vs canonical 1.03) compresses rally time ≈ g/(g−1)-fold — analytically ~1.95× fewer ticks
  in the rally-dominated regime. This is an ENVIRONMENT change, not an optimization: the
  mid-speed learning band compresses and interceptors weaken relatively (their speed
  threshold arrives sooner), so raw numbers inflate. Guardrails mirror the rally cap:
  training dynamics only; **anchor evals always run canonical physics** (MatchParams{}), so
  cross-run skill curves stay comparable; honest metric = anchor skill per wall-clock hour.
  Train/deploy distribution shift is mild (velocity is observed; same speed range) but
  real — the evo3 A/B (same founders, 1.03 vs 1.06) measures it. Knobs consolidated into
  `MatchParams` {speedBonus, speedRefTicks, rallyCap, speedup}.
- **Update (2026-07-06) — win-speed shaping (guardrailed):** raw
  "multiplier for faster wins" is Goodhart-prone (scaling a negative margin punishes
  defensive resilience; a large bonus can reorder wins below losses). Implemented instead
  as `fitness = margin + β·max(0, 1 − ticks/T_ref)` **on wins only**: any win (≥ +1/11)
  still beats any loss (≤ −1/11) for every β; losses are never shaped; anchor evals stay
  unshaped (they count wins). Opt-in (`--speed-bonus`, default 0) so runs stay comparable;
  suggested β=0.2, T_ref=30k ticks. First use: escalation step if evo2 ends at 0% vs laggy.
- **Update (2026-07-06):** third reporting anchor added (`interceptor`, the instant one) —
  the "consistently beat interceptor" goal needs both rungs visible. Long GA run launched:
  pop 128, games 8, 3000 gens, eval-every 25 ×24 matches, seed 2, 18 threads (2 reserved),
  out `models/evo2` (separate dir so gen numbering can't collide with the first run).

- **Update (2026-07-06) — memory seam added (format v2).** Paddle velocity isn't a state
  variable (massless, velocity-commanded paddles), so opponent motion is only inferable
  from history. Rather than widen the raw observation, memory lives agent-side:
  `aipong-mlp 2` declares `obs stack K` (input 6·K, frames concatenated oldest→newest,
  fresh-episode padding by repeating the earliest frame); the core stays single-frame.
  v1 files still load (≡ stack 1). The same seam — per-episode agent state cleared by
  `Agent::reset()`, already wired into the frontend's new-game path — is where a future
  recurrent v3 plugs in. Kept out of scope now: actually training with K>1.
- **Update (2026-07-10) — the "libmvec SIMD tanh" claim above was half right.** A fresh
  disassembly pass (`nm -u` on `agents.cpp.o`) found `act()` linking only scalar `tanhf`,
  never a vector tanh symbol — `-ffast-math` unlocked vectorization for
  `batch_forward.cpp`'s lane loop (Decision 13) but never for `act()`'s per-output serial
  reduce: a transcendental call sitting inside a reduction loop is exactly the shape gcc
  won't auto-vectorize, `-ffast-math` or not. So the "701k ticks/s (~1.75×)" MLP-hot-path
  number above is real, but it came entirely from `-ffast-math` reassociating the dot
  product, not from vectorized tanh — that gap sat unnoticed until this pass. Corrected by
  Decision 23 (act() flattened to a transposed-weight/flat-MAC shape; scalar path
  vectorizes tanh too, a further 2.1–2.3× on top of this one).

## Decision 12 — The Elo ladder (league/): deployment champions by rating, not final fitness

Motivated by four observed champion-slot flickers across evo2/evo3: self-play fitness picks
who *trains*, the ladder picks who *ships*. `./pong ladder [--dir models/evoN]...` plays a
round-robin over checkpoints + scripted anchors on **canonical physics** (no training knobs)
and rates with **Bradley–Terry via the MM algorithm** — order-free and deterministic, unlike
sequential Elo updates; +0.5/+0.5 smoothing per played pair keeps undefeated specs finite;
**RallyCap matches are excluded from rating** (Decision 4: no winner — zero-sum Elo can't
score a mutual loss) but counted per spec as a stalemate column. Scale pinned: p:classic =
1000. Outputs `ratings.jsonl` + `matches.jsonl` (provenance; enables future incremental
rating). Maiden anchors-only table (per-pair 2, seed 1): Interceptor 1369 (undefeated),
P-hard 1235, laggy 1117, P-classic 1000, P-easy 866 — a ~500-Elo calibrated yardstick.
Full checkpoint sweep (evo2 64 + evo3's set) runs at evo3 completion; its top mlp: entry
becomes the GUI default and evo4's founder anchor.

## Decision 13 — evo4 readiness: float32 + batched SoA fitness path + economics knobs + strided memory

Design picks: float32, SoA batching, economics knobs, plus the strided-stacking plan
(executed inline), with the compatibility matrix specified as hard requirements:
- **float32 inference** in MlpAgent + BatchedMlp (format stays double text; converted at
  load). Decision drift vs the double history ≈ 1e-7 logits — accepted cross-version class.
- **Batched SoA fitness path** (`playMatchesBatched`, train/batch_forward.cpp, fast-math
  scoped): 16 lanes, weights stored lane-contiguous [nOut][nIn][B], the lane loop is the
  SIMD dimension. Bench (under evo3 contention): scalar 512k → batched 1.16M ticks/s 1-thread
  (~2.25×), 1.98M → 4.15M at 4 threads. **Lane lifecycle:** every lane owns a complete
  independent State (own serve timer, scores, seed schedule) — lockstep applies to WHEN
  forwards are computed, never to game time; finished lanes refill immediately from the job
  queue with a fresh initialState(seed); tail drain wastes ≤B matches of occupancy (~1.5%).
  Results are bitwise independent of thread count AND lane width (pinned by test).
- **Economics knobs** (runtime, training-only; anchor evals stay canonical): `--serve-hold 0`
  (skip the dead 0.75 s wait), `--win-score 5` (short games; margin renormalizes by the
  active win score).
- **Strided memory (evo4)**: `obs stack K stride S` grammar (caps 32/60/512-tick window),
  MlpAgent + batched lane rings sample identically (slot f = t − S·(K−1−f), earliest-frame
  padding). `widenGenome` = Net2Net function-preserving stack widening; `--init-from` a
  stack-1 dir auto-widens founders. **Compatibility guarantees (tested):** widened genome ≡
  parent bit-identically (200 randomized states); widened stack-6 batched matches reproduce
  the stack-1 originals tick-for-tick; mixed architectures (evo3 vs evo4) play each other in
  GUI/ladder/matches since history is agent-private; evo3.5 (stack-1 control, no opponent
  memory) = plain `--init-from models/evo3` — the anticipation A/B's control arm.

## Decision 14 — evo5: opponent diversity in fitness (the evo4 lesson)

evo4 (stack-6 memory) exposed the failure mode memory enables: with 100% of fitness from
the evolving peer meta, intent-reading collapses into META-reading — champions overfit
peer motion signatures and by gen ~1149-2999 lost to ALL canonical anchors (even P-classic)
while shaped fitness held. Fix: fitness games become a
mixed distribution — `--pool-dir/--pool-games` (frozen cross-lineage checkpoints, any
architecture, scalar path) + `--anchor-games` (rotating {p:hard, laggy, instant}, training
physics) carved out of gamesPerEval; ~25% out-of-meta at 6+1+1. Motion features now pay
only if they generalize across peer, archival, and scripted movers. Epistemics: trained-on
anchors become in-distribution, so two HELD-OUT variants exist solely in evals
(`p:heldout` deadband 0.05/duty 0.85; `interceptor:heldout` lag 36 ticks) — never in any
fitness, reported as vs_held_* in logs and the analysis block. Guards: peers keep ≥1 game
(CLI rejects otherwise); picks seeded from the TagOpp stream (thread-invariant, pinned by
test alongside held-out reporting).

## Decision 15 — Live training dashboard (`./pong trainview`)

Native ImGui viewer (the "experiment control panel" Decision 1 anticipated) that tails
`models/<run>/train_log.jsonl` and plots the two things a run is judged by: the scoring
metric over time (evolve best+mean fitness; ppo episode return) and success % vs the
deterministic bots (all anchors incl. evo5's held-out variants). Schema (evolve vs ppo) is
auto-detected from the row keys; the run is pickable live from a combo over
`models/*/train_log.jsonl`. Read-only, so it can watch a container mid-training; refreshes
~2.5 Hz, consuming only newline-complete rows. Parsing/tailing/series-extraction live in the
header-only, doctest-covered `train/train_log_view.hpp` (`test_trainview`); the .cpp is
window + `ImDrawList` rendering only, plus a headless `--dump` mode that prints each series'
range/last (verification without a display). ImGui is now a shared `imgui_lib` static lib
across `pong_imgui` + `pong_trainview`. Launch: `./pong trainview [models/evoN]`.
`./pong evolve` and `./pong ppo` now **auto-spawn** the dashboard (detached, `--rm`,
pointed at the run's `--out` dir) via a display-guarded `launch_trainview` helper — skipped
cleanly with no display or `AIPONG_NO_GUI=1`; the passed run is tailed even before its log
exists (train_view appends an unmatched `--out` dir to the run list).

## Decision 16 — The memory experiment's null result: evo4–6 "champions" are the widened memoryless evo3.5 policy

Forensic finding (aiming study + md5 audit, `tools/aiming.cpp`): `evo4/gen_0574` ≡ `evo5/gen_1249`
≡ `evo6/gen_0074` are **byte-identical**, and all three are the function-preservingly-widened
`evo3.5/gen_2974` — first-layer history-input columns are **exactly zero** (|history weight| = 0),
the newest-frame slot equals the stack-1 evo3.5 champion bit-for-bit. So the three "memory
champions" are one *memoryless* policy: elitism carried the evo3.5 founder through all three
memory runs unchanged and the ladder kept re-selecting it. **Mechanism:** widening drops the
strong founder into a *fitness valley* — the first mutations to engage the zero history weights
only add noise to a near-optimal policy, lowering self-play fitness, so they're selected against;
no smooth mutation path exists from the memoryless optimum to a better memory-using genome.
**Consequences:** (1) the earlier leaderboard's "top-4 cluster" was one policy entered 4× (its
Elo spread was Bradley–Terry noise) — corrected to distinct policies: evo3.5 **2014** > instant
Interceptor 1843 > evo3 1578 > evo2 1413 > best-PPO 1278. (2) The champion-slot "flicker/drift"
observed all through evo4–6 was self-play-fitness NOISE in which genome ranks #1 (the frozen
strong founder vs transient peer-exploiting mutants), **not** loss of skill — the ladder always
recovered the founder. (3) Memory added a large, initially-useless input dimension, so more
mutants are anchor-weak peer-exploiters ⇒ harder flicker (worst in evo4, first memory run).
**Fix for a genuine memory run:** don't widen (random-init the history weights so there's no
memoryless fallback optimum), or shape fitness toward things only memory enables (deceptive /
adaptive opponents where anticipation pays), or use gradients (PPO engages new inputs smoothly
where mutation can't). **Aiming study (evo3.5 vs instant Interceptor, 300 matches):** 95% win;
aim is opponent-aware (`r(offset, opp_y)` = 0.18 vs evo2's −0.01) but modest; kills are
ATTRITIONAL, not quick-corner — mean 56-hit rallies, ball ≈ 4 cu/s, 0% one-shot, ~4% of returns
unreturnable-by-reachability. The offense is real and hard (beating a perfect defender is
impossible without it), but the mechanism is patient rally control + ball-speed accumulation
(canonical 1.03×/hit) until geometry cracks — not a pre-planned corner shot.

## Decision 17 — Repo reorganization: src/ consolidation, results/ split, one docs/ tree (goal: navigable root)

Requirement (2026-07-09): the root had ~12 loosely-sorted directories, three
identically-named READMEs, and `models/` silently mixing three artifact kinds; the goal was
user-facing dirs one level down and all source in one organized folder. Landed on branch
`reorg/readable-layout`, each step `./pong build && ./pong test` green:

- **This file:** entries had been inserted at a fixed anchor after #10 (file order was
  1–10, 16, 15, …, 11); now chronological with an index. New entries append at the end.
- **`src/`:** `git mv` of core/ agents/ record/ match/ train/ league/ renderers/ bindings/
  tools/ tests/ → `src/` — history preserved, ZERO source edits: cross-module includes are
  bare filenames resolved via CMake include dirs, which moved with their libraries. Only
  CMakeLists paths, the `pong` script, and the two `sys.path` lines that compute `build/`
  relative to `__file__` (ppo.py, test_bindings.py — now `../../build`) changed.
- **`models/` = weights only:** the ladder's default `outDir` was `models/ladder`
  (ladder.hpp) — the code-level root of 16 `ladder_*` output dirs (~40 MB of ratings/matches
  JSONL, zero weights) accumulating in `models/`. Default is now `results/ladder`; old sweeps
  moved to `results/ladder/<name>` (`ladder_probe_out` → `probe`). Curated input sets
  grouped: `*_founders` → `models/founders/<run>`, `*_pool` → `models/pools/<run>`,
  `_leaderboard` → `models/leaderboard`, `ladder_probe` (an input set, despite the output
  prefix) → `models/probes`.
- **INVARIANT — training runs are an append-only namespace.** `models/<run>/` stays flat and
  immutable-by-path: `src/train/ppo.py`'s default curriculum pool hard-codes six checkpoints,
  trainview scans `models/*/train_log.jsonl`, ladder provenance JSONL stores `mlp:models/...`
  specs, and the leaderboard table cites them. Add new runs freely; never rename/move old
  ones. Pre-move `ladder_*` provenance files self-describe with old output paths — they are
  records; the model paths inside them remain valid.
- **`docs/`:** training.md / evaluation.md / formats.md / leaderboard.md replace content that
  was scattered and partially DUPLICATED across README.md, models/README.md, and
  datasets/README.md (the two nested READMEs deleted; every topic single-sourced, README.md
  rewritten around a layout map + demo gif at docs/media/pong.gif). Live pointers in
  `pong`/rallydb/recorder/rally_index updated; historical mentions in this file are records, deliberately untouched.
- **Alternatives considered:** docs-only tidy (rejected — leaves eval outputs and curated
  sets tangled with checkpoints in `models/`); grouping runs under `models/runs/` or by
  experiment family (rejected — breaks the append-only invariant for cosmetic gain).
- **Failure modes accepted:** anything OUTSIDE the repo (shell history, aliases, notebooks)
  hard-coding pre-move paths breaks; three same-named `ladder.cpp`/`evolve.cpp` pairs remain
  (`src/tools/` mains vs library sources) — the binary-name = filename convention, kept.

## Decision 18 — Eval-stride bug: in-training anchor evals misloaded stack>1 genomes (fix + `reeval` recovery)

Found (2026-07-09) while investigating "why does evomem still trail the memoryless champion?"
— much of the trail was measurement artifact. `evolve()`'s anchor eval built matches with
`MatchParams{}`: canonical physics as intended (the Decision 11/13 guardrail), but
`MatchParams.stride` defaults to 1 and `playMatchVsSpec` loads the genome at `mp.stride`, so
every stack-6 **stride-6** genome was evaluated with its 0.52 s history window collapsed to
0.10 s of near-duplicate frames — out-of-distribution input, garbage play, in the eval only.
- **Evidence:** evomem_long's train_log reported 0% vs every anchor incl. both held-outs for
  all 62 evals over 1,557 gens, while the same checkpoints rate 1649 Elo on the ladder and
  gen_1149 beats p:classic 100% (memuse, 80 matches). Smoking gun: identical weights with the
  file header's `stride 6` → 100% vs p:classic; header forced to `stride 1` → 0%.
- **Blast radius: reporting only.** Selection always used the training `MatchParams` (correct
  stride) and checkpoints bake stride into the file, so ladder / memuse / aiming / GUI played
  correctly all along. Stack-1 runs (evo2/evo3/evo3.5) were never affected. What was garbage:
  train_log anchor + held-out columns and the live dashboard, for every memory-USING genome
  in every stack>1 run (a widened zero-history champion is stride-invariant, which is why
  evo4–6's frozen founder logged plausibly while its memory-using rivals logged 0%).
- **Fix (TDD, watched it fail 1 != 6 first):** `evalParams(cfg)` — canonical unshaped physics
  with `stride = cfg.stride` — at the single eval call site; the doctest pins the full
  contract (training knobs must NOT leak in; stride must ride along).
- **Recovery — `./pong reeval <run-dir> [matches] [out]`** (src/tools/reeval.cpp): replays
  every `gen_*.txt` vs the 3 reporting + 2 held-out anchors on canonical physics with
  trainer-mirrored eval seeds → `results/reeval/<run>.jsonl` + a table. evomem_long's 62
  checkpoints take ~17 s.
- **Corrected history (reeval, 24 matches/anchor):** evomem_long was ≥96% vs p:classic AND
  both held-outs from gen 24 onward; vs laggy mostly 0% with its run-maximum 25% at the FINAL
  checkpoint (still climbing when interrupted at 1,557 of 3,000 budgeted gens); vs instant
  interceptor 0% throughout. evo4 never lost p:classic (~88–100% through gen 2999) — Decision
  14's "lost to ALL canonical anchors (even P-classic)" was overstated by this bug; the real,
  still-valid evo4 finding is erosion on the interceptor rungs (instant 92%→~0% by gen ~350,
  laggy 100%→0% by gen ~1000).
- **Watch-item (not done):** `--resume` restarts generation numbering at 0 — a
  same-`--out` resume overwrites the original segment's `gen_*.txt` and appends log rows that
  restart at gen 0 (evomem_long's log is strictly consecutive 0–1556, so it never resumed).
  Until renumbering lands, resume interrupted runs into a fresh `--out`; documented in
  docs/training.md.

## Decision 19 — Trainview becomes the control panel: pause/stop/eval channel + curriculum telemetry

Requirement: pause training from the dashboard and get the cores back, plus resume — scope
settled as the full bundle.
This deliberately ends Decision 15's read-only stance: trainview is now the experiment
control panel Decision 1 anticipated, with a one-token write surface.
- **Channel (files-as-interfaces):** `<run>/control` token — `pause` | `run` | `stop` |
  `eval` — polled at generation tops (GA) / update tops (PPO); `<run>/status.json` heartbeat
  (`state`/`gen`/`gens_total`/`ts_unix`) back. Both dirs are gitignored data dirs already.
- **Pause parks between generations**: no worker threads exist there, so cores are fully
  released (~2 MB population held); no RNG is drawn, so results are **pause-pattern-
  invariant** — pinned bitwise by doctest. Latency = one generation/update.
- **Stop is graceful and lossless**: GA writes population.txt/hof.txt (full `--resume`
  state — the evomem_long "killed at 1557, population from 1549" loss mode is gone); PPO
  exports latest.txt. `eval` forces one anchor eval + checkpoint, then self-clears. Stale
  control files are removed at startup, so a run can never begin paused. Pause-in-place
  also sidesteps #18's resume-renumbering wart for the reclaim-my-machine case.
- **Live-verified end to end:** GA parked at gen 88 (log frozen across the pause), resumed,
  stopped with continuation state saved and a clean container exit; PPO parked at step
  26,624, resumed, stopped with latest.txt exported.
- **Curriculum telemetry (GA log rows):** `fit_peer`/`fit_pool`/`fit_anchor` — population-
  mean shaped margin per opponent class (the live "is the curriculum giving gradient"
  signal #18's investigation had to reconstruct forensically) — and `champ_age`
  (generations since the best genome changed, FNV hash). Backwards-compatible key
  additions; -2 sentinel matches the vs_* convention.
- **Dashboard:** Pause/Resume/Stop/Eval-now buttons; state badge with stall detection
  (running-but-silent > 15 s); ETA from status `gens_total` + logged pace (PPO via sps);
  faint reeval-overlay curves when `results/reeval/<run>.jsonl` exists; EMA smoothing
  toggle; a third panel plotting the class margins. `--dump` extended to print status,
  telemetry, and reeval row counts — headless verification stays possible.

## Decision 20 — Launcher: recipes as data, boot menu, GUI New-run via host-executed launch queue (full GUI launcher on a recipe layer; native binary rejected)

Requirement (2026-07-09): too many training CLIs to remember, a boot-up menu wanted, and an
open question whether "just an executable" made sense. Brainstormed options; the decision: the
full GUI launcher (option B), grounded on a recipe layer, plus a TTY boot menu on bare `./pong`
(Enter = play). A native binary was considered and REJECTED: it trades the no-host-toolchain
/ docker-hermeticity invariant every prior decision has kept (CLAUDE.md rule 3 — everything
builds and runs in docker) for convenience the menu already delivers; a host desktop shim was
declined for the same reason.

- **The load-bearing constraint: a container dies with its PID 1.** Whoever launches a
  trainer owns its lifetime, so the GUI must NEVER spawn one directly — it only writes
  request files to `results/queue/`. A HOST-side executor (`./pong watch [--once]`, also
  folded into `./pong trainview` as a background poll loop for the session) claims requests
  and `docker run -d`s them detached: trainers outlive every UI, including the dashboard that
  launched them. Extends Decision 19's files-as-interfaces write surface from a one-token
  control channel to a launch channel.
- **Recipes are data, not code.** `recipes/*.recipe` — a line format (`desc:`, `trainer:`,
  `out:`, repeatable `args:` lines concatenated in order) parsed by the doctested
  `src/train/recipes.hpp`, the single source of truth `./pong` reads with `sed`/`grep` (no
  JSON lib; Decision 9's watch-item stays parked) and the GUI reads to populate the New-run
  form. The command blocks that used to live in docs/training.md MOVED into four seed
  recipes — evo-basic, evomem, ppo-scratch, ppo-warmstart — docs now point at the recipe
  files instead of duplicating their flags.
- **`./pong train <name> [overrides]`** composes recipe args + CLI overrides with ordinary
  argv semantics: appended, last flag wins. The GUI's launch-request protocol
  (`results/queue/req_<ts>.launch`: `recipe:`, `out:`, `resume:`, `args:`, repeatable
  `override:`) diverges on purpose — `args:` is a FULL replacement of the recipe's args, not
  an append, because one flag breaks the "last wins" assumption: `--pool-dir` accumulates
  (`cfg.poolDirs.push_back(next())`, `src/tools/evolve.cpp:39`) instead of overwriting like
  every other flag in that parse loop, so appending a GUI edit on top of a recipe's own
  `--pool-dir` would silently grow the pool instead of replacing it. `override:` still
  appends last, for small additions like the New-run form's seed field.
- **The one carve-out inside that override contract, hardened across the executor's three
  review rounds:** the validated `--out` is appended LAST in the executor's argv regardless
  of `args:`/`override:` order, so no spelling of `--out` — `--out=...`, or an argparse
  abbreviation on the ppo side (`--o`, unambiguous against ppo.py's flag set) — can redirect
  where a run lands. `out:` is the only door, checked against `models/`|`results/` with `..`
  path segments rejected before the trainer ever sees an argv. Where the override contract
  (last flag wins) and output containment collide, containment wins.
- **Review-driven hardening** (three rounds on the executor, two on the GUI): `..` traversal
  and override-smuggled `--out` were each empirically demonstrated against the first cut,
  then closed; the watcher was hardened to survive a stray per-request failure under `set -e`
  (a bad `sed`/`mv` on one malformed request no longer kills the poll loop); GUI request
  writes are write-tmp-then-`fs::rename` atomic, so the executor's `req_*.launch` glob never
  sees a half-written file; same-dir resume — Decision 18's destructive wart — is hard-gated
  in the New-run form (Launch disables, red warning) with a `<dir>2` suggestion offered the
  moment a picked resume source matches the out dir; the New-run tab itself is now reachable
  on a fresh checkout (it had been nested inside an "existing runs" branch that never
  rendered with zero runs).
- **Provenance.** Both trainers write `<out>/config.json` at startup — `writeRunConfig` in
  `src/train/evolve.cpp` (argv + start time); ppo.py's block additionally dumps the full
  argparse `vars(args)` dict. The dashboard displays it verbatim for the selected run, so a
  GUI-launched run stays exactly as reproducible as one pasted into a terminal.
- **Residual risks accepted (watch-items, none blocking):** the `<dir>2` resume suggestion
  doesn't check that dir is actually fresh (a stale `<dir>2` from an earlier attempt can
  still same-dir-collide); a failed `fs::rename` of a request file drops its error code — the
  stranded `.tmp` is invisible to both the executor's glob and the GUI's pending/failed scan;
  the container-name sanitizer (`tr -c 'A-Za-z0-9' '_'`) can collide two distinct `--out`
  dirs onto one Docker name, but fails LOUDLY via `.failed` (docker's own name-in-use error),
  never silently; a permanently unclaimable request (e.g. a stuck `mv`) retries every poll
  with no surfaced error; `docker run -d` returns success once the container starts, so an
  in-container trainer death after that point is invisible to `.failed` — `status.json`
  staleness (Decision 19's stall detection) is the actual detection path; and requests queued
  with zero executor running produce no error at all, only the GUI's "pending Ns — launcher
  not running?" hint once a request ages past 5 s. The boot menu is TTY-only by design
  (non-interactive stdin keeps the old play default so scripts never hang on a prompt).
- **Whole-branch review (2026-07-09):** evolve's silent random-founders fallback on a failed
  `--resume` load predates this branch and wants a fail-loudly follow-up decision; the stop
  path prints "done: final best fitness" after a control-stop too (cosmetic); a `docker wait`
  reaper or `./pong watch --selftest` replaying the refusal matrix would close the remaining
  launch-vs-death observability gap; reeval's seed mirroring is exact for seed-1 runs only.
- **Update (2026-07-10):** the fail-loudly follow-up and the "done: final best fitness"
  cosmetic both landed — closed by #22 (fail-loud resume + honest stop landed with tests).

## Decision 21 — Play picker: curated models with champions-canonical Elo in the game GUI

Requirement (2026-07-09): the Play GUI should list available models by name and Elo
automatically, refreshed whenever models get re-ranked on the ladder. Three calls settled
the shape: **curated list only** (`models/leaderboard/*.txt` + every run's `best.txt`/
`latest.txt` — 20 files today, not the ~850 individually-rated checkpoints spread across
every sweep in `results/ladder/`), **champions-sweep-canonical Elo** (fall back to the
newest-by-mtime sweep for anything the `champions` sweep hasn't rated), and a **flat
Elo-descending list with a filter box** that only appears once the list outgrows a
screenful (~15 entries) — no grouping, no always-on search.

- **Ratings index (`src/league/ratings_index.hpp`, doctested in `test_league` — league owns
  the ratings format, Decision 12).** `loadRatings(ladderDir)` walks every
  `<ladderDir>/*/ratings.jsonl`, skips header rows, and resolves per-spec conflicts
  champions-first, else newest sweep by file mtime; `curatedModels(modelsDir, ratings)`
  lists the curated files and looks each one up. **Pool-relativity is why the source sweep
  is always rendered next to the number, not just the Elo alone** (Decisions 12/16): the
  same checkpoint, `evo3_5/gen_2974`, rates **2014** in the 9-spec `champions` sweep and
  **1576** in the 355-spec `final` sweep — one bare number would misrepresent it either
  way, so every label reads `name — elo · sweep` (e.g. `evo3_5/gen_2974 — 2014 · champions`).
- **The `_leaderboard` alias.** Champions sweeps run before the Decision 17 reorg wrote
  their provenance as `mlp:models/_leaderboard/...`; the files live at
  `models/leaderboard/...` today. `normalizeRatingSpec` rewrites the old prefix at load
  time — confirmed necessary against the live sweep, not just hypothetical:
  `results/ladder/champions/ratings.jsonl` still carries the pre-reorg
  `mlp:models/_leaderboard/evo3_5_gen2974.txt` spec verbatim.
- **Repo-relative lookup-key contract.** `ratings.jsonl` specs are always written
  repo-relative (`./pong ladder` runs from the repo root, so every spec is literally
  `mlp:models/...`), independent of how a caller spells its `modelsDir` argument. The first
  cut keyed the lookup as `fs::relative(f, fs::path(modelsDir).parent_path())`, which
  collapses to an empty base — and the key to bare `"mlp:"` — for a bare relative dir like
  the GUI's own `"models"` (`fs::path("models").parent_path()` is empty), silently
  rendering every model unrated. Caught in review before Task 2 wired it into `main.cpp`;
  fixed to `"mlp:models/" + fs::relative(f, modelsDir)`, independent of whether `modelsDir`
  itself is absolute or relative, pinned by a test using the GUI's call shape (a bare relative modelsDir).
- **Rescan on combo open, no daemon.** `main.cpp` rebuilds the entry list from
  `loadRatings`/`curatedModels` only inside `ImGui::IsWindowAppearing()` — when a paddle
  combo is actually opened, never per-frame and never on a timer — so a fresh ladder run's
  numbers show up on the next click with nothing watching `results/ladder/` in the
  background.
- **`AIPONG_MODEL` stays inject-only.** The env model is still probed once at startup and
  appended to the entry list if not already present, but it does not preselect either
  paddle — unchanged from pre-picker behavior and kept on purpose (an early spec draft's
  "preselects" was a slip against actual behavior, corrected during planning).
- **Layout (mid-execution): panel moves below the court, popup height gets capped.**
  A wider combo (name + Elo + sweep per row, sometimes a filter box) prompted moving
  the control strip out of the court entirely: a pinned, non-move/non-resize `panelH`-px
  strip docked at the window bottom, the court scaled into the region above it
  (`courtH = max(200, H − panelH)`, a faint boundary line at `y = courtH`), and the combo
  popup's height constrained to `min(320, courtH·0.6)` so a long entry list scrolls inside
  the popup instead of growing off-screen. Shipped first at `panelH = 230`; review measured
  the idle panel's actual content (two combos, seed + New game, sim-speed, pause, record,
  separators, live status text) at 235–250 px — taller than that first number — and bumped
  it to **270** so the strip doesn't scroll at rest.
- **Watch-item:** the `champions` sweep is a manually-run snapshot, not a live view — a
  newly-promoted champion shows `(unrated)` or a stale number until someone re-runs
  `./pong ladder --dir models/leaderboard --out results/ladder/champions`. Nothing
  auto-triggers that (Decision 20's launcher scope is training runs, not ladder sweeps);
  the picker's rescan-on-open only refreshes the GUI's view of whatever is already on disk.

## Decision 22 — Interview polish: reviewed-backlog burn-down, CI, fail-loud resume, sweep corrections

Requirement (2026-07-10): the repo is public on GitHub for interviewers — polish
all code and docs for that audience, not docs alone. Four calls settled the shape:
**full sweep**, not docs-only; **add CI** (GitHub Actions) with a **README badge**;
**the stale hero gif gets re-recorded by hand** (capture commands handed off — still
pending, needs a display, not headless-automatable); **push to origin when green**
(pre-approved). Landed on branch
`polish/interview-pass`, `./pong build && ./pong test` green at every commit.

- **League ratings robustness** (`league: ratings robustness — writer-parser contract
  test, sentinel skip, atomic writes`): `loadRatings`'s ingest lambda
  (`src/league/ratings_index.hpp`) now skips `games <= 0` rows — the `elo = -1e9`
  sentinel `runLadder` writes for all-rally-cap specs — so an unrated sentinel from a
  newer sweep can never overwrite a real rating from an older one; `ratings.jsonl`/
  `matches.jsonl` (`src/league/ladder.cpp`) write to `.tmp` then
  `std::filesystem::rename` atomically, so a reader scanning `results/ladder/` never
  observes a torn file. A new writer↔parser contract test runs a real 2-bot ladder and
  reads it back through `loadRatings`, pinning the round-trip (`p:classic` anchors at
  1000.0) instead of asserting on the format in the abstract.

- **Shared `sizesInBounds` gate — review caught the first fix covering only one of
  three untrusted text parsers.** The initial pass (`polish: weight-loader sanity
  bounds; evolve fails loudly on bad --resume, reports stop honestly`) put the
  [1,100000]-per-layer / ≤8,000,000-total-param bounds check inline in
  `MlpAgent::loadStream` only. `readPopulation` (the `--resume` path) and
  `readModelFlat` (the `--init-from` path), both in `src/train/evolve.cpp`, still
  sized their allocations off an unvalidated sizes line — a corrupted `population.txt`
  could `bad_alloc` the process before ever reaching the new `resumeFailed` return,
  defeating the fail-loud fix four lines above it. Closed (`fix: shared sizesInBounds
  gate — --init-from and --resume parsers validate before allocating`) by extracting
  `sizesInBounds()` as the one shared helper (`src/agents/agents.hpp`/`.cpp`) and
  calling it from all three sites — `loadStream`, `readPopulation`, `readModelFlat` —
  before any allocation. `readPopulation` also bounds the population-count header
  (`count == 0 || count > 10000 || count*n > 16'000'000`) before sizing the genome
  grid. `MlpAgent::loadWeights` — fed only by already-vetted or program-constructed
  sizes (trainer hot loop, bench, tests), never raw text — deliberately does NOT call
  the shared gate; a comment at `agents.cpp:96-101` states that trust boundary
  explicitly so it doesn't read as a missed spot.

- **`EvolveResult.resumeFailed` and `.stopped`.** A failed `--resume` load (missing,
  corrupted, or architecture-mismatched `population.txt`) used to fall through to
  `founders.clear()` and silently random-restart training — Decision 20's
  whole-branch review flagged this as a needed follow-up. Now it returns immediately
  with `resumeFailed = true`, and the CLI (`src/tools/evolve.cpp`) prints the refusal
  to stderr and exits 1 instead of training on nothing the operator never asked for. A
  control-stop now sets `.stopped`, and the CLI prints "stopped by operator (control
  file) — population saved in `<out>` for --resume" instead of the misleading "done:
  final best fitness" line Decision 20 also flagged as cosmetic. Both follow-ups are
  closed; Decision 20's entry carries an "Update (2026-07-10)" line saying so. NOT
  covered by this fix: the sibling `hof.txt` load on the same resume — see
  watch-items.

- **Recipes: rtrim in both parser layers, `out:` required in both layers.**
  `src/train/recipes.hpp` rtrims `desc`/`trainer`/`out` (plus a `std::string(key)` →
  `std::char_traits<char>::length(key)` nit riding along) and `Recipe::valid` now
  requires non-empty `out`. The first commit (`polish: recipes rtrim + out-validation,
  shellcheck directive, escaping coverage`) only trimmed the C++ side; review caught
  that `pong`'s bash-side `load_recipe` still read `RCP_TRAINER`/`RCP_OUT` via bare
  `sed` with no trailing-whitespace strip, so a hand-edited recipe with trailing
  spaces would validate under one parser and not the other. Closed one commit later
  (`fix: bash-side recipe field trim — both parser layers now agree on trailing
  whitespace`). That same first commit also restores the `# shellcheck disable=SC2086`
  directive that had gone missing above the launch executor's docker line, and adds a
  `writeRunConfig` escaping test (`src/tests/test_train.cpp`) — coverage only, the
  escaping itself was already correct.

- **User-reported continuation UX** (`fix: New-run continuation UX — resumable-only
  resume list, ppo warm-start picker, teaching refusals`). Surfaced in live use:
  `evo-basic` recipe, out `models/ppo4`, resume `models/ppo3` in the New-run tab → a
  correct but unhelpful refusal ("no population.txt"), because `models/ppo3` is a PPO
  run and PPO has no `--resume`, only warm-start. The resume combo (evolve recipes
  only) now lists only run dirs containing `population.txt`, showing "(no resumable GA
  runs)" instead of an empty-looking combo when none qualify; PPO recipes get a
  "warm-start from" combo in its place (`"(recipe default)"` + every
  `best.txt`/`latest.txt` + curated leaderboard models), writing `override:
  --init-model <picked>` after the seed override so argparse's last-wins beats the
  recipe's own `--init-model`; refusal messages now name the right path instead of
  just the failure (e.g. "resume: is for GA runs — continue a ppo run with
  --init-model..."); a `[evolve]`/`[ppo]` trainer tag sits on the recipe description
  line; and a successful launch now `rm -f`s its claimed request file, which
  previously lingered in `results/queue/claimed/` even on success.

- **CI + README badge + mermaid dependency graph** (`ci: docker build+test workflow,
  README badge + mermaid dependency graph`). `.github/workflows/ci.yml` builds the
  docker image via `buildx` with a GitHub Actions layer cache
  (`cache-from`/`cache-to: type=gha`) — the torch install in `docker/Dockerfile` is
  the expensive layer worth caching — then configures/builds/tests inside the
  container, mirroring `./pong build && ./pong test` but with
  `-DAIPONG_MARCH_NATIVE=OFF`: a CI runner isn't the training machine, and a
  `-march=native` binary built on one runner's CPU could SIGILL on another. README
  gets the badge under the title and a mermaid graph in the Layout section (`core →
  agents/record → match → {train, league, bindings, arena}`, plus dashed provenance
  edges for `train_log` → trainview and `ratings.jsonl` → arena) — every solid edge
  checked against `target_link_libraries` in CMakeLists.txt before committing, not
  just drawn from memory of the architecture.

- **Fresh-eyes sweeps (code: 2 Important/9 Minor; docs: 7 Important/5 Minor) → one
  fix-wave commit** (`polish: fresh-eyes sweep fixes — measured 95% headline,
  dep-chain prose, K-cap, layer-agreement on out, warm-start combo gating`). Two
  independent read-only review passes, read as a first-time reader would: a code sweep over `src/`, `pong`,
  `recipes/`, CMakeLists.txt, and a docs sweep over README-as-first-contact plus every
  doc link/path and DECISIONS.md's index-vs-heading agreement. 18 of the 23 findings
  landed in this one commit (7 of 11 code, 11 of 12 docs); the rest are adjudicated
  skips — real behavior decisions, not mechanical text, flagged as follow-ups below.
  Notable corrections: the headline claim "beats the perfect defender 99%" was
  mislabeled — 99% is evo3.5's overall round-robin record across the 9-spec champions
  field, not the Interceptor matchup — corrected to the actually-**measured 95% over
  300 matches vs the instant Interceptor** (Decision 16's aiming study), in
  `docs/leaderboard.md`; README's dependency-chain prose said `core → agents →
  {record, match} → ...`, contradicting both its own mermaid diagram three lines below
  and CMakeLists (`pong_record` links only `pong_core`) — corrected to `core →
  {agents, record} → match → {train, league}`; a stale "1 ≤ K ≤ 8" in
  `docs/formats.md` contradicted its own K ≤ 32 line and the actual enforced cap
  (`agents.cpp:104`) — fixed to K ≤ 32 there and in `agents.hpp`'s docstring; README's
  `../ui_pong` link 404s on GitHub (a local sibling dir, never pushed) — de-linked to
  plain code text; `docs/training.md`'s basic Evolution/PPO examples still
  hand-duplicated recipe flags instead of pointing at the recipes, contradicting
  Decision 20's own "docs point at recipe files" claim — swapped for `./pong train
  evo-basic` / `./pong train ppo-scratch`; and `Recipe::valid` (C++) didn't check
  `out` non-empty, contradicting the bash-side gate landed in the burn-down above — a
  hand-authored recipe missing `out:` would show as selectable in the GUI yet be
  refused the moment it launched.

- **Watch-items:** `hof.txt` on `--resume` has no fail-loud guard — a
  missing/corrupted/mismatched hof silently leaves the hall-of-fame empty rather than
  setting `resumeFailed` (`src/train/evolve.cpp:340`); deliberately out of scope here
  since this pass's contract was `population.txt` only, and choosing
  fail-the-whole-resume vs. warn-and-continue-empty is a real behavior call that wants
  its own decision. A related gap in that same fix: an architecture-mismatched resume
  source (e.g. a stack-1 population fed to the stack-6 evomem recipe) still passes the
  launch executor's existence check (`pong`'s `population.txt`-present gate) — the
  architecture check only runs inside the container — so it launches detached and dies
  invisibly: `resumeFailed` exits 1, but `docker run -d` discards stderr and neither
  `status.json` nor `.failed` ever gets written. Strictly better than the old silent
  random-restart, but the same observability class as Decision 20's `docker wait`
  reaper follow-up; fix candidates are recorded there. CI's workflow YAML was only
  structurally sanity-checked locally (no `pyyaml` in the build image) — full
  validation happens on the first push-triggered run. The hero gif at
  `docs/media/pong.gif` still shows the pre-Decision-21 panel-over-court layout —
  a by-hand re-record is pending once a display is available. Sweep follow-ups flagged but
  not fixed (behavior decisions, not mechanical corrections): claimed-request cleanup
  in `pong`'s `process_queue` only fires on the successful-launch path — the other 7
  exit branches (every refusal/failure) still leave a stranded copy in
  `results/queue/claimed/`; the atomic-rename `std::error_code`s in
  `src/league/ladder.cpp` (the `ratings.jsonl`/`matches.jsonl` writes) are captured
  but never checked or surfaced, so a failed rename fails silently —
  `src/tools/ladder.cpp` does no renaming itself; its unconditional `:74` "written:"
  print has no way to know one failed; and the warm-start combo's new entry-count
  filter has no popup height cap the way Decision 21's picker does (`main.cpp`'s
  `SetNextWindowSizeConstraints`, absent from `train_view.cpp`'s combo) — ImGui
  already caps an unconstrained combo popup at ~8 items then scrolls (default
  `ImGuiComboFlags_HeightRegular`, verified v1.90.4), so this is a comfort delta, not
  unbounded growth: no filter box, and a shorter default height than Decision 21's
  picker gives a large `models/` tree.

## Decision 23 — Scalar forward flattened: act() vectorizes at last (measured 2.1–2.3×)

Disassembly follow-up on Decision 11's "libmvec SIMD tanh" claim (see its 2026-07-10 Update
above): `nm -u build/CMakeFiles/pong_agents.dir/src/agents/agents.cpp.o | grep -i tanh` showed
only `U tanhf` — the scalar libm symbol — even though `-ffast-math` has been scoped onto
`agents.cpp` since Decision 11 landed. **Root cause:** `MlpAgent::act()`'s per-layer loop is
`for(o){ serial-reduce over i; tanh(acc); }` — a transcendental call sits inside the
reduction, which is exactly the shape gcc won't auto-vectorize (there's no flat SIMD
dimension for the o-loop to walk). `batch_forward.cpp`'s batched path never had this problem:
its SIMD dimension is the lane axis, orthogonal to the per-lane reduction (Decision 13).

**Fix (`src/agents/agents.{hpp,cpp}`).** `Layer::w` is now stored TRANSPOSED — `[nIn][nOut]`
instead of the wire format's row-major `[nOut][nIn]` — built once at load time
(`loadWeights`; the on-disk/flat text contract, Decision 6, is unchanged, only the in-memory
layout). `act()`'s inner loop is reshaped flat: bias-init an `nOut`-wide accumulator (`acc_`,
preallocated alongside the existing `bufA_`/`bufB_` scratch — no per-call allocation), then
`for(i){ x=in[i]; for(o) acc[o] += w[i*nOut+o]*x; }` — now a flat, contiguous o-loop that
vectorizes — followed by a SEPARATE flat pass `for(o) out[o] = tanh(acc[o])` (hidden layers
only; output layer stays linear), which is the loop libmvec now vectorizes.
`batch_forward.cpp` is untouched — different layout, already optimal (Decision 13).

**Verified vectorized, not just faster.** The same `nm -u` check on the post-fix object file
now lists `_ZGVbN4v_tanhf` (SSE, 4-wide) and `_ZGVdN8v_tanhf` (AVX2, 8-wide) alongside scalar
`tanhf` (the last is the vectorization loop's remainder/epilogue, not a sign it failed).

**Measured** (median of 3 uncontended runs each, `./pong bench` / `./pong bench --stack 6`,
idle machine — `docker ps` empty, load average < 1; twelve before/after runs
in total). `./pong bench` drives `MlpAgent` end to end
through `MatchRunner`, so these numbers include physics + match bookkeeping, not just the
isolated forward call:
- **6-32-32-3** (unstacked): mlp vs mlp 951k → 2.10M ticks/s 1-thread (**2.21×**), 3.81M →
  7.82M at 4 threads (**2.06×**).
- **36-32-32-3** (evo4's live stride-6 shape, `--stack 6`, Decision 13): mlp vs mlp 819k →
  1.89M ticks/s 1-thread (**2.30×**), 3.03M → 6.74M at 4 threads (**2.23×**).
- The batched SoA path and the three scripted-agent workloads (no MLP involved) moved ≤ ±11%
  across the same twelve runs — noise, confirming the fix is isolated to the scalar forward
  exactly as intended (`batch_forward.cpp` untouched).

**Numerics class + the golden pin.** Same ~1e-7 reassociation class Decisions 11/13 already
accepted (float32 + `-ffast-math` reduction reordering) — this change adds a genuinely
different summation order (input-major instead of output-major) on top of that, so it is
ALSO a cross-version bit-compat break, same precedent, same acceptance. A throwaway
logit-diff probe (old row-major-serial-reduce vs. new transposed-flat forward, identical
weights/inputs, same float32/`-ffast-math`/`-march=native` codegen, 400 LCG-perturbed
states across both shapes) measured the drift directly rather than assuming it: max
|Δlogit| ≈ 8.6e-7, mean ≈ 1.6e-7 — the same order as Decision 11/13's ~1e-7, and **zero**
argmax flips in that 400-sample draw. The committed golden pin (`src/tests/test_agents.cpp`,
`MlpAgent::act: bit-exact logit pin` — FNV-1a over `act()`'s Move sequence across 200
LCG-perturbed states × {stack 1, stack 6}, plus a 20-state argmax table per stack) was
regenerated against the flattened `act()` and came back **byte-identical** to its pre-fix
constants: 0/400 sampled Moves flipped, both argmax tables unchanged. That is a legitimate
result, not a non-event — the drift is real and directly measured, it simply never crossed a
decision boundary in this sample, the same "well-separated decisions tolerate reassociation
noise" property `test_train.cpp`'s "BatchedMlp matches the scalar agent on well-separated
decisions" already relies on. The pin now stands as a live regression guard for any FUTURE
numerics change to `act()`; had the hash or either table actually changed here, that was the
designed STOP-and-review signal — none was needed this time.

**Alternatives considered — Eigen / libxsmm / BLAS for the forward pass: rejected,
reasoned not benchmarked.** These layers are GEMV-scale (36×32, 32×32, 32×3 — a few hundred
to ~1k MACs each) called millions of times per generation; general GEMM-oriented libraries
carry per-call dispatch cost (kernel selection, threading setup) that plausibly dominates
compute at this size, and none of Eigen/BLAS/libxsmm is a dependency of this repo today
(checked — no reference in `docker/Dockerfile` or `CMakeLists.txt`). The hand-rolled SoA
shape this fix lands (contiguous, preallocated, no per-call allocation) is already the right
shape for this problem size; pulling in a matrix library for a ~1k-parameter net buys build/
image weight against a payoff that isn't clearly there at this scale. Flagged as a
watch-item to revisit if the network ever grows materially past 6-32-32-3-class dimensions.

**Not-yet (deferred, watch-items):**
- **SLEEF / approx-tanh, quantization (int8/fp16).** Deferred until this fix is re-profiled
  inside an actual training run — `act()` is the trainer's real per-tick hot path and it just
  moved 2.1–2.3×; re-measuring generation wall-clock before stacking another numerics-
  affecting change on top is the right order of operations (measure-first, Decision 11's own
  rule, vindicated a third time by this decision).
- **Thread pinning (`taskset` / `--cpuset-cpus`): measured, not assumed — and it lost.**
  Three runs of `./pong bench --stack 6` under `docker run --cpuset-cpus=0-3` (pinned to 4
  cores, matching the 4-thread workload) against the same unpinned baseline: "mlp vs mlp (4
  threads)" fell from a 6.74M ticks/s median to 4.20M (≈ **−38%**), "mlp BATCHED (4t)" fell
  from 5.40M to 3.35M (≈ **−38%**) — a regression, not the small gain a dedicated, quiet
  training box would predict. This machine is a shared desktop (20 cores; Opera/Discord/VS
  Code/etc. also running — `perf-report.md` has the `ps` snapshot), and pinning to a narrow
  low-numbered core range plausibly collided with THAT load rather than isolating from it;
  the default scheduler's freedom to roam all 20 cores wins here. Left as a watch-item to
  re-test on the actual dedicated/quiet training box, not adopted from this measurement.

- **Update (2026-07-10) — Rec 2 (batched anchor evals) re-measured and killed, Decision-11 style:**
  the research's ~2.2× eval-batching projection predated #23. Fresh numbers: eval passes were
  2.0% of evomem_long2's wall-clock pre-#23 (0.40 s overhead per eval gen, 1-in-25 gens), ~0.9%
  post-#23, and the scalar-vs-batched forward gap is now 1.06× — ceiling on total training time
  ≈ 0.05%. Not built. If eval cost ever matters, the lever is eval-match count/roster (canonical
  full-cap matches are inherently long), not the kernel.

## Decision 24 — Training profile: the hot path is the forward's MACs; profiler + thread choice

The task: run timing tests to see exactly which step is taking the longest — add them in the
training and inference code — then do more granular timing. Built an opt-in, determinism-safe
profiler and drilled down to the instruction class. Machine: i5-14600K (6 P-cores + 8 E-cores,
20 threads; AVX2 + FMA, no AVX-512). Workload: the `evomem` stack-6 (36-32-32-3) recipe.

**Instrumentation (src/train/prof.hpp, AIPONG_PROFILE=1; committed, interview-visible telemetry):**
- Two layers: six sequential **generation phases** (job-build / batched games / scalar games /
  selection+breed / anchor eval / log+status+callback) that genuinely sum to the per-gen wall
  clock — the last bucket exists so nothing between t0 and gen-end is silently unaccounted; it
  measures 0.0%, which is the accounting's own check. And a per-tick **inference split**
  (obs-assembly / MLP forward / physics step) inside the batched kernel, summed across worker
  threads (CPU-time — the ratio is the signal). Dumps to stderr from the evolve CLI. A CI step
  (`.github/workflows/ci.yml`) runs the trainer with the profiler on vs off and diffs the logs,
  giving the bit-identical-results invariant real teeth rather than a one-time manual check.
- Zero cost when off (one cached env check; per-tick hooks are one predictable branch). Timing
  only — no RNG, state, or control-flow touched — so results are **bit-identical on vs off**
  (verified: 12-gen best/mean/fit_pool sequences match exactly; the thread-invariance doctests
  pass either way). This is why hot-path timing was safe to commit rather than kept on a branch.
- Sub-forward granularity can't be timed inline: a forward is ~0.4–5 µs and `steady_clock::now()`
  is ~20 ns, so per-region reads would swamp the ratio. `bench --granular` instead uses **amortized
  differential timing** — three nested variants of the real SoA kernel (transpose+argmax / +MAC /
  +tanh), each over 1e6 iterations in batch_forward.cpp so codegen matches the production forward's
  -ffast-math. Cross-checked: 5381 ns/forward measured vs ~5774 ns implied by the batched-1t
  throughput (8e9 / ticks·s⁻¹), the ~7% gap being the obs/step the isolated kernel omits — faithful.

**Findings (measured, idle machine):**
- **Generation phases:** batched peer/HoF games **86.7%**, scalar pool/anchor games **11.4%**,
  anchor eval 1.2%, selection+breed 0.7%, job-build 0.0%. Training is almost entirely genome-vs-
  genome match play; the deque push/pop and per-call sizing flagged as suspects earlier are in the
  0–2% noise (job-build 0.0%; history assembly is inside `obs` below at 2.2%).
- **Inference split (per tick):** MLP forward **95.9%**, obs assembly (incl. the stack-6 history
  ring) 2.2%, physics step (incl. substepping) 1.9%. Physics and observation handling are
  effectively free — optimizing substepping or the history ring would move nothing.
- **Granular forward (the 96%):** MAC accumulate **76.3%**, tanh (2 hidden layers) **20.9%**,
  transpose+argmax 2.8%. MAC time by layer (MAC-count-weighted): L0 36×32 ≈ 51%, L1 32×32 ≈ 45%,
  L2 32×3 ≈ 4%. So the single dominant cost in all of training is the two 32-wide hidden layers'
  vectorized multiply-accumulates.

**Consequences.**
- The forward is the only lever that moves training throughput — which is exactly what #23's
  flatten-and-vectorize already 2.1–2.3×'d. Anything targeting physics, history assembly, job
  allocation, or selection is chasing <2%.
- **tanh is 20.9% of the forward**, so an approximate tanh (the deferred quality-affecting option)
  caps at ~20% of the forward ≈ ~12–15% of training for a realistic 2–3× tanh, and it needs an
  anchor-skill A/B (not just a bit-diff). Bounded, not free — parked with that number attached.
- **MACs are 76%** and already SoA-vectorized. Remaining free-determinism-tier lever the granular
  data points at: **batched lane width** (currently 16 = 2 AVX2 registers; more independent lanes
  could hide FMA latency for better ILP, and lane width is already proven bit-invariant by
  `playMatchesBatched: identical results for any thread count and lane width`). Measure a
  `--lanes` sweep before committing. int8/AVX-VNNI would speed the MACs but changes the on-disk
  format (CLAUDE.md's highest-ceremony boundary) and is quality-affecting — last resort.
- **Thread count (measured 2026-07-10, idle):** 20t 0.651 s/gen, 18t 0.666, 16t 0.677, 14t 0.696,
  12t 0.740, 10t 0.819 — the curve is shallow (E-cores help a little, P-core HT barely).
  **Decision: stay at 18**, not 20 — the ~2% is not worth saturating a machine in interactive use
  (dashboard/OS headroom). Pinned with rationale in `recipes/evomem.recipe`. (Last night's −38%
  taskset result was a shared-load confound, not a pinning verdict.)

## Decision 25 — Granular MAC timing + the lane-width A/B that an isolated bench got backwards

The task: more granular MAC timing. Drilled `bench --granular` inside the forward
(the 96% of the hot loop; the MAC is 76% of the forward, #24) with layer-prefix differential timing (measured, not the earlier analytic
split), then chased the lever it exposed — and the chase is the lesson.

**Measured forward breakdown (stack-6 36-32-32-3, 16 lanes, idle i5-14600K):**
- L0 MAC (36×32) ≈ 35%, L1 MAC (32×32) ≈ 37%, L2 MAC (32×3) ≈ 3%, tanh (L0+L1) ≈ 23%,
  transpose+argmax ≈ 2%. Measured L0≈L1 (not the analytic 51/45 — L1's dependent-input access
  runs a bit slower per MAC), so the two hidden layers are the cost in roughly equal parts.
- The MAC runs at **~18 GFLOP/s, ≈11% of the ~169 GFLOP/s single-P-core AVX2+FMA peak**, streaming
  weights at ~36 GB/s. **Arithmetic intensity 0.5 FLOP/byte** — the SoA layout gives every lane its
  own genome's weights, so each MAC reads a fresh weight (no reuse). That's a streaming workload.

**The lane-width trap (why isolated benchmarks lie).** A single-thread sweep showed ns/lane
*falling* with lane count (1116→362→236 ns/lane at B=4→16→64) — the FMA-latency/ILP signature, i.e.
16 lanes = 2 AVX2 accumulators under-feed the FMA units, and wider should help. A multi-thread bench
agreed: 18t×32 lanes beat 18t×16 by ~13%. So the batched default moved to 32 — then came the
honest test: an **end-to-end training A/B** (resume the same population + seed, `--eval-every 0`,
median s/gen). It **reversed**: 16 lanes **0.620 s/gen**, 32 lanes **0.640 s/gen** — 32 is ~3%
*slower*. Fitness stayed bit-identical (lane-invariance doctest holds end-to-end), so it's purely
throughput. **Reverted to 16.**
- Why the bench lied: `runMlpBatched` reused **two** genomes across all lanes, so weights sat
  cache-hot and wider lanes won on ILP with no bandwidth cost. Real training streams a **128-genome
  population** — at 32 lanes × 18 threads the doubled unique-weight working set thrashes shared
  L3/DRAM, and the 0.5-FLOP/byte MAC is weight-**bandwidth**-bound, so the extra ILP doesn't pay
  (plus wider setLane/refill/construction overhead the isolated kernel omits). Fixing the bench to
  stream 128 distinct genomes narrowed but did **not** close the gap — the full generation loop has
  costs no kernel micro-benchmark captures. **The end-to-end `./pong train` A/B is authoritative;
  isolated-kernel A/Bs of this workload are not.**
- Bench changes kept: distinct-genome streaming in `runMlpBatched` (was cache-hot 2-genome), the
  measured per-layer/MAC/tanh/roofline breakdown, and the single-thread sweep labelled "isolated,
  does not predict training." Removed the multi-thread "does wider help" lines that pointed the
  wrong way.

**Consequence.** Lane width stays 16 (the change is one number in evolve.hpp, re-open only with a
training A/B). The MAC being weight-bandwidth-bound at 0.5 FLOP/byte reframes the remaining compute
lever: **int8 weights cut weight traffic 4×**, attacking the actual bound (not a compute bound) —
but it changes the aipong-mlp on-disk format (CLAUDE.md's highest-ceremony boundary) and is
quality-affecting (needs an anchor-skill A/B), so it stays deferred with that justification attached.
This is the third time this week that measuring-first reversed a plausible optimization (Rec 2
batched-anchor-evals ≈0.05%; taskset −38%; now lane width) — the repo's measure-don't-assert rule
earning its keep.

- **Follow-up (two questions) — measured, `bench --granular` extended:**
  - *"Which step of L0/L1 is most expensive — the raw matrix multiply or the allocation?"*
    The **raw multiply**, overwhelmingly. A `DoMAC` gate that runs bias-init with the FMA accumulate
    skipped isolates the two: **raw multiply ≈ 4.3 µs vs bias-init ≈ 80 ns — the multiply is ~50× the
    setup** (both stable over 3 runs). The multiply is ~75–85% of the forward — the exact % wobbles
    run-to-run because the denominator carries the noisy tanh differential (consistent with #24's
    76%). There is **no per-forward allocation at all** — the weight and activation buffers are
    `BatchedMlp` members assigned once in the constructor, and `setLane` only writes into them per
    match. So L0 (36×32) and L1 (32×32), ~36% of the forward each, are each dominated by their
    `for i {for lane: out += w·x}` inner loop; bias, transpose, and argmax together are a few percent.
    The lever is the multiply, not setup.
  - *"Is there a cheaper tanh?"* Yes, ~4.7× cheaper, at an accuracy cost that makes it a
    train-time decision, not a drop-in. Timed (in the -ffast-math TU) per element over the
    pre-activation range: libm `std::tanh` (vectorized `_ZGVdN8v_tanhf`) **0.61 ns**, a clamped
    Padé `x(27+x²)/(27+9x²)` **0.13 ns (4.6–4.8×)** — but **max abs error 2.4e-2** vs true tanh
    (~2.4% of tanh's [-1,1] range). tanh is ~12–20% of the forward (isolated 0.61 ns × 1024
    evals/forward ≈ 12%; the noisier in-context differential runs higher), so a swap saves
    ~10–15% of the forward ≈ ~9–13% of training. But it **changes the activation the network
    computes**: existing champions would play differently (need re-rating/retraining), and whether
    the GA trains as well under a 2.4e-2-error tanh is unknown without an **anchor-skill training
    A/B** — the #25 discipline. A higher-order Padé / minimax polynomial trades some of the 4.7×
    back for lower error if pursued. Deferred with the numbers attached; not a free win.

- **Follow-up² ("no more efficient way to do the raw multiply?" / "a lower-error tanh?"):**
  - *Multiply:* essentially no — and measurement killed the obvious idea. The multiply is already
    at its codegen ceiling (SoA + FMA + `-ffast-math`); it's bandwidth-bound (#25), so the lever
    "should" be fewer weight bytes. Measured the same forward with weights at 4/2/1 bytes (convert
    to float in the MAC): fp32 5479 ns → **16-bit 1.02× (a wash) → 8-bit 0.86× (SLOWER)**. The
    int→float **convert cost exceeds the bandwidth saved** for a single thread (whose working set
    isn't bandwidth-starved). So narrower *storage* doesn't help. A real int8 win would need native
    **AVX-VNNI `vpdpbusd`** (int8×int8→int32 in one instr, no convert) — which requires *full*
    quantization: int8 activations too, int32 accumulate, a wholly different numerics path — high
    quality risk and uncertain payoff at this scale. And like #25's lane width, this is a
    single-thread reading; under 18-thread bandwidth contention int8 *might* claw back, but that's
    untested and not worth betting on. Conclusion: the multiply is as efficient as it cheaply gets.
  - *tanh:* yes — the 2.4e-2 error isn't the only option. A **Padé[7/6]** rational
    `x(945+x²(105+x²)) / (945+x²(420+15x²))`, output-clamped, measures **max err 1.3e-3 (≈17×
    lower than the crude Padé's 2.4e-2) at 0.17 ns/elem — still 3.5× faster than libm's 0.60 ns**. So the
    tanh lever survives at a far more palatable accuracy (1.3e-3 is plausibly training-A/B-safe
    where 2.4e-2 was a stretch); higher-order rationals drop it further if needed. Still changes the
    activation, so a swap still needs the anchor-skill A/B — but this is the version worth trying.

## Decision 26 — Loss-side fitness shaping: bounded survival + concession-proximity

**Problem (signal side of the plateau).** GA fitness was whole-match point margin — one scalar per
match. Against a pool champion 400+ Elo above the population every match is a loss, and every loss
scores identically whether the paddle just grazed the ball or sat at the far wall. The gradient is
flat exactly where the population lives (the #16 fitness valley, on the *signal* side rather than the
opponent side), so mutation has nothing to climb. A flat gradient isn't a compute problem — a fresh
10,000-generation run reproduced the same plateau (instant-interceptor still 0%), so the lever is the
fitness signal, not the generation count.

**The lever (four rules).** Add a dense, sub-margin gradient *among losses*: a loss that
lasts longer and concedes closer to the paddle scores higher than a blowout — but never higher than a
win. From the policy's (always-left) perspective, `margin = (pts_for − pts_against)/winScore`:
- **win** (`margin > 0`): `+ speedBonus·max(0, 1 − ticks/T_ref)` — the #11 win-speed bonus, unchanged.
- **loss** (`margin < 0`, `lossBonus > 0`): `+ eff_loss·[ (1−w_d)·survive + w_d·prox ]`, where
  `survive = clamp(ticks/T_ref, 0, 1)` (slower loss → less-bad) and `prox = clamp(1 − avgMiss, 0, 1)`
  (`avgMiss` = mean |ball.y − paddle.y| in cu over the policy's conceded points → closer → less-bad).
  `w_d = lossProxWeight` defaults **0.7** — proximity-weighted, because a near-miss is the policy's
  own defense whereas survival time is partly the opponent's doing.

**Win > loss ALWAYS — the hard guarantee (rule 3).** `eff_loss = min(lossBonus, 2/winScore − 1e-9)`,
and `survive`, `prox`, `lossProxWeight` are each clamped to [0,1] so `q ∈ [0,1]`. Best shaped loss =
`−1/winScore + eff_loss < +1/winScore` = worst win. So no flag value — any `lossBonus`, any weight —
can make a loss outrank a win; the shaping only re-ranks *within* wins and *within* losses, and the
GA can never be tricked into preferring to lose. This extends #11's discipline to the loss side.
RallyCap stays −1 (unshaped, short-circuited before `shapeFitness`) so slow-losing can't degrade into
stalling. Pinned by adversarial doctests (lossBonus=100, lossProxWeight=1.2 still rank below the
narrowest win).

**Passive core telemetry, no new persistent state.** The proximity term needs the ball-to-paddle
distance at each concession, so `State`/`Snapshot` gained `missSum[2]`/`missN[2]` (cu, [0]=left
[1]=right), accumulated in `scoreIfOut`. It never feeds back into physics — trajectories, bounces,
scores, and outcomes stay bitwise-identical (core-purity rule holds; every existing sim doctest passes
unchanged). The accumulators reset per match in `initialState` and `shapeFitness` is stateless, so
there is **no serialization and no format change** — `aipong-pop` and `--resume` are untouched, and
results stay thread-invariant (pinned by a threads-1-vs-4 doctest with shaping on).

**Training-only, flag-gated.** Loss shaping is training-only: `evalParams` keeps `lossBonus = 0`, so
anchor/held-out evals stay unshaped and cross-run-comparable (the #11/#13 contract). `--loss-bonus`
defaults 0 → fitness byte-identical to today, so every existing recipe and checkpoint is unaffected;
`recipes/evomem-rich.recipe` is the opt-in A/B arm (evomem + `--loss-bonus 0.2 --loss-prox-weight
0.7`, one variable).

**The A/B (result pending — measure, don't assert).** Fresh evomem-rich vs the evomem baseline, same
seed/budget/pool. The comparison uses **unshaped metrics only** — the shaped `fit_pool` of the rich
arm is mechanically inflated by the bonus and is not comparable. Cross-arm: anchor win-rates (already
unshaped), the champion's mean concession distance (`avgMiss` trend — the direct behavioral target),
and final ladder Elo. Success = defensive improvement that *transfers* (Elo/win-rate rise above
baseline), not a higher self-inflated score. Failure mode to watch: `avgMiss` shrinks but Elo/win-rate
don't move ⇒ the shaping taught graceful losing, and the complementary lever — a **graduated
opponent pool** — attacks the *opponent* side of the same `fit_pool` plateau this decision attacks
from the signal side. Instead of the pool's current uniform draw over three fixed champions 400+
Elo above the population (against which every game is an identical loss, the #16 fitness valley),
it samples a **graded ladder of ~6 frozen checkpoints spanning weak→strong** with a
**frontier-following win-rate-EMA sampler**: a per-rung EMA (all rungs init 0.5, updated
`ema = 0.9·ema + 0.1·[won]` once per generation) up-weights the rungs the population currently wins
15–85% of — floor 0.15 so no rung starves — so training always faces a ~beatable rung and the
sampler slides up the ladder as the population improves. It is a straight port of `ppo.py`'s
opponent curriculum (Decision 14's PPO track) to the C++ GA. The EMA is sequential,
cross-generation, outcome-dependent state — new surface on the GA's bitwise-thread-invariance and
exact-`--resume` contracts — kept deterministic by sampling from the current EMA during the
single-threaded job-build and folding results back in fixed job-index order, and kept resume-safe
by a `<outDir>/pool_ema.txt` sidecar (whitespace floats, written every checkpoint, restored on
`--resume`, init 0.5 if absent) so the doctest-pinned `aipong-pop` population format is untouched.
A new `--pool-adaptive` flag gates it (default off ⇒ byte-identical to every current recipe). The
A/B that would decide it: a fresh from-scratch `evomem-graded` run (evomem args + `--pool-dir
models/pools/evomem_graded --pool-adaptive`, the `--pool-games` budget held so only the curriculum
varies) to ~3000 gens at seed 1, against the `evomem_10k` baseline — success = `fit_pool` climbs
past its ≈ −0.66 floor, the vs-laggy/instant-interceptor win-rates lift off 0%, and final champion
ladder Elo rises above baseline. Spec'd, deferred, not yet run.

## Decision 27 — evomem_10k: 10,000 gens reproduce the ~3,000-gen plateau — gens 3000 vindicated

Open question after evomem_long/evomem_long2 (#18) stopped short of their 3,000-gen budget: would
more generations crack the instant interceptor? Ran `evomem.recipe` unchanged except
`--gens 10000 --out models/evomem_10k` (`config.json` confirms the argv is the recipe verbatim plus
that one override — same pop 128 / games 8 / pool+anchor diversity / speed-bonus 0.2 / rally-cap 500 /
speedup 1.06 / serve-hold 0 / win-score 5 / seed 1 / threads 18).

**Result: a flat plateau, not continued improvement.** Final (gen 9999): best fitness **+0.505**,
**0%** vs the instant interceptor (peak **4% @ gen 9149**, never sustained), vs laggy peaked
**75% @ gen 8999** then regressed to **21%** by the final generation; vs p:classic saturated at 100%
by gen 699 and never dropped. Best-fitness across gens [2900,3100] (mean 0.525, range 0.324–0.882) is
statistically indistinguishable from the final 100 generations (mean 0.537, range 0.347–0.776) — 7,000
additional generations, more than 3× the original budget, moved nothing measurable.

**Vindicates `evomem.recipe`'s `--gens 3000`** (traced to #11's update, the evo2 template): the
original stopping point was not premature — training had already saturated whatever this fitness
landscape lets mutation + truncation-selection reach by roughly gen 3000. **Reconfirms #16's standing
conclusion a second, independent way**: pure/diversified self-play GA does not crack the instant
interceptor, not because any run was cut short, but because the mechanism itself plateaus — consistent
with #16's own prescribed alternatives (shape fitness toward deceptive/adaptive opponents, or use
gradient-based methods that engage new inputs mutation can't reach; #26's loss shaping is the first of
these now shipped, A/B pending), neither of which is "run the GA longer."

**Not done:** a second seed to check whether gen-3000 saturation is seed-specific or systematic (this
is a single-seed result); no forensic lineage/md5 audit of the gen-9000ish laggy 75%→21% regression
the way #16 audited evo4–6 — the shape rhymes with #16's champion-slot flicker and is a plausible next
forensic target if this run is revisited.

## Decision 28 — Island (deme) co-evolution: N architectures cross-play under per-island selection (`coEvolve()` / `./pong coevolve`)

The GA plateaus (#16/#27), and every run since Decision 10 has trained the identical `32,32`
hidden-shape net (6-32-32-3 unstacked, 36-32-32-3 at stack 6) — architecture was never the varied
dimension, only the optimizer was (fitness shaping #26, opponent pool #14, generation budget #27),
so whether `32,32` itself caps the ceiling or the optimizer does was never separable. Co-evolution
is the separating experiment: run N single-architecture populations ("islands") side by side, each
an ordinary `evolve()`-shaped GA with its own selection and hall-of-fame so within-island
comparisons stay apples-to-apples — but every generation, a slice of each genome's fitness games
are played against the OTHER islands' current champions, so architectures co-adapt instead of
training in isolation. Selection stays entirely WITHIN each island, so a fast-converging small net
can never crowd a slower-maturing deep net's population out the way one shared species would —
ratio preservation is a structural property of running N independent island loops, not a tuned
parameter (a doctest pins it: every island still returns a champion of its own shape after
cross-play).

**What shipped (`src/train/coevolve.{hpp,cpp}`, `src/tools/coevolve.cpp`).** `coEvolve()` lands
alongside an **untouched** `evolve()` — the one edit to `evolve.{hpp,cpp}` moves `flatSize`/
`randomGenome` to namespace scope and adds `playMatchVsGenome` (self and opponent each loaded with
their OWN, possibly different, sizes; mirrors `playMatchVsSpec`'s in-memory branch), a
behavior-preserving refactor whose regression gate is `evolve()`'s pre-existing deeper-net and
thread-invariance doctests, not new ones written for the occasion. Each genome plays **3 self / 3
cross / 2 anchor** games by default (`--self-games`/`--cross-games`/`--anchor-games`; cross
defaults to one fewer than the island count, one draw per other island): self-play stays the
existing batched SoA path (uniform arch, unchanged); cross-island games run through the new scalar
`playMatchVsGenome`; anchor games reuse the existing scalar `playMatchVsSpec`. Every generation,
each island's top-25% elites + HoF are copied into a **frozen pre-mutation snapshot** before any
match is built, so every cross-island opponent that generation reads the same data regardless of
thread scheduling — results are bit-identical for any `--threads` (pinned by a doctest at 1 vs 4
threads, the same invariance contract `evolve()` already carries). `./pong coevolve --islands
"32,32;64,64;32,32,32;64,64,64" ...` drives it (that four-arch split doubles as the CLI's own
default); islands must number ≥2 and have distinct hidden-layer shapes (`archLabel`, e.g.
`h32-32`), checked before training starts.

**I/O — the existing tools, unchanged.** Each island writes a standard run dir,
`<outDir>/<archLabel>/`, carrying `population.txt`/`hof.txt`/`best.txt`/`status.json`/
`train_log.jsonl` in **`evolve()`'s own log schema** (`fit_peer`←`fitSelf`, `fit_pool`←`fitCross`,
`fit_anchor`←`fitAnchor`, the same `vs_p_classic`/`vs_laggy`/`vs_interceptor`/`vs_held_*`/
`champ_age` keys) — trainview and `./pong ladder` read a co-evolution run with no changes of their
own. The base `--out` dir additionally gets `crosswin.jsonl`, one row per eval generation holding
the full island×island champion-vs-champion margin matrix. `--resume` warm-starts every island
from `<resumeDir>/<label>/population.txt` (+ `hof.txt`) — population continues, generation
numbering restarts at 0, mirroring `evolve()`'s own `--resume` (Decision 11, including the
Decision-18-noted renumbering wart) — and a missing, corrupted, or architecture-mismatched island
sets `resumeFailed` and refuses the WHOLE run rather than silently restarting that one island from
random, extending Decision 22's fail-loud contract per-island. `./pong coevolve` does not
auto-spawn trainview the way `evolve`/`ppo` do (Decision 15) — N per-island dirs don't fit a
single-run dashboard's `--out` contract, so operators watch per-island `status.json`/
`train_log.jsonl` directly, or point trainview at one island dir at a time.

**Measurement plan — no run yet; this records the mechanism, not a result.** Per-island absolute
interceptor win-rates (`vs_p_classic`/`vs_laggy`/`vs_interceptor`, canonical unshaped physics with
the genome's stride, mirroring `evolve()`'s `evalParams` contract from Decision 18; unaffected by
co-evolutionary cycling since the scripted anchors are a fixed external reference) plus the
cross-island win matrix (who is beating whom under co-adaptation) land in `train_log.jsonl`/
`crosswin.jsonl` every `--eval-every` generations. The headline comparison is a final `./pong
ladder --dir models/coevo/h32-32 --dir models/coevo/h64-64 --dir models/coevo/h32-32-32 --dir
models/coevo/h64-64-64` for cross-architecture Elo on the anchor-pinned scale (Decision 12). The
rollout plan runs a 50-generation calibration pass before committing to the full
10,000-generation run, to measure s/gen and decide whether the deferred batched cross-arch path
(below) is worth building first.

**Scope.** Cross-island games are **scalar** in v1 — one match at a time, like today's pool games
— not the batched SoA path; a batched cross-arch kernel (group jobs by `(selfArch, oppArch)`, one
`BatchedMlp` per side) is spec'd but deliberately deferred behind the calibration run's measured
s/gen rather than built speculatively (#25's measure-before-build discipline). Loss shaping (#26)
flows through for free — `--loss-bonus`/`--loss-prox-weight` ride the shared `MatchParams` into
every island's matches — but **defaults off**, so an architecture comparison isn't also an
unrequested fitness-shaping A/B. No migration between islands: architectures stay pure by
construction, so any skill difference traces to capacity and cross-play, never to genome leakage
across architectures.

## Decision 29 — Co-evolution dashboard (`pong_coview`/`./pong coview`) + `coEvolve` control channel

Trainview (#15/#19) is a single-run tailer — it scans `models/*` one level deep and plots one
run's curves. A co-evolution run (#28) is N nested island dirs (`<outDir>/<label>/train_log.jsonl`,
written in `evolve()`'s own schema) plus a base `crosswin.jsonl` island×island matrix — trainview
can't show either, and the whole point of running islands side by side is a per-species comparison
trainview was never built for.

**The comparison metric is the crux.** Self-play fitness is NOT comparable across architectures
under co-adaptation: each island's fitness is scored relative to its OWN opponent mix (self +
cross-island + anchor), so a higher number can just mean that island faced softer opponents that
generation — the Red-Queen effect behind the deep run's fitness 0.88 with interceptor win-rate
~0%, a high score that says nothing about absolute skill. Only performance against the fixed
scripted bots is apples-to-apples across architectures. So coview's headline panel is each
species' **average margin vs the bots** (unshaped), with self-play best fitness plotted separately
and labeled as a foil, not a scoreboard.

**What shipped.** `pong_coview` / `./pong coview [dir]` (default `models/coevo`), a new binary
paralleling `pong_trainview`. `train_plot.hpp` — the `Series` struct and `drawPlot` — is extracted
verbatim out of `train_view.cpp` into a shared header both dashboards include (behavior-preserving;
trainview renders identically). `coview_data.hpp` adds a crosswin-matrix parser
(`parseCrossWinLine`/`readLastCrossWin`) and `avgMarginSeries`, which builds the per-species
"avg margin vs bots" line from each eval row's `marg_*` triple, falling back to `vs_*`
win-rate×100 for logs predating `marg_*` — the mode (margin vs win-rate) is decided ONCE for the
whole series, not row-by-row, so a plot never silently switches units mid-curve. `coEvolve()`
gains `marg_p_classic`/`marg_laggy`/`marg_interceptor` (the same reporting-only anchor-eval pass
that already computed win-rates now also records its raw margin — no extra matches) and a
pause/stop/eval control channel on the base `<outDir>/control`, mirroring `evolve()`'s
`readControl`/`writeStatus` between generations: it finally wires up `CoEvolveResult::stopped`
(present since #28 but never set) and writes a base aggregate `status.json` heartbeat coview
polls. `./pong coevolve` now auto-launches coview (`launch_coview`, display-guarded, the same
pattern as #15's `launch_trainview`) when a display is available.

**Scope.** coview monitors and controls a run; it does not launch one — no recipe/queue, unlike
the Decision-20 GUI launcher. `--resume` keeps coEvolve's existing #11/#28 warm-start (population
continues, generation numbering restarts at 0), so stop→continue is a CLI relaunch, not a
dashboard button. The crosswin heatmap shows only the latest eval generation's matrix, not
history. `evolve()` gains nothing further here — #28's one edit to it stands unchanged; this task
chain only touches `coEvolve()`'s I/O/telemetry plus the two new dashboard files. Determinism:
control is polled between generations only, same contract as #19's, pinned by a pause-invariance
doctest (pause then resume reproduces the run bit-identically); `marg_*` logging reads out of the
existing reporting-only eval pass and feeds nothing back into fitness or selection.
