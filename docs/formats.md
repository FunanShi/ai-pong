# Data formats — the training interfaces

These are **contracts** (see `CLAUDE.md`): checkpoints and recordings outlive code changes,
so breaking a format here invalidates trained models or recorded data. Change them only with
a `DECISIONS.md` entry and a version bump.

## The observation vector

`observation(snapshot, side)` (`src/core/pong_core.hpp`) → 6 doubles
`{ball x, ball y, vx, vy, own paddle y, opp paddle y}` in cu / cu/s, court frame C (y down),
x-mirrored for the right side so **every policy plays left** — symmetry halves the learning
problem. Paddle *velocities* don't exist in this sim (massless, velocity-commanded paddles);
opponent motion is exactly what frame stacking (below) lets a policy infer.

## Model weights — `aipong-mlp` v1 / v2

Whitespace-separated text, loaded by `MlpAgent` (`src/agents/`).

**v2 (current):**

    aipong-mlp 2
    obs stack K [stride S]
    6·K 32 32 3
    <layer 1: weights row-major [n_out × n_in]> <biases [n_out]>
    <layer 2: ...>

`stride S` (optional, default 1): frames are sampled every S ticks — slot f of the input
holds the observation from t − S·(K−1−f); slot K−1 is now. evo4 default: `obs stack 6
stride 6` = a 0.5 s game-time window (the opponent-intent memory). Caps: K ≤ 32, S ≤ 60,
window (K−1)·S+1 ≤ 512 ticks. Fresh episodes pad with the earliest frame. Inference
arithmetic is float32 (Decision 13). Cross-architecture play is fully supported — any
mix of stack/stride models can meet in the GUI, ladder, and matches, since history is
each agent's private machinery.

**v1 (legacy, still loads):** same without the `obs` line; input width fixed at 6 (≡ `obs stack 1`).

- Sizes line: input width **must equal 6·K**, output **must be 3** (logits {0: None, 1: Up, 2: Down};
  argmax wins, ties to the lower index). Hidden activation tanh, linear output. 1 ≤ K ≤ 32.
- **Frame stacking (v2):** the network input is the last K observations concatenated
  **oldest→newest** — slots [0,6) hold frame t−K+1, …, slots [6(K−1), 6K) hold frame t.
  A fresh episode pads by repeating the earliest available frame. History is per-agent
  state: cleared by `Agent::reset()` (the frontend calls it on every new game), and one
  `MlpAgent` instance must stay on one side within a match (the buffer stores
  side-mirrored observations).
- **Reserved:** v3 for recurrent policies — hidden-state weights, same `reset()` lifecycle.

Generate a random one (wiring test, plays badly on purpose):

    ./pong genmodel 42            # → models/random.txt  (stack 1)
    ./pong genmodel 42 2          # → 12-input, stack 2
    AIPONG_MODEL=models/random.txt ./pong

## Recorded matches — `aipong_match` v1 (JSONL)

Written by the frontend's **Record match** button (or any headless match runner)
via the `MatchRecorder` class in `src/record/recorder.hpp`. One file per match:
`datasets/match_<unix_s>_seed<seed>.jsonl`.

Structure (one JSON object per line):

1. **Metadata** — `{"aipong_match":1, "seed":…, "left":"<controller>", "right":"<controller>",
   "dt_s":0.01667, "win_score":11, "rally_max":1000, "speedup":1.03, "ball_speed_max":100.0,
   "started_unix_s":…}`. The sim constants are embedded so files stay interpretable if
   defaults change later.
2. **Tick lines**, one per stepped tick — the **pre-step** state with the actions **applied
   that tick** (the (s_t, a_t) alignment behavior cloning (BC) training would consume —
   a documented, not-yet-built consumer):
   - `t` — tick index (each tick = `dt_s` of sim time)
   - `b` — `[x, y, vx, vy]` ball position (cu) and velocity (cu/s), court frame C (y down)
   - `ly`, `ry` — paddle center heights (cu)
   - `al`, `ar` — actions: 0 = None, 1 = Up, 2 = Down
   - `ph` — phase: `S`erving / `P`laying / game`O`ver
   - `rally` — paddle hits so far this rally
   - `ev` (only when it happens) — `pointL` / `pointR` / `rallycap`, meaning the action on
     THIS line caused that outcome (derived from the post-step state). These are the reward
     markers for future RL / return-weighted training.
3. **Summary** — `{"end":true, "outcome":"LeftWin|RightWin|RallyCap|None", "score":[l,r],
   "ticks":N, "truncated":bool}`. `truncated:true` means the recording was stopped by the
   operator (Stop button, controller/seed change, window close) rather than by the game
   ending — filter on this in training pipelines.

Per-side observation vectors are **not** stored — derive them with
`observation(snapshot, side)` (the mirror transform), so one file serves both paddles'
training views.

## The rally database — `aipong_rally_index` v1 (`datasets/rallies.jsonl`)

The training unit is the **rally** (serve → point/cap/stop): an independent episode with a
binary outcome. The rally-database tool, `rallydb` (invoked as `./pong rallies`), derives a
sortable index over the raw match files — rows point into them (line spans); tick data is
never copied.

    ./pong rallies build                 # scan datasets/*.jsonl -> datasets/rallies.jsonl
    ./pong rallies list --sort hits --top 20
    ./pong rallies list --winner none --end rally_cap
    ./pong rallies list --player "Human" --sort vmax
    ./pong rallies extract match_...jsonl 3        # rally header + its raw tick lines (JSONL)

Row fields: `match` + `rally` (0-based) identify the rally; `left`/`right` = controller
labels; `winner` ∈ left|right|**none** (rally-cap and truncated rallies have no winner —
Decision 4); `end` ∈ `left_missed` | `right_missed` | `rally_cap` | `truncated`
(left_missed = left failed to return ⇒ right won); `lines` = [start,end] 1-based inclusive
span in the source file; `ticks` + `dur_s` (sim seconds; includes the 0.75 s serve hold);
`hits` = paddle contacts; `vmax` = max ball speed reached (cu/s); `served_to` = receiving
side. Rebuild after recording new matches — the index is derived data, safe to delete.

---

`models/`, `datasets/`, and `results/` are all gitignored: weights, recordings, and eval
outputs stay local.
