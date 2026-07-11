# Training — GA self-play, PPO, and island co-evolution

Every named run is a checked-in recipe file — `./pong train` lists them.

Three tracks, one model format (`docs/formats.md`), one ladder: everything any trainer
exports competes head-to-head in `./pong ladder` (`docs/evaluation.md`). Evolution trains a
genetic algorithm (GA) via self-play; PPO trains with Proximal Policy Optimization (PPO), a
policy-gradient algorithm; island co-evolution is a GA variant that runs several network
architectures as separate populations, cross-playing each other to compare capacities under
co-adaptation. Current standings: `docs/leaderboard.md`. Method and rationale per run:
`DECISIONS.md` #10–#16.

Either trainer **auto-launches the live dashboard** (`./pong trainview`) — scoring metric +
win-% vs the scripted bots, updating as it runs. It tails `models/<run>/train_log.jsonl`,
so it can watch a container mid-training; skipped cleanly with no display or `AIPONG_NO_GUI=1`.
*Caveat:* anchor/held-out columns logged by stack>1 runs **before 2026-07-09** are invalid
(eval-stride bug, `DECISIONS.md` #18) — recover the true curves with `./pong reeval <run-dir>`;
the dashboard overlays recovered curves automatically (faint lines) when
`results/reeval/<run>.jsonl` exists.

## Evolution track (C++ GA)

GA self-play — pure self-play fitness (peers + hall of fame), scripted anchors for progress
reporting only, deterministic for any thread count (recipe + rationale: `DECISIONS.md` #10):

    ./pong train evo-basic                   # recipes/evo-basic.recipe: --gens 300 --seed 1
    AIPONG_MODEL=models/evo/best.txt ./pong  # then play the result

Checkpoints land in `models/evo/` (`gen_*.txt`, `best.txt`, `train_log.jsonl`).

### All the bells and whistles — the `evomem` recipe

0.5 s memory + opponent-diversity fitness + training-economics knobs (`DECISIONS.md`
#14/#16):

- **Random-init memory** — no `--init-from`, so there's no memoryless fallback for mutation
  to collapse into (Decision 16's fix).
- **A curriculum pool of frozen memoryless champions**, so memory only pays if it beats what
  memorylessness already can.
- **Training-time speedups** (rally cap, faster per-hit speedup, no serve hold, first-to-5,
  win-speed shaping) that only apply during training — anchor evals always replay canonical
  physics.

    ./pong train evomem                        # the full recipe: recipes/evomem.recipe
    ./pong train evomem --seed 2 --out models/evomem_s2     # overrides append, last-wins

Run it in **long continuous stretches, not short re-founded segments** — segmenting (ladder →
re-found from the top-8 every 500 gens) narrows the population back to 8 seeds each cycle and
stalls progress; a single continuous run reaches a higher ladder Elo than restarting from the
top-8 each segment (`DECISIONS.md` #18).

To resume an interrupted run (keeps the full population, no re-founding):

**Warning:** a resumed run restarts generation numbering at 0, so a same-`--out` resume
overwrites the original segment's `gen_*.txt` and appends log rows that restart at gen 0; to
keep the first segment's artifacts, resume into a fresh `--out` (`DECISIONS.md` #18).

    ./pong train evomem --resume models/evomem_long --out models/evomem_long2

Check whether a stack>1 champion is actually *using* its memory window (structural weight
mass, functional win-rate drop, and behavioral decision-divergence when history is ablated —
`DECISIONS.md` #16, tool guide in `docs/evaluation.md`):

    ./pong memuse models/evomem/gen_XXXX.txt 80 p:classic

## PPO track (Python + pybind11)

pybind11 `VecPong` bindings + CleanRL-style PPO (`src/train/ppo.py`), actor pinned to the
deployment shape, opponent pool = scripted bots + GA checkpoints, exports to the same model
format — both tracks compete on the same ladder:

    ./pong pytest                            # bindings smoke tests
    ./pong ppo --smoke                       # 20k-step wiring check
    ./pong train ppo-scratch                 # real run → models/ppo/latest.txt (recipes/ppo-scratch.recipe)

### All the bells and whistles — memory + curriculum + warm start

0.5 s memory + adaptive curriculum + warm-start from the best prior checkpoint:

- **Memory** — `src/train/ppo.py`'s `FrameStack` mirrors `MlpAgent`'s stack-6/stride-6 window
  in Python.
- **Adaptive curriculum** — the default opponent pool is a difficulty ladder (scripted tiers +
  archive models spanning evo2-mid → the reigning champion) sampled by a per-opponent
  win-rate EMA that favors the current learning frontier.
- **Warm start** — `--init-model` does a function-preserving warm-start (old weights land in
  the newest-frame slot, zeros elsewhere — the actor plays identically to the source at step
  0, then keeps training under the new memory + curriculum).

    ./pong train ppo-warmstart                 # the full recipe: recipes/ppo-warmstart.recipe
    ./pong train ppo-warmstart --seed 2 --out models/ppo3_s2     # overrides append, last-wins

Drop `--init-model` for a from-scratch run with the same memory + curriculum; add
`--no-curriculum` to fall back to uniform opponent sampling (ablation baseline) and
`--pool "..."` (comma-separated specs) to override the default ladder entirely.

## Island co-evolution — the capacity comparison (`DECISIONS.md` #28/#29)

To ask *which network capacity wins* under co-adaptation, run several architectures as
separate GA demes that co-evolve against each other. Each island is a single-arch GA
population; each generation every genome plays its own peers + hall of fame, a rotation of the
*other* islands' frozen elites (cross-play), and the scripted anchors. Selection is
per-island, so the demes never homogenize — and the run stays **bit-identical for any thread
count**, like the single-species trainer.

    ./pong coevolve --islands "32,32;64,64;32,32,32;64,64,64" --gens 10000 --seed 1
    # four demes (base / wide / deep / wide+deep) → models/coevo/<label>/

`./pong coview` (auto-launched, like trainview) makes the comparison legible: **per-species
average margin vs the scripted bots** is the headline — self-play fitness is *not* comparable
across architectures under co-adaptation (each island's fitness is relative to its own
opponent mix, the Red-Queen effect), so it is shown only as a labeled foil — alongside the
live cross-island win matrix. The same **Pause / Stop / Eval** buttons drive the run via
`models/coevo/control`; Stop checkpoints every island, and continuing after a stop is a
`./pong coevolve --resume …` relaunch.

## Controlling a running trainer (`DECISIONS.md` #19)

Both trainers poll `models/<run>/control` between generations (GA) / updates (PPO) — a
one-token file the dashboard's **Pause / Resume / Stop / Eval now** buttons write, or any
shell can:

    echo pause > models/evomem_long2/control   # park it: cores released, ~1 gen latency
    echo run   > models/evomem_long2/control   # resume exactly where it parked
    echo eval  > models/evomem_long2/control   # force one anchor eval + checkpoint (one-shot)
    echo stop  > models/evomem_long2/control   # graceful: GA saves population.txt/hof.txt
                                               # (lossless --resume state); PPO exports latest.txt

Pausing draws no RNG and reorders nothing — results are **bitwise identical for any pause
pattern** (pinned by test). A stale control file is deleted at startup, so a run can never
begin paused. The trainer heartbeats `models/<run>/status.json`
(`state`/`gen`/`gens_total`/`ts_unix`) each generation — the dashboard's badge, stall
detection, and ETA read it. Prefer pause over kill+`--resume` when you just need the
machine back: it also sidesteps the resume renumbering wart above.

GA log rows now carry curriculum telemetry: `fit_peer` / `fit_pool` / `fit_anchor`
(population-mean shaped margin per opponent class — watch when the pool game starts giving
gradient) and `champ_age` (generations since the best genome changed). The dashboard plots
the classes as a third panel.

## After a run

Rate the checkpoints and pick a deployment champion (`docs/evaluation.md`):

    ./pong ladder --dir models/<run> --dir models/leaderboard

`models/<run>/` directories are an **append-only namespace**: their paths are baked into
`src/train/ppo.py`'s default pool, old ladder provenance JSONL, and the leaderboard —
add new runs freely, never rename or move old ones.
