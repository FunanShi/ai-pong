# Evaluating a trained model

Four tools, each answering a different question. All results are **JSONL on disk** —
nothing is printed-and-discarded except `memuse`/`aiming`'s stdout (paste it into
`DECISIONS.md` or project notes if it's worth keeping; ladder output is kept automatically).

Ladder sweeps land in `results/ladder/<name>/` (pick the name with `--out`; the bare default
is `results/ladder/` itself). Everything under `results/` is derived data — reproducible from
checkpoints + seeds, safe to delete, and nothing is auto-deleted: old sweeps accumulate until
you clean them up. *(The `DECISIONS.md` #17 reorg (2026-07-09) moved old sweeps: they were
written to `models/ladder_<name>`; they now live at `results/ladder/<name>`, with
`ladder_probe_out` → `probe`. The hand-picked probe checkpoint set `models/ladder_probe` — an
input, despite its old prefix — is now `models/probes/`.)*

## "How strong is it, and against what?" — `./pong ladder`

Round-robins every model it's pointed at (plus the built-in scripted bots) on canonical
physics and computes Bradley–Terry Elo pinned to P-controller-classic = 1000:

    ./pong ladder --dir models/evomem_long --per-pair 4 --out results/ladder/evomem --top 10

    # compare across runs / against the current leaderboard in one sweep
    # (--dir must be a directory that already has checkpoints in it — a fresh
    # --out models/evomem from `evolve` won't exist yet until you've run it):
    ./pong ladder --dir models/evomem_long --dir models/leaderboard --per-pair 8 --out results/ladder/check

Prints a ranked table and a `deployment champion: ... -> AIPONG_MODEL=... ./pong` line you can
paste straight into a shell. Every run also writes `<out>/ratings.jsonl` (one line per model:
`spec`, `elo`, `games`, `wins`, `caps`) and `<out>/matches.jsonl` (one line per match) — so you
can re-slice a sweep later without replaying it, e.g. `results/ladder/champions/ratings.jsonl`
is the raw data behind the leaderboard table in `docs/leaderboard.md`.

The Play GUI's model picker shows each curated model's Elo from the `champions` sweep when
it has one (most recent sweep otherwise) — refresh the canonical numbers with
`./pong ladder --dir models/leaderboard --out results/ladder/champions`.

**Comparing co-evolution islands:** each island writes a normal run dir
(`models/coevo/<label>/`), so ladder them together for the apples-to-apples capacity check
the in-training margin view (`./pong coview`) can't give — the fixed-bot Elo of the
base/wide/deep/wide+deep champions side by side:

    ./pong ladder --dir models/coevo/h32-32 --dir models/coevo/h64-64 \
                  --dir models/coevo/h32-32-32 --dir models/coevo/h64-64-64 \
                  --dir models/leaderboard --out results/ladder/coevo

## "What did a finished run *really* do against the anchors?" — `./pong reeval`

Train logs written by stack>1 runs before the eval-stride fix (`DECISIONS.md` #18;
2026-07-09) under-report every memory-using genome: the in-training eval loaded genomes at
stride 1, collapsing their history window (evomem_long logged 0% vs everything while its
checkpoints beat p:classic 100%). Checkpoint *files* always carried the correct stride, so
replaying them recovers the true curves:

    ./pong reeval models/evomem_long            # 62 checkpoints ≈ 17 s
    ./pong reeval models/evo4 12 results/reeval/evo4_quick.jsonl

Plays every `gen_*.txt` against the three reporting anchors + both held-outs on canonical
physics with trainer-mirrored eval seeds (exact for seed-1 runs — reeval hard-codes master
seed 1; other seeds' curves are unbiased but not row-identical); prints a table and writes
`results/reeval/<run>.jsonl` (same `vs_*` keys as `train_log.jsonl`). Logs written after
the fix are trustworthy as-is.

## "Is a memory model actually using its history?" — `./pong memuse`

For `--stack` > 1 models: is the history window earning its keep, or is it dead weight?
`./pong memuse <model.txt> [matches] [opponent]` (default 60 matches vs the instant
Interceptor; pass a beatable opponent like `p:classic` if the model can't contest the
Interceptor yet — see `DECISIONS.md` #16 for why that matters):

    ./pong memuse models/evomem_long/gen_1149.txt 80 p:classic

Reports three independent checks and a verdict: **(A) structural** — weight mass on the history
inputs vs the current frame; **(B) functional** — win rate with real history vs. history
artificially flattened to "no motion" (a drop means memory has value); **(C) behavioral** — the
% of in-match decisions that change when history is ablated. This is how we caught evo4–6's
"memory champions" being a widened memoryless policy with exactly 0% on all three checks.

## "How does it beat the perfect defender — smart aim, or luck?" — `./pong aiming`

`./pong aiming <model.txt> [matches]` plays the model against the instant Interceptor and
measures whether its contact angle correlates with the opponent's position, what fraction of
winning shots are geometrically unreturnable (outside the defender's reachable set), and the
ball-speed/rally-length regime of the kills:

    ./pong aiming models/evo3_5/gen_2974.txt 300
