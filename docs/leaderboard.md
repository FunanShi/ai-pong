# Strongest models — leaderboard

Distinct policies rated head-to-head on **canonical physics** (round-robin, 80 games each,
Bradley–Terry Elo pinned to P-controller-classic = 1000). Scripted bots interleaved as fixed
rulers. **One policy — evo3.5's — dominates, and it beats the perfect-prediction Interceptor.**

| rank | model | Elo | notes |
|---:|---|---:|---|
| 1 | `models/evo3_5/gen_2974.txt` | **2014** | evo3.5 — GA self-play, **memoryless**. Beats the perfect defender 95% over 300 matches (`DECISIONS.md` #16 aiming study). The project's strongest policy. |
| · | *Interceptor (instant)* | *1843* | *perfect-prediction bot — the bar #1 clears* |
| 2 | `models/evo3/gen_0699.txt` | 1578 | evo3 — pre-memory GA |
| · | *Interceptor (laggy)* | *1500* | |
| 3 | `models/evo2/gen_1549.txt` | 1413 | evo2 — the first long/rated GA run |
| 4 | `models/ppo2/ckpt_upd00400.txt` | 1278 | best PPO (memory + curriculum) — undertrained vs the GA runs |
| · | *P-controller hard / classic / easy* | *1226 / 1000 / 761* | *tracking baselines; classic pins the scale* |

Play the strongest:

    AIPONG_MODEL=models/evo3_5/gen_2974.txt ./pong

**The memory experiment's verdict (evo4–evo6).** Those runs' ladder-champions — `evo4/gen_0574`,
`evo5/gen_1249`, `evo6/gen_0074` — are *byte-identical to one another and behaviorally identical
to `evo3.5/gen_2974`*: the widened evo3.5 policy carrying **exactly zero weight on its history
inputs** (verified — `DECISIONS.md` #16). Elitism carried that memoryless founder through all three
memory runs unchanged; no memory-using genome ever beat it — function-preserving widening drops
the founder into a fitness valley that mutation never crossed. So the 0.5 s memory window never
engaged, and the project's ceiling is a *memoryless* policy. Absolute Elo is pool-relative; the
ordering and the "beats the Interceptor" line are the robust facts.

**evomem follow-up** (from-scratch, random-init memory — the Decision-16 fix): a genuine
memory-using policy is now reachable (`./pong memuse <model> 80 p:classic` confirms real
history dependence, unlike evo4–6). A single continuous run reached a higher ladder Elo than
restarting from the top-8 each segment (`DECISIONS.md` #18); recipes in `docs/training.md`.

Re-rate the table: `./pong ladder --dir models/leaderboard` (the curated champion weights live
in `models/leaderboard/`; raw data behind this table: `results/ladder/champions/ratings.jsonl`).
Reproduce the runs themselves: `docs/training.md`. Method per run: `DECISIONS.md` #10–#16.
