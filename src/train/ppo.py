#!/usr/bin/env python3
"""PPO self-play trainer for AI Pong (CleanRL-style single file — DECISIONS.md #10).

Actor is EXACTLY the aipong-mlp deployment shape (6-32-32-3, tanh hidden) so exports are
byte-compatible with the C++ ladder; the critic (6-64-64-1) exists only at training time
and is amputated at export. Opponent pool = scripted bots + any mlp: checkpoints found
(GA checkpoints included), resampled per env per episode — the anti-overfitting lesson.

  ./pong ppo --total-steps 3000000            # train
  ./pong ppo --smoke                          # 20k-step wiring check
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build"))
import aipong  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

STACK, STRIDE = 6, 6             # 0.5 s memory window — matches the evo4/evo5 architecture
WINDOW = (STACK - 1) * STRIDE + 1
ACTOR_SIZES = [6 * STACK, 32, 32, 3]   # deployment contract — must match aipong-mlp export


class FrameStack:
    """Python-side mirror of MlpAgent's strided history: slot f holds the obs from
    t − STRIDE·(STACK−1−f); fresh episodes pad with their earliest frame."""
    def __init__(self, n):
        self.buf = np.zeros((n, WINDOW, 6), dtype=np.float32)
        self.count = np.zeros(n, dtype=np.int64)

    def reset_env(self, i):
        self.count[i] = 0

    def push(self, obs):
        n = self.buf.shape[0]
        idx = self.count % WINDOW
        self.buf[np.arange(n), idx] = obs
        self.count += 1
        newest = self.count - 1
        oldest = np.maximum(0, self.count - WINDOW)
        out = np.empty((n, 6 * STACK), dtype=np.float32)
        for f in range(STACK):
            logical = np.maximum(newest - (STACK - 1 - f) * STRIDE, oldest)
            out[:, f * 6:(f + 1) * 6] = self.buf[np.arange(n), logical % WINDOW]
        return out


def mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


def export_actor(actor, path, sizes=ACTOR_SIZES):
    """Write aipong-mlp v2 with the stack/stride header — C++ MlpAgent reproduces the
    FrameStack semantics, so exports play identically in the ladder and GUI.
    torch Linear.weight is [out,in] row-major — exactly the format's layout."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lin = [m for m in actor.modules() if isinstance(m, nn.Linear)]
    assert [l.in_features for l in lin] == sizes[:-1] and [l.out_features for l in lin] == sizes[1:]
    with open(path, "w") as f:
        f.write(f"aipong-mlp 2\nobs stack {STACK} stride {STRIDE}\n")
        f.write(" ".join(str(s) for s in sizes) + "\n")
        for l in lin:
            w = l.weight.detach().numpy().reshape(-1)
            b = l.bias.detach().numpy().reshape(-1)
            f.write(" ".join(f"{x:.9g}" for x in w) + "\n")
            f.write(" ".join(f"{x:.9g}" for x in b) + "\n")


def warm_start(actor, path):
    """Function-preserving init from an aipong-mlp export (stack 1 or STACK): old input
    weights -> newest-frame columns, zeros elsewhere, deeper layers verbatim — the actor
    plays identically to the source at init (the widenGenome trick, torch-side).
    Assumes the project's 4-entry sizes line. The critic stays fresh: expect a few noisy
    early updates while value estimates calibrate (clip bounds the policy damage)."""
    toks = open(path).read().split()
    assert toks[0] == "aipong-mlp"
    i = 2
    if int(toks[1]) == 2:
        assert toks[i] == "obs" and toks[i + 1] == "stack"
        i += 3
        if toks[i] == "stride":
            i += 2
    sizes = [int(t) for t in toks[i:i + 4]]; i += 4
    lin = [m for m in actor.modules() if isinstance(m, nn.Linear)]
    assert sizes[1:] == [l.out_features for l in lin], "hidden/output shape mismatch"
    assert sizes[0] in (6, 6 * STACK), "source must be stack-1 or same-stack"
    with torch.no_grad():
        for li, l in enumerate(lin):
            n_in = sizes[li] if li > 0 else sizes[0]
            n_out = sizes[li + 1]
            W = torch.tensor([float(t) for t in toks[i:i + n_out * n_in]]).reshape(n_out, n_in)
            i += n_out * n_in
            b = torch.tensor([float(t) for t in toks[i:i + n_out]])
            i += n_out
            if li == 0:
                l.weight.zero_()
                l.weight[:, -n_in:] = W          # newest-frame slot = last 6 columns
            else:
                l.weight.copy_(W)
            l.bias.copy_(b)
    print(f"warm-started actor from {path} (input {sizes[0]} -> {6 * STACK}, function-preserving)")


def evaluate(actor, spec, n=24, seed=1234, max_ticks=2_000_000):
    """Greedy (argmax — deployment semantics) win rate of the actor vs one anchor spec."""
    env = aipong.VecPong(n, seed, spec)
    ret = np.zeros(n); done_once = np.zeros(n, dtype=bool); wins = 0
    fs = FrameStack(n)
    x = fs.push(env.observations())
    for _ in range(max_ticks // n + 1):
        with torch.no_grad():
            act = actor(torch.as_tensor(x)).argmax(dim=1).numpy().astype(np.int32)
        obs, rew, done = env.step(act)
        for i in np.nonzero(done)[0]:
            fs.reset_env(int(i))                 # auto-reset env: next push starts the new episode
        x = fs.push(obs)
        ret += np.where(done_once, 0.0, rew)
        newly = done & ~done_once
        wins += int(np.sum(ret[newly] > 0))
        done_once |= done
        if done_once.all():
            break
    return wins / n


def analysis_block(out, upd, n_updates, gstep, total, sps, recent, eval_hist):
    """Formatted analysis at eval checkpoints — sibling of the evolve tool's block."""
    eta_min = (total - gstep) / max(sps, 1) / 60.0
    e = eval_hist[-1]
    def last5(key):
        return " ".join(f"{100 * h[key]:3.0f}" for h in eval_hist[-5:])
    def best(key):
        i = max(range(len(eval_hist)), key=lambda j: eval_hist[j][key])
        return f"{100 * eval_hist[i][key]:3.0f}% @u{eval_hist[i]['update']}"
    avg = float(np.mean(recent)) if recent else float("nan")
    print(f"+---- {out} | update {upd}/{n_updates} | step {gstep:,}/{total:,} | ETA {eta_min:.1f}m")
    print(f"| return   mean(last {len(recent)} ep) {avg:+.2f}          (stochastic, training pool)")
    print(f"| anchors  p:classic {100 * e['vs_p_classic']:3.0f}%   laggy {100 * e['vs_laggy']:3.0f}%"
          f"   instant {100 * e['vs_interceptor']:3.0f}%     (greedy, canonical)")
    print(f"| laggy    last: {last5('vs_laggy')}      best {best('vs_laggy')}")
    print(f"| instant  last: {last5('vs_interceptor')}      best {best('vs_interceptor')}")
    print(f"| pace     {sps:,} steps/s")
    print("+" + "-" * 69, flush=True)


def read_control(out):
    """Operator control channel (DECISIONS.md #19): first token of <out>/control.
    pause|run|stop|eval; absent/unreadable/unrecognized -> run."""
    try:
        with open(os.path.join(out, "control")) as f:
            tok = f.read().split()
        return tok[0] if tok and tok[0] in ("pause", "stop", "eval") else "run"
    except OSError:
        return "run"


def write_status(out, state, done, total):
    """Heartbeat for dashboards: <out>/status.json. Same keys as the GA trainer —
    gen/gens_total carry step/total_steps here (generic work units for the ETA)."""
    if not out:
        return
    try:
        with open(os.path.join(out, "status.json"), "w") as f:
            f.write(json.dumps({"state": state, "gen": done, "gens_total": total,
                                "ts_unix": int(time.time())}) + "\n")
    except OSError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--rollout", type=int, default=256)          # steps per env per update
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pool", type=str, default="")              # comma specs; "" = auto
    p.add_argument("--out", type=str, default="models/ppo")
    p.add_argument("--eval-every", type=int, default=20)        # updates between anchor evals
    p.add_argument("--eval-matches", type=int, default=24)
    p.add_argument("--no-curriculum", action="store_true")     # uniform opponent sampling
    p.add_argument("--init-model", type=str, default="")       # warm-start actor from an export
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.envs, args.rollout, args.total_steps = 8, 128, 20_000
        args.eval_every, args.eval_matches = 8, 4   # exercise the analysis block too

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.pool:
        pool = args.pool.split(",")
    else:
        # Curriculum ladder: scripted tiers + archive models spanning ~1100 → 1550 elo
        # (ladder-rated; see results/ladder/final + results/ladder/evo5). Held-out variants stay out.
        pool = ["p:easy", "p:classic", "p:hard", "interceptor:laggy", "interceptor"]
        for f in ["models/evo2/gen_0549.txt",   # ~1104: soft archive rung
                  "models/evo2/gen_1474.txt",   # ~1331
                  "models/evo3/gen_0699.txt",   # ~1454
                  "models/evo3_5/gen_2974.txt", # ~1576-class
                  "models/evo4/gen_0574.txt",   # old champion
                  "models/evo5/gen_1249.txt"]:  # reigning champion
            if os.path.exists(f):
                pool.append(f"mlp:{f}")
    print(f"opponent pool: {pool}")

    # Adaptive curriculum (--no-curriculum → uniform): per-opponent win-rate EMA; sampling
    # favors the learning frontier (EMA in [0.15, 0.85]) with a floor so nothing starves.
    pool_ema = np.full(len(pool), 0.5)
    env_opp = np.zeros(args.envs, dtype=np.int64)

    def sample_opp():
        if args.no_curriculum:
            return int(rng.integers(len(pool)))
        w = np.where((pool_ema >= 0.15) & (pool_ema <= 0.85), 1.0, 0.15)
        return int(rng.choice(len(pool), p=w / w.sum()))

    env = aipong.VecPong(args.envs, args.seed, pool[0])
    for i in range(args.envs):
        env_opp[i] = sample_opp()
        env.set_opponent(i, pool[env_opp[i]])
    env.reset_all()

    actor = mlp(ACTOR_SIZES)
    critic = mlp([6 * STACK, 64, 64, 1])
    if args.init_model:
        warm_start(actor, args.init_model)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    n_updates = args.total_steps // (args.envs * args.rollout)
    fstack = FrameStack(args.envs)
    obs_t = torch.as_tensor(fstack.push(env.observations()))
    ep_ret = np.zeros(args.envs); recent = []
    logf = None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        try:                                             # stale-state guard: never start paused
            os.remove(os.path.join(args.out, "control"))
        except OSError:
            pass
        with open(os.path.join(args.out, "config.json"), "w") as f:
            f.write(json.dumps({"trainer": "ppo", "argv": " ".join(sys.argv[1:]),
                                "config": {k: v for k, v in vars(args).items()},
                                "started_unix": int(time.time())}) + "\n")
        logf = open(os.path.join(args.out, "train_log.jsonl"), "a")
        write_status(args.out, "running", 0, args.total_steps)
    t0 = time.time(); gstep = 0
    eval_hist = []
    stopped = False

    for upd in range(1, n_updates + 1):
        # ---- operator control (DECISIONS.md #19): checked between updates only ----
        force_eval = False
        if args.out:
            ctl = read_control(args.out)
            if ctl == "pause":                           # park: cores released, torch pool idle
                write_status(args.out, "paused", gstep, args.total_steps)
                while read_control(args.out) == "pause":
                    time.sleep(0.5)
                ctl = read_control(args.out)
                write_status(args.out, "running", gstep, args.total_steps)
            if ctl == "stop":                            # graceful: export, mark, leave
                export_actor(actor, os.path.join(args.out, "latest.txt"))
                write_status(args.out, "stopped", gstep, args.total_steps)
                print("control: stop — exported latest.txt", flush=True)
                stopped = True
                break
            if ctl == "eval":                            # one-shot forced eval, self-clears
                force_eval = True
                with open(os.path.join(args.out, "control"), "w") as f:
                    f.write("run")
        O = torch.zeros(args.rollout, args.envs, 6 * STACK)
        A = torch.zeros(args.rollout, args.envs, dtype=torch.long)
        LP = torch.zeros(args.rollout, args.envs)
        R = torch.zeros(args.rollout, args.envs)
        D = torch.zeros(args.rollout, args.envs)
        V = torch.zeros(args.rollout, args.envs)

        for t in range(args.rollout):
            O[t] = obs_t
            with torch.no_grad():
                logits = actor(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                LP[t] = dist.log_prob(a)
                V[t] = critic(obs_t).squeeze(-1)
            A[t] = a
            obs, rew, done = env.step(a.numpy().astype(np.int32))
            R[t] = torch.as_tensor(rew)
            D[t] = torch.as_tensor(done, dtype=torch.float32)
            gstep += args.envs
            ep_ret += rew
            for i in np.nonzero(done)[0]:
                i = int(i)
                recent.append(ep_ret[i])
                j = env_opp[i]                       # curriculum bookkeeping for the ended episode
                pool_ema[j] = 0.9 * pool_ema[j] + 0.1 * (1.0 if ep_ret[i] > 0 else 0.0)
                ep_ret[i] = 0.0
                env_opp[i] = sample_opp()
                env.set_opponent(i, pool[env_opp[i]])  # applies next episode
                fstack.reset_env(i)                  # step already returned the new episode's obs
            obs_t = torch.as_tensor(fstack.push(obs))
            recent = recent[-200:]

        # GAE (auto-reset envs: bootstrap 0 on done — the returned obs is the new episode)
        with torch.no_grad():
            next_v = critic(obs_t).squeeze(-1)
            adv = torch.zeros_like(R); lastgae = torch.zeros(args.envs)
            for t in reversed(range(args.rollout)):
                nonterm = 1.0 - D[t]
                nv = next_v if t == args.rollout - 1 else V[t + 1]
                delta = R[t] + args.gamma * nv * nonterm - V[t]
                lastgae = delta + args.gamma * args.gae_lambda * nonterm * lastgae
                adv[t] = lastgae
            ret = adv + V

        b_obs = O.reshape(-1, 6 * STACK); b_act = A.reshape(-1); b_lp = LP.reshape(-1)
        b_adv = adv.reshape(-1); b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_obs.shape[0]; mb = n // args.minibatches
        idx = np.arange(n)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, mb):
                j = torch.as_tensor(idx[s:s + mb])
                logits = actor(b_obs[j])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(b_act[j])
                ratio = (lp - b_lp[j]).exp()
                l1 = -b_adv[j] * ratio
                l2 = -b_adv[j] * ratio.clamp(1 - args.clip, 1 + args.clip)
                pol_loss = torch.max(l1, l2).mean()
                v = critic(b_obs[j]).squeeze(-1)
                v_loss = 0.5 * (v - b_ret[j]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + args.vf_coef * v_loss - args.ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()),
                                         args.max_grad_norm)
                opt.step()

        sps = int(gstep / (time.time() - t0))
        avg = float(np.mean(recent)) if recent else float("nan")
        line = {"update": upd, "step": gstep, "ep_return_mean": avg, "sps": sps,
                "pool_ema": [round(float(e), 2) for e in pool_ema]}
        did_eval = False
        if force_eval or (args.eval_every and upd % args.eval_every == 0):
            for name, spec in [("vs_p_classic", "p:classic"),
                               ("vs_laggy", "interceptor:laggy"),
                               ("vs_interceptor", "interceptor")]:
                line[name] = evaluate(actor, spec, n=args.eval_matches, seed=10_000 + upd)
            export_actor(actor, os.path.join(args.out, f"ckpt_upd{upd:05d}.txt"))
            export_actor(actor, os.path.join(args.out, "latest.txt"))
            eval_hist.append(line)
            did_eval = True
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in line.items() if not isinstance(v, list)), flush=True)
        if did_eval:
            analysis_block(args.out, upd, n_updates, gstep, args.total_steps, sps,
                           recent, eval_hist)
        if logf:
            logf.write(json.dumps(line) + "\n"); logf.flush()
        write_status(args.out, "running", gstep, args.total_steps)

    if args.out and not stopped:
        export_actor(actor, os.path.join(args.out, "latest.txt"))
        write_status(args.out, "finished", gstep, args.total_steps)
        print(f"done: exported {args.out}/latest.txt "
              f"(play it: AIPONG_MODEL={args.out}/latest.txt ./pong)")


if __name__ == "__main__":
    main()
