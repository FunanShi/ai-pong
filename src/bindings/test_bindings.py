#!/usr/bin/env python3
"""Smoke tests for the aipong bindings — run inside the container:
   python3 src/bindings/test_bindings.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build"))
import aipong
import numpy as np

assert aipong.OBS_DIM == 6 and aipong.N_ACTIONS == 3

# shapes + value sanity
v = aipong.VecPong(4, 3, "p:easy")
obs = v.observations()
assert obs.shape == (4, 6), obs.shape
assert (obs[:, 0] >= 0).all() and (obs[:, 0] <= 1).all(), "ball x out of court"
obs2, rew, done = v.step(np.zeros(4, dtype=np.int32))
assert obs2.shape == (4, 6) and rew.shape == (4,) and done.shape == (4,)

# determinism: same seed + same action script → identical trajectories
def run(seed, n_steps=600):
    e = aipong.VecPong(2, seed, "p:classic")
    acc = []
    for t in range(n_steps):
        a = np.full(2, t % 3, dtype=np.int32)
        o, r, d = e.step(a)
        acc.append(o.copy())
    return np.concatenate(acc)
a, b = run(5), run(5)
assert np.array_equal(a, b), "same seed diverged"
assert not np.array_equal(run(5), run(6)), "different seeds identical"

# an idle policy eventually concedes points: negative reward + done arrive
v = aipong.VecPong(2, 7, "p:classic")
saw_neg, saw_done = False, False
for _ in range(200_000):
    o, r, d = v.step(np.zeros(2, dtype=np.int32))
    saw_neg |= bool((r < 0).any())
    saw_done |= bool(d.any())
    if saw_neg and saw_done:
        break
assert saw_neg and saw_done, "no scoring events observed"

# opponent swap machinery + mlp spec loading
v.set_opponent(0, "interceptor:laggy")
v.reset_all()
assert v.opponent(0) == "interceptor:laggy"
assert v.opponent_loaded(0)
v.set_opponent(1, "mlp:/nonexistent/file.txt")
v.reset_all()
assert not v.opponent_loaded(1), "bogus model should not load"

print("bindings OK: shapes, determinism, scoring events, opponent swaps")
