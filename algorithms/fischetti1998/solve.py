import math
import time

T0 = time.time()

import numpy as np

_BIG = np.int64(1 << 60)


def _parse(instance):
    n = int(instance["n"])
    d0 = int(instance["depot"]) - 1
    t0 = int(instance["t0"])
    D = np.zeros((n, n), dtype=np.int64)
    items = instance["travel_times"].items()
    keys = []
    vals = []
    for k, v in items:
        keys.append(k)
        vals.append(v)
    ij = np.array([[int(a) for a in k.strip("() ").split(",")] for k in keys], dtype=np.int64).reshape(-1, 2) - 1
    vals = np.array(vals, dtype=np.int64)
    I, J = ij[:, 0], ij[:, 1]
    D[I, J] = vals
    D[J, I] = vals
    p = np.zeros(n, dtype=np.int64)
    for k, v in instance["prizes"].items():
        idx = int(k) - 1
        if 0 <= idx < n and idx != d0:
            p[idx] = int(v)
    return n, d0, t0, D, p


def _length(T, D):
    return int(D[T, np.roll(T, -1)].sum())


def _prize(T, p):
    return int(p[T].sum())


def _fill(T, D, p, t0, rng=None):
    """Greedily insert unvisited cities (ratio prize / insertion cost) while the budget allows."""
    n = D.shape[0]
    L = _length(T, D)
    while True:
        inT = np.zeros(n, dtype=bool)
        inT[T] = True
        cand = np.flatnonzero(~inT & (p > 0))
        if cand.size == 0:
            break
        Tn = np.roll(T, -1)
        base = D[T, Tn]
        C = D[np.ix_(T, cand)] + D[np.ix_(cand, Tn)].T - base[:, None]
        pos = C.argmin(axis=0)
        ins = C[pos, np.arange(cand.size)]
        feas = (L + ins) <= t0
        if not feas.any():
            break
        w = p[cand] / np.maximum(ins, 1).astype(np.float64)
        if rng is not None:
            w = w * (0.8 + 0.4 * rng.random(cand.size))
        w = np.where(feas, w, -1.0)
        q = int(np.argmax(w))
        T = np.insert(T, int(pos[q]) + 1, int(cand[q]))
        L += int(ins[q])
    return T, L


def _initial(n, d0, t0, D, p):
    cand = np.array([i for i in range(n) if i != d0], dtype=np.int64)
    k = cand.size
    out = D[d0, cand]
    mid = D[np.ix_(cand, cand)]
    L2 = out[:, None] + mid + out[None, :]
    ps = p[cand][:, None] + p[cand][None, :]
    iu = np.triu(np.ones((k, k), dtype=bool), 1)
    feas = iu & (L2 <= t0)
    if feas.any():
        best_p = int(np.where(feas, ps, -1).max())
        sub = feas & (ps == best_p)
        Lm = np.where(sub, L2, _BIG)
    else:
        Lm = np.where(iu, L2, _BIG)
    a, b = np.unravel_index(int(np.argmin(Lm)), Lm.shape)
    T = np.array([d0, cand[a], cand[b]], dtype=np.int64)
    return T


def _two_opt(T, D):
    """Best-improvement 2-opt on the cyclic tour; position 0 (depot) stays fixed."""
    while True:
        m = len(T)
        if m < 4:
            return T
        Tn = np.roll(T, -1)
        M = D[np.ix_(T, T)] + D[np.ix_(Tn, Tn)]
        e = D[T, Tn]
        delta = M - e[:, None] - e[None, :]
        mask = np.triu(np.ones((m, m), dtype=bool), 2)
        mask[0, m - 1] = False
        delta = np.where(mask, delta, 0)
        k = int(np.argmin(delta))
        i, j = divmod(k, m)
        if delta[i, j] >= 0:
            return T
        T = T.copy()
        T[i + 1 : j + 1] = T[i + 1 : j + 1][::-1]


def _or_opt(T, D):
    """Best-improvement Or-opt: move a segment of 1..3 cities (not the depot), optionally reversed."""
    while True:
        m = len(T)
        if m < 5:
            return T
        Tn = np.roll(T, -1)
        Eab = D[T, Tn]
        best = (0, None)
        jj = np.arange(m)[None, :]
        for l in (1, 2, 3):
            if m - l < 3:
                break
            starts = np.arange(1, m - l + 1)
            kk = starts.size
            s0 = T[starts]
            s1 = T[starts + l - 1]
            pp = T[starts - 1]
            nn = T[(starts + l) % m]
            R = D[pp, s0] + D[s1, nn] - D[pp, nn]
            fwd = D[np.ix_(T, s0)].T + D[np.ix_(s1, Tn)] - Eab[None, :]
            rev = D[np.ix_(T, s1)].T + D[np.ix_(s0, Tn)] - Eab[None, :]
            st = starts[:, None]
            invalid = (jj >= st - 1) & (jj <= st + l - 1)
            best_ins = np.minimum(fwd, rev)
            best_ins = np.where(invalid, _BIG, best_ins)
            j_best = best_ins.argmin(axis=1)
            cost = best_ins[np.arange(kk), j_best]
            gain = R - cost
            q = int(np.argmax(gain))
            if gain[q] > best[0]:
                best = (int(gain[q]), (int(starts[q]), l, int(j_best[q]), bool(rev[q, j_best[q]] < fwd[q, j_best[q]])))
        if best[1] is None:
            return T
        i, l, j, reverse = best[1]
        seg = T[i : i + l]
        if reverse:
            seg = seg[::-1]
        rem = np.concatenate([T[:i], T[i + l :]])
        pos_a = j if j < i else j - l
        T = np.concatenate([rem[: pos_a + 1], seg, rem[pos_a + 1 :]])


def _local_search(T, D):
    while True:
        L0 = _length(T, D)
        T = _two_opt(T, D)
        T = _or_opt(T, D)
        if _length(T, D) >= L0:
            return T


def _perturb_remove(T, rng):
    m = len(T)
    kmax = min(max(1, int(0.15 * m)), m - 3)
    if kmax < 1:
        return None
    k = int(rng.integers(1, kmax + 1))
    keep = np.ones(m, dtype=bool)
    if rng.random() < 0.5:
        s = int(rng.integers(1, m - k + 1))
        keep[s : s + k] = False
    else:
        keep[rng.choice(np.arange(1, m), size=k, replace=False)] = False
    return T[keep]


def _perturb_swap(T, D, p, t0, rng):
    """Force-insert an unvisited city (prize-weighted), then drop low-value cities until feasible."""
    n = D.shape[0]
    inT = np.zeros(n, dtype=bool)
    inT[T] = True
    cands = np.flatnonzero(~inT & (p > 0))
    if cands.size == 0:
        return None
    w = p[cands].astype(np.float64)
    c = int(rng.choice(cands, p=w / w.sum()))
    Tn = np.roll(T, -1)
    ins = D[T, c] + D[c, Tn] - D[T, Tn]
    pos = int(np.argmin(ins))
    T = np.insert(T, pos + 1, c)
    L = _length(T, D)
    while L > t0:
        m = len(T)
        if m <= 3:
            return None
        Tp = np.roll(T, 1)
        Tn = np.roll(T, -1)
        S = D[Tp, T] + D[T, Tn] - D[Tp, Tn]
        score = (p[T] + 1.0) / (np.maximum(S, 0) + 1.0)
        score = score * (0.7 + 0.6 * rng.random(m))
        score[0] = np.inf
        score[T == c] = np.inf
        k = int(np.argmin(score))
        T = np.delete(T, k)
        L = _length(T, D)
    return T


def _ils(state, D, p, t0, deadline, rng):
    cur, curP, curL = state["best"]
    start = time.time()
    span = max(deadline - start, 1e-6)
    pos_p = p[p > 0]
    mean_p = float(pos_p.mean()) if pos_p.size else 1.0
    t_hi = 0.2 * mean_p
    t_lo = 0.01
    stall = 0
    while True:
        now = time.time()
        if now >= deadline:
            break
        frac = min(1.0, (now - start) / span)
        temp = t_hi * (1.0 - frac) + t_lo * frac
        if rng.random() < 0.5:
            T2 = _perturb_remove(cur, rng)
        else:
            T2 = _perturb_swap(cur, D, p, t0, rng)
        if T2 is None:
            continue
        T2 = _local_search(T2, D)
        T2, L2 = _fill(T2, D, p, t0, rng)
        T2 = _local_search(T2, D)
        L2 = _length(T2, D)
        if L2 > t0 or len(T2) < 3:
            continue
        P2 = _prize(T2, p)
        accept = P2 > curP or (P2 == curP and L2 <= curL)
        if not accept and temp > 1e-9:
            accept = rng.random() < math.exp(max(-700.0, (P2 - curP) / temp))
        if accept:
            cur, curP, curL = T2, P2, L2
        bestP, bestL = state["best_P"], state["best_L"]
        if P2 > bestP or (P2 == bestP and L2 < bestL):
            state["best"] = (T2.copy(), P2, L2)
            state["best_P"], state["best_L"] = P2, L2
            stall = 0
        else:
            stall += 1
        if stall > 300:
            cur, curP, curL = state["best"]
            stall = 0


def _output(T, p):
    cities = [int(x) + 1 for x in T]
    tour = cities + [cities[0]]
    edges = [[tour[k], tour[k + 1]] for k in range(len(cities))]
    return {
        "objective_value": int(p[T].sum()),
        "visited_nodes": cities,
        "edges": edges,
        "tour": tour,
    }


def solve(instance: dict, time_limit_s: int) -> dict:
    budget = min(0.8 * time_limit_s, time_limit_s - 10)
    if budget <= 0:
        budget = 0.5 * time_limit_s
    deadline = T0 + budget

    n, d0, t0, D, p = _parse(instance)
    rng = np.random.default_rng(20260611)

    T = _initial(n, d0, t0, D, p)
    T, L = _fill(T, D, p, t0)
    L = _length(T, D)
    if len(T) < 3 or L > t0:
        # No feasible tour with >= 3 cities was found by the constructor.
        return _output(T, p)

    state = {"best": (T.copy(), _prize(T, p), L)}
    state["best_P"], state["best_L"] = state["best"][1], state["best"][2]
    try:
        T = _local_search(T, D)
        T, _ = _fill(T, D, p, t0)
        L = _length(T, D)
        if L <= t0 and len(T) >= 3 and _prize(T, p) >= state["best_P"]:
            P = _prize(T, p)
            if P > state["best_P"] or L < state["best_L"]:
                state["best"] = (T.copy(), P, L)
                state["best_P"], state["best_L"] = P, L
        _ils(state, D, p, t0, deadline, rng)
    except Exception:
        pass

    T_best, _, _ = state["best"]
    return _output(T_best, p)


if __name__ == "__main__":
    import argparse, json

    def _plain(value):
        if hasattr(value, "tolist"):  # numpy arrays and scalars
            return value.tolist()
        raise TypeError(f"{type(value).__name__} is not JSON serializable: return dicts, lists, str, int and float")

    parser = argparse.ArgumentParser()
    for name in ("--problem", "--instance", "--output", "--time-limit"):
        parser.add_argument(name, required=True)
    args = parser.parse_args()

    with open(args.instance) as file:
        instance = json.load(file)

    solution = solve(instance, int(args.time_limit))

    with open(args.output, "w") as file:
        json.dump(solution, file, default=_plain, allow_nan=False)
