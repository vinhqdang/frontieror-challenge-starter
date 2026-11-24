"""Stochastic capacitated facility location (Bodur et al. 2017 formulation).

Approach:
  1. Greedy feasible open set (cheapest opening cost per unit of capacity until total capacity covers the
     largest scenario demand).
  2. Sample-average MIP over a random scenario sample (plus the mean-demand scenario) to pick an open set.
  3. Iterated local search (close / open / swap) on a sample of scenarios, using one warm-started HiGHS LP
     whose capacity-row bounds encode the open set and whose demand-row bounds encode the scenario.
  4. Exact evaluation of the best open set on ALL scenarios; output sparse y (only nonzero shipments).
"""

import math
import random
import time

T0 = time.time()

import numpy as np

import highspy

_INF = highspy.kHighsInf


def _deadline(time_limit_s):
    tl = float(time_limit_s)
    return T0 + max(min(0.80 * tl, tl - 10.0), 0.5 * tl)


class _Transport:
    """Transportation LP min sum c_ij y_ij, sum_i y_ij >= d_j, sum_j y_ij <= capS_i, y >= 0.

    Capacity rows are set to 0 for closed facilities, which forces their flows to zero.
    """

    def __init__(self, cost, cap):
        I, J = cost.shape
        self.I, self.J = I, J
        self.cap = cap
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        n = I * J
        h.addVars(n, np.zeros(n), np.full(n, _INF))
        h.changeColsCost(n, np.arange(n, dtype=np.int32), np.asarray(cost, dtype=float).ravel())
        starts, idx = [], []
        for j in range(J):
            starts.append(len(idx))
            idx.extend(int(i) * J + j for i in range(I))
        for i in range(I):
            starts.append(len(idx))
            idx.extend(i * J + j for j in range(J))
        nnz = len(idx)
        lo = np.concatenate([np.zeros(J), np.full(I, -_INF)])
        up = np.concatenate([np.full(J, _INF), np.zeros(I)])
        h.addRows(
            I + J,
            lo,
            up,
            nnz,
            np.array(starts, dtype=np.int32),
            np.array(idx, dtype=np.int32),
            np.ones(nnz),
        )
        self.h = h
        self.dem_rows = np.arange(J, dtype=np.int32)
        self.cap_rows = np.arange(J, J + I, dtype=np.int32)
        self.inf_J = np.full(J, _INF)
        self.neg_inf_I = np.full(I, -_INF)
        self.inf_J_lower = None

    def set_open(self, open_mask):
        capS = np.where(open_mask, self.cap, 0.0)
        self.h.changeRowsBounds(self.I, self.cap_rows, self.neg_inf_I, capS)

    def solve(self, demand):
        """Solve the transport LP for one demand vector; returns (objective, y matrix I x J)."""
        self.h.changeRowsBounds(self.J, self.dem_rows, np.asarray(demand, dtype=float), self.inf_J)
        self.h.run()
        if self.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError("transport LP not optimal")
        obj = float(self.h.getInfo().objective_function_value)
        return obj

    def solution(self):
        return np.array(self.h.getSolution().col_value, dtype=float).reshape(self.I, self.J)


def _build_mip(cost, cap, f, D_sample, w_sample, max_total):
    """Sample-average MIP: binary x_i, continuous y_sij. Returns highspy model (not solved)."""
    I, J = cost.shape
    M = D_sample.shape[0]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    ncols = I + M * I * J
    lb = np.zeros(ncols)
    ub = np.full(ncols, _INF)
    ub[:I] = 1.0
    h.addVars(ncols, lb, ub)
    h.changeColsIntegrality(I, np.arange(I, dtype=np.int32), np.array([highspy.HighsVarType.kInteger] * I))
    costs = np.zeros(ncols)
    costs[:I] = f
    ycost = np.zeros((M, I, J))
    for s in range(M):
        ycost[s] = w_sample[s] * cost
    costs[I:] = ycost.ravel()
    h.changeColsCost(ncols, np.arange(ncols, dtype=np.int32), costs)

    starts, idx, vals = [], [], []
    lo, up = [], []
    base = I
    for s in range(M):
        # demand rows (s, j): sum_i y_sij >= d_sj
        for j in range(J):
            starts.append(len(idx))
            idx.extend(base + i * J + j for i in range(I))
            vals.extend([1.0] * I)
            lo.append(float(D_sample[s, j]))
            up.append(_INF)
        # capacity rows (s, i): sum_j y_sij - cap_i x_i <= 0
        for i in range(I):
            starts.append(len(idx))
            idx.extend(base + i * J + j for j in range(J))
            vals.extend([1.0] * J)
            idx.append(i)
            vals.append(-float(cap[i]))
            lo.append(-_INF)
            up.append(0.0)
        base += I * J
    # total capacity >= max total demand
    starts.append(len(idx))
    idx.extend(range(I))
    vals.extend(float(c) for c in cap)
    lo.append(float(max_total))
    up.append(_INF)

    nnz = len(idx)
    h.addRows(
        len(lo),
        np.array(lo),
        np.array(up),
        nnz,
        np.array(starts, dtype=np.int32),
        np.array(idx, dtype=np.int32),
        np.array(vals, dtype=float),
    )
    return h


class _Searcher:
    def __init__(self, f, cap, cost, D, p, max_total, deadline, rng):
        self.f = f
        self.cap = cap
        self.cost = cost
        self.D = D
        self.p = p
        self.I, self.J = cost.shape
        self.K = D.shape[0]
        self.max_total = max_total
        self.deadline = deadline
        self.rng = rng
        self.tp = _Transport(cost, cap)

    def feasible(self, mask):
        return float(self.cap[mask].sum()) >= self.max_total

    def greedy(self):
        order = np.argsort(self.f / self.cap, kind="stable")
        mask = np.zeros(self.I, dtype=bool)
        total = 0.0
        for i in order:
            mask[i] = True
            total += self.cap[i]
            if total >= self.max_total:
                break
        return mask

    def sample_value(self, mask, Ds, ws, bound=math.inf):
        """Sample-average objective f.x + sum_s w_s LP_s; returns None when it cannot beat `bound`."""
        self.tp.set_open(mask)
        total = float(self.f[mask].sum())
        if total >= bound:
            return None
        acc = 0.0
        for s in range(Ds.shape[0]):
            acc += ws[s] * self.tp.solve(Ds[s])
            if total + acc >= bound:
                return None
        return total + acc

    def improve(self, mask, Ds, ws, cur_val, swap_samples=150):
        """First-improvement local search with close / open / swap moves."""
        improved = True
        while improved and time.time() < self.deadline:
            improved = False
            opened = np.flatnonzero(mask)
            closed = np.flatnonzero(~mask)
            moves = [("c", int(i)) for i in opened] + [("o", int(j)) for j in closed]
            swaps = [(int(i), int(j)) for i in opened for j in closed]
            if len(swaps) > swap_samples:
                pick = self.rng.sample(range(len(swaps)), swap_samples)
                swaps = [swaps[t] for t in pick]
            moves += [("s", i, j) for i, j in swaps]
            self.rng.shuffle(moves)
            for mv in moves:
                if time.time() >= self.deadline:
                    break
                new = mask.copy()
                if mv[0] == "c":
                    new[mv[1]] = False
                elif mv[0] == "o":
                    new[mv[1]] = True
                else:
                    new[mv[1]] = False
                    new[mv[2]] = True
                if not self.feasible(new):
                    continue
                val = self.sample_value(new, Ds, ws, bound=cur_val - 1e-9 * max(1.0, abs(cur_val)))
                if val is not None and val < cur_val:
                    mask, cur_val = new, val
                    improved = True
                    break
        return mask, cur_val

    def perturb(self, mask):
        new = mask.copy()
        for _ in range(3):
            opened = np.flatnonzero(new)
            closed = np.flatnonzero(~new)
            if len(closed) == 0 or len(opened) == 0:
                break
            i = self.rng.choice(list(opened))
            j = self.rng.choice(list(closed))
            new[i] = False
            new[j] = True
        if not self.feasible(new):
            return mask
        return new


def _exact_solution(tp, mask, D, cost, f, p):
    """Solve the transport LP for every scenario; returns (objective, sparse y dict)."""
    I, J = cost.shape
    K = D.shape[0]
    tp.set_open(mask)
    y_out = {}
    transport = 0.0
    for k in range(K):
        tp.solve(D[k])
        Y = np.maximum(tp.solution(), 0.0)
        Y[Y < 1e-12] = 0.0
        transport += float(p[k]) * float(np.sum(cost * Y))
        ii, jj = np.nonzero(Y)
        if ii.size:
            scen = {}
            for i, j in zip(ii.tolist(), jj.tolist()):
                scen.setdefault(str(i), {})[str(j)] = float(Y[i, j])
            y_out[str(k)] = scen
        else:
            y_out[str(k)] = {}
    opening = float(f[mask].sum())
    return opening + transport, y_out


def solve(instance: dict, time_limit_s: int) -> dict:
    deadline = _deadline(time_limit_s)
    I = len(instance["facilities"])
    J = int(instance["num_customers"])
    K = len(instance["scenarios"])
    f = np.array([float(fa["opening_cost"]) for fa in instance["facilities"]])
    cap = np.array([float(fa["capacity"]) for fa in instance["facilities"]])
    cost = np.array(instance["transportation_costs"], dtype=float).reshape(I, J)
    D = np.array([sc["demands"] for sc in instance["scenarios"]], dtype=float).reshape(K, J)
    p = np.array([float(sc["probability"]) for sc in instance["scenarios"]])
    max_total = float(D.sum(axis=1).max())
    rng = random.Random(12345)

    searcher = _Searcher(f, cap, cost, D, p, max_total, deadline, rng)
    best_mask = searcher.greedy()
    best_val = None

    try:
        # Estimate per-LP cost to reserve time for the exact final evaluation.
        t_probe = time.time()
        searcher.tp.set_open(best_mask)
        probe_n = min(K, 5)
        for k in range(probe_n):
            searcher.tp.solve(D[k])
        t_lp = max((time.time() - t_probe) / max(probe_n, 1), 1e-5)
        reserve = 1.5 * K * t_lp + 1.0
        search_end = deadline - reserve
        now = time.time()

        # Evaluation sample for local search.
        M_ls = min(K, 80)
        ls_idx = np.array(rng.sample(range(K), M_ls)) if M_ls < K else np.arange(K)
        Ds_ls = D[ls_idx]
        pw = p[ls_idx]
        ws_ls = pw / pw.sum()

        # Sample-average MIP for an initial open set.
        M_mip = min(K, max(10, min(60, 150000 // max(1, I * J))))
        mip_idx = np.array(rng.sample(range(K), M_mip)) if M_mip < K else np.arange(K)
        D_mip = np.vstack([D[mip_idx], D.mean(axis=0, keepdims=True)])
        w_mip = np.full(D_mip.shape[0], 1.0 / D_mip.shape[0])
        mip_budget = 0.35 * max(search_end - time.time(), 0.0)
        if mip_budget > 0.5:
            try:
                mip = _build_mip(cost, cap, f, D_mip, w_mip, max_total)
                mip.setOptionValue("time_limit", float(mip_budget))
                mip.setOptionValue("threads", 2)
                mip.setOptionValue("mip_rel_gap", 1e-4)
                mip.run()
                sol = np.array(mip.getSolution().col_value, dtype=float)
                if sol.size >= I:
                    mask = sol[:I] > 0.5
                    if mask.any() and searcher.feasible(mask):
                        cand = mask
                        v_cand = searcher.sample_value(cand, Ds_ls, ws_ls)
                        v_best = searcher.sample_value(best_mask, Ds_ls, ws_ls)
                        if v_cand is not None and (v_best is None or v_cand < v_best):
                            best_mask = cand
            except Exception:
                pass

        # Iterated local search on the sample.
        searcher.deadline = search_end
        cur_mask = best_mask
        cur_val = searcher.sample_value(cur_mask, Ds_ls, ws_ls)
        cur_mask, cur_val = searcher.improve(cur_mask, Ds_ls, ws_ls, cur_val)
        best_mask_ls, best_val_ls = cur_mask.copy(), cur_val
        while time.time() < search_end:
            cand = searcher.perturb(best_mask_ls)
            cand_val = searcher.sample_value(cand, Ds_ls, ws_ls)
            if cand_val is None:
                continue
            cand, cand_val = searcher.improve(cand, Ds_ls, ws_ls, cand_val)
            if cand_val < best_val_ls:
                best_mask_ls, best_val_ls = cand, cand_val
        best_mask = best_mask_ls
    except Exception:
        pass

    # Exact evaluation of the chosen open set on all scenarios.
    objective, y_out = _exact_solution(searcher.tp, best_mask, D, cost, f, p)
    open_idx = [int(i) for i in np.flatnonzero(best_mask)]
    x_out = {str(i): (1 if best_mask[i] else 0) for i in range(I)}
    return {
        "objective_value": float(objective),
        "open_facilities": open_idx,
        "x": x_out,
        "y": y_out,
    }


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
