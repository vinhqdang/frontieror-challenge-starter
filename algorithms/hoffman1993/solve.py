import itertools
import time

import numpy as np
import highspy

T0 = time.time()

CHECK_TOL = 1e-5  # tolerance used by the published checker


def _deadline(time_limit_s):
    tl = float(time_limit_s)
    return T0 + max(min(0.80 * tl, tl - 10.0), 0.5 * tl)


def _parse(instance):
    dims = instance["dimensions"]
    m = int(dims["num_rows"])
    n = int(dims["num_cols"])
    cost_raw = list(instance["cost_vector"])[:n]
    cost = np.asarray(cost_raw, dtype=np.float64)

    cols = list(instance["constraint_matrix_A"]["columns"])[:n]
    cols += [[] for _ in range(n - len(cols))]
    lens = np.fromiter((len(c) for c in cols), dtype=np.int64, count=n)
    flat = np.fromiter(itertools.chain.from_iterable(cols), dtype=np.int64, count=int(lens.sum()))
    ent = np.repeat(np.arange(n, dtype=np.int64), lens)
    ok = (flat >= 0) & (flat < m)
    key = np.unique(ent[ok] * m + flat[ok])  # sorted by column, then row, no duplicates
    ent_col = key // m
    row = key % m
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(ent_col, minlength=n))

    has_base = bool(instance.get("has_base_constraints", False))
    base = instance.get("base_constraints") or {}
    D = None
    d1 = np.zeros(0)
    d2 = np.zeros(0)
    nb = 0
    if has_base and base:
        d_rows = base["D_matrix"]["rows"]
        d1_raw = list(base["lower_bounds_d1"])
        d2_raw = list(base["upper_bounds_d2"])
        nb = min(int(base["num_constraints"]), len(d_rows), len(d1_raw), len(d2_raw))
        if nb > 0:
            D = np.asarray([list(d_rows[b])[:n] for b in range(nb)], dtype=np.float64)
            D = D.reshape(nb, n)
            d1 = np.asarray(d1_raw[:nb], dtype=np.float64)
            d2 = np.asarray(d2_raw[:nb], dtype=np.float64)
    return {
        "m": m,
        "n": n,
        "cost_raw": cost_raw,
        "cost": cost,
        "ptr": ptr,
        "ent_col": ent_col,
        "row": row,
        "nb": nb,
        "D": D,
        "d1": d1,
        "d2": d2,
    }


def _columns_kept(data):
    """Drop duplicate row sets (keep cheapest) when there are no base constraints."""
    n = data["n"]
    if data["nb"] > 0:
        return list(range(n))
    ptr, row, cost = data["ptr"], data["row"], data["cost"]
    best = {}
    for j in range(n):
        k = row[ptr[j] : ptr[j + 1]].tobytes()
        b = best.get(k)
        if b is None or cost[j] < cost[b]:
            best[k] = j
    return sorted(best.values())


def _build_lp(data, kept):
    m, nb = data["m"], data["nb"]
    k = len(kept)
    newid = np.full(data["n"], -1, dtype=np.int64)
    newid[np.asarray(kept, dtype=np.int64)] = np.arange(k, dtype=np.int64)

    sel = newid[data["ent_col"]] >= 0
    rows = [data["row"][sel]]
    cols = [newid[data["ent_col"][sel]]]
    vals = [np.ones(int(sel.sum()), dtype=np.float64)]
    R = m + nb
    if nb > 0:
        Dk = data["D"][:, np.asarray(kept, dtype=np.int64)]
        bi, bj = np.nonzero(Dk)
        rows.append(m + bi)
        cols.append(bj)
        vals.append(Dk[bi, bj])
    rows = np.concatenate(rows).astype(np.int32)
    cols = np.concatenate(cols).astype(np.int64)
    vals = np.concatenate(vals)
    order = np.lexsort((rows, cols))
    rows, cols, vals = rows[order], cols[order], vals[order]
    start = np.zeros(k + 1, dtype=np.int32)
    start[1:] = np.cumsum(np.bincount(cols, minlength=k))

    lp = highspy.HighsLp()
    lp.num_col_ = k
    lp.num_row_ = R
    lp.col_cost_ = data["cost"][np.asarray(kept, dtype=np.int64)].copy()
    lp.col_lower_ = np.zeros(k)
    lp.col_upper_ = np.ones(k)
    lp.row_lower_ = np.concatenate([np.ones(m), data["d1"]])
    lp.row_upper_ = np.concatenate([np.ones(m), data["d2"]])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = k
    lp.a_matrix_.num_row_ = R
    lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = rows
    lp.a_matrix_.value_ = vals
    lp.integrality_ = [highspy.HighsVarType.kInteger] * k
    return lp


def _valid(data, chosen):
    """Exact partition check plus base bounds, with the checker's tolerance."""
    if len(set(chosen)) != len(chosen):
        return False
    m, n = data["m"], data["n"]
    if not chosen:
        return False
    idx = np.concatenate([data["row"][data["ptr"][j] : data["ptr"][j + 1]] for j in chosen])
    counts = np.bincount(idx, minlength=m)
    if counts.shape[0] != m or not np.all(counts == 1):
        return False
    if data["nb"] > 0:
        x = np.zeros(n)
        x[np.asarray(chosen, dtype=np.int64)] = 1.0
        Dx = data["D"] @ x
        if np.any(Dx < data["d1"] - CHECK_TOL) or np.any(Dx > data["d2"] + CHECK_TOL):
            return False
    return True


def _run_highs(data, kept, lp, time_s, feasibility_only):
    n_cols = len(kept)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 2)
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("time_limit", float(max(0.5, time_s)))
    if n_cols > 5000:
        # HiGHS presolve (probing) can overrun the time limit on large models.
        h.setOptionValue("presolve", "off")
    if feasibility_only:
        lp.col_cost_ = np.zeros(len(kept))
        h.setOptionValue("mip_heuristic_effort", 0.5)
    h.passModel(lp)
    h.run()
    if h.getInfo().primal_solution_status < 2:
        return None
    x = np.asarray(h.getSolution().col_value, dtype=np.float64)
    chosen = [kept[i] for i in np.nonzero(x[: len(kept)] > 0.5)[0]]
    return chosen if _valid(data, chosen) else None


def _solve_mip(data, deadline):
    kept = _columns_kept(data)
    lp = _build_lp(data, kept)
    budget = deadline - time.time()
    if budget < 1.0:
        return None
    reserve = min(5.0, 0.1 * budget)
    chosen = _run_highs(data, kept, lp, budget - reserve, False)
    if chosen is None and deadline - time.time() > 1.0:
        chosen = _run_highs(data, kept, lp, deadline - time.time(), True)
    return chosen


def _greedy(data):
    """Last resort: cheapest-per-row disjoint columns (may not cover every row)."""
    ptr, row, cost = data["ptr"], data["row"], data["cost"]
    lens = np.diff(ptr)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(lens > 0, cost / np.maximum(lens, 1), np.inf)
    covered = np.zeros(data["m"], dtype=bool)
    chosen = []
    for j in np.argsort(ratio, kind="stable"):
        if lens[j] == 0:
            continue
        r = row[ptr[j] : ptr[j + 1]]
        if not covered[r].any():
            covered[r] = True
            chosen.append(int(j))
    return chosen


def _output(data, chosen):
    chosen = sorted(set(int(j) for j in chosen))
    vv = {str(j): 0.0 for j in range(data["n"])}
    for j in chosen:
        vv[str(j)] = 1.0
    obj = float(sum(float(data["cost_raw"][j]) for j in chosen))
    return {"objective_value": obj, "selected_rotations": chosen, "variable_values": vv}


def solve(instance, time_limit_s):
    deadline = _deadline(time_limit_s)
    data = _parse(instance)
    chosen = None
    try:
        chosen = _solve_mip(data, deadline)
    except Exception:
        chosen = None
    if chosen is None:
        try:
            chosen = _greedy(data)
        except Exception:
            chosen = []
    return _output(data, chosen)


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
