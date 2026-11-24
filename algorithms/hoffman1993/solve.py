import itertools
import multiprocessing as mp
import time

import numpy as np
import scipy.sparse as sp
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
    """Drop empty columns and, when there are no base constraints, duplicate row sets (keep cheapest)."""
    n = data["n"]
    ptr, row, cost = data["ptr"], data["row"], data["cost"]
    if data["nb"] > 0:
        return [j for j in range(n) if ptr[j + 1] > ptr[j]]
    best = {}
    for j in range(n):
        if ptr[j + 1] == ptr[j]:
            continue
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


class _Search:
    """Exact-cover depth-first branch and bound (iterative, resumable)."""

    def __init__(self, data, kept):
        self.data = data
        m, n = data["m"], data["n"]
        self.m = m
        ptr, row = data["ptr"], data["row"]
        self.cost = data["cost"]
        self.size = np.diff(ptr).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.ratio = np.where(self.size > 0, self.cost / np.maximum(self.size, 1.0), np.inf)
        ones = np.ones(len(row), dtype=np.float64)
        self.AT = sp.csr_matrix((ones, row, ptr), shape=(n, m))  # n x m
        self.A = self.AT.T.tocsr()  # m x n: row r -> columns covering r
        self.a_ptr = self.A.indptr
        self.a_idx = self.A.indices
        self.ratio_nnz = self.ratio[self.a_idx]
        self.nonempty = np.nonzero(np.diff(self.a_ptr) > 0)[0]
        self.nonempty_start = self.a_ptr[self.nonempty]
        self.D = data["D"]
        self.nb = data["nb"]
        self.d2 = data["d2"]
        self.best = None
        self.best_cost = np.inf
        self.nodes = 0
        self.done = False
        self.stack = []
        active = np.zeros(n, dtype=bool)
        active[np.asarray(kept, dtype=np.int64)] = True
        root = self._expand(
            active,
            np.zeros(m, dtype=bool),
            0.0,
            np.zeros(self.nb) if self.nb else None,
            (),
        )
        if root is None:
            self.done = True
        else:
            self.stack.append(root)

    def offer(self, chosen):
        """Accept an external solution if it is valid and better than the incumbent."""
        chosen = [int(j) for j in chosen]
        c = float(self.cost[np.asarray(chosen, dtype=np.int64)].sum()) if chosen else np.inf
        if c < self.best_cost - 1e-9 and _valid(self.data, chosen):
            self.best = chosen
            self.best_cost = c
            return True
        return False

    def _expand(self, active, covered, cost_, dx, chosen):
        """Return a frame for an open node, or None if the node is a leaf or is pruned."""
        self.nodes += 1
        unc = ~covered
        if not unc.any():
            self.offer(chosen)
            return None
        cnt = self.A @ active.astype(np.float64)
        if np.any(cnt[unc] == 0):
            return None
        vals = np.where(active[self.a_idx], self.ratio_nnz, np.inf)
        rowmin = np.full(self.m, np.inf)
        if self.nonempty.size:
            rowmin[self.nonempty] = np.minimum.reduceat(vals, self.nonempty_start)
        lb = cost_ + float(rowmin[unc].sum())
        if lb >= self.best_cost - 1e-9:
            return None
        r = int(np.argmin(np.where(unc, cnt, np.inf)))
        cands = self.a_idx[self.a_ptr[r] : self.a_ptr[r + 1]]
        cands = cands[active[cands]]
        order = np.lexsort((-self.size[cands], self.cost[cands]))
        cands = cands[order]
        return [active, covered, cost_, dx, chosen, cands, 0]

    def _child(self, frame, c):
        active, covered, cost_, dx, chosen = frame[0], frame[1], frame[2], frame[3], frame[4]
        ndx = None
        if dx is not None:
            ndx = dx + self.D[:, c]
            if np.any(ndx > self.d2 + 1e-9):
                return None
        p0, p1 = self.data["ptr"][c], self.data["ptr"][c + 1]
        rows_c = self.data["row"][p0:p1]
        hit = np.zeros(self.m, dtype=np.float64)
        hit[rows_c] = 1.0
        conflict = self.AT @ hit  # > 0 for every column sharing a row with c (incl. c)
        nact = active & (conflict == 0)
        ncov = covered.copy()
        ncov[rows_c] = True
        return self._expand(nact, ncov, cost_ + float(self.cost[c]), ndx, chosen + (int(c),))

    def run(self, deadline, stop_on_first=False):
        """Search until the deadline. Returns True when the whole tree has been explored."""
        while self.stack:
            if time.time() >= deadline:
                return False
            if stop_on_first and self.best is not None:
                return False
            f = self.stack[-1]
            if f[6] >= len(f[5]):
                self.stack.pop()
                continue
            c = int(f[5][f[6]])
            f[6] += 1
            child = self._child(f, c)
            if child is not None:
                self.stack.append(child)
        self.done = True
        return True


def _mip_worker(conn, data, kept, incumbent, time_s):
    try:
        lp = _build_lp(data, kept)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 2)
        h.setOptionValue("mip_rel_gap", 0.0)
        h.setOptionValue("time_limit", float(max(0.5, time_s)))
        h.passModel(lp)
        if incumbent:
            pos = {c: i for i, c in enumerate(kept)}
            x = np.zeros(len(kept))
            x[[pos[c] for c in incumbent]] = 1.0
            sol = highspy.HighsSolution()
            sol.col_value = x.tolist()
            h.setSolution(sol)
        h.run()
        result = None
        if h.getInfo().primal_solution_status >= 2:
            x = np.asarray(h.getSolution().col_value, dtype=np.float64)
            cand = [int(kept[i]) for i in np.nonzero(x[: len(kept)] > 0.5)[0]]
            if _valid(data, cand):
                result = cand
        conn.send(result)
    except Exception:
        try:
            conn.send(None)
        except Exception:
            pass
    finally:
        conn.close()


def _run_mip_process(data, kept, incumbent, time_s, kill_at):
    """HiGHS MIP in a forked child; the child is hard-killed at kill_at if still running."""
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_mip_worker, args=(child, data, kept, incumbent, time_s), daemon=True)
    p.start()
    child.close()
    result = None
    try:
        wait = kill_at - time.time()
        if wait > 0 and parent.poll(wait):
            result = parent.recv()
    except Exception:
        result = None
    finally:
        if p.is_alive():
            p.kill()
        p.join(1.0)
        parent.close()
    return result


def _solve_exact(data, deadline):
    kept = _columns_kept(data)
    if not kept:
        return None
    search = None
    try:
        search = _Search(data, kept)
        start = _first_fit(data)
        if start:
            search.offer(start)
        span = max(deadline - T0, 1.0)
        if not search.done:
            search.run(T0 + 0.40 * span)
        if not search.done and search.best is None:
            search.run(T0 + 0.85 * span, stop_on_first=True)
    except Exception:
        pass
    if search is None:
        return None
    if search.done:
        return search.best

    now = time.time()
    if deadline - now > 3.0:
        try:
            res = _run_mip_process(
                data,
                kept,
                search.best,
                deadline - now - 2.0,
                deadline - 0.5,
            )
            if res is not None:
                search.offer(res)
        except Exception:
            pass

    try:
        search.run(deadline - 0.5)
    except Exception:
        pass
    return search.best


def _first_fit(data):
    """Columns taken in index order whenever disjoint from those already taken; a valid partition or None."""
    ptr, row = data["ptr"], data["row"]
    covered = np.zeros(data["m"], dtype=bool)
    chosen = []
    for j in range(data["n"]):
        r = row[ptr[j] : ptr[j + 1]]
        if r.size and not covered[r].any():
            covered[r] = True
            chosen.append(j)
            if covered.all():
                break
    return chosen if _valid(data, chosen) else None


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
        chosen = _solve_exact(data, deadline)
    except Exception:
        chosen = None
    if chosen is None:
        try:
            chosen = _first_fit(data)
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
