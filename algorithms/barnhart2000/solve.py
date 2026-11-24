"""Origin-destination integer multicommodity flow (barnhart2000).

Pipeline: greedy insertion under several commodity orderings -> rip-up-and-reroute local
search -> arc-flow MIP in HiGHS, warm-started from the best heuristic solution. Each
commodity is either rejected or carried on one simple path; every candidate is validated
(capacity, path continuity, simple path) before it can become the answer.
"""

import heapq
import random
import time

T0 = time.time()

_INF = float("inf")
_HOP_EPS = 1e-7
_MIP_MAX_VARS = 300_000


def _deadline_for(time_limit_s):
    return T0 + min(0.80 * time_limit_s, time_limit_s - 10)


class _Solver:
    def __init__(self, instance):
        net = instance["network"]
        comm_list = instance["commodities"]["commodity_list"]

        self.node_index = {}
        for node in net.get("nodes", []):
            self._nid(node)

        self.arc_ids, self.from_node, self.to_node = [], [], []
        self.tail, self.head, self.cap, self.cost = [], [], [], []
        for arc in net["arcs"]:
            self.arc_ids.append(arc["arc_id"])
            self.from_node.append(arc["from_node"])
            self.to_node.append(arc["to_node"])
            self.tail.append(self._nid(arc["from_node"]))
            self.head.append(self._nid(arc["to_node"]))
            self.cap.append(float(arc["capacity"]))
            self.cost.append(max(float(arc["cost"]), 0.0))
        self.A = len(self.arc_ids)

        self.comm_ids, self.origin, self.dest, self.dem, self.pen, self.valid = [], [], [], [], [], []
        for c in comm_list:
            o = self._nid(c["origin"])
            t = self._nid(c["destination"])
            d = float(c["demand"])
            self.comm_ids.append(c["commodity_id"])
            self.origin.append(o)
            self.dest.append(t)
            self.dem.append(d)
            # Rejection cost in the checker is artificial_arc_cost * demand.
            self.pen.append(d * float(c["artificial_arc_cost"]))
            self.valid.append(o != t)
        self.K = len(comm_list)
        self.N = len(self.node_index)

        self.out = [[] for _ in range(self.N)]
        self.inc = [[] for _ in range(self.N)]
        for a in range(self.A):
            self.out[self.tail[a]].append(a)
            self.inc[self.head[a]].append(a)

        self.wroute = [self.cost[a] + _HOP_EPS for a in range(self.A)]
        self.ratio = [self.pen[k] / max(self.dem[k], 1e-9) for k in range(self.K)]

        self.routed = [None] * self.K
        self.resid = list(self.cap)
        self.best_obj = _INF
        self.best_routed = None

    def _nid(self, node):
        idx = self.node_index.get(node)
        if idx is None:
            idx = len(self.node_index)
            self.node_index[node] = idx
        return idx

    # ----- basic state handling -------------------------------------------------
    def set_state(self, routed):
        self.routed = list(routed)
        self.resid = list(self.cap)
        for k, path in enumerate(self.routed):
            if path is not None:
                d = self.dem[k]
                for a in path:
                    self.resid[a] -= d

    def route(self, k, path):
        d = self.dem[k]
        for a in path:
            self.resid[a] -= d
        self.routed[k] = tuple(path)

    def unroute(self, k):
        path = self.routed[k]
        if path is None:
            return
        d = self.dem[k]
        for a in path:
            self.resid[a] += d
        self.routed[k] = None

    def objective_of(self, routed):
        total = 0.0
        for k in range(self.K):
            path = routed[k]
            if path is None:
                total += self.pen[k]
            else:
                total += self.dem[k] * sum(self.cost[a] for a in path)
        return total

    def validate_of(self, routed):
        load = [0.0] * self.A
        for k in range(self.K):
            path = routed[k]
            if path is None:
                continue
            if not self.valid[k]:
                return False
            cur = self.origin[k]
            seen = {cur}
            for a in path:
                if self.tail[a] != cur:
                    return False
                cur = self.head[a]
                if cur in seen:
                    return False
                seen.add(cur)
            if cur != self.dest[k]:
                return False
            d = self.dem[k]
            for a in path:
                load[a] += d
        return all(load[a] <= self.cap[a] + 1e-9 for a in range(self.A))

    def record(self):
        obj = self.objective_of(self.routed)
        if obj < self.best_obj - 1e-12 and self.validate_of(self.routed):
            self.best_obj = obj
            self.best_routed = list(self.routed)
        return obj

    def restore_best(self):
        if self.best_routed is not None:
            self.set_state(self.best_routed)

    # ----- shortest paths -------------------------------------------------------
    def dijkstra(self, s, t, w, need=None):
        dist = {s: 0.0}
        prev = {}
        heap = [(0.0, s)]
        resid = self.resid
        out = self.out
        head = self.head
        tail = self.tail
        while heap:
            du, u = heapq.heappop(heap)
            if u == t:
                break
            if du > dist[u]:
                continue
            for a in out[u]:
                wa = w[a]
                if wa == _INF:
                    continue
                if need is not None and resid[a] < need:
                    continue
                v = head[a]
                nd = du + wa
                if nd < dist.get(v, _INF):
                    dist[v] = nd
                    prev[v] = a
                    heapq.heappush(heap, (nd, v))
        if t not in dist:
            return None
        path = []
        v = t
        while v != s:
            a = prev[v]
            path.append(a)
            v = tail[a]
        path.reverse()
        return path

    def try_insert(self, k):
        if not self.valid[k] or self.routed[k] is not None:
            return False
        d = self.dem[k]
        path = self.dijkstra(self.origin[k], self.dest[k], self.wroute, need=d)
        if path is None:
            return False
        pc = d * sum(self.cost[a] for a in path)
        if pc >= self.pen[k]:
            return False
        self.route(k, path)
        return True

    def greedy(self, order):
        self.routed = [None] * self.K
        self.resid = list(self.cap)
        for k in order:
            self.try_insert(k)

    # ----- rip-up and reroute ---------------------------------------------------
    def ripup(self, k):
        d = self.dem[k]
        pen = self.pen
        dem = self.dem
        cost = self.cost
        users = [[] for _ in range(self.A)]
        for j, path in enumerate(self.routed):
            if path is not None and dem[j] > 0:
                for a in path:
                    users[a].append(j)

        w = [_INF] * self.A
        picked = [None] * self.A
        for a in range(self.A):
            if self.cap[a] < d:
                continue
            base = cost[a] + _HOP_EPS
            if self.resid[a] >= d:
                w[a] = base
                continue
            deficit = d - self.resid[a]
            freed = 0.0
            chosen = []
            ps = 0.0
            for j in sorted(users[a], key=self.ratio.__getitem__):
                chosen.append(j)
                freed += dem[j]
                ps += pen[j]
                if freed >= deficit:
                    break
            if freed < deficit:
                continue
            w[a] = base + ps
            picked[a] = chosen

        path = self.dijkstra(self.origin[k], self.dest[k], w)
        if path is None:
            return False
        removed = set()
        for a in path:
            if picked[a] is not None:
                removed.update(picked[a])
        removed.discard(k)

        old = self.objective_of(self.routed)
        snap_routed = list(self.routed)
        snap_resid = list(self.resid)
        for j in removed:
            self.unroute(j)
        if not self._fits(path, d):
            self.routed = snap_routed
            self.resid = snap_resid
            return False
        self.route(k, path)
        for j in sorted(removed, key=lambda j: -pen[j]):
            self.try_insert(j)
        if self.objective_of(self.routed) < old - 1e-9:
            return True
        self.routed = snap_routed
        self.resid = snap_resid
        return False

    def _fits(self, path, d):
        return all(self.resid[a] >= d - 1e-9 for a in path)

    def local_search(self, end_time):
        while True:
            improved = False
            cands = [k for k in range(self.K) if self.valid[k] and self.routed[k] is None and self.pen[k] > 0]
            cands.sort(key=lambda k: -self.pen[k])
            for k in cands:
                if time.time() >= end_time:
                    return improved
                if self.routed[k] is not None:
                    continue
                if self.ripup(k):
                    improved = True
            if not improved:
                return False

    # ----- arc-flow MIP ---------------------------------------------------------
    def _reach(self, src, ok, forward):
        seen = {src}
        stack = [src]
        adj = self.out if forward else self.inc
        while stack:
            u = stack.pop()
            for a in adj[u]:
                if not ok[a]:
                    continue
                v = self.head[a] if forward else self.tail[a]
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def _bfs_path(self, s, t, arcs):
        adj = {}
        for a in arcs:
            adj.setdefault(self.tail[a], []).append(a)
        prev = {s: None}
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            if u == t:
                break
            for a in adj.get(u, ()):
                v = self.head[a]
                if v not in prev:
                    prev[v] = a
                    queue.append(v)
        if t not in prev:
            return None
        path = []
        v = t
        while v != s:
            a = prev[v]
            path.append(a)
            v = self.tail[a]
        path.reverse()
        return path

    def mip(self, deadline):
        import numpy as np
        import highspy

        A = self.A
        cap, cost, tail, head = self.cap, self.cost, self.tail, self.head

        cands = []
        total_vars = 0
        for k in range(self.K):
            if not self.valid[k]:
                continue
            if time.time() > deadline - 1.0:
                return False
            d = self.dem[k]
            ok = [cap[a] >= d and tail[a] != head[a] for a in range(A)]
            fwd = self._reach(self.origin[k], ok, True)
            if self.dest[k] not in fwd:
                continue
            bwd = self._reach(self.dest[k], ok, False)
            allowed = [a for a in range(A) if ok[a] and tail[a] in fwd and head[a] in bwd]
            if not allowed:
                continue
            cands.append((k, allowed))
            total_vars += len(allowed) + 1
            if total_vars > _MIP_MAX_VARS:
                return False
        if not cands:
            return False

        node_row = {}
        row_lo, row_hi = [], []

        def get_row(key, lo, hi):
            r = node_row.get(key)
            if r is None:
                r = len(row_lo)
                node_row[key] = r
                row_lo.append(lo)
                row_hi.append(hi)
            return r

        col_cost, col_cons, col_arc, col_kpos = [], [], [], []
        r_col = {}
        kcols = {}
        for kpos, (k, allowed) in enumerate(cands):
            o, t, d = self.origin[k], self.dest[k], self.dem[k]
            ro = get_row((kpos, o), 1.0, 1.0)
            rt = get_row((kpos, t), -1.0, -1.0)
            r_col[kpos] = len(col_cost)
            kcols[kpos] = []
            col_cost.append(self.pen[k])
            col_cons.append([(ro, 1.0), (rt, -1.0)])
            col_arc.append(-1)
            col_kpos.append(kpos)
            for a in allowed:
                ru = get_row((kpos, tail[a]), 0.0, 0.0)
                rv = get_row((kpos, head[a]), 0.0, 0.0)
                kcols[kpos].append((a, len(col_cost)))
                col_cost.append(d * cost[a])
                col_cons.append([(ru, 1.0), (rv, -1.0)])
                col_arc.append(a)
                col_kpos.append(kpos)

        n_cons = len(row_lo)
        used = sorted({a for a in col_arc if a >= 0})
        cap_row = {a: n_cons + i for i, a in enumerate(used)}
        m = n_cons + len(used)
        inf = highspy.kHighsInf
        lo_all = row_lo + [-inf] * len(used)
        hi_all = row_hi + [cap[a] for a in used]

        starts = [0]
        idx = []
        vals = []
        for j in range(len(col_cost)):
            for r, v in col_cons[j]:
                idx.append(r)
                vals.append(v)
            a = col_arc[j]
            if a >= 0:
                idx.append(cap_row[a])
                vals.append(float(self.dem[cands[col_kpos[j]][0]]))
            starts.append(len(idx))
        n = len(col_cost)

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m
        lp.col_cost_ = np.array(col_cost, dtype=np.float64)
        lp.col_lower_ = np.zeros(n, dtype=np.float64)
        lp.col_upper_ = np.ones(n, dtype=np.float64)
        lp.row_lower_ = np.array(lo_all, dtype=np.float64)
        lp.row_upper_ = np.array(hi_all, dtype=np.float64)
        mat = highspy.HighsSparseMatrix()
        mat.format_ = highspy.MatrixFormat.kColwise
        mat.num_col_ = n
        mat.num_row_ = m
        mat.start_ = np.array(starts, dtype=np.int32)
        mat.index_ = np.array(idx, dtype=np.int32)
        mat.value_ = np.array(vals, dtype=np.float64)
        lp.a_matrix_ = mat
        lp.integrality_ = [highspy.HighsVarType.kInteger] * n

        # Warm start from the current heuristic solution.
        x0 = np.zeros(n, dtype=np.float64)
        for kpos, (k, _allowed) in enumerate(cands):
            path = self.routed[k]
            x0[r_col[kpos]] = 1.0 if path is None else 0.0
            if path is not None:
                pset = set(path)
                for a, j in kcols[kpos]:
                    if a in pset:
                        x0[j] = 1.0

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 2)
        h.passModel(lp)
        h.setOptionValue("time_limit", float(max(deadline - time.time() - 0.3, 0.5)))
        try:
            sol = highspy.HighsSolution()
            sol.col_value = x0.tolist()
            sol.value_valid = True
            h.setSolution(sol)
        except Exception:
            pass
        h.run()

        sol = h.getSolution()
        if not sol.value_valid or len(sol.col_value) != n:
            return False
        xv = np.asarray(sol.col_value, dtype=np.float64)

        new = [None] * self.K
        for kpos, (k, _allowed) in enumerate(cands):
            if xv[r_col[kpos]] > 0.5:
                continue
            arcs_x = [a for a, j in kcols[kpos] if xv[j] > 0.5]
            path = self._bfs_path(self.origin[k], self.dest[k], arcs_x)
            if path:
                new[k] = tuple(path)
        new = self._repair(new)
        self.set_state(new)
        self.record()
        return True

    def _repair(self, routed):
        routed = list(routed)
        load = [0.0] * self.A
        for k, path in enumerate(routed):
            if path is not None:
                for a in path:
                    load[a] += self.dem[k]
        for a in range(self.A):
            if load[a] <= self.cap[a] + 1e-9:
                continue
            users = [k for k, p in enumerate(routed) if p is not None and a in p]
            users.sort(key=self.ratio.__getitem__)
            for k in users:
                if load[a] <= self.cap[a] + 1e-9:
                    break
                for b in routed[k]:
                    load[b] -= self.dem[k]
                routed[k] = None
        return routed

    # ----- path-based set packing MIP -------------------------------------------
    def gen_pool(self, end_time, max_paths=4):
        """Candidate simple paths per commodity: successive shortest paths with penalised reuse."""
        cap, cost = self.cap, self.cost
        pool = {}
        for k in range(self.K):
            if not self.valid[k] or self.pen[k] <= 0:
                continue
            if time.time() >= end_time:
                break
            d = self.dem[k]
            base = [(1.0 + cost[a]) if cap[a] >= d else _INF for a in range(self.A)]
            used = {}
            paths = []
            seen = set()
            for _ in range(max_paths):
                w = [_INF if base[a] == _INF else base[a] + 2.0 * used.get(a, 0) for a in range(self.A)]
                p = self.dijkstra(self.origin[k], self.dest[k], w)
                if p is None:
                    break
                key = tuple(p)
                if key in seen:
                    break
                seen.add(key)
                paths.append(key)
                for a in p:
                    used[a] = used.get(a, 0) + 1
            pool[k] = paths
        return pool

    def path_mip(self, pool_end, time_limit, max_paths=4):
        import numpy as np
        import highspy

        pool = self.gen_pool(pool_end, max_paths)
        cols = []  # (k, path, cost_coefficient)
        for k in range(self.K):
            cur = self.routed[k]
            if cur is not None:
                cols.append((k, cur, self.dem[k] * sum(self.cost[a] for a in cur) - self.pen[k]))
            for p in pool.get(k, ()):
                if p == cur:
                    continue
                c = self.dem[k] * sum(self.cost[a] for a in p) - self.pen[k]
                if c < -1e-9:
                    cols.append((k, p, c))
        if not cols:
            return False

        k_rows = {}
        arc_rows = {}
        for k, _p, _c in cols:
            if k not in k_rows:
                k_rows[k] = len(k_rows)
        used_arcs = sorted({a for _k, p, _c in cols for a in p})
        for i, a in enumerate(used_arcs):
            arc_rows[a] = len(k_rows) + i
        m = len(k_rows) + len(used_arcs)
        inf = highspy.kHighsInf
        row_lo = [-inf] * m
        row_hi = [1.0] * len(k_rows) + [self.cap[a] for a in used_arcs]

        starts = [0]
        idx = []
        vals = []
        costs = []
        x0 = []
        for k, p, c in cols:
            idx.append(k_rows[k])
            vals.append(1.0)
            d = float(self.dem[k])
            for a in p:
                idx.append(arc_rows[a])
                vals.append(d)
            starts.append(len(idx))
            costs.append(c)
            x0.append(1.0 if self.routed[k] == p else 0.0)
        n = len(cols)

        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = m
        lp.col_cost_ = np.array(costs, dtype=np.float64)
        lp.col_lower_ = np.zeros(n, dtype=np.float64)
        lp.col_upper_ = np.ones(n, dtype=np.float64)
        lp.row_lower_ = np.array(row_lo, dtype=np.float64)
        lp.row_upper_ = np.array(row_hi, dtype=np.float64)
        mat = highspy.HighsSparseMatrix()
        mat.format_ = highspy.MatrixFormat.kColwise
        mat.num_col_ = n
        mat.num_row_ = m
        mat.start_ = np.array(starts, dtype=np.int32)
        mat.index_ = np.array(idx, dtype=np.int32)
        mat.value_ = np.array(vals, dtype=np.float64)
        lp.a_matrix_ = mat
        lp.integrality_ = [highspy.HighsVarType.kInteger] * n

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 2)
        h.setOptionValue("time_limit", float(max(time_limit, 0.5)))
        h.passModel(lp)
        try:
            sol = highspy.HighsSolution()
            sol.col_value = x0
            sol.value_valid = True
            h.setSolution(sol)
        except Exception:
            pass
        h.run()

        sol = h.getSolution()
        if not sol.value_valid or len(sol.col_value) != n:
            return False
        new = [None] * self.K
        for j, (k, p, _c) in enumerate(cols):
            if sol.col_value[j] > 0.5:
                new[k] = tuple(p)
        new = self._repair(new)
        self.set_state(new)
        self.record()
        return True

    # ----- output ---------------------------------------------------------------
    def output(self, routed):
        commodities = []
        for k in range(self.K):
            path = routed[k]
            commodities.append(
                {
                    "commodity_id": self.comm_ids[k],
                    "rejected": path is None,
                    "path_arcs": []
                    if path is None
                    else [{"from": self.from_node[a], "to": self.to_node[a], "arc_id": self.arc_ids[a]} for a in path],
                }
            )
        return {"objective_value": float(self.objective_of(routed)), "commodities": commodities}


def _run(s, start, deadline):
    span = max(deadline - start, 0.0)
    rng = random.Random(20240601)
    valid_ks = [k for k in range(s.K) if s.valid[k]]

    s.greedy(sorted(valid_ks, key=lambda k: -s.pen[k]))
    s.record()
    s.greedy(sorted(valid_ks, key=lambda k: -s.ratio[k]))
    s.record()

    greedy_end = start + 0.10 * span
    it = 0
    while time.time() < greedy_end and it < 200:
        if it % 2 == 0:
            key = {k: -s.pen[k] * rng.uniform(0.7, 1.3) for k in valid_ks}
        else:
            key = {k: -s.ratio[k] * rng.uniform(0.7, 1.3) for k in valid_ks}
        s.greedy(sorted(valid_ks, key=key.__getitem__))
        s.record()
        it += 1

    s.restore_best()
    s.local_search(start + 0.55 * span)
    s.record()
    s.restore_best()

    now = time.time()
    remaining = deadline - now
    if remaining > 2.0:
        try:
            s.path_mip(now + 0.20 * remaining, 0.35 * remaining)
        except Exception:
            pass
        s.restore_best()
        now = time.time()
        remaining = deadline - now
        if remaining > 3.0:
            try:
                s.path_mip(now + 0.15 * remaining, 0.6 * remaining, max_paths=10)
            except Exception:
                pass
            s.restore_best()

    remaining = deadline - time.time() - 0.5
    if remaining > 2.0:
        try:
            s.mip(deadline - 0.5)
        except Exception:
            pass
        s.restore_best()
        if deadline - time.time() > 1.0:
            s.local_search(deadline - 0.5)
            s.record()
            s.restore_best()


def _all_rejected(instance):
    commodities = []
    total = 0.0
    for c in instance["commodities"]["commodity_list"]:
        total += float(c["demand"]) * float(c["artificial_arc_cost"])
        commodities.append({"commodity_id": c["commodity_id"], "rejected": True, "path_arcs": []})
    return {"objective_value": float(total), "commodities": commodities}


def solve(instance: dict, time_limit_s: int) -> dict:
    start = time.time()
    deadline = _deadline_for(time_limit_s)
    try:
        s = _Solver(instance)
    except Exception:
        return _all_rejected(instance)
    try:
        _run(s, start, deadline)
    except Exception:
        pass
    if s.best_routed is None or not s.validate_of(s.best_routed):
        return _all_rejected(instance)
    return s.output(s.best_routed)


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
