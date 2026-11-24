"""Dial-a-Ride Problem (Cordeau 2006): ALNS over exact route feasibility.

Route feasibility for a sequence (origin, requests..., destination) is a system of difference constraints on
service-start times: chain (B_next >= B_prev + s_prev + t), windows (e <= B <= l), ride time
(B_drop - B_pick - s_pick <= L) and route duration (B_end - B_start <= T). The least solution above the lower
bounds is found by fixpoint iteration; the sequence is feasible iff that fixpoint exists within the upper bounds.
Insertion candidates are prefiltered with exact window/capacity propagation and then checked with that fixpoint.
"""

import math
import random
import time

T0 = time.time()

_MARG = 1e-7


class _Model:
    def __init__(self, instance):
        self.n = n = int(instance["num_users"])
        self.K = int(instance["num_vehicles"])
        self.Q = float(instance["vehicle_capacity"])
        self.L = float(instance["maximum_ride_time"])
        self.T = float(instance["maximum_route_duration"])
        N = 2 * n + 2
        self.N = N
        self.origin = 0
        self.dest = 2 * n + 1
        by_id = {int(nd["node_id"]): nd for nd in instance["nodes"]}
        X = [0.0] * N
        Y = [0.0] * N
        self.e = [0.0] * N
        self.l = [0.0] * N
        self.ds = [0.0] * N
        self.q = [0] * N
        for v in range(N):
            nd = by_id[v]
            X[v] = float(nd["x"])
            Y[v] = float(nd["y"])
            self.e[v] = float(nd["earliest_time"])
            self.l[v] = float(nd["latest_time"])
            self.ds[v] = float(nd["service_duration"])
            self.q[v] = int(nd["load"])
        D = [[0.0] * N for _ in range(N)]
        for i in range(N):
            xi, yi = X[i], Y[i]
            row = D[i]
            for j in range(N):
                if i != j:
                    row[j] = math.sqrt((xi - X[j]) ** 2 + (yi - Y[j]) ** 2)
        self.D = D
        arcs = [D[i][j] for i in range(N) for j in range(N) if i != j]
        self.avg_arc = max(1e-6, sum(arcs) / len(arcs))
        self._info = {}
        self._ins = {}

    # ------------------------------------------------------------------ feasibility
    def sched(self, seq):
        """Earliest feasible service-start times for (origin,)+seq+(dest,), or None if infeasible."""
        n = self.n
        q = self.q
        load = 0
        picked = set()
        for v in seq:
            if v <= n:
                load += q[v]
                if load > self.Q:
                    return None
                picked.add(v)
            else:
                if v - n not in picked:
                    return None
                load += q[v]
                if load < 0:
                    return None
        nodes = (self.origin,) + tuple(seq) + (self.dest,)
        m = len(nodes)
        pos = {v: k for k, v in enumerate(nodes)}
        pairs = [(pos[v - n], k) for k, v in enumerate(nodes) if n < v <= 2 * n]
        e, l, ds, D = self.e, self.l, self.ds, self.D
        L, T = self.L, self.T
        B = [e[v] for v in nodes]
        for _ in range(m + 5):
            changed = False
            for k in range(m - 1):
                a = nodes[k]
                b = nodes[k + 1]
                need = B[k] + ds[a] + D[a][b] + _MARG
                if need > B[k + 1]:
                    B[k + 1] = need
                    changed = True
            for ip, idr in pairs:
                need = B[idr] - ds[nodes[ip]] - L + _MARG
                if need > B[ip]:
                    B[ip] = need
                    changed = True
            need = B[m - 1] - T + _MARG
            if need > B[0]:
                B[0] = need
                changed = True
            for k in range(m):
                if B[k] > l[nodes[k]]:
                    return None
            if not changed:
                return B
        return None

    def info(self, seq):
        """Cached (cost, nodes, Ef, Lt, Pl, B) for a feasible route, or None."""
        if seq in self._info:
            return self._info[seq]
        B = self.sched(seq)
        res = None
        if B is not None:
            e, l, ds, D, q = self.e, self.l, self.ds, self.D, self.q
            nd = (self.origin,) + tuple(seq) + (self.dest,)
            m = len(nd)
            cost = 0.0
            for k in range(m - 1):
                cost += D[nd[k]][nd[k + 1]]
            Ef = [0.0] * m
            Ef[0] = e[nd[0]]
            for k in range(m - 1):
                a = nd[k]
                b = nd[k + 1]
                Ef[k + 1] = max(e[b], Ef[k] + ds[a] + D[a][b])
            Lt = [0.0] * m
            Lt[m - 1] = l[nd[m - 1]]
            for k in range(m - 2, -1, -1):
                a = nd[k]
                b = nd[k + 1]
                Lt[k] = min(l[a], Lt[k + 1] - ds[a] - D[a][b])
            Pl = [0] * m
            for k in range(1, m):
                Pl[k] = Pl[k - 1] + q[nd[k]]
            res = (cost, nd, Ef, Lt, Pl, B)
        if len(self._info) > 300000:
            self._info.clear()
        self._info[seq] = res
        return res

    def route_cost(self, seq):
        inf = self.info(seq)
        return math.inf if inf is None else inf[0]

    # ------------------------------------------------------------------ insertion
    def best_insert(self, p, seq):
        """Cheapest feasible insertion of request p into route seq: (delta, new_seq) or None."""
        key = (p, seq)
        if key in self._ins:
            return self._ins[key]
        res = self._best_insert(p, seq)
        if len(self._ins) > 400000:
            self._ins.clear()
        self._ins[key] = res
        return res

    def _best_insert(self, p, seq):
        inf = self.info(seq)
        if inf is None:
            return None
        _, nd, Ef, Lt, Pl, _ = inf
        n, Q = self.n, self.Q
        dr = p + n
        qp = self.q[p]
        e, l, ds, D = self.e, self.l, self.ds, self.D
        m = len(nd)
        dsp, dsd = ds[p], ds[dr]
        cands = []
        for i in range(m - 1):
            if Pl[i] + qp > Q:
                continue
            A = nd[i]
            Bn = nd[i + 1]
            arr = max(e[p], Ef[i] + ds[A] + D[A][p])
            if arr > l[p]:
                continue
            last_node = p
            last_dep = arr + dsp
            for j in range(i, m - 1):
                if j > i:
                    v = nd[j]
                    if Pl[j] + qp > Q:
                        break
                    a = max(e[v], last_dep + D[last_node][v])
                    if a > l[v]:
                        break
                    last_node = v
                    last_dep = a + ds[v]
                a_d = max(e[dr], last_dep + D[last_node][dr])
                if a_d > l[dr]:
                    continue
                Ct = nd[j + 1]
                a_n = max(e[Ct], a_d + dsd + D[dr][Ct])
                if a_n > Lt[j + 1]:
                    continue
                if j == i:
                    delta = D[A][p] + D[p][dr] + D[dr][Bn] - D[A][Bn]
                else:
                    C = nd[j]
                    delta = (D[A][p] + D[p][Bn] - D[A][Bn]) + (D[C][dr] + D[dr][Ct] - D[C][Ct])
                cands.append((delta, i, j))
        if not cands:
            return None
        cands.sort()
        for delta, i, j in cands:
            ns = list(seq)
            ns.insert(j, dr)
            ns.insert(i, p)
            t = tuple(ns)
            if self.sched(t) is not None:
                return (delta, t)
        return None


# ---------------------------------------------------------------------- solution helpers
def _objective(M, routes, unassigned):
    total = 0.0
    for seq in routes:
        total += M.route_cost(seq)
    return total + _PEN_FACTOR * M.avg_arc * len(unassigned)


_PEN_FACTOR = 100.0
_SA_START = 0.05
_SA_END = 0.001


def _remove(routes, rem, n):
    out = []
    for seq in routes:
        out.append(tuple(x for x in seq if (x if x <= n else x - n) not in rem))
    return out


def _construct(M, rng):
    K = M.K
    routes = [() for _ in range(K)]
    unassigned = []
    order = sorted(range(1, M.n + 1), key=lambda p: (M.e[p], M.l[p]))
    for p in order:
        best = None
        for k in range(K):
            r = M.best_insert(p, routes[k])
            if r is not None and (best is None or r[0] < best[0]):
                best = (r[0], k, r[1])
        if best is None:
            unassigned.append(p)
        else:
            routes[best[1]] = best[2]
    return routes, unassigned


def _repair(M, routes, pending, rng, mode, noise):
    K = M.K
    routes = list(routes)
    failed = []
    opts = {p: [M.best_insert(p, routes[k]) for k in range(K)] for p in pending}
    pending = list(pending)
    while pending:
        scored = []
        still = []
        for p in pending:
            cand = []
            for k in range(K):
                r = opts[p][k]
                if r is not None:
                    cand.append((r[0] + noise * rng.random(), k, r[1]))
            if not cand:
                failed.append(p)
                continue
            still.append(p)
            cand.sort(key=lambda c: c[0])
            if mode == "regret":
                regret = cand[1][0] - cand[0][0] if len(cand) > 1 else 1e9
                score = (regret, -cand[0][0])
            else:
                score = (-cand[0][0], 0.0)
            scored.append((score, p, cand[0][1], cand[0][2]))
        pending = still
        if not scored:
            break
        scored.sort(key=lambda s: s[0], reverse=True)
        _, p, k, new_seq = scored[0]
        routes[k] = new_seq
        pending.remove(p)
        for p2 in pending:
            opts[p2][k] = M.best_insert(p2, routes[k])
    return routes, failed


def _destroy(M, routes, rng, q, op):
    n = M.n
    assigned = [x for seq in routes for x in seq if x <= n]
    if not assigned:
        return set()
    q = min(q, len(assigned))
    if op == 0:  # random requests
        return set(rng.sample(assigned, q))
    if op == 1:  # worst removal (largest saving), noisy
        scored = []
        for seq in routes:
            if not seq:
                continue
            base = M.route_cost(seq)
            for p in seq:
                if p > n:
                    continue
                ns = tuple(x for x in seq if x != p and x != p + n)
                saving = base - M.route_cost(ns)
                if saving == math.inf or saving != saving:
                    saving = 0.0
                scored.append((saving * (1.0 + 0.3 * rng.random()), p))
        scored.sort(reverse=True)
        return set(p for _, p in scored[:q])
    if op == 2:  # Shaw relatedness on positions and time windows
        e, D = M.e, M.D
        seed = rng.choice(assigned)
        rel = []
        for p in assigned:
            if p == seed:
                continue
            r = D[seed][p] + D[seed + n][p + n] + abs(e[seed] - e[p]) + abs(e[seed + n] - e[p + n])
            rel.append((r * (1.0 + 0.5 * rng.random()), p))
        rel.sort()
        out = {seed}
        for _, p in rel:
            if len(out) >= q:
                break
            out.add(p)
        return out
    # route removal: whole routes until q requests are removed
    nonempty = [seq for seq in routes if seq]
    rng.shuffle(nonempty)
    out = set()
    for seq in nonempty:
        if len(out) >= q:
            break
        out.update(x for x in seq if x <= n)
    return out


def _alns(M, rng, routes, unassigned, deadline, t_begin):
    n = M.n
    routes = list(routes)
    unassigned = list(unassigned)
    cur_obj = _objective(M, routes, unassigned)
    best = None
    if not unassigned:
        best = (cur_obj, list(routes))
    t_start_sa = _SA_START * M.avg_arc
    t_end_sa = _SA_END * M.avg_arc
    D_W = [1.0, 1.0, 1.0, 1.0]
    R_W = [1.0, 1.0]
    D_S = [0.0] * 4
    D_U = [0] * 4
    R_S = [0.0] * 2
    R_U = [0] * 2
    qmin = max(1, int(0.1 * n))
    qmax = max(qmin, int(0.4 * n))
    it = 0
    span = max(1e-3, deadline - t_begin)
    while time.time() < deadline:
        it += 1
        frac = min(1.0, (time.time() - t_begin) / span)
        temp = t_start_sa * (t_end_sa / t_start_sa) ** frac
        d_op = _roulette(rng, D_W)
        r_op = _roulette(rng, R_W)
        qn = rng.randint(qmin, qmax)
        rem = _destroy(M, routes, rng, qn, d_op)
        if not rem:
            continue
        new_routes = _remove(routes, rem, n)
        pending = list(unassigned) + sorted(rem)
        noise = 0.5 * M.avg_arc * rng.random() * 0.5
        new_routes, failed = _repair(M, new_routes, pending, rng, "regret" if r_op == 1 else "greedy", noise)
        new_un = sorted(failed)
        new_obj = _objective(M, new_routes, new_un)
        accepted = False
        if new_obj < cur_obj or rng.random() < math.exp(-(new_obj - cur_obj) / max(temp, 1e-9)):
            accepted = True
        score = 0
        if not new_un and (best is None or new_obj < best[0] - 1e-9):
            score = 3
            best = (new_obj, list(new_routes))
        elif accepted and new_obj < cur_obj - 1e-9:
            score = 2
        elif accepted:
            score = 1
        D_S[d_op] += score
        D_U[d_op] += 1
        R_S[r_op] += score
        R_U[r_op] += 1
        if accepted:
            routes = new_routes
            unassigned = new_un
            cur_obj = new_obj
        if it % 50 == 0:
            for i in range(4):
                if D_U[i]:
                    D_W[i] = max(0.05, 0.8 * D_W[i] + 0.2 * D_S[i] / D_U[i])
                D_S[i] = 0.0
                D_U[i] = 0
            for i in range(2):
                if R_U[i]:
                    R_W[i] = max(0.05, 0.8 * R_W[i] + 0.2 * R_S[i] / R_U[i])
                R_S[i] = 0.0
                R_U[i] = 0
    return best


def _roulette(rng, weights):
    total = sum(weights)
    x = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if x <= acc:
            return i
    return len(weights) - 1


def _output(M, routes):
    n, K = M.n, M.K
    dest = M.dest
    ds = M.ds
    sol_routes = {}
    service = {}
    total = 0.0
    for k in range(K):
        seq = routes[k]
        inf = M.info(seq)
        nd = (M.origin,) + tuple(seq) + (dest,)
        sol_routes[str(k)] = [int(v) for v in nd]
        if inf is None:
            continue
        B = inf[5]
        total += inf[0]
        for idx, v in enumerate(nd):
            if v == M.origin:
                service[f"depot_start_{k}"] = float(B[idx])
            elif v == dest:
                service[f"depot_end_{k}"] = float(B[idx])
            else:
                service[str(v)] = float(B[idx])
    ride = {}
    for p in range(1, n + 1):
        if str(p) in service and str(p + n) in service:
            ride[str(p)] = service[str(p + n)] - (service[str(p)] + ds[p])
    return {
        "objective_value": float(total),
        "routes": sol_routes,
        "service_times": service,
        "ride_times": ride,
    }


def solve(instance: dict, time_limit_s: int) -> dict:
    deadline = T0 + min(0.80 * float(time_limit_s), float(time_limit_s) - 10.0)
    M = _Model(instance)
    rng = random.Random(20260)
    routes, unassigned = _construct(M, rng)
    best_routes = list(routes)
    try:
        best = _alns(M, rng, routes, unassigned, deadline, time.time())
        if best is not None:
            best_routes = best[1]
    except Exception:
        pass
    return _output(M, best_routes)


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
