import time

T0 = time.time()

import math
import random
from collections import deque

EPS = 1e-9
KN = 30  # neighbour list length
KLS = 20  # neighbours scanned by local search
KINS = 25  # neighbours scanned by cheapest insertion


def solve(instance: dict, time_limit_s: int) -> dict:
    deadline = T0 + max(0.0, min(0.80 * time_limit_s, time_limit_s - 10))
    rng = random.Random(20240601)

    n = int(instance["num_customers"])
    cap = instance["vehicle_capacity"]
    raw = instance["distance_matrix"]
    cust = {c["id"]: c for c in instance["customers"]}
    N = 2 * n
    # expanded node -> original location index (linehaul i and backhaul n+i share location i)
    orig = [0] + list(range(1, n + 1)) + list(range(1, n + 1))
    D = [[raw[a][b] for b in orig] for a in orig]
    dem = [0] * (N + 1)  # linehaul i: delivery; backhaul n+i: pickup
    for i in range(1, n + 1):
        dem[i] = cust[i]["delivery_demand"]
        dem[n + i] = cust[i]["pickup_demand"]

    def ok(seq):
        """Load and adjacency feasibility of one route (depot excluded from seq)."""
        load = 0
        for u in seq:
            if u <= n:
                load += dem[u]
        if load > cap:
            return False
        prev = 0
        for u in seq:
            if u <= n:
                load -= dem[u]
                if prev == u + n:  # arc n+i -> i is forbidden
                    return False
            else:
                load += dem[u]
            if load > cap:
                return False
            prev = u
        return True

    def rcost(seq):
        if not seq:
            return 0
        c = D[0][seq[0]] + D[seq[-1]][0]
        for k in range(len(seq) - 1):
            c += D[seq[k]][seq[k + 1]]
        return c

    def verify(routes):
        """Mirror of the checker's hard constraints; returns total distance or None."""
        seen = [False] * (N + 1)
        total = 0
        for seq in routes:
            if not seq or not ok(seq):
                return None
            for u in seq:
                if not (1 <= u <= N) or seen[u]:
                    return None
                seen[u] = True
            total += rcost(seq)
        if not all(seen[1:]):
            return None
        return total

    nbrs = [[]]
    for u in range(1, N + 1):
        row = D[u]
        others = [v for v in range(1, N + 1) if v != u]
        others.sort(key=row.__getitem__)
        nbrs.append(others[:KN])

    class State:
        def __init__(self, routes):
            self.routes = [list(r) for r in routes if r]
            self.rc = [rcost(r) for r in self.routes]
            self.rid = [-1] * (N + 1)
            for r, seq in enumerate(self.routes):
                for u in seq:
                    self.rid[u] = r
            self.cost = sum(self.rc)

        def set(self, r, seq):
            c = rcost(seq)
            self.cost += c - self.rc[r]
            self.routes[r] = seq
            self.rc[r] = c
            for u in seq:
                self.rid[u] = r

        def add(self, seq):
            r = len(self.routes)
            c = rcost(seq)
            self.routes.append(seq)
            self.rc.append(c)
            self.cost += c
            for u in seq:
                self.rid[u] = r

        def delta(self, changes):
            d = 0
            for r, seq in changes.items():
                if not ok(seq):
                    return None
                d += rcost(seq) - (self.rc[r] if r >= 0 else 0)
            return d

        def apply(self, changes):
            for r, seq in changes.items():
                if r < 0:
                    self.add(seq)
                else:
                    self.set(r, seq)

    # ---------------- neighbourhood moves: return (delta, changes) or None ----------------

    def mv_relocate(st, u, v):
        r1 = st.rid[u]
        r2 = st.rid[v]
        A = st.routes[r1]
        Ar = A[: A.index(u)] + A[A.index(u) + 1 :]
        best = None
        if r1 == r2:
            q = Ar.index(v)
            for k in (q, q + 1):
                s = Ar[:k] + [u] + Ar[k:]
                if s == A:
                    continue
                ch = {r1: s}
                d = st.delta(ch)
                if d is not None and (best is None or d < best[0]):
                    best = (d, ch)
            return best
        B = st.routes[r2]
        q = B.index(v)
        for k in (q, q + 1):
            ch = {r1: Ar, r2: B[:k] + [u] + B[k:]}
            d = st.delta(ch)
            if d is not None and (best is None or d < best[0]):
                best = (d, ch)
        return best

    def mv_swap(st, u, v):
        r1 = st.rid[u]
        r2 = st.rid[v]
        if r1 == r2:
            s = list(st.routes[r1])
            i = s.index(u)
            j = s.index(v)
            s[i], s[j] = s[j], s[i]
            ch = {r1: s}
        else:
            a = list(st.routes[r1])
            b = list(st.routes[r2])
            i = a.index(u)
            j = b.index(v)
            a[i] = v
            b[j] = u
            ch = {r1: a, r2: b}
        d = st.delta(ch)
        return None if d is None else (d, ch)

    def mv_2opt(st, u, v):
        r = st.rid[u]
        A = st.routes[r]
        i = A.index(u)
        j = A.index(v)
        if i > j:
            i, j = j, i
        s = A[:i] + A[i : j + 1][::-1] + A[j + 1 :]
        ch = {r: s}
        d = st.delta(ch)
        return None if d is None else (d, ch)

    def mv_2opt_star(st, u, v):
        r1 = st.rid[u]
        r2 = st.rid[v]
        A = st.routes[r1]
        B = st.routes[r2]
        p = A.index(u)
        q = B.index(v)
        best = None
        for a in (p, p + 1):
            for b in (q, q + 1):
                ch = {r1: A[:a] + B[b:], r2: B[:b] + A[a:]}
                d = st.delta(ch)
                if d is not None and (best is None or d < best[0]):
                    best = (d, ch)
        return best

    def mv_pair(st, u, v):
        """Move customer i (both its nodes i and n+i, in that order, adjacent) next to v."""
        i = u if u <= n else u - n
        w = i + n
        if v == i or v == w:
            return None
        cur = {}
        for x in (i, w):
            r = st.rid[x]
            seq = cur.get(r, st.routes[r])
            cur[r] = [y for y in seq if y != x]
        rv = st.rid[v]
        base = cur.get(rv, st.routes[rv])
        q = base.index(v)
        best = None
        for k in (q, q + 1):
            ch = dict(cur)
            ch[rv] = base[:k] + [i, w] + base[k:]
            d = st.delta(ch)
            if d is not None and (best is None or d < best[0]):
                best = (d, ch)
        return best

    def mv_newroute(st, u):
        r = st.rid[u]
        Ar = [y for y in st.routes[r] if y != u]
        ch = {r: Ar, -1: [u]}
        d = st.delta(ch)
        return None if d is None else (d, ch)

    def local_search(st, nodes, deadline_s):
        dq = deque()
        inq = [False] * (N + 1)
        for x in nodes:
            if not inq[x]:
                inq[x] = True
                dq.append(x)
        cnt = 0
        while dq:
            cnt += 1
            if (cnt & 31) == 0 and time.time() > deadline_s:
                break
            u = dq.popleft()
            inq[u] = False
            if st.rid[u] < 0:
                continue
            found = mv_newroute(st, u)
            if found is not None and found[0] >= 0:
                found = None
            if found is None:
                for v in nbrs[u][:KLS]:
                    if st.rid[v] < 0:
                        continue
                    cands = [mv_relocate(st, u, v), mv_swap(st, u, v), mv_pair(st, u, v)]
                    if st.rid[v] == st.rid[u]:
                        cands.append(mv_2opt(st, u, v))
                    else:
                        cands.append(mv_2opt_star(st, u, v))
                    best = None
                    for c in cands:
                        if c is not None and c[0] < 0 and (best is None or c[0] < best[0]):
                            best = c
                    if best is not None:
                        found = best
                        break
            if found is not None:
                ch = found[1]
                st.apply(ch)
                for seq in ch.values():
                    for x in seq:
                        if not inq[x]:
                            inq[x] = True
                            dq.append(x)

    # ---------------- construction ----------------

    def clarke_wright():
        rts = {x: [x] for x in range(1, N + 1)}
        owner = list(range(N + 1))
        sav = []
        for u in range(1, N + 1):
            du = D[0][u]
            row = D[u]
            for v in range(1, N + 1):
                if v != u:
                    s = du + D[0][v] - row[v]
                    if s > 0:
                        sav.append((s, u, v))
        sav.sort(reverse=True)
        for _, u, v in sav:
            ru = owner[u]
            rv = owner[v]
            if ru == rv:
                continue
            A = rts[ru]
            B = rts[rv]
            if A[-1] != u or B[0] != v:
                continue
            new = A + B
            if not ok(new):
                continue
            rts[ru] = new
            del rts[rv]
            for x in B:
                owner[x] = ru
        return list(rts.values())

    def singles():
        out = []
        for i in range(1, n + 1):
            if ok([i, n + i]):
                out.append([i, n + i])
            else:
                out.append([i])
                out.append([n + i])
        return out

    # ---------------- ruin and recreate ----------------

    def ruin(st, k):
        present = [x for x in range(1, N + 1) if st.rid[x] >= 0]
        mode = rng.random()
        rem = set()
        if mode < 0.25:  # random nodes
            rem.update(rng.sample(present, min(k, len(present))))
        elif mode < 0.55:  # related (neighbour-based) removal
            seed = rng.choice(present)
            rem.add(seed)
            frontier = deque([seed])
            while len(rem) < k:
                if not frontier:
                    rem.add(rng.choice(present))
                    continue
                x = frontier.popleft()
                for y in nbrs[x]:
                    if len(rem) >= k:
                        break
                    if st.rid[y] >= 0 and y not in rem and rng.random() < 0.7:
                        rem.add(y)
                        frontier.append(y)
        elif mode < 0.8:  # worst removal (largest saving from taking the node out)
            gains = []
            for x in present:
                r = st.rid[x]
                seq = st.routes[r]
                g = st.rc[r] - rcost([y for y in seq if y != x])
                gains.append((g * rng.uniform(0.8, 1.2), x))
            gains.sort(reverse=True)
            for _, x in gains:
                if len(rem) >= k:
                    break
                rem.add(x)
        else:  # whole routes
            rids = [r for r in range(len(st.routes)) if st.routes[r]]
            rng.shuffle(rids)
            for r in rids:
                if len(rem) >= k:
                    break
                rem.update(st.routes[r])
        for x in list(rem):  # customers' pickup and delivery are often removed together
            if rng.random() < 0.5:
                y = x + n if x <= n else x - n
                if st.rid[y] >= 0:
                    rem.add(y)
        touched = {st.rid[x] for x in rem}
        for r in touched:
            st.set(r, [y for y in st.routes[r] if y not in rem])
        for x in rem:
            st.rid[x] = -1
        return rem, touched

    def recreate(st, rem):
        order = list(rem)
        if rng.random() < 0.5:
            order.sort(key=lambda x: -dem[x])
        else:
            rng.shuffle(order)
        for x in order:
            best = None
            for v in nbrs[x][:KINS]:
                r = st.rid[v]
                if r < 0:
                    continue
                seq = st.routes[r]
                q = seq.index(v)
                for k in (q, q + 1):
                    s = seq[:k] + [x] + seq[k:]
                    d = rcost(s) - st.rc[r]
                    if best is not None and d >= best[0]:
                        continue
                    if ok(s):
                        best = (d, r, s)
            if best is None or 2 * D[0][x] < best[0]:
                st.add([x])
            else:
                st.set(best[1], best[2])

    # ---------------- main search ----------------

    holder = {"routes": None, "cost": None}

    def record(st):
        c = st.cost
        if holder["cost"] is None or c < holder["cost"] - EPS:
            holder["routes"] = [list(r) for r in st.routes if r]
            holder["cost"] = c

    try:
        init = clarke_wright()
        if verify(init) is None:
            raise ValueError("invalid construction")
    except Exception:
        init = singles()
    if verify(init) is None:
        init = singles()
    holder["routes"] = init
    holder["cost"] = verify(init)

    try:
        st = State(init)
        local_search(st, range(1, N + 1), deadline)
        record(st)
        cur = st.cost
        avg = cur / max(1, N + len(st.routes))
        t_hi = 1.0 * avg
        t_lo = 0.05 * avg
        t_start = time.time()
        span = max(1e-3, deadline - t_start)
        while time.time() < deadline:
            frac = (time.time() - t_start) / span
            temp = max(1e-9, t_hi * (t_lo / max(t_hi, 1e-9)) ** frac)
            snap = [list(r) for r in st.routes if r]
            k = max(2, int(N * rng.uniform(0.05, 0.20)))
            rem, ruin_touched = ruin(st, k)
            recreate(st, rem)
            local_search(st, list(rem), deadline)
            new = st.cost
            if new < cur - EPS or rng.random() < math.exp(-(new - cur) / temp):
                cur = new
                record(st)
                st = State(st.routes)
            else:
                st = State(snap)
    except Exception:
        pass

    final = holder["routes"]
    total = verify(final)
    if total is None:
        final = singles()
        total = verify(final)

    routes_out = []
    detail = []
    depot = {"node": 0, "role": "depot", "customer": 0, "quantity": 0}
    for seq in final:
        routes_out.append([0] + list(seq) + [0])
        stops = [dict(depot)]
        for u in seq:
            if u <= n:
                stops.append({"node": u, "role": "linehaul", "customer": u, "quantity": dem[u]})
            else:
                stops.append({"node": u, "role": "backhaul", "customer": u - n, "quantity": dem[u]})
        stops.append(dict(depot))
        detail.append(stops)
    return {
        "objective_value": float(total),
        "routes": routes_out,
        "routes_detailed": detail,
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
