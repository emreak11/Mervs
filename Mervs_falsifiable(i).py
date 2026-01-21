import numpy as np
import matplotlib.pyplot as plt


"""
Mervs_falsifiable.py
-------------------
Goal: Make the simulation falsifiable (non self-confirming) by:
  - Removing direct morphology -> epsilon/tau mappings from the core experiment
  - Introducing a controlled perturbation: neighbor rewiring rate (rho)
  - Adding null switches:
        Null-1: fixed neighbors (rho = 0)
        Null-2: rewiring but global-random replacement (rho > 0, global)
        Alt:    rewiring with LOCAL replacement (rho > 0, local), mimicking local rearrangements
  - Using a single primary output: boundary/defect rate proxy (phase mismatches on edges)

Core scientific claim we can test:
  "Changing neighbor renewal statistics (rho) changes defect rate, even with fixed epsilon and tau."
If this does not happen (or cannot be distinguished from nulls), the hypothesis is falsified in this model class.
"""

# -------------------------
# 1) Tissue geometry and neighbor graph
# -------------------------
def build_grid_positions(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])  # (N,2)

def build_neighbor_graph_grid(nx: int, ny: int, mode: str = "von_neumann"):
    """Initial neighbor list for each node (directed list, but symmetric for grid)."""
    def idx(i, j): return i * ny + j
    N = nx * ny
    nbrs = [set() for _ in range(N)]
    for i in range(nx):
        for j in range(ny):
            a = idx(i, j)
            if mode == "von_neumann":
                candidates = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
            elif mode == "moore":
                candidates = [(i+di, j+dj) for di in (-1,0,1) for dj in (-1,0,1) if not (di==0 and dj==0)]
            else:
                raise ValueError("mode must be von_neumann or moore")
            for ii, jj in candidates:
                if 0 <= ii < nx and 0 <= jj < ny:
                    b = idx(ii, jj)
                    nbrs[a].add(b)
    return nbrs  # list[set[int]]

def _precompute_local_candidates(pos: np.ndarray, local_radius: float) -> list[np.ndarray]:
    """For each node i, candidate nodes within Euclidean distance <= local_radius (excluding itself)."""
    N = pos.shape[0]
    cand = []
    for i in range(N):
        d = np.linalg.norm(pos - pos[i], axis=1)
        ids = np.where((d <= local_radius) & (d > 0))[0]
        cand.append(ids)
    return cand

def rewire_neighbors(
    nbrs: list,
    pos: np.ndarray,
    rng: np.random.Generator,
    rho: float,
    mode: str,
    local_candidates: list[np.ndarray] | None = None,
):
    """
    Rewire neighbor graph in-place with rate rho.
    - rho is probability per node per rewire event (per call).
    - mode:
        * "global": choose new neighbor uniformly from all nodes
        * "local":  choose new neighbor from local_candidates[i]
    Keeps degree ~ constant by replacing one neighbor with another.
    Ensures symmetry by rewiring both directions (undirected-like).
    """
    if rho <= 0:
        return

    N = len(nbrs)
    for i in range(N):
        if rng.random() >= rho:
            continue
        if len(nbrs[i]) == 0:
            continue

        # pick an existing neighbor to drop
        j_drop = rng.choice(np.fromiter(nbrs[i], dtype=int))

        # candidate pool for new neighbor
        if mode == "global":
            pool = np.arange(N, dtype=int)
        elif mode == "local":
            if local_candidates is None:
                raise ValueError("local_candidates required for local mode")
            pool = local_candidates[i]
            if pool.size == 0:
                continue
        else:
            raise ValueError("mode must be 'global' or 'local'")

        # sample until we find a valid new neighbor (avoid self, avoid duplicates)
        # bounded attempts to prevent infinite loops in tight local pools
        for _ in range(50):
            j_new = int(rng.choice(pool))
            if j_new == i:
                continue
            if j_new in nbrs[i]:
                continue
            # accept
            break
        else:
            continue  # failed to find a new neighbor

        # apply symmetric rewiring
        nbrs[i].discard(j_drop)
        nbrs[i].add(j_new)

        nbrs[j_drop].discard(i)
        nbrs[j_new].add(i)

# -------------------------
# 2) Delayed Kuramoto with dynamic neighbors (rewiring)
# -------------------------
def simulate_delayed_kuramoto_dynamic(
    nbrs,               # list[set[int]] neighbor graph (can be rewired in place)
    pos,                # (N,2) positions (only needed for local candidate computation outside)
    eps: float,         # coupling strength (fixed; not morphology-derived)
    tau: float,         # delay (fixed; not morphology-derived)
    omega: np.ndarray,  # (N,) natural frequencies
    T: float = 300.0,
    dt: float = 0.05,
    noise_sigma: float = 0.0,
    seed: int = 0,
    rewiring_rho: float = 0.0,
    rewiring_mode: str = "local",      # "global" or "local"
    rewiring_interval: int = 5,        # apply rewiring every K steps
    local_candidates: list[np.ndarray] | None = None,
):
    rng = np.random.default_rng(seed)
    N = omega.shape[0]
    steps = int(np.round(T / dt))
    delay_steps = int(np.round(tau / dt))
    if delay_steps < 0:
        raise ValueError("tau must be >= 0")

    theta = rng.uniform(0, 2*np.pi, size=N)

    # ring buffer for delayed theta(t - tau)
    hist = np.zeros((delay_steps + 1, N), dtype=float)
    hist[:] = theta
    out = np.zeros((steps, N), dtype=float)

    for t in range(steps):
        # rewiring event
        if rewiring_rho > 0 and (t % rewiring_interval == 0):
            rewire_neighbors(
                nbrs=nbrs,
                pos=pos,
                rng=rng,
                rho=rewiring_rho,
                mode=rewiring_mode,
                local_candidates=local_candidates,
            )

        theta_del = hist[0]

        # compute coupling from current neighbors with row-normalized equal weights
        coupling = np.zeros(N, dtype=float)
        for i in range(N):
            ni = nbrs[i]
            if not ni:
                continue
            # equal weights over neighbors
            # sum_j (1/deg(i)) * sin(theta_j(t-tau) - theta_i(t))
            deg = len(ni)
            # vectorize over neighbors
            js = np.fromiter(ni, dtype=int)
            coupling[i] = np.sin(theta_del[js] - theta[i]).sum() / deg

        dtheta = omega + eps * coupling
        if noise_sigma > 0:
            dtheta += noise_sigma * rng.normal(size=N) / np.sqrt(dt)

        theta = theta + dt * dtheta
        theta = np.mod(theta, 2*np.pi)

        out[t] = theta

        # update buffer
        hist[:-1] = hist[1:]
        hist[-1] = theta

    return out

# -------------------------
# 3) Primary metric: defect/boundary proxy
# -------------------------
def defect_rate_proxy(theta_t, nbrs, threshold=np.pi/2):
    """
    Edge-based proxy: fraction of neighbor edges with large phase mismatch.
    Counts directed edges (i->j). Since graph is maintained symmetric, this is ~2x undirected count.
    """
    cnt = 0
    tot = 0
    N = len(nbrs)
    for i in range(N):
        for j in nbrs[i]:
            d = np.angle(np.exp(1j * (theta_t[j] - theta_t[i])))
            if np.abs(d) > threshold:
                cnt += 1
            tot += 1
    return cnt / max(tot, 1)

def defect_window_stats(theta, nbrs, threshold=np.pi/2, tail_steps=300):
    """
    Compute mean/std of defect proxy over a tail window to avoid transient bias.
    """
    tail = theta[-tail_steps:]
    vals = np.array([defect_rate_proxy(tail[t], nbrs, threshold=threshold) for t in range(tail.shape[0])])
    return float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0), vals

# -------------------------
# 4) Experiment runner with null switches
# -------------------------
def run_rewiring_sweep(
    nx=20,
    ny=6,
    mode="von_neumann",
    eps=1.2,
    tau=1.0,
    mean_period=30.0,          # time-units
    omega_cv=0.02,
    T=300.0,
    dt=0.05,
    tail_steps=300,
    threshold=np.pi/2,
    rhos=(0.0, 0.02, 0.05, 0.1, 0.2),
    n_reps=20,
    rewiring_interval=5,
    local_radius=2.1,          # ~ includes manhattan-2 neighbors at dx=dy=1
    noise_sigma=0.0,
    seed0=0,
):
    """
    Produces results for three conditions:
      - null_fixed: rho=0 (neighbors fixed)
      - null_global: rho>0 with global replacement
      - alt_local: rho>0 with local replacement
    """
    pos = build_grid_positions(nx, ny, dx=1.0, dy=1.0)
    N = nx * ny

    mean_omega = 2*np.pi / mean_period
    base_rng = np.random.default_rng(seed0)
    omega = mean_omega * (1.0 + omega_cv * base_rng.normal(size=N))

    local_candidates = _precompute_local_candidates(pos, local_radius=local_radius)

    results = []

    for rho in rhos:
        for condition in ("null_fixed", "null_global", "alt_local"):
            print(f"Running rho={rho}, condition={condition}", flush=True)

            # enforce rho for fixed condition
            rho_eff = 0.0 if condition == "null_fixed" else float(rho)
            mode_eff = "global" if condition == "null_global" else "local"

            rep_means = []
            for rep in range(n_reps):
                # fresh neighbor graph each replicate to avoid cross-run contamination
                nbrs = build_neighbor_graph_grid(nx, ny, mode=mode)
                theta = simulate_delayed_kuramoto_dynamic(
                    nbrs=nbrs,
                    pos=pos,
                    eps=eps,
                    tau=tau,
                    omega=omega,
                    T=T,
                    dt=dt,
                    noise_sigma=noise_sigma,
                    seed=seed0 + 1000 * rep + 17,
                    rewiring_rho=rho_eff,
                    rewiring_mode=mode_eff,
                    rewiring_interval=rewiring_interval,
                    local_candidates=local_candidates,
                )
                m, s, _ = defect_window_stats(theta, nbrs, threshold=threshold, tail_steps=tail_steps)
                rep_means.append(m)

            rep_means = np.asarray(rep_means, dtype=float)
            results.append({
                "rho": float(rho_eff),
                "condition": condition,
                "defect_mean": float(rep_means.mean()),
                "defect_std": float(rep_means.std(ddof=1) if rep_means.size > 1 else 0.0),
                "n_reps": int(n_reps),
                "eps": float(eps),
                "tau": float(tau),
                "mean_period": float(mean_period),
                "rewiring_interval": int(rewiring_interval),
                "local_radius": float(local_radius),
                "threshold": float(threshold),
            })

    return results

def plot_sweep_results(results):
    """
    Simple plot: defect_mean vs rho for each condition.
    """
    # group
    conds = sorted(set(r["condition"] for r in results))
    plt.figure()
    for c in conds:
        xs = [r["rho"] for r in results if r["condition"] == c]
        ys = [r["defect_mean"] for r in results if r["condition"] == c]
        es = [r["defect_std"] for r in results if r["condition"] == c]
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        es = np.asarray(es)[order]
        plt.errorbar(xs, ys, yerr=es, marker="o", linestyle="-", capsize=3, label=c)

    plt.xlabel("rewiring rate rho (effective)")
    plt.ylabel("defect rate (tail-window mean)")
    plt.title("Defect rate vs neighbor rewiring (nulls vs alt)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":


    # Default run: fast-ish sweep (adjust n_reps upward for stronger stats)
    results = run_rewiring_sweep(
        rhos=(0.0, 0.02, 0.05, 0.1, 0.2),
        n_reps=20,
        T=300.0,
        dt=0.05,
        tail_steps=300,
        rewiring_interval=5,
        local_radius=2.1,
        noise_sigma=0.0,
    )
    for r in results:
        print(r)
    plot_sweep_results(results)





