import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# 1) Geometry -> weights L_ij (hücre ağları için en minimal düzeyde 2D modelleme "3D ?")
# -------------------------
def build_grid_positions(nx, ny, dx=1.0, dy=1.0):
    """2D grid positions (pseudo-tissue)."""
    xs = np.arange(nx) * dx
    ys = np.arange(ny) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.column_stack([X.ravel(), Y.ravel()])  # (N,2)
    return pos

def build_neighbor_graph_grid(nx, ny, mode="von_neumann"):  #von neumann for 4-connectivity, moore for 8-connectivity, "should z-axis ?")
    """komşu hücre"""
    def idx(i, j): return i * ny + j
    N = nx * ny
    nbrs = [[] for _ in range(N)]
    for i in range(nx):
        for j in range(ny):
            a = idx(i, j)
            candidates = []
            if mode == "von_neumann":
                candidates = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
            elif mode == "moore":
                candidates = [(i+di, j+dj) for di in (-1,0,1) for dj in (-1,0,1) if not (di==0 and dj==0)]
            else:
                raise ValueError("mode must be von_neumann or moore")
            for ii, jj in candidates:
                if 0 <= ii < nx and 0 <= jj < ny:
                    b = idx(ii, jj)
                    nbrs[a].append(b)
    return nbrs

def morphology_to_L(nbrs, pos, cell_radius, anisotropy=0.0, axis=np.array([1.0, 0.0])):
    """
    Build weighted adjacency matrix L_ij from morphology.
    Daha basit (ileri seviye için adımlara bak): L_ij proportional to 'contact potential' based on distance,
    plus optional anisotropy (directional bias).
    """
    N = pos.shape[0]
    L = np.zeros((N, N), dtype=float)

    axis = axis / (np.linalg.norm(axis) + 1e-12)

    for i in range(N):
        for j in nbrs[i]:
            rij = pos[j] - pos[i]
            d = np.linalg.norm(rij) + 1e-12

            # Base contact weight: closer -> larger, use radius as scale.
            # For grid, d is apprx. dx/dy, still ok as proxy.
            base = np.clip((2.0 * cell_radius - d) / (2.0 * cell_radius), 0.0, 1.0)

            # Optional anisotropy: bias coupling along a preferred axis
            # (models elongated cells/packing)
            dir_cos = np.dot(rij / d, axis)  # [-1,1]
            bias = 1.0 + anisotropy * (dir_cos**2)  # >=1 if anisotropy>0

            L[i, j] = base * bias

    # Normalize rows so sum_j L_ij = 1 (keeps scale interpretable)
    row_sums = L.sum(axis=1, keepdims=True) + 1e-12
    L = L / row_sums
    return L
#ileri adımlarda bias için verileri değiştirip farkı ortaya koymak gerekliyor

# -------------------------
# 2) Morphology -> epsilon_i and tau_ij mappings (minimal, testable)
# -------------------------
def epsilon_from_morphology(cell_radius, eps0=1.0, alpha=0.0, r_ref=1.0):
    """
    Example: epsilon increases with contact area proxy ~ r^2.
    Keep it simple: eps = eps0 * (1 + alpha*(r/r_ref - 1))
    """
    return eps0 * (1.0 + alpha * (cell_radius / r_ref - 1.0))

def tau_from_morphology(cell_radius, tau0=1.0, beta=0.0, r_ref=1.0):
    """
    Example: effective delay increases with radius (longer trafficking distance),
    or decreases (if contact stability increases). Choose sign via beta.
    """
    return tau0 * (1.0 + beta * (cell_radius / r_ref - 1.0))

# -------------------------
# 3) Delayed Kuramoto simulation
# -------------------------
def simulate_delayed_kuramoto(
    L,                 # (N,N) coupling weights
    eps,               # scalar or (N,) epsilon
    tau,               # scalar delay (for now)
    omega,             # (N,) natural frequencies
    T=200.0,
    dt=0.05,
    noise_sigma=0.0,
    seed=0
):
    """
    Euler integration with discrete delay buffer.
    theta history stored with a ring buffer.
    """
    rng = np.random.default_rng(seed)
    N = L.shape[0]
    steps = int(np.round(T / dt))
    delay_steps = int(np.round(tau / dt)) # essential for delay implementation (furhter log.)
    if delay_steps < 0:
        raise ValueError("tau must be >= 0")

    eps_vec = np.full(N, eps, dtype=float) if np.isscalar(eps) else np.asarray(eps, dtype=float)
    theta = rng.uniform(0, 2*np.pi, size=N)

    # history buffer: store theta at past times for delay lookup
    hist = np.zeros((delay_steps + 1, N), dtype=float)
    hist[:] = theta  # initialize constant history
    out = np.zeros((steps, N), dtype=float)

    for t in range(steps):
        # delayed state: theta(t - tau)
        theta_del = hist[0]  # oldest in buffer

        # coupling term: sum_j L_ij * sin(theta_j(t-tau) - theta_i(t))
        phase_diff = theta_del[None, :] - theta[:, None]   # (N,N) = theta_del[j] - theta[i]
        coupling = (L * np.sin(phase_diff)).sum(axis=1)    # (N,)

        dtheta = omega + eps_vec * coupling

        if noise_sigma > 0:
            dtheta += noise_sigma * rng.normal(size=N) / np.sqrt(dt)

        theta = theta + dt * dtheta
        # wrap to [0, 2pi) for numerical stability
        theta = np.mod(theta, 2*np.pi)

        out[t] = theta

        # update delay buffer: push new theta, pop oldest
        hist[:-1] = hist[1:]
        hist[-1] = theta

    return out  # (steps, N)

# -------------------------
# 4) Metrics: synchrony R, phase gradient, wave speed proxy, defect proxy
# -------------------------
def synchrony_R(theta_t):
    """Kuramoto order parameter R(t) from phases at time t: theta_t shape (N,)."""
    return np.abs(np.mean(np.exp(1j * theta_t)))

def unwrap_phase_field(theta_t, pos):
    """
    Convert phases to a continuous field for gradient estimation.
    Minimal trick: unwrap along sorted x then y (works ok for smooth waves).
    """
    # sort by x then y
    order = np.lexsort((pos[:, 1], pos[:, 0]))
    theta_sorted = theta_t[order]
    theta_unwrapped = np.unwrap(theta_sorted)
    return theta_unwrapped, order

def phase_gradient_slope(theta_t, pos):
    """
    Fit theta ~ a*x + b (1D slope along x). Assumes wave mostly along x.
    """
    theta_unwrapped, order = unwrap_phase_field(theta_t, pos)
    x = pos[order, 0]
    A = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A, theta_unwrapped, rcond=None)[0]
    return a  # slope dtheta/dx

def wave_speed_from_slope_and_frequency(slope, mean_omega):
    """
    For traveling waves: phase = kx - wt => slope ~ k, mean_omega ~ w.
    Speed v = w/k. Use mean_omega as proxy.
    """
    k = slope
    if np.abs(k) < 1e-9:
        return np.inf
    return mean_omega / k

def defect_rate_proxy(theta_t, nbrs, threshold=np.pi/2):
    """
    Proxy: count edges with large phase mismatch.
    Higher mismatch frequency ~ more defects.
    """
    N = len(nbrs)
    cnt = 0
    tot = 0
    for i in range(N):
        for j in nbrs[i]:
            # smallest circular difference
            d = np.angle(np.exp(1j*(theta_t[j] - theta_t[i])))
            if np.abs(d) > threshold:
                cnt += 1
            tot += 1
    return cnt / max(tot, 1)

# -------------------------
# 5) Experiment runner: sweep morphology -> eps/tau -> metrics
# -------------------------
def run_morphology_sweep():
    nx, ny = 20, 6
    pos = build_grid_positions(nx, ny, dx=1.0, dy=1.0)
    nbrs = build_neighbor_graph_grid(nx, ny, mode="von_neumann")
    N = nx * ny

    # Base oscillator frequencies around a mean (segmentation clock proxy)
    mean_omega = 2*np.pi / 30.0  # ~ period 30 time-units (arbitrary)
    omega = mean_omega * (1.0 + 0.02*np.random.default_rng(0).normal(size=N))

    # Sweep a morphology parameter: cell_radius or anisotropy
    radii = [0.8, 1.0, 1.2]
    anisotropies = [0.0, 0.5]

    results = []
    for r in radii:
        for aniso in anisotropies:
            L = morphology_to_L(nbrs, pos, cell_radius=r, anisotropy=aniso)

            eps = epsilon_from_morphology(r, eps0=1.2, alpha=1.0, r_ref=1.0)
            tau = tau_from_morphology(r, tau0=1.0, beta=0.3, r_ref=1.0)

            theta = simulate_delayed_kuramoto(
                L=L, eps=eps, tau=tau, omega=omega,
                T=300.0, dt=0.05, noise_sigma=0.0, seed=1
            )

            # take last window for metrics
            tail = theta[-200:]  # last 200 steps
            R_vals = np.array([synchrony_R(tail[t]) for t in range(tail.shape[0])])
            R_mean = float(R_vals.mean())

            slope = phase_gradient_slope(tail[-1], pos)
            v = wave_speed_from_slope_and_frequency(slope, mean_omega)

            defect = defect_rate_proxy(tail[-1], nbrs, threshold=np.pi/2)

            results.append({
                "radius": r,
                "anisotropy": aniso,
                "epsilon": float(eps),
                "tau": float(tau),
                "R_mean": R_mean,
                "phase_slope": float(slope),
                "wave_speed_proxy": float(v),
                "defect_proxy": float(defect),
            })

    return results

if __name__ == "__main__":
    res = run_morphology_sweep()
    for row in res:
        print(row)



def plot_run(theta, pos, nbrs, dt, title=""):
    """
    theta: (steps, N)
    pos: (N,2)
    """
    steps, N = theta.shape

    # 1) R(t)
    R = np.array([np.abs(np.mean(np.exp(1j * theta[t])) ) for t in range(steps)])

    # 2) defect proxy over time
    def defect_rate_proxy(theta_t, nbrs, threshold=np.pi/2):
        cnt = 0
        tot = 0
        for i in range(len(nbrs)):
            for j in nbrs[i]:
                d = np.angle(np.exp(1j*(theta_t[j] - theta_t[i])))
                if np.abs(d) > threshold:
                    cnt += 1
                tot += 1
        return cnt / max(tot, 1)

    defect = np.array([defect_rate_proxy(theta[t], nbrs) for t in range(steps)])

    t_axis = np.arange(steps) * dt

    # --- Plot 1: R(t)
    plt.figure()
    plt.plot(t_axis, R)
    plt.xlabel("time")
    plt.ylabel("R(t)")
    plt.title(f"Synchrony R(t) {title}".strip())
    plt.tight_layout()

    # --- Plot 2: defect(t)
    plt.figure()
    plt.plot(t_axis, defect)
    plt.xlabel("time")
    plt.ylabel("defect proxy")
    plt.title(f"Defect proxy {title}".strip())
    plt.tight_layout()

    # --- Plot 3: phase snapshot on tissue (scatter)
    theta_last = theta[-1]
    plt.figure()
    sc = plt.scatter(pos[:, 0], pos[:, 1], c=theta_last)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Phase snapshot θ(x,y) {title}".strip())
    plt.colorbar(sc, label="phase (rad)")
    plt.tight_layout()

    # --- Plot 4: phase vs x (unwrapped fit)
    order = np.lexsort((pos[:, 1], pos[:, 0]))
    x = pos[order, 0]
    th_sorted = theta_last[order]
    th_unwrapped = np.unwrap(th_sorted)

    plt.figure()
    plt.plot(x, th_unwrapped, marker=".", linestyle="none")
    plt.xlabel("x (sorted)")
    plt.ylabel("unwrapped phase")
    plt.title(f"Phase gradient along x {title}".strip())
    plt.tight_layout()

    plt.show()

def demo_one_run():
    # --- build tissue
    nx, ny = 20, 6
    pos = build_grid_positions(nx, ny, dx=1.0, dy=1.0)
    nbrs = build_neighbor_graph_grid(nx, ny, mode="von_neumann")
    N = nx * ny

    # --- choose a morphology setting
    r = 1.0
    aniso = 0.5
    L = morphology_to_L(nbrs, pos, cell_radius=r, anisotropy=aniso)

    # --- oscillator params
    mean_omega = 2*np.pi / 30.0
    rng = np.random.default_rng(0)
    omega = mean_omega * (1.0 + 0.02 * rng.normal(size=N))

    eps = epsilon_from_morphology(r, eps0=1.2, alpha=1.0, r_ref=1.0)
    tau = tau_from_morphology(r, tau0=1.0, beta=0.3, r_ref=1.0)

    # --- simulate
    T = 300.0
    dt = 0.05
    theta = simulate_delayed_kuramoto(
        L=L, eps=eps, tau=tau, omega=omega,
        T=T, dt=dt, noise_sigma=0.0, seed=1
    )

    title = f"(r={r}, aniso={aniso}, eps={eps:.3f}, tau={tau:.3f})"
    plot_run(theta, pos, nbrs, dt, title=title)

if __name__ == "__main__":
    demo_one_run()

