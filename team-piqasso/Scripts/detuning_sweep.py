"""
Detuning sweep: H → H + Δ a†a
==============================
Runs the joint T_X / T_Z optimizer at 8 values of the storage-mode detuning
Δ ∈ [0, 2] MHz and plots the optimal ε_d and g_2 as functions of Δ.

Everything in the optimizer (CMA-ES, loss, bounds, epochs) is identical to
TzTx_optimization.py.  The only change is the detuning term Δ a†a in the
Hamiltonian passed to every simulation call.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from concurrent.futures import ThreadPoolExecutor

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ── Fixed system parameters ────────────────────────────────────────────────────
na      = 12    # reduced from 15 for speed
nb      = 4     # reduced from 5 for speed
kappa_b = 10.0
kappa_a = 1.0

NOTEBOOK_EPS_D = 4.0
NOTEBOOK_G2    = 1.0
TARGET_RATIO   = 320.0

BATCH_SIZE    = 8     # reduced from 10 for speed
N_EPOCHS      = 20    # reduced from 50 for speed
SIGMA0        = 0.5
RATIO_PENALTY = 8.0

BOUNDS = np.array([
    [0.5, 8.0],
    [0.2, 4.0],
])

# ── Current detuning (set per sweep iteration) ─────────────────────────────────
DELTA = 0.0   # MHz  — overwritten in the sweep loop below

# ── Global operators ───────────────────────────────────────────────────────────
a_s  = dq.destroy(na)
a_op = dq.tensor(a_s, dq.eye(nb))
b_op = dq.tensor(dq.eye(na), dq.destroy(nb))

parity_s  = (1j * jnp.pi * a_s.dag() @ a_s).expm()
parity_op = dq.tensor(parity_s, dq.eye(nb))
a2_op     = dq.powm(a_op, 2)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTIC ESTIMATES  (unchanged from TzTx_optimization.py)
# ══════════════════════════════════════════════════════════════════════════════

def analytic_alpha(eps_d: float, g2: float) -> float:
    kappa_2 = 4.0 * g2**2 / kappa_b
    if kappa_2 < 1e-12:
        return 0.0
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    inner = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
    return float(np.sqrt(max(inner, 0.0)))

def analytic_Tx(eps_d: float, g2: float) -> float:
    alpha = analytic_alpha(eps_d, g2)
    if alpha < 1e-6:
        return 1e-3
    return 1.0 / (kappa_a * alpha**2)

def analytic_Tz(eps_d: float, g2: float) -> float:
    alpha = analytic_alpha(eps_d, g2)
    return float(np.exp(2.0 * alpha**2)) / kappa_a


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM BUILDER  — adds DELTA * a†a to the Hamiltonian
# ══════════════════════════════════════════════════════════════════════════════

def build_system(eps_d: float, g2: float):
    alpha = analytic_alpha(eps_d, g2)

    H = (jnp.conj(g2) * a_op.dag() @ a_op.dag() @ b_op
         + g2 * a_op @ a_op @ b_op.dag()
         - eps_d * b_op.dag()
         - jnp.conj(eps_d) * b_op
         + DELTA * a_op.dag() @ a_op)          # ← detuning term

    L_b = jnp.sqrt(kappa_b) * b_op
    L_a = jnp.sqrt(kappa_a) * a_op

    ket_p  = dq.coherent(na, alpha)
    ket_m  = dq.coherent(na, -alpha)
    cat_x  = (ket_p + ket_m).unit()
    psi0_x = dq.tensor(cat_x, dq.fock(nb, 0))

    return H, [L_b, L_a], alpha, psi0_x, ket_p, ket_m


# ══════════════════════════════════════════════════════════════════════════════
# LIFETIME MEASUREMENTS  (unchanged from TzTx_optimization.py)
# ══════════════════════════════════════════════════════════════════════════════

def _exp_model(p, t):
    A, tau, C = p
    return A * jnp.exp(-t / tau) + C

def _fit_lifetime(ts: np.ndarray, signal: np.ndarray) -> float:
    A0   = float(signal[0] - signal[-1])
    C0   = float(signal[-1])
    tau0 = float(ts[-1] - ts[0])
    def residuals(p):
        return _exp_model(p, ts) - signal
    result = least_squares(
        residuals, [A0, tau0, C0],
        bounds=([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf]),
        loss="soft_l1", f_scale=0.05,
    )
    return float(result.x[1])

def _measure_Tx_inner(H, jumps, psi0_x, alpha, tfinal, n_pts):
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    result = dq.mesolve(H, jumps, psi0_x, tsave,
                        exp_ops=[parity_op],
                        options=dq.Options(progress_meter=False))
    sxt = np.array(result.expects[0, :].real)
    ts  = np.array(tsave)
    if sxt[-1] > 0.9 * sxt[0]:
        Tx = 5.0 * tfinal
    else:
        try:
            Tx = _fit_lifetime(ts, sxt)
        except Exception:
            Tx = 0.0
    return {"T_X": Tx, "sxt": sxt, "ts": ts, "alpha": alpha, "tfinal": tfinal}

def _measure_Tz_inner(H, jumps, ket_p, ket_m, alpha, tfinal, n_pts):
    psi0_z = dq.tensor(ket_p, dq.fock(nb, 0))
    tsave  = jnp.linspace(0.0, tfinal, n_pts)
    res = dq.mesolve(H, jumps, psi0_z, tsave,
                     options=dq.Options(progress_meter=False),
                     exp_ops=[a2_op, a_op, a_op.dag(), a_op.dag() @ a_op])
    a2_exp   = res.expects[0, :]
    a_exp    = res.expects[1, :]
    adag_exp = res.expects[2, :]
    num_exp  = jnp.maximum(res.expects[3, :].real, 1e-12)
    phi  = jnp.angle(a2_exp) / 2
    Xphi = 0.5 * (jnp.exp(1j * phi) * adag_exp + jnp.exp(-1j * phi) * a_exp) / jnp.sqrt(num_exp)
    szt = np.array(jnp.real(Xphi))
    ts  = np.array(res.tsave)
    if szt[-1] > 0.9 * szt[0]:
        Tz = 5.0 * tfinal
    else:
        try:
            Tz = _fit_lifetime(ts, szt)
        except Exception:
            Tz = 0.0
    return {"T_Z": Tz, "szt": szt, "ts": ts, "alpha": alpha, "tfinal": tfinal}

def measure_joint(eps_d: float, g2: float) -> dict:
    H, jumps, alpha, psi0_x, ket_p, ket_m = build_system(eps_d, g2)
    tx_est    = 1.0 / max(kappa_a * alpha**2, 1e-6)
    tz_est    = float(np.exp(2.0 * alpha**2)) / kappa_a
    tx_window = float(np.clip(3.0 * tx_est, 0.3, 15.0))
    tz_window = float(np.clip(3.0 * tz_est, 50.0, 200.0))
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_x = ex.submit(_measure_Tx_inner, H, jumps, psi0_x, alpha, tx_window, 20)
        fut_z = ex.submit(_measure_Tz_inner, H, jumps, ket_p, ket_m, alpha, tz_window, 25)
        out_x = fut_x.result()
        out_z = fut_z.result()
    Tx    = out_x["T_X"]
    Tz    = out_z["T_Z"]
    ratio = Tz / max(Tx, 1e-9)
    return {"T_X": Tx, "T_Z": Tz, "ratio": ratio,
            "sxt": out_x["sxt"], "ts_x": out_x["ts"],
            "szt": out_z["szt"], "ts_z": out_z["ts"],
            "alpha": alpha}


# ══════════════════════════════════════════════════════════════════════════════
# CMA-ES OPTIMIZER  (unchanged from TzTx_optimization.py)
# ══════════════════════════════════════════════════════════════════════════════

def clip_bounds(x):
    return np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])

def _eval_candidate(x):
    x = clip_bounds(x)
    try:
        out       = measure_joint(float(x[0]), float(x[1]))
        Tx        = out["T_X"]
        Tz        = out["T_Z"]
        ratio     = out["ratio"]
        ratio_err = (ratio - TARGET_RATIO) / TARGET_RATIO
        val       = (-(Tx + Tz / TARGET_RATIO) / 2.0
                     + RATIO_PENALTY * ratio_err**2)
    except Exception:
        Tx, Tz, ratio, val = 0.0, 0.0, 0.0, 1e6
        out = {"alpha": analytic_alpha(float(x[0]), float(x[1]))}
    return {"T_X": Tx, "T_Z": Tz, "ratio": ratio, "x": x, "loss": val,
            "alpha": out["alpha"]}

def optimize_joint(eps_d_init=NOTEBOOK_EPS_D, g2_init=NOTEBOOK_G2,
                   n_epochs=N_EPOCHS, batch_size=BATCH_SIZE,
                   sigma0=SIGMA0, verbose=True):
    theta0 = np.array([eps_d_init, g2_init])
    if verbose:
        print(f"  Warm-start θ₀ = (eps_d={eps_d_init:.3f}, g_2={g2_init:.3f})", flush=True)
    optimizer   = SepCMA(mean=theta0, sigma=sigma0, bounds=BOUNDS,
                         population_size=batch_size)
    max_workers = min(batch_size, os.cpu_count() or 4)
    best_obj, best_theta, best_Tx, best_Tz = -np.inf, theta0.copy(), 0.0, 0.0

    for epoch in range(n_epochs):
        candidates    = [optimizer.ask() for _ in range(batch_size)]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            batch_metrics = list(pool.map(_eval_candidate, candidates))
        optimizer.tell([(bm["x"], bm["loss"]) for bm in batch_metrics])

        best_idx = min(range(batch_size), key=lambda i: batch_metrics[i]["loss"])
        bm = batch_metrics[best_idx]
        bx = bm["x"]
        obj_ep = bm["T_X"] + bm["T_Z"] / TARGET_RATIO
        if obj_ep > best_obj:
            best_obj, best_theta = obj_ep, bx.copy()
            best_Tx, best_Tz     = bm["T_X"], bm["T_Z"]

        if verbose:
            print(f"    Epoch {epoch+1:2d}/{n_epochs}:  "
                  f"T_X={bm['T_X']:.3f} µs  T_Z={bm['T_Z']:.1f} µs  "
                  f"ratio={bm['ratio']:.1f}  "
                  f"eps_d={float(bx[0]):.3f}  g_2={float(bx[1]):.3f}",
                  flush=True)

    return best_theta, best_Tx, best_Tz


# ══════════════════════════════════════════════════════════════════════════════
# DETUNING SWEEP
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DELTA_VALUES = np.linspace(0.0, 2.0, 8)

    eps_d_results = []
    g2_results    = []
    Tx_results    = []
    Tz_results    = []

    print("=" * 65)
    print("  Detuning sweep: H + Δ a†a,  Δ ∈ [0, 2] MHz,  8 points")
    print("=" * 65, flush=True)

    for i, delta_val in enumerate(DELTA_VALUES):
        DELTA = float(delta_val)
        print(f"\n[{i+1}/8]  Δ = {DELTA:.4f} MHz", flush=True)
        print("-" * 65, flush=True)

        best_theta, best_Tx, best_Tz = optimize_joint(verbose=True)
        eps_d_opt, g2_opt = best_theta

        eps_d_results.append(float(eps_d_opt))
        g2_results.append(float(g2_opt))
        Tx_results.append(float(best_Tx))
        Tz_results.append(float(best_Tz))

        print(f"  → Best: eps_d={eps_d_opt:.4f}  g_2={g2_opt:.4f}  "
              f"T_X={best_Tx:.4f} µs  T_Z={best_Tz:.2f} µs", flush=True)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Δ (MHz)':>10}  {'ε_d* (MHz)':>12}  {'g_2* (MHz)':>12}  "
          f"{'T_X (µs)':>10}  {'T_Z (µs)':>10}")
    print("-" * 65)
    for d, e, g, tx, tz in zip(DELTA_VALUES, eps_d_results, g2_results,
                                Tx_results, Tz_results):
        print(f"{d:10.4f}  {e:12.4f}  {g:12.4f}  {tx:10.4f}  {tz:10.2f}")

    # ── Plot ε_d vs Δ ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        r"Effect of detuning $\Delta$ on optimal control parameters"
        "\n"
        r"$H \;\to\; H + \Delta\, a^\dagger a$,  $\Delta \in [0,\,2]$ MHz",
        fontsize=13,
    )

    ax = axes[0]
    ax.plot(DELTA_VALUES, eps_d_results, 'o-', color='steelblue',
            ms=8, lw=2, markerfacecolor='white', markeredgewidth=2)
    for d, e in zip(DELTA_VALUES, eps_d_results):
        ax.annotate(f"{e:.2f}", xy=(d, e), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.set(xlabel=r'$\Delta$ (MHz)',
           ylabel=r'Optimal $\epsilon_d^*$ (MHz)',
           title=r'Drive amplitude $\epsilon_d^*$ vs detuning $\Delta$')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 2.1)

    ax = axes[1]
    ax.plot(DELTA_VALUES, g2_results, 's-', color='crimson',
            ms=8, lw=2, markerfacecolor='white', markeredgewidth=2)
    for d, g in zip(DELTA_VALUES, g2_results):
        ax.annotate(f"{g:.2f}", xy=(d, g), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=8)
    ax.set(xlabel=r'$\Delta$ (MHz)',
           ylabel=r'Optimal $g_2^*$ (MHz)',
           title=r'Two-photon coupling $g_2^*$ vs detuning $\Delta$')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 2.1)

    plt.tight_layout()
    plt.savefig('detuning_sweep.png', dpi=150, bbox_inches='tight')
    print("\nFigure saved to detuning_sweep.png", flush=True)
    plt.show()
