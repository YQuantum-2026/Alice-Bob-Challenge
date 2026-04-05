# YQuantum 2026: Alice & Bob Cat Qubit Optimization Challenge

## The Challenge

**"When your cat keeps secrets, can you still train it to do what you want?"**

We were given a dissipative cat qubit system — a superconducting storage mode stabilized by engineered two-photon dissipation through a lossy buffer mode — and asked to build an **online optimizer** that tunes 4 complex control parameters ($g_2$ and $\varepsilon_d$) to simultaneously maximize both logical lifetimes ($T_Z$, $T_X$) while hitting a target bias ratio, all under realistic hardware drift.

The full system Hamiltonian:

$$\frac{H}{\hbar} = g_2^* \hat{a}^2 \hat{b}^\dagger + g_2 (\hat{a}^\dagger)^2 \hat{b} - \varepsilon_d \hat{b}^\dagger - \varepsilon_d^* \hat{b}$$

with Lindblad jump operators $L_b = \sqrt{\kappa_b}\,\hat{b}$ (engineered buffer loss) and $L_a = \sqrt{\kappa_a}\,\hat{a}$ (unwanted storage loss).

The challenge has five parts:

1. **Reward Function Design** — Construct fast, informative reward signals that correlate with the true (expensive) lifetime measurements. The naive approach requires 2 full Lindblad simulations + 2 curve fits per evaluation; we need something cheaper for online optimization.
2. **Online Optimizer Selection** — Choose and implement an optimizer that works with expensive, noisy, black-box evaluations in a 4D parameter space.
3. **Drift and Noise Modeling** — The real hardware drifts: amplitude fluctuations, frequency shifts, Kerr nonlinearities appearing, SNR degradation, even two-level system (TLS) defects coupling in. The optimizer must track and adapt.
4. **Moon Cat Extension** — Extend to the "moon cat" Hamiltonian (Rousseau et al. 2025, arXiv:2502.07892) with an additional squeezing parameter $\lambda$, expanding the search to 5D.
5. **Single-Qubit Gate Extension** — Optimize a Zeno gate (single-photon drive on the storage mode) simultaneously with stabilization quality.

### Feedback

Key feedback that shaped our approach:
- `compute_alpha` (the adiabatic elimination formula $\alpha = \sqrt{2(\varepsilon_2 - \kappa_a/4)/\kappa_2}$) is a **heuristic only** — it becomes unreliable near thresholds and doesn't match what experiments actually measure.
- Must use **vacuum-based state preparation**: let the system find its own cat states from vacuum, then measure what actually happened. This matches the Alice & Bob experimental protocol (Reglade et al. 2024).


## Running

```bash
# Setup
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

# Run optimization benchmark
python run.py                        # MEDIUM profile
python run.py --profile hpc          # HPC profile (Palmetto GPU)
python run.py --enable all           # All sections
```

## What We Built and Why

### 8 Reward Functions

We implemented a progression of reward functions, from fast heuristics to physically rigorous measurements:

- **Proxy reward** — Single-point expectation value measurement. JIT-compiled, vmap-batched, differentiable through dynamiqs/JAX. Fastest option (~1 mesolve per evaluation) but uses heuristic $\alpha$ from adiabatic elimination. Design insight from Sivak et al. (2025, arXiv:2511.08493).
- **Multipoint proxy** — Measures at $N$ time points and uses geometric mean of per-point $T$ estimates. More robust to noise than single-point.
- **Log-derivative estimation** — Estimates $T$ from the slope of $\log\langle O\rangle$ vs $t$ via OLS regression. Amplitude-invariant (doesn't need to know the initial value $A$). Can be enabled on any lifetime-based reward.
- **Photon number proxy** — Measures steady-state $\langle a^\dagger a\rangle$ as a proxy for cat size $|\alpha|^2$. Simplest reward: 1 mesolve from vacuum.
- **Fidelity-based** — Overlap with the target even cat state $|C_+\rangle$. Uses data-driven $\alpha$ from simulation (not `compute_alpha` heuristic).
- **Parity decay** — Uses parity operator $P = e^{i\pi \hat{a}^\dagger\hat{a}}$ as logical $X$, which is completely $\alpha$-independent. Vacuum-based state prep with data-driven $\alpha$ for $Z$ measurement.
- **Enhanced proxy** — Extends the standard proxy with physics-motivated guardrails: buffer occupation penalty, code space confinement (using **data-driven $\alpha$** from a vacuum settle, not the heuristic formula), and alpha stability margin. When confinement is active, falls back to Python-loop batching.
- **Liouvillian spectral gap** — Extracts lifetimes directly from eigenvalues of the Lindbladian superoperator $\mathcal{L}$. No mesolve, no curve fitting. Exact for small Hilbert spaces. Runtime warning if eigenmode swap is detected (small $\alpha$ regime).
- **Vacuum reward (primary, alpha-free)** — The physically correct approach matching Alice & Bob's experimental protocol. Starts from vacuum, lets the system find its own cat states, estimates $\alpha$ from actual $\langle a^\dagger a\rangle$, measures parity ($X$) and quadrature ($Z$) decay. Ref: Reglade et al., Nature 629 (2024).

### 5 Optimizers

- **CMA-ES** — Population-based evolutionary strategy. Our primary optimizer. Handles noisy, non-differentiable rewards well. Dominates on benchmarks (Pack et al. 2025, arXiv:2509.08555).
- **Hybrid CMA-ES + Adam** — Alternates global exploration (CMA-ES) with local gradient refinement (Adam). Injects refined point back as CMA-ES mean after each gradient phase.
- **REINFORCE** — Vanilla policy gradient with factorized Gaussian policy $\pi(x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$. Single score-function gradient step with baseline variance reduction and entropy bonus. Ref: Sivak et al. (2025, arXiv:2511.08493).
- **PPO-Clip** — Proximal Policy Optimization with clipped importance sampling ratio and multiple gradient epochs per batch. Prevents destructively large policy updates. Ref: Schulman et al. (2017, arXiv:1707.06347).
- **Bayesian optimization** — Gaussian process surrogate with acquisition function. Best when evaluations are very expensive and the budget is small.

### 10 Drift Scenarios

We model realistic hardware perturbations the optimizer must track:

- Amplitude drift (slow/fast sinusoidal) — $g_2$ magnitude fluctuations
- Frequency drift — detuning shift $\delta\omega\,\hat{a}^\dagger\hat{a}$ on the storage mode
- Kerr drift — nonlinear frequency shift $K(\hat{a}^\dagger\hat{a})^2$
- Step drift — sudden parameter jump (simulates recalibration events)
- SNR degradation — growing measurement noise over time
- White noise — Gaussian noise on control parameters
- Multi-drift — amplitude + frequency + Kerr simultaneously
- TLS coupling — two-level system defect coupling to storage (doubles Hilbert space)

All reward types (including vacuum, parity, fidelity, spectral) now support full Hamiltonian drift perturbations (detuning $\Delta\hat{n}$ + Kerr $K\hat{n}^2$) via the drift-aware reward wrapper. Sigma floor enforcement in CMA-ES prevents premature convergence under drift.

### Moon Cat Extension

Implements the Rousseau et al. (2025, arXiv:2502.07892) squeezing Hamiltonian extension, adding a 5th control parameter $\lambda$ that modulates a squeezing interaction $g_2 \lambda\,\hat{a}^\dagger\hat{a}\,\hat{b}$. Side-by-side comparison with standard cat.

### Single-Qubit Gate Extension

Zeno gate: $H_\text{gate} = \varepsilon_Z(\hat{a}^\dagger + \hat{a})$. Optimizes gate fidelity simultaneously with stabilization quality. Ref: iQuHACK 2025 Alice & Bob challenge.

### Alpha-Free Compliance

Following the feedback from Alice & Bob, we moved all lifetime-measuring rewards away from the heuristic `compute_alpha` formula. The vacuum reward, fidelity reward, parity reward, and the enhanced proxy's confinement penalty now use **data-driven $\alpha$ estimation**: start from vacuum, let the system settle, measure the actual $\langle a^\dagger a\rangle$, and use $|\alpha_\text{est}| = \sqrt{\langle a^\dagger a\rangle}$. This matches what experiments do and avoids the adiabatic elimination approximation breaking down near thresholds.

### Comprehensive Test Suite (221 tests)

- Unit tests for all optimizers (ask/tell interface, convergence, special features)
- Physics tests (Hermiticity, eigenvalue correctness, dimension checks)
- Numerical stability tests (NaN/Inf guards, edge cases, conditioning)
- Reward correlation tests (proxy ranking vs full measurement ranking)
- Integration tests (full optimizer $\times$ reward $\times$ drift pipeline)
- Convergence tests (all optimizers converge on proxy reward)
- Vacuum reward tests (parity decay, quadrature decay, photon number growth)
- Log-derivative estimation tests (recovery, amplitude invariance, edge cases)

### Visualization Pipeline

- Wigner function animations showing cat state formation in phase space
- Lifetime decay curves $\langle Z_L\rangle(t)$ and $\langle X_L\rangle(t)$ with exponential fits
- Reward convergence curves across optimizers and reward types
- Parameter tracking under drift (optimizer vs true trajectory)
- Lifetime comparison heatmaps (reward $\times$ optimizer)
- Drift tracking matrices (multi-panel optimizer $\times$ drift scenario)

## Project Structure

```
src/
  cat_qubit.py          # Core simulation: operators, Hamiltonian, lifetimes
  reward/               # 8 reward functions + drift-aware wrapper (package)
  config.py             # Preset profiles (LOCAL/MEDIUM/HPC/EXPERIMENTAL)
  benchmark.py          # Benchmark runner: sweep reward x optimizer x drift
  drift.py              # 10 drift models
  moon_cat.py           # Moon cat extension (Rousseau et al. 2025)
  gates.py              # Single-qubit Zeno gate extension
  visualization/        # Wigner function animations + phase-space viz (package)
  plotting/             # Publication-quality comparison plots (package)
  optimizers/
    base.py             # OnlineOptimizer ABC (ask/tell interface)
    cmaes_opt.py        # CMA-ES (SepCMA/CMA)
    hybrid_opt.py       # CMA-ES + Adam hybrid
    reinforce_opt.py    # REINFORCE policy gradient (Sivak et al. 2025)
    ppo_opt.py          # PPO-Clip (Schulman et al. 2017)
    bayesian_opt.py     # Bayesian optimization (GP + acquisition)
notebooks/              # Analysis notebooks
run.py                  # CLI entry point
slurm/                  # HPC job scripts (Palmetto)
```

## Key References

- Berdou et al., "One hundred second bit-flip time in a two-photon dissipative oscillator," PRX Quantum 4, 020350 (2023). [arXiv:2204.09128](https://arxiv.org/abs/2204.09128)
- Reglade et al., "Quantum control of a cat-qubit with bit-flip times exceeding ten seconds," Nature 629, 778-783 (2024). [arXiv:2307.06617](https://arxiv.org/abs/2307.06617)
- Marquet et al., "Preserving phase coherence and linearity in cat qubits with exponential bit-flip suppression," Phys. Rev. X 15, 011070 (2025). [arXiv:2409.17556](https://arxiv.org/abs/2409.17556)
- Rousseau et al., "Moon cats," (2025). [arXiv:2502.07892](https://arxiv.org/abs/2502.07892)
- Sivak et al., "RL Control of QEC," (2025). [arXiv:2511.08493](https://arxiv.org/abs/2511.08493)
- Pack et al., "Benchmarking Optimization Algorithms for Automated Calibration of Quantum Devices," (2025). [arXiv:2509.08555](https://arxiv.org/abs/2509.08555)

## Resources

| Resource | Link |
|---|---|
| Dynamiqs Docs | https://www.dynamiqs.org |
| Challenge Repo | https://github.com/YQuantum-2026/Alice-Bob-Challenge |
| Alice & Bob | https://alice-bob.com/ |
