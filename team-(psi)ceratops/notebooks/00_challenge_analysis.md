# YQuantum 2026 — Alice & Bob Challenge: Comprehensive Analysis

> **Purpose:** Deep breakdown of every challenge area with objectives, attack strategies,
> implementation approaches, and pros/cons — to be read before writing any code.

---

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [Reward Function Design](#2-reward-function-design)
3. [Choice of Online Optimizer](#3-choice-of-online-optimizer)
4. [Drift and Noise Modeling](#4-drift-and-noise-modeling)
5. [Extension to Moon Cats](#5-extension-to-moon-cats)
6. [Extension to Single-Qubit Gates](#6-extension-to-single-qubit-gates)
7. [Recommended Strategy](#7-recommended-strategy)

---

## 1. The Core Problem

### 1.1 What We're Optimizing

A **dissipative cat qubit** encodes quantum information in superpositions of coherent states
|α⟩ and |-α⟩ inside a superconducting resonator (the "storage" mode). The cat state is
stabilized by engineering **two-photon dissipation** via a lossy auxiliary "buffer" mode.

The full system Hamiltonian (storage mode $\hat{a}$, buffer mode $\hat{b}$):

$$\frac{H}{\hbar} = g_2^* \hat{a}^2 \hat{b}^\dagger + g_2 (\hat{a}^\dagger)^2 \hat{b} - \varepsilon_d \hat{b}^\dagger - \varepsilon_d^* \hat{b}$$

Dissipation (Lindblad jump operators):

- Buffer loss: $L_b = \sqrt{\kappa_b} \cdot \hat{b}$ (engineered, fast, $\kappa_b = 10$ MHz)
- Storage loss: $L_a = \sqrt{\kappa_a} \cdot \hat{a}$ (unwanted, slow, $\kappa_a = 1$ MHz)

When $\kappa_b \gg g_2, \varepsilon_d$ (adiabatic elimination), the buffer can be traced out, yielding effective single-mode dynamics with:

- Effective two-photon loss rate: $\kappa_2 = 4|g_2|^2 / \kappa_b$
- Effective two-photon drive: $\varepsilon_2 = 2 g_2 \varepsilon_d / \kappa_b$
- Stabilized cat size: $\alpha^2 = \frac{2}{\kappa_2}\left(\varepsilon_2 - \frac{\kappa_a}{4}\right)$ (valid when $\varepsilon_2 > \kappa_a/4$)

### 1.2 The Two Objectives

The optimizer must simultaneously:

1. **Maximize lifetimes $T_X$ and $T_Z$:**
   - $T_Z$ (bit-flip lifetime): time for $\langle Z_L \rangle$ to decay from $|{+z}\rangle$. Scales **exponentially** with $|\alpha|^2$.
   - $T_X$ (phase-flip lifetime): time for $\langle X_L \rangle$ to decay from $|{+x}\rangle$. Scales **linearly** (inversely) with $|\alpha|^2$.
   - Typical values: $T_Z \sim 100$–$200\;\mu\text{s}$, $T_X \sim 0.1$–$1\;\mu\text{s}$ (for $\alpha \approx 2$, $\kappa_a = 1$ MHz).

2. **Hit a target bias ratio:** $\eta = T_Z / T_X$.
   - This is the "biased noise" property that makes cat qubits useful for error correction.
   - Larger $\alpha \to$ higher $\eta$ (exponentially), but $T_X$ gets shorter.
   - The optimizer must find the sweet spot balancing both.

### 1.3 The Control Knobs

**4 real parameters** (the complex $g_2$ and $\varepsilon_d$):

| Knob | Physical meaning | Default | Typical range |
| ---- | ---------------- | ------- | ------------- |
| $\text{Re}(g_2)$ | Two-photon coupling magnitude | 1.0 | $[0.1,\;5.0]$ |
| $\text{Im}(g_2)$ | Two-photon coupling phase | 0.0 | $[-2.0,\;2.0]$ |
| $\text{Re}(\varepsilon_d)$ | Buffer drive magnitude | 4.0 | $[0.5,\;20.0]$ |
| $\text{Im}(\varepsilon_d)$ | Buffer drive phase | 0.0 | $[-5.0,\;5.0]$ |

These knobs control $\alpha$ (cat size), which controls $T_X$, $T_Z$, and $\eta$.

### 1.4 What Makes This Hard

1. **Measuring $T_X$ and $T_Z$ is expensive:** Requires preparing states, evolving for different durations, measuring expectation values at multiple time points, then fitting exponential decays. Two full simulations + two curve fits per evaluation.

2. **Hardware drifts:** Real devices experience parameter fluctuations (amplitude shifts, frequency drifts, new nonlinearities appearing). The optimizer must continuously adapt.

3. **Conflicting objectives:** Larger $\alpha$ gives better $T_Z$ but worse $T_X$. The optimal point depends on the target bias $\eta$.

4. **Measurement difficulty:** The logical Z operator $\hat{\sigma}_z = |\alpha\rangle\langle\alpha| - |-\alpha\rangle\langle -\alpha|$ requires knowing $\alpha$ accurately. On real hardware, $\alpha$ estimation is unreliable. The logical X operator (parity, $e^{i\pi \hat{a}^\dagger \hat{a}}$) is more robust but still noisy.

### 1.5 Relevant Literature Summary

| Paper | Key Contribution | Relevance |
|-------|-----------------|-----------|
| Sivak et al. 2023 (Nature) | First beyond-break-even QEC via model-free RL | Proved model-free online optimization works for quantum control |
| Sivak et al. 2025 | PPO-style policy gradients for surface code calibration | Surrogate reward design (detection rate, not error rate). Drift tracking. Scalability proof. |
| Pack et al. 2025 | Benchmarked 6 optimizers for quantum device calibration | **CMA-ES dominates** across all regimes. Justifies our baseline choice. |
| Berdou et al. 2022 | 100s bit-flip time in cat qubit | Experimental params. Same Hamiltonian as our challenge. |
| Rousseau et al. 2025 | Moon cat (squeezed cat): 160× T_X improvement | Extension (moon cat). Adds λ control knob. |

---

## 2. Reward Function Design

### 2.1 Objective

Define a scalar reward $R(\mathbf{x})$ that captures both lifetime maximization and bias targeting, where $\mathbf{x} = [\text{Re}(g_2),\;\text{Im}(g_2),\;\text{Re}(\varepsilon_d),\;\text{Im}(\varepsilon_d)]$.

Requirements (from challenge description):
- **Experimentally efficient:** Minimize measurement overhead (fewer simulation runs per evaluation).
- **Robust to uncertainty:** Works even without precise knowledge of α.
- **Captures both objectives:** Lifetimes AND bias.

### 2.2 Full Measurement Reward

**How it works:** Run two full Lindblad simulations per evaluation — one from $|{+z}\rangle$ to measure $T_Z$ decay, one from $|{+x}\rangle$ to measure $T_X$ decay. Fit exponential decays to extract $T_Z$, $T_X$.

$$R_\text{full}(\mathbf{x}) = \log(T_X) + \log(T_Z) - \lambda \left(\log\frac{T_Z}{T_X} - \log\eta_\text{target}\right)^2$$

**Implementation:**

1. Build Hamiltonian $H(\mathbf{x})$ and jump operators
2. Simulate from $|{+z}\rangle$ for $t \in [0, 200\;\mu\text{s}]$, measure $\langle Z_L\rangle(t)$ at 100 points
3. Fit: $\langle Z_L\rangle(t) = A \cdot e^{-t/T_Z} + C$ using `scipy.optimize.least_squares`
4. Simulate from $|{+x}\rangle$ for $t \in [0, 1\;\mu\text{s}]$, measure $\langle X_L\rangle(t)$ at 100 points
5. Fit: $\langle X_L\rangle(t) = A \cdot e^{-t/T_X} + C$
6. Compute reward from $T_Z$, $T_X$

| Pros | Cons |
|------|------|
| Directly measures what we care about | **Very expensive:** 2 mesolve calls + 2 curve fits per evaluation |
| Ground-truth validation | **Not JIT-compatible:** scipy curve_fit breaks JAX compilation |
| Easy to understand and verify | **Not differentiable:** Cannot use gradient-based optimizers |
| Robust to probe time choice | **Slow:** With CMA-ES population of 24, that's 48 mesolve calls per epoch |
| | Fit can fail when T is very large or very small relative to sim window |

**When to use:** Validation only. Periodically evaluate the full reward to verify that the
proxy reward is tracking correctly.

### 2.3 Single-Point Proxy Reward

**Key insight (from Sivak et al. 2025):** You don't need to measure the full decay curve. For an exponential decay $\langle \hat{O}\rangle(t) = e^{-t/T}$, a **single measurement at a fixed probe time $t_\text{probe}$** is a monotonic function of $T$:

- $\langle \hat{O}\rangle(t_\text{probe}) = e^{-t_\text{probe}/T}$
- Larger $T \to$ larger $\langle \hat{O}\rangle(t_\text{probe})$
- This is fully differentiable, JIT-compatible, and requires only ONE time point.

$$R_\text{proxy}(\mathbf{x}) = w_z \log\langle Z_L\rangle(t_z) + w_x \log\langle X_L\rangle(t_x) - \lambda\left(\log\frac{\langle Z_L\rangle}{\langle X_L\rangle} - C_\text{target}\right)^2$$

where $t_z \sim 50\;\mu\text{s}$ (order of $T_Z$) and $t_x \sim 0.3\;\mu\text{s}$ (order of $T_X$).

**Implementation:**

1. Build $H(\mathbf{x})$, jump operators
2. Simulate from $|{+z}\rangle$ to time $t_z$ (only need final state, `tsave=[0, t_z]`)
3. Compute $\langle Z_L\rangle$ at $t_z$
4. Simulate from $|{+x}\rangle$ to time $t_x$
5. Compute $\langle X_L\rangle$ at $t_x$
6. Combine into reward

| Pros | Cons |
|------|------|
| **Fast:** 2 mesolve calls, NO curve fitting | Probe time choice is critical (see below) |
| **JIT-compatible:** Fully within JAX | Not a direct measure of T_X, T_Z |
| **Differentiable:** Enables jax.grad for gradient-based optimization | Monotonic but nonlinear mapping to true T — reward landscape may be distorted |
| 2× faster than full measurement (no multi-point sim) | Probe time must match expected lifetime order of magnitude |
| Matches real experimental constraints (finite measurement budget) | Bias proxy (ratio of expectations) ≠ true bias (ratio of lifetimes) |

**Critical design consideration — probe time selection:**

| $t_\text{probe} \ll T$ | $t_\text{probe} \sim T$ | $t_\text{probe} \gg T$ |
| ----------------------- | ----------------------- | ---------------------- |
| $\langle \hat{O}\rangle \approx 1$ for all $T$ — no discrimination | $\langle \hat{O}\rangle$ varies most with $T$ — **optimal** | $\langle \hat{O}\rangle \approx 0$ for all $T$ — no discrimination |

The optimal probe time is $t_\text{probe} \approx T$, but $T$ is what we're trying to optimize! This creates a chicken-and-egg problem. Solutions:

- **Fixed conservative probe times:** $t_z = 50\;\mu\text{s}$, $t_x = 0.3\;\mu\text{s}$ (works if $T$ stays in a known range)
- **Adaptive probe times:** Periodically run full measurement, update probe times based on current $T$ estimates
- **Multi-point proxy:** Measure at 2–3 time points instead of 1 (intermediate between full and single-point)

### 2.4 Photon Number / Cat Size Proxy

**Idea:** Instead of measuring lifetimes directly, optimize for the cat size $\alpha$, which determines both $T_X$ and $T_Z$. Measure $\langle \hat{n}\rangle = \langle \hat{a}^\dagger \hat{a}\rangle$ as a proxy for $|\alpha|^2$.

$$R_\text{photon}(\mathbf{x}) = -\left|\langle \hat{n}\rangle(t_\text{ss}) - n_\text{target}\right|^2$$

where $n_\text{target} = |\alpha_\text{target}|^2$ is chosen to achieve the desired bias.

| Pros | Cons |
|------|------|
| **Very fast:** Single mesolve, single expectation | Only an indirect proxy for lifetimes |
| JIT-compatible and differentiable | Doesn't distinguish between good and bad cat states with same ⟨n⟩ |
| Physically intuitive | Doesn't capture how κ_a affects lifetime at fixed α |
| No α estimation needed (⟨n⟩ is directly measurable) | Ignores buffer mode dynamics that affect lifetime |

**When to use:** As a fast pre-screen to narrow the parameter search space before using
more expensive rewards.

### 2.5 Fidelity-Based Reward

**Idea:** Measure fidelity between the evolved state and a target cat state at a fixed time.

$$R_\text{fidelity}(\mathbf{x}) = F\!\left(\rho_a(t_\text{ss}),\; |\text{cat}_\text{target}\rangle\langle\text{cat}_\text{target}|\right)$$

| Pros | Cons |
|------|------|
| Well-defined, differentiable | Requires knowing the target α to construct |cat_target⟩ |
| Captures state quality beyond just photon number | Doesn't directly measure lifetimes |
| Available in dynamiqs: `dq.fidelity()` | Fidelity to a specific cat ≠ lifetime quality |
| | Target state depends on control params (circular) |

### 2.6 Parity Decay Rate Proxy

**Idea:** The parity operator $\hat{P} = e^{i\pi \hat{a}^\dagger \hat{a}}$ is the logical X operator for cat qubits. Its decay rate is directly related to $T_X$. Measure parity at a short time after initialization.

$$R_\text{parity}(\mathbf{x}) = \left|\langle \hat{P}\rangle(t_\text{probe})\right|$$

This is attractive because parity is robust (doesn't require α estimation) and directly
relates to the phase-flip channel.

| Pros | Cons |
|------|------|
| Robust to α uncertainty | Only captures T_X, not T_Z directly |
| Experimentally practical (parity can be measured via photon counting) | Must combine with another metric for T_Z |
| JIT-compatible and differentiable | |

### 2.7 Recommendation

**Start with the single-point proxy** as the primary optimization reward.
Use **full measurement** every N epochs for validation and probe time calibration.
Discuss the photon, fidelity, and parity approaches in the analysis as alternatives explored.

The **composite reward formula:**

```python
def reward_proxy(x, t_probe_z=50.0, t_probe_x=0.3, target_bias=100.0, w_lifetime=1.0, w_bias=0.5):
    # Simulate from |+z⟩, measure ⟨Z_L⟩(t_probe_z)
    ez = simulate_and_measure_z(x, t_probe_z)
    # Simulate from |+x⟩, measure ⟨X_L⟩(t_probe_x)
    ex = simulate_and_measure_x(x, t_probe_x)
    
    # Lifetime component (higher expectations = longer lifetimes)
    lifetime_score = w_lifetime * (jnp.log(jnp.maximum(ez, 1e-10)) + jnp.log(jnp.maximum(ex, 1e-10)))
    
    # Bias component (proxy: ratio of log-expectations ~ ratio of 1/T values)
    bias_proxy = jnp.log(jnp.maximum(ez, 1e-10)) / jnp.log(jnp.maximum(ex, 1e-10))
    # Note: log(exp(-t_z/T_Z)) / log(exp(-t_x/T_X)) = (t_z/T_Z) / (t_x/T_X) = (t_z * T_X) / (t_x * T_Z)
    # So bias_proxy = (t_z / t_x) / eta ... need to invert and scale
    bias_penalty = -w_bias * (bias_proxy - bias_target_proxy)**2
    
    return lifetime_score + bias_penalty
```

---

## 3. Choice of Online Optimizer

### 3.1 Objective

Select and implement optimizer(s) that efficiently find and track optimal control parameters $\mathbf{x} = [\text{Re}(g_2),\;\text{Im}(g_2),\;\text{Re}(\varepsilon_d),\;\text{Im}(\varepsilon_d)]$ under non-stationary conditions (drift).

### 3.2 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**How it works:** Maintains a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{C})$ over the search space.
Each epoch: sample a population of candidates, evaluate them, update the distribution toward better-performing candidates. $\sigma$ adapts via cumulative step-size adaptation (CSA).

**Implementation:** Already in the challenge notebook using `SepCMA` from `cmaes` library.

```python
from cmaes import SepCMA

optimizer = SepCMA(
    mean=np.array([1.0, 0.0, 4.0, 0.0]),  # [g2_re, g2_im, eps_d_re, eps_d_im]
    sigma=0.5,
    bounds=np.array([[0.1, 5.0], [-2.0, 2.0], [0.5, 20.0], [-5.0, 5.0]]),
    population_size=24,
    seed=42
)

for epoch in range(N_EPOCHS):
    xs = np.array([optimizer.ask() for _ in range(optimizer.population_size)])
    rewards = batched_reward(jnp.array(xs))  # vmap'd reward function
    solutions = [(xs[i], -rewards[i]) for i in range(len(xs))]  # CMA-ES MINIMIZES
    optimizer.tell(solutions)
```

**Key API details:**
- `SepCMA.ask()` → returns one candidate (numpy array). Call pop_size times.
- `SepCMA.tell(solutions)` → list of (candidate, fitness) tuples. **Lower = better** (minimization).
- `SepCMA.should_stop()` → convergence check (6 criteria: tolfun, tolx, divergence, stagnation, condition number).
- `optimizer.mean` → current best estimate.
- `optimizer._sigma` → current step size (can be manually set for drift handling).

**Variants available:**
- `SepCMA`: Diagonal covariance (O(n) memory). Used in challenge notebook. Fine for 4D.
- `CMA`: Full covariance (O(n²) memory). Better for correlated parameters. Also fine for 4D.
- `CMA(lr_adapt=True)`: LRA-CMA, auto-adjusts learning rate. Better for noisy/multimodal landscapes.

**Drift handling strategies:**
1. **Sigma floor:** `optimizer._sigma = max(optimizer._sigma, 0.05)` after each tell(). Prevents convergence.
2. **Periodic sigma inflation:** Every K epochs, `optimizer._sigma *= 2`. Restores exploration.
3. **Full restart with warm start:** When `should_stop()` triggers, re-initialize with current mean + fresh sigma.
4. **Natural tracking:** For slow drift, CMA-ES tracks without modification because σ doesn't fully collapse if the landscape keeps changing.

| Pros | Cons |
|------|------|
| **Derivative-free:** Works with ANY reward function (full, proxy, noisy) | **Sample-inefficient:** Needs pop_size evaluations per epoch |
| **Proven for quantum calibration:** Pack et al. 2025 showed CMA-ES dominates | No gradient information — wastes available information when rewards are differentiable |
| **Already in starter code:** Minimal implementation effort | Population-based → 24 simulations per epoch (expensive with full reward) |
| **Self-adapting:** σ and C adapt automatically | Convergence in 4D: ~50-200 epochs typically |
| **Robust to noise, local optima** | Drift tracking requires manual sigma management |
| Trivially parallelizable via JAX vmap | |
| **Realistic:** Works on real hardware where gradients aren't available | |

### 3.3 JAX Gradient-Based Optimization (Adam via optax)

**How it works:** Compute the gradient $\partial R / \partial \mathbf{x}$ by differentiating through the entire Lindblad simulation using JAX's reverse-mode autodiff. Update parameters with Adam.

**Requires:** A differentiable reward function (proxy reward, NOT full measurement
with scipy fitting).

**Implementation:**

```python
import jax
import optax

# Define differentiable loss (negative proxy reward)
@jax.jit
def loss_fn(params):
    # params = jnp.array([g2_re, g2_im, eps_d_re, eps_d_im])
    return -reward_proxy(params)  # negate because optax minimizes

optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(params)

grad_fn = jax.jit(jax.grad(loss_fn))

for epoch in range(N_EPOCHS):
    grads = grad_fn(params)
    grads = grads.conj()  # CRITICAL: dynamiqs returns Wirtinger derivatives for complex params
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
```

**Critical dynamiqs detail:** When differentiating with respect to complex parameters,
dynamiqs/JAX returns Wirtinger derivatives. For gradient descent, you must conjugate the
gradients: `grads = grads.conj()`. (From dynamiqs docs.)

**Gradient computation through mesolve:**

```python
result = dq.mesolve(
    H, jump_ops, rho0, tsave,
    gradient=dq.gradient.BackwardCheckpointed(),  # Memory-efficient backprop
    options=dq.Options(progress_meter=False)
)
```

`BackwardCheckpointed(ncheckpoints=None)` uses O(log(n_steps)) checkpoints. For a 75×75
density matrix with 100 time steps, this uses ~4 MB of checkpoint memory — trivial.

| Pros | Cons |
|------|------|
| **Extremely sample-efficient:** 1 forward + 1 backward pass per step | **Only works with differentiable rewards** (proxy, not full measurement) |
| **Fast convergence:** Gradient gives exact direction to improve | **Can get stuck in local optima** (no population-based exploration) |
| **Leverages JAX ecosystem:** JIT compilation, GPU acceleration | **Not realistic for hardware:** Can't differentiate through a real device |
| Novel approach for this problem (papers use CMA-ES or PPO, not direct grad) | First-call JIT compilation of grad_fn is slow (minutes) |
| Natural drift tracking: gradient always points toward current optimum | Sensitive to learning rate choice |
| Novel: not used in cited references | Need Wirtinger conjugation for complex params |

### 3.4 Hybrid CMA-ES + Gradient Refinement

**How it works:** Use CMA-ES for global exploration (avoid local optima), then refine
the best solution with a few Adam steps using gradients.

```
1. Run CMA-ES for K epochs → find approximate optimum
2. Take the CMA-ES mean as starting point
3. Run Adam for M steps with small learning rate → refine
4. Use refined point as new CMA-ES mean, reset sigma → repeat
```

| Pros | Cons |
|------|------|
| Combines global exploration with local gradient refinement | More complex implementation |
| CMA-ES handles non-differentiable reward components | Switching logic needs tuning |
| Gradient refinement converges faster than CMA-ES alone | Two optimizers to maintain |

### 3.5 PPO-Style Policy Gradients (Sivak et al. 2025)

**How it works:** Parameterize a policy $\pi(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ over control parameters. Use Monte Carlo gradient estimation on the reward to update $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$.

This is what the cited papers actually used. The policy is NOT a neural network — it's
just a Gaussian with learnable mean and variance (same structure as CMA-ES but with
different update rules).

| Pros | Cons |
|------|------|
| Matches the cited literature exactly | More complex to implement than CMA-ES |
| Entropy regularization maintains exploration naturally | Hyperparameter sensitive (clipping, learning rates) |
| Scalable to high dimensions (40,000+ params in Sivak) | Overkill for 4 parameters |
| Explicitly designed for non-stationary objectives | No advantage over CMA-ES in low dimensions |

### 3.6 Bayesian Optimization

**How it works:** Fit a Gaussian process surrogate model to (x, R(x)) pairs. Use an
acquisition function (UCB, EI) to choose the next x to evaluate.

| Pros | Cons |
|------|------|
| **Most sample-efficient:** Each evaluation maximally informative | O(n³) cost per step in number of observations |
| Handles noisy rewards naturally | Does not scale well to many epochs (GP grows) |
| Can model non-stationary objectives with time-varying kernels | Complex implementation |
| | Overkill for this problem where simulation is relatively cheap |

### 3.7 Recommendation

**Implement CMA-ES and Adam/gradient and benchmark both.** This gives:
- CMA-ES: robust baseline, works with any reward, realistic for hardware
- Adam: fast convergence, demonstrates differentiable quantum simulation, novelty factor
- The comparison itself is a valuable contribution (technical depth for judging)

If time permits, add the hybrid approach as a third option.

---

## 4. Drift and Noise Modeling

### 4.1 Objective

Formulate phenomenological models for environmental drift. Evaluate how well each optimizer
adapts. Systematically characterize optimizer responsiveness.

### 4.2 Buffer Drive Amplitude Shift

**Physics:** The effective g₂ coupling drifts due to fluctuations in the pump power or
SQUID operating point.

**Model:**

$$g_{2,\text{eff}}(\text{epoch}) = g_2 \cdot \left(1 + A \sin(2\pi f \cdot \text{epoch} + \varphi)\right)$$

**Effect:** Changes effective $\kappa_2$ and $\varepsilon_2$, shifting the stabilized cat size $\alpha$. Both $T_X$ and $T_Z$ change. This is the most natural and common drift mechanism.

**Implementation (following pi pulse pattern from challenge notebook):**
```python
@jit
def loss_under_drift(p):
    N_KNOBS = 4
    # Control parameters (optimizer's knobs)
    g2_re, g2_im, eps_d_re, eps_d_im = p[0], p[1], p[2], p[3]
    # Drift offsets (injected by training loop, not controlled by optimizer)
    g2_re_drift, g2_im_drift = p[N_KNOBS+0], p[N_KNOBS+1]
    # Effective parameters = control - drift
    g2_re_eff = g2_re - g2_re_drift
    g2_im_eff = g2_im - g2_im_drift
    # ... build H with effective params, simulate, return reward
```

| Pros | Cons |
|------|------|
| Most physically motivated | Simple sinusoidal may not capture real drift statistics |
| Directly changes the parameters the optimizer controls | Only affects g₂, not the Hamiltonian structure |
| Easy to implement (follows starter code pattern exactly) | |
| Good for benchmarking optimizer tracking speed | |

**Characterization metrics:**
- **Tracking error:** |x_optimizer - x_optimal(epoch)| averaged over time
- **Response time:** Epochs to recover after a step change in drift
- **Critical drift frequency:** Fastest sinusoidal drift the optimizer can track

### 4.3 Storage Resonator Frequency Shift

**Physics:** The storage mode frequency shifts due to thermal fluctuations, TLS interactions,
or charge noise. This adds a detuning term to the Hamiltonian.

**Model:** $H_\text{drift} = \Delta(\text{epoch}) \cdot \hat{a}^\dagger \hat{a}$, where $\Delta(\text{epoch})$ varies with time.

**Effect:** Detunes the two-photon resonance condition, reducing stabilization efficiency.
The cat state becomes less stable, reducing both $T_X$ and $T_Z$.

**Implementation:** Requires modifying the Hamiltonian (not just control parameters):
```python
H_total = H_base(g2, eps_d) + delta_drift * a.dag() @ a
```

**Compensation knob:** The optimizer can add a tunable detuning $\Delta_d \cdot \hat{a}^\dagger \hat{a}$ to cancel the drift:
```python
H_total = H_base(g2, eps_d) + (delta_drift - delta_compensate) * a.dag() @ a
```
This adds a 5th control knob ($\Delta_d$) for the optimizer.

| Pros | Cons |
|------|------|
| Physically realistic (frequency drift is ubiquitous) | Requires Hamiltonian modification (not just parameter offset) |
| Natural compensation knob (tunable detuning) | Adds complexity to the loss function |
| Tests optimizer's ability to use new knobs | 5th parameter may slow CMA-ES |

### 4.4 Kerr Nonlinearity

**Physics:** Self-Kerr nonlinearity in the storage mode appears due to junction aging, thermal effects, or parasitic coupling. Adds $K \cdot (\hat{a}^\dagger \hat{a})^2$ to the Hamiltonian.

**Model:** $H_\text{Kerr} = K(\text{epoch}) \cdot (\hat{a}^\dagger \hat{a})^2$, where $K$ slowly increases or oscillates.

**Effect:** Distorts the cat state (no longer coherent state superpositions). Reduces $T_Z$ more than $T_X$ because it breaks the symmetry of the double-well potential.

**Implementation:**
```python
n_a = a.dag() @ a  # number operator
H_total = H_base(g2, eps_d) + K_drift * n_a @ n_a
```

Note: Berdou et al. 2022 achieved Kerr < 1 Hz in their experiment (negligible), but
it can appear as a drift mechanism.

| Pros | Cons |
|------|------|
| Tests robustness against Hamiltonian structure changes | No obvious compensation knob (can't "undo" Kerr easily) |
| Physically motivated (known issue in superconducting circuits) | Optimizer can only partially mitigate by adjusting α |
| Interesting regime: small K distorts but doesn't destroy stabilization | More complex physics |

### 4.5 TLS Coupling

**Physics:** Two-level system (TLS) defects in the substrate couple resonantly to the
storage mode, creating a parasitic energy exchange channel.

**Model:** Add a TLS mode with coupling:

$$H_\text{TLS} = \frac{\omega_\text{TLS}}{2}\hat{\sigma}_z + g_\text{TLS}\left(\hat{a}^\dagger \hat{\sigma}_- + \hat{a}\,\hat{\sigma}_+\right)$$

with TLS decay rate $\gamma_\text{TLS}$.

**Effect:** Additional loss channel through the TLS. Effectively increases $\kappa_a$.

| Pros | Cons |
|------|------|
| Very realistic (TLS are a major noise source) | Requires extending Hilbert space (×2 for TLS) |
| Tests fundamentally different kind of drift (new loss channel, not parameter shift) | More expensive simulation |
| | Implementation complexity higher |

### 4.6 Measurement SNR Degradation

**Physics:** The measurement apparatus degrades over time (amplifier drift, cable losses),
adding noise to the observed reward values.

**Model:** $R_\text{observed} = R_\text{true} + \mathcal{N}(0, \sigma_\text{noise}(\text{epoch}))$, where $\sigma_\text{noise}$ increases with epoch.

**Effect:** The optimizer receives increasingly noisy reward signals, making it harder to
distinguish good from bad parameters.

**Implementation:**
```python
noise_std = base_noise + noise_growth * epoch
reward_noisy = reward_true + jax.random.normal(key, shape=()) * noise_std
```

| Pros | Cons |
|------|------|
| Easy to implement (just add noise to reward) | Doesn't test physics understanding |
| Tests optimizer robustness to noisy objectives | Less interesting for the physics-focused judging |
| CMA-ES is naturally noise-robust | Trivial for gradient-based (just average over samples) |

### 4.7 Recommendation

**Implement amplitude drift first** — it follows the starter code pattern directly.
Then add **frequency drift** to demonstrate Hamiltonian-level drift with a compensation
knob. If time permits, add **Kerr drift** for a fundamentally different drift type.

Create a **systematic benchmark:**
- Each drift type × each optimizer × slow/medium/fast drift frequency
- Plot: tracking error vs drift frequency (identifies critical frequency)
- Plot: reward recovery time after step change

---

## 5. Extension to Moon Cats

### 5.1 Objective

Explore the "moon cat" variant (Rousseau et al. 2025) where an additional squeezing
interaction deforms the circular cat blobs into crescent-shaped states, enhancing bit-flip
protection by a factor of 160×.

### 5.2 Physics

Standard cat Hamiltonian + additional term:

$$H_\text{moon} = H_\text{standard} + g_2 \lambda\, \hat{a}^\dagger \hat{a}\, \hat{b}$$

where $\lambda$ is a new (5th) real control knob that controls the degree of squeezing.

**Key results from Rousseau et al. 2025:**

- Bit-flip scaling exponent $\gamma = 4.3$ (vs $\sim 1$ for standard cat)
- $T_X = 22$ seconds at $\bar{n} = 4.1$ photons
- $T_Z = 1.3\;\mu\text{s}$ at $\bar{n} = 4.1$ photons
- $160\times$ improvement in $T_X$ vs standard cat at same photon number
- Z-gate infidelity reduced by factor of 2

### 5.3 Implementation

```python
def build_moon_cat_hamiltonian(a, b, g2, eps_d, lam, kappa_b):
    """Moon cat = standard cat + squeezing interaction."""
    H_standard = (jnp.conj(g2) * a @ a @ b.dag() 
                  + g2 * a.dag() @ a.dag() @ b
                  - eps_d * b.dag() 
                  - jnp.conj(eps_d) * b)
    H_squeeze = g2 * lam * a.dag() @ a @ b
    return H_standard + H_squeeze
```

Control knobs expand to 5: $[\text{Re}(g_2),\;\text{Im}(g_2),\;\text{Re}(\varepsilon_d),\;\text{Im}(\varepsilon_d),\;\lambda]$

| Pros | Cons |
|------|------|
| Dramatic improvement in T_X (160×) according to literature | 5th control knob increases optimization difficulty |
| Directly referenced in challenge as exploration direction | Need to verify dynamiqs can handle the modified Hamiltonian |
| Shows depth of analysis (judging criteria) | Less starter code to build on |
| Novel: online optimization of moon cat params hasn't been done | May need larger Hilbert space (higher photon numbers) |

### 5.4 Recommendation

**Implement as a stretch goal after the core solution works.** The moon cat is a natural
extension that shows ambition and depth. Compare: standard cat vs moon cat, optimized
with the same optimizer, under the same drift conditions.

---

## 6. Extension to Single-Qubit Gates

### 6.1 Objective

Maintain bias and lifetimes while actively performing single-qubit gates on the cat qubit
(not just idling). This is the most challenging extension.

### 6.2 Physics

The iQuHack-2025 project (Task 1.3) implemented the **Zeno gate:**

$$H_\text{gate} = \varepsilon_Z (\hat{a}^\dagger + \hat{a})$$

applied to the storage mode. During the gate, single-photon loss and two-photon stabilization
compete. The optimizer must maintain stabilization quality while the gate Hamiltonian is active.

### 6.3 Implementation Approaches

**Gate-aware reward**
Add the gate Hamiltonian to the system and optimize control parameters to maintain
lifetime/bias during the gate.

**Time-dependent control**
Optimize a time-varying control sequence: [g₂(t), ε_d(t)] that changes during the gate
to compensate for the gate's effect on stabilization.

| Pros | Cons |
|------|------|
| Most impactful for real quantum computing | Significantly more complex |
| Directly relevant to fault-tolerant QC | Time-dependent control adds many parameters |
| Leverages iQuHack-2025 Zeno gate code | May require much longer optimization |

### 6.4 Recommendation

**Skip unless all other tasks are completed.** This is the hardest extension and
the least likely to produce clean results in a hackathon timeframe. Mention it in the
presentation as future work.

---

## 7. Recommended Strategy

### 7.1 Priority Ordering

Based on judging criteria (Technical Soundness > Depth of Analysis > Originality > Impact > Clarity):

| Priority | Task | Why |
|----------|------|-----|
| **1 (must)** | Core: CMA-ES + full reward, no drift | End-to-end baseline. Without this, nothing else matters. |
| **2 (must)** | Proxy reward + comparison with full | Key technical insight. Shows measurement efficiency thinking. |
| **3 (must)** | Gradient optimizer + comparison with CMA-ES | Originality differentiator. Shows JAX/dynamiqs mastery. |
| **4 (should)** | Amplitude drift + frequency drift | Required by challenge. Shows robustness analysis. |
| **5 (nice)** | Moon cat extension | Shows ambition and depth. |
| **6 (stretch)** | Kerr drift, TLS coupling | Extra depth. |
| **7 (skip)** | Single-qubit gates | Too complex for hackathon. Mention as future work. |

### 7.2 What "Done" Looks Like

A complete submission includes:

1. **Working end-to-end optimizer** (CMA-ES) that tunes g₂ and ε_d to achieve target bias and maximize lifetimes
2. **Two reward functions** (full and proxy) with analysis of their trade-offs
3. **Two optimizers** (CMA-ES and gradient-based) with benchmarking comparison
4. **At least one drift model** with tracking demonstration
5. **Clear plots** showing: reward convergence, parameter trajectories, T_X/T_Z/η evolution, drift tracking
6. **A narrative notebook** that tells the story: problem → approach → results → insights

### 7.3 Dynamiqs API Gotchas to Remember

| Gotcha | Solution |
| ------ | -------- |
| Default precision is `float32` | Use `dq.set_precision('double')` if numerical issues arise |
| No scalar `+` with operators | Use `dq.eye(n)` for identity: `delta * dq.eye(n)` not `delta + operator` |
| `*` is element-wise, not matmul | Always use `@` for operator products |
| `progress_meter=True` breaks JIT | Set `options=dq.Options(progress_meter=False)` inside `@jit` functions |
| Complex gradients need conjugation | Apply `grads.conj()` before passing to optax optimizer |
| CMA-ES minimizes (lower = better) | Negate reward when passing to `tell()` |
| `a.dag()` works | Alternative to `dq.dag(a)` |
| Batching uses broadcasting dims | Add `[:, None, None]` for parameter sweeps |
