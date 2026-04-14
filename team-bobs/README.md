# Learning Cats: Adaptive Control for Dissipative Cat Qubits

This project was developed for the **Alice & Bob Quantum Control Challenge**. We implement an advanced control framework for the stabilization of dissipative cat qubits, focusing on maximizing bit-flip suppression ($T_X$) and phase coherence ($T_Z$) in the presence of dynamic parameter drift.

## Challenge Objective

The "Core Challenge" focuses on designing and benchmarking an online optimization algorithm to maintain the performance of a cat qubit under varying hardware conditions. The primary metric is the **bias ratio** ($\eta = T_Z/T_X$), where high bit-flip protection ($T_X \gg 1 \mu s$) is prioritized.

## Methodology

Our approach combines high-performance physics simulations with derivative-free optimization and reinforcement learning.

### 1. High-Performance Simulation (JAX + Dynamiqs)
Stabilizing cat qubits requires solving the Lindblad master equation over long timescales. We utilized:
*   **Dynamiqs**: A JAX-native library for GPU-accelerated quantum dynamics.
*   **JIT-Compiled Core**: The entire simulation and metric extraction loop is JIT-compiled to minimize switch overhead between classical optimization and quantum simulation.
*   **2-Point Fast Lifetime Fit**: To avoid expensive full-decay curve analysis, we implemented a robust 2-point exponential fitting method to estimate $T_X$ and $T_Z$ in a single forward pass.

### 2. Global Optimization (CMA-ES)
*   We utilized the **Separable CMA-ES (Sep-CMA)** algorithm to locate the optimal control parameters ($g_2$, $\epsilon_d$) for steady-state operation. This provided a baseline for the maximum achievable bias in static conditions.
*   Implemented a reward function to guide optimization
*   Reward function tested with a reliable and costly method (exponential fit fot T_x and T_z) to ensure reasonable parameters can be achived (RL_Loss_Function_Test.ipynb).

### 3. Online Drift Compensation (PPO Reinforcement Learning)
To address time-varying drift (e.g., changes in drive amplitude or phase), we developed a **Proximal Policy Optimization (PPO)** agent using `Stable-Baselines3`:
*   **Gymnasium Environment**: A custom `CatQubitEnv` that simulates realistic synthetic drift.
*   **Adaptive Observation**: The agent monitors the current state and recent performance to predict optimal parameter adjustments.
*   **Parallel Training**: Leveraging multi-environment vectorization to accelerate sample collection on high-end GPUs.

## Reward Engineering

The reward function was carefully designed to reconcile the logarithmic nature of bit-flip protection:
$$R = -\left| \ln\left(\frac{T_Z}{100 \cdot T_X}\right) \right| + \text{bonus}(T_Z)$$
This objective ensures the agent prioritizes reaching the target bias threshold while continuously pushing for higher absolute coherence lifetimes.

## Key Features

*   **Drift-Aware Control**: Ability to track and compensate for amplitude and phase fluctuations in real-time.
*   **Computational Efficiency**: Using adaptive measurement windows to focus simulation power where it matters most.
*   **Robustness**: Effective across different "flavors" of cat qubit parameters.

## Results

The optimized controller successfully maintains a bias ratio $>100$ even under significant synthetic step-drift in the drive amplitude ($\epsilon_d$). The transition from CMA-ES to RL allowed for a significantly more responsive control loop compared to standard periodic recalibration.

---
**Team BOBS**
*Quantum Control & Machine Learning*

**Slide Show**
https://docs.google.com/presentation/d/1hzKO0FUQ8FnlWalzt8DRbu-AxPX3RoWvHRRT6qh_zUo/edit?usp=sharing
