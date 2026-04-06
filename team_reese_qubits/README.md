# yQuantum — Robust Control of Cat Qubits
**Team Reese Qubits**

## 🚀 Overview

This project explores how to automatically tune control parameters in a quantum system (cat qubits) to improve stability, error protection, and robustness under drift.

We use optimization (CMA-ES) to learn control parameters that:
- reduce system loss
- suppress bit-flip errors
- preserve quantum coherence
- recover from disturbances in real time

The goal is to move toward **adaptive, self-correcting quantum control systems**.

---

## 🧠 Key Idea

Instead of manually calibrating quantum systems, we treat control as an optimization problem:

- The system produces feedback (loss, observables)
- An optimizer adjusts parameters (e.g., drive, detuning)
- The system converges to a stable, protected regime

---

## 📊 Results

- **Robust convergence**
  - Loss improved: `-0.8 → -1.37`

- **Improved quantum properties**
  - Bit-flip protection increased (`s_z ↑`)
  - Coherence improved (`s_x ↑`)

- **Recovery under drift**
  - Introduced +30% perturbation in `g₂`
  - System recovered in ~15 epochs

- **Control insight**
  - Adding detuning improved recovery speed
  - Shows importance of richer control parameters

---

## 🔬 What This Means

Our system doesn’t just optimize performance — it learns a **stable operating regime**:

- balances drive and dissipation
- protects logical states
- adapts to changes in system dynamics

This is critical for real-world quantum devices, where drift and noise are unavoidable.

---

## 🏗️ How We Built It

- Simulated a bosonic (cat qubit) system
- Defined a loss function based on physical observables
- Used **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) for optimization
- Tuned parameters such as:
  - two-photon drive (`g₂`)
  - drive amplitude (`ε_d`)
  - detuning (`Δ`)

---

## ⚙️ Tech Stack

- Python
- NumPy / JAX (depending on implementation)
- Scientific simulation tools
- CMA-ES optimizer

---

## ⚠️ Challenges

- Balancing **coherence vs stability** (usually a tradeoff)
- Handling **drift and perturbations**
- Designing a meaningful loss function tied to physics
- Ensuring convergence is not just numerical, but physically meaningful

---

## 🏆 Accomplishments

- Achieved simultaneous improvement in:
  - bit-flip protection
  - coherence (rare to get both)
- Demonstrated **recovery under system drift**
- Showed that **adding control knobs (detuning)** improves adaptability

---

## 📚 What We Learned

- Optimization can discover **non-obvious physical regimes**
- Control parameters like detuning are critical for stability
- Robustness is just as important as raw performance
- Adaptive control is key for scalable quantum systems

---

## 🔮 What's Next

- Integrate real-time feedback (reinforcement learning / adaptive control)
- Scale to larger quantum systems
- Combine with quantum error correction (QEC)
- Test on hardware or more realistic noise models

---

## 📎 Submission

This repository includes:
- Code for simulation and optimization
- Results demonstrating convergence and robustness
- Documentation for reproducibility

---

## 👥 Team Reese Qubits

- Luis Mendez  
- Terrell Osborne  
- Akshay Pespunuri  
- Malachi Collins  

---

## 💡 Tagline

**“Learning to stabilize quantum systems — automatically.”**