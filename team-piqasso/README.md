# Piqasso -- Alice & Bob Cat-Qubit Challenge (YQuantum 2026)

## Team

**Batu Yalcin, Tymofii Baranenko, Leon Katarzynski, Jack Ploof, and Alex Wong**

## Overview

This repository contains Team Piqasso's submission to the Alice & Bob dissipative cat-qubit stabilization challenge at YQuantum 2026. The challenge explores model-free, measurement-based control strategies for the stabilization and control of dissipative cat qubits.

**Guiding question**: *How does drift in the system impact our system and (physically) how can we counteract it to ensure stability?*

We characterize cat qubits through end-to-end Lindblad simulations using the `dynamiqs` package, implement CMA-ES and PPO optimizers over the Hamiltonian control parameters g2 and epsilon_d, explore crescent (moon) cat qubits with cross-Kerr interactions, analyze the effects of drift/detuning/Kerr nonlinearities, and examine single-qubit gate implementations.

## Repository Structure

```
.
├── README.md                        ← this file
├── LICENSE                          ← MIT license (Team Piqasso's original work)
├── Main_PiqassoNotebook.ipynb       ← main submission notebook
├── Conclusion_Jupyter.ipynb         ← conclusion and future directions
├── results.txt                      ← optimization results log
├── *.png / *.gif                    ← figures referenced by the notebook
│
├── challenge/                       ← upstream Alice & Bob challenge materials (unmodified)
│   ├── 1-challenge.ipynb            ← original challenge notebook
│   ├── resources/                   ← tutorials (dynamiqs, cat qubits, open quantum systems)
│   └── *.py                         ← scratch/graph scripts
│
└── team-piqasso/                    ← team submission folder
    ├── README.md                    ← detailed submission notes and run instructions
    ├── Main_PiqassoNotebook.ipynb   ← copy of main notebook
    ├── Notebooks/                   ← topic notebooks (Drift, Detuning, AddZ, etc.)
    ├── Scripts/                     ← Python utilities (CMA-ES, PPO, optimizers, analysis)
    ├── Graphs/                      ← figures and PPO result exports
    ├── Gifs/                        ← Wigner function animations for crescent cats
    ├── licenses/                    ← attribution and third-party license inventory
    └── outputs/                     ← PPO training outputs
```

## Submission Contents

### 1. End-to-End Measurement and Optimization

- **CMA-ES optimizer** over Re/Im(g2), Re/Im(epsilon_d) using `dynamiqs` Lindblad simulations (`Scripts/cmas.py`, `four-parameter-optimizer.ipynb`)
- **PPO reinforcement learning optimizer** with surrogate and Lindblad backends (`ppo_batched_parallel_search.py`, `rl_refinement_notebook.ipynb`)
- Reward function: `0.5 * log(Tx * Tz) - lambda_eta * (log(eta) - log(eta_target))^2` with hard gating on bias range
- Both optimizers converge to similar parameter regions; comparative analysis included in the main notebook

### 2. Crescent (Moon) Cat Qubits

- Extended Hamiltonian with cross-Kerr term `lambda * a_dag * a * b`
- Wigner function GIF animations generated from optimized parameters (`team-piqasso/Gifs/`, `challenge/build_crescent_wigner_gifs.py`)

### 3. Drift Analysis and Correction

- Phase drift on g2, constant detuning, and Kerr nonlinearity effects on lifetimes (`Notebooks/Drift.ipynb`, `Notebooks/Detuning.ipynb`)
- Correction strategies using moon cat qubits for two-level system interactions

### 4. Single-Qubit Gates

- Hamiltonian representations of Z and X gates
- Rabi oscillation simulations (`Notebooks/AddZ.ipynb`)

### 5. Conclusion and Future Directions

- Discussion of experimental constraints (buffer mode necessity, dissipative stabilization, single-photon loss)
- Four-photon cat qubits and pair-cat qubits as future directions (`Conclusion_Jupyter.ipynb`)

## Key Results

| Optimizer | Best Reward | Tz (us) | Tx (us) | Bias (Tz/Tx) | Target Bias Error |
|-----------|-------------|---------|---------|---------------|-------------------|
| CMA-ES    | +5.63       | 224.6   | 0.67    | 337.4         | 5.4%              |
| PPO       | +4.61       | 75.9    | 0.23    | 331.3         | 3.5%              |

Target bias: 320. Full optimization logs are in `results.txt`.

## Getting Started

**Python 3.10+** is required. Key dependencies:

- `dynamiqs`, `jax`, `numpy`, `scipy`, `matplotlib`, `torch`, `cmaes`
- Optional: `qutip` (for `cat_size_proof.py`)

See `team-piqasso/licenses/THIRD_PARTY_LICENSES.md` for a complete dependency inventory, and `team-piqasso/README.md` for detailed run instructions for each notebook and script.

## Challenge Context

This submission is for the **Alice & Bob x YQuantum 2026 Challenge** on dissipative cat-qubit stabilization and control. The challenge materials in `challenge/` are adapted from [iQuHACK/2025-Alice-and-Bob](https://github.com/iQuHACK/2025-Alice-and-Bob). Tutorial resources in `challenge/resources/` cover dynamiqs, cat qubits, and open quantum systems.

## Acknowledgments

- **Alice & Bob** for the challenge materials and the `dynamiqs` quantum simulation package
- AI-assisted coding tools (ChatGPT, Claude) were used for coding assistance
- See `team-piqasso/licenses/ACKNOWLEDGMENTS.md` for full attribution

## License

The team's original code in `team-piqasso/` and the repository root is released under the **MIT License** (see `LICENSE.md`). The challenge materials in `challenge/` are provided by Alice & Bob and are subject to their own licensing terms. See `team-piqasso/licenses/` for a detailed third-party license inventory.
