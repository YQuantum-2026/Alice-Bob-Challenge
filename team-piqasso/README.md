# Team Piqasso Submission Notes

This folder contains a standalone exploratory notebook:

- `four-parameter-optimizer.ipynb`

## What it does

The notebook builds a simple optimization loop for the four real control knobs behind the complex cat-stabilization parameters:

- `Re(g_2)`
- `Im(g_2)`
- `Re(epsilon_d)`
- `Im(epsilon_d)`

It uses `dynamiqs` to simulate logical decay, estimates proxy lifetimes `T_X` and `T_Z`, and then applies CMA-ES to improve a reward that balances:

- longer lifetimes
- a target bias `T_Z / T_X`

## Outputs

The notebook generates:

- reward versus epoch
- optimizer trajectories for all four controls
- estimated `T_X`, `T_Z`, and bias across epochs
- scatter plots of sampled controls colored by reward
- final logical decay curves for the best candidate

## How to run

1. Install the repository requirements from the root:
   `pip install -r requirements.txt`
2. Open `team-piqasso/four-parameter-optimizer.ipynb`.
3. Run the cells from top to bottom.

This is a baseline model meant to be easy to inspect and extend rather than a fully optimized final submission.
