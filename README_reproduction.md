# Adaptive hypergraph epidemic reproduction code

This repository did not include the authors' original simulation code. The code
here is an independent reproduction scaffold based on the equations and
parameters reported in the paper:

`Adaptive Epidemic Dynamics on Hypergraphs with Group-Level Immunization and Rewiring`.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The core model only requires `numpy`. Plotting scripts require `matplotlib`.

## Quick verification

```bash
python scripts/smoke_test.py
```

## Reproduction scripts

The scripts under `scripts/` use `--quick` for a lightweight run. Remove
`--quick` and increase `--reps` to approach the paper's 200 independent Monte
Carlo simulations.

```bash
python scripts/reproduce_fig2.py --quick
python scripts/reproduce_fig3.py --quick
python scripts/reproduce_fig4_fig5.py --quick
python scripts/reproduce_fig6.py --quick
python scripts/run_all.py --quick
```

Outputs are written to `outputs/data/` and `outputs/figures/`.

## Important reproduction notes

- The paper reports QS simulation with a history of 50 active configurations
  and probability 0.2. This implementation uses the standard convention:
  absorption is always replaced by a stored active configuration, and `0.2` is
  the probability of refreshing the QS history.
- The spontaneous isolation algorithm in the paper sorts candidate hyperedges
  by activity in descending order, while parts of the text interpret SI as
  removing severely suppressed low-activity hyperedges. The default follows the
  algorithm (`descending`), and scripts expose `--si-sort ascending` for
  sensitivity checks.
- For Figs. 4-5, the paper is ambiguous about whether immunization is applied
  once, continuously under a simultaneous budget, or cumulatively over time.
  The reproduction scripts apply the selected strategy once after a burn-in
  period, which makes SI meaningful because hyperedge activities have already
  adapted. This is a deliberate, documented convention rather than a claim that
  it matches hidden author code.
- The empirical congressional cosponsorship data used in Fig. 7 is not bundled
  with the paper or current workspace, so the code provides synthetic
  reproductions for Figs. 2-6 and a loader hook for uniform edge-list data.
