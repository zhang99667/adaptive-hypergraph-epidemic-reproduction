# Adaptive Hypergraph Epidemic Reproduction

Independent reproduction code for the paper:

**Adaptive Epidemic Dynamics on Hypergraphs with Group-Level Immunization and Rewiring**

The original author code was not included in the local workspace or arXiv source package. This repository provides a clean, runnable reproduction scaffold based on the equations, algorithms, and parameters reported in the paper.

## What Is Included

- Homogeneous MMCA for the adaptive simplicial SIS model.
- Stochastic Monte Carlo / quasistationary simulation on random uniform hypergraphs.
- Hyperedge-level interventions:
  - random hyperedge immunization,
  - targeted immunization by infection pressure,
  - spontaneous isolation by activity threshold.
- Random and degree-preferential rewiring.
- Reproduction scripts for Figs. 2-6 style experiments.
- Quick-run output figures and CSV files under `outputs/`.
- A detailed Chinese research report: `adaptive_hypergraph_research_report.html`.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Quick Run

```bash
python scripts/smoke_test.py
python scripts/audit_no_figure_inputs.py
python scripts/run_all.py --quick
```

Outputs are written to:

```text
outputs/data/
outputs/figures/
```

## Full Reproduction

The paper reports 200 independent MC simulations. To approach that setting, remove `--quick` and use larger repetition counts:

```bash
python scripts/reproduce_fig2.py --reps 200
python scripts/reproduce_fig4_fig5.py --reps 200
python scripts/reproduce_fig6.py
```

Running all scripts without `--quick` may take a long time.

## Important Notes

- The publisher PDF is intentionally not committed, because this repository is public.
- The reproduction scripts do not digitize, read, or copy data from the paper figures. Run `python scripts/audit_no_figure_inputs.py` to check this guardrail.
- The empirical congressional cosponsorship hypergraph used for Fig. 7 is not bundled with the paper or this workspace, so Fig. 7 is not reproduced here.
- The spontaneous isolation strategy in the paper has a small ambiguity: the algorithm sorts candidate hyperedges by activity in descending order, while the text sometimes interprets SI as targeting severely suppressed low-activity hyperedges. The default follows the algorithm, and scripts expose `--si-sort ascending` for sensitivity checks.
- For Figs. 4-5, immunization is applied once after a burn-in period. The paper does not fully specify whether hidden code applies interventions once, continuously, or cumulatively, so this convention is documented explicitly.

## Repository Layout

```text
src/adaptive_hypergraph/     Core model and simulation code
scripts/                     Reproduction entry points
outputs/data/                Quick-run CSV outputs
outputs/figures/             Quick-run PNG outputs
docs/original_request.md     Original user research request
docs/reproduction_audit.md   Guardrail against figure-derived reproduction
README_reproduction.md       Extra reproduction notes
```
