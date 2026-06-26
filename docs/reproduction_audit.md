# Reproduction Audit

This repository is intended to reproduce the model from the paper equations and
reported simulation settings, not by digitizing or copying the paper's plotted
points.

## What Was Checked

- `src/` and `scripts/` do not read the local publisher PDF.
- `src/` and `scripts/` do not read extracted page images or arXiv figure PDFs.
- Reproduction scripts do not call image readers such as `imread` or
  `Image.open`.
- Reproduction scripts do not read precomputed figure CSV files as inputs.
- Hard-coded numeric values in the scripts are experiment settings stated in the
  paper text and captions, such as `n=1500`, `d=3`, `mu=0.1`, `eta=0.6`,
  `gamma=0.02`, `I(0)=0.3`, `<k>=6,9,12`, and the scan ranges used for figures.

## Guardrail

Run:

```bash
python scripts/audit_no_figure_inputs.py
```

Expected output:

```text
audit ok: no paper figure/PDF extraction inputs found in tracked source
```

## Remaining Scientific Caveat

The author code and raw empirical congressional cosponsorship hypergraph are not
public in the provided materials. Therefore this repository is an independent
implementation from the paper's equations and algorithm descriptions. Where the
paper leaves implementation details ambiguous, the convention is documented in
`README.md` and `README_reproduction.md`.
