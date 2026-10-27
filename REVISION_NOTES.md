# Revision notes (October 2026)

Corrections applied after a review of the method and of the paper. Every number added or changed
in the paper and README comes from a file in `results/` written by a script, and a check script
confirmed they match. The general conclusion is unchanged: the extrapolation rule does at least
as well as XGBoost and Random Forest.

Commits: `d09ecfd` (method), `6438494` (paper form), `18a0d82` (repository form).

## Done

### A. Method

| # | Change | Output |
|---|--------|--------|
| 1 | Paired moving-block bootstrap on the F1 difference, rule − model (30-day blocks, same indices for both methods, B = 2000). Replaces "no per-asset ranking is statistically supported" in the paper and README, with a sentence on why overlapping marginal intervals are not enough. | `results/paired_differences.csv` |
| 2 | Event metrics: median lead of the first alert (capped at 30 days), false-alarm days, false-alarm episodes per year. New "Event-level metrics" subsection; the 30-day ceiling is stated. | `results/event_metrics.csv` |
| 3 | Continuous score for the rule, 1/(1+τ_t) (same ranking as −τ_t), its average precision per asset, and its curve on the SPY precision-recall figure. "No learned parameters" replaced by: lookback of 5 fixed by hand, horizon checked on the train set. | `run_info.json`, `figures/pr_curve_spy.*` |
| 4 | Protocol decomposition: (a) threshold on the test set with embargo and refit, (b) threshold on validation without embargo. The paper states that Random Forest measures the protocol effect better, since XGBoost moves by up to 0.047 under 1e-6 perturbations. | `results/protocol_decomposition.csv` |
| 5 | Noise test: multiplicative noise P·(1+ε), σ ∈ [0.004, 0.20], instead of σ·sd(P). The result adds little, so the section moved to Appendix A. | `results/noise_curves.csv`, `results/target_analysis.json`, `figures/target_noise.*` |
| 6 | Limitations: MA_cross_momentum is in price units (MA50_accel in dollars); hyperparameters were chosen on the SPY train set (to 2018), which overlaps walk-forward folds 2 and 3. | paper, README |

### B. Paper form

- 7: indicator function written `\mathbf{1}` (macro `\ind`) instead of `\mathbb{1}`.
- 8: wide tables use `\small` and two-line headers; `xurl` loaded for the URL.
- 9: affiliation set to "MSc student, CentraleSupélec – ESSEC".
- 10: protocol-effect and per-asset F1 figures moved into Sections 5.1 and 5.3; the other figures stay in the appendix.
- 11: contribution 1 (reproducible pipeline) removed; the Reproducibility paragraph covers it.
- 12: references checked.
  - Bhootra: year corrected 2023 → 2024 (*Managerial Finance* 50(6):1066–1088, doi:10.1108/MF-04-2023-0257).
  - Avramov, Kaplanski and Subrahmanyam (2018): title confirmed against the SSRN working paper; number 3111334 added.

### C. Repository form

- 14: notebook 01 imports `add_target` and `add_target_ma_cross` from `src/target.py`; it runs end to end and was re-executed. The README note about broken imports is removed.
- 15: `XGB_PARAMS` and `RFC_PARAMS` in `scripts/run_main.py` hold the selected values (results identical).
- 16: `data/CHECKSUMS.sha256` for the cached CSVs; check with `python scripts/verify_data.py` or `cd data && sha256sum -c CHECKSUMS.sha256`.
- 17: README retitled, link to the paper, three result bullets at the top, numbers updated.

### Verification

`run_main.py`, `target_analysis.py`, `perturbation.py` and `make_figures.py` run without error.
Rerunning them leaves every CSV and PNG unchanged.

## Differences from the reviewer's control values

- TLT rule − XGBoost: +0.166 [0.003, 0.341], against the expected [0.00, 0.32]. All other control values match.
- "3 to 5 times more false alarms" holds pooled over the five assets (4.0× XGBoost, 3.0× Random Forest) but ranges from 1.9 to 7.5 per asset. The paper reports both.
- The models alert earlier only on SPY and DIA. On QQQ and GLD all methods reach the 30-day ceiling; on TLT the rule leads by one day.
- The embargo effect is small and changes sign (−0.010 / +0.016 for Random Forest); almost all of the protocol effect comes from where the threshold is selected.

## Left to do

- [x] **13. Compile the paper with pdflatex and commit `paper/GC_prediction_paper.pdf`.** Compiled with pdfTeX (TeX Live 2026) and committed.
- [ ] After compiling, check the PDF: no "⊮" glyph, no table or URL past the right margin (the walk-forward, data, event and main results tables are the ones to watch).
- [ ] Optional: cite the published version of Avramov et al. (*Review of Financial Economics*, 2021, "Moving Average Distance as a Predictor of Equity Returns") instead of the 2018 working paper.
- [ ] Optional: rename the repository, for example `golden-cross-evaluation`.
- [ ] Note: the cached CSVs in `data/` are not committed, so the checksums only help a reviewer who has the same files.
