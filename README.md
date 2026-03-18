# Golden Cross Transition Predictor

A machine learning project investigating whether **the choice of training asset
systematically affects cross-asset generalizability** of MA crossover transition
predictors in financial time series.

## Research Question

Does training asset selection drive the cross-asset generalizability of ML-based
Golden Cross / Death Cross transition predictors — and if so, what asset properties
explain this effect?
For now, we are exploring and testing whether this claim may be true.

## Project Structure

```
project/
├── src/
│   ├── data_loader.py          # yfinance data loader
│   ├── features.py             # Feature engineering pipeline
│   ├── models.py               # RFCModel and XGBModel classes
│   └── asset_analyzer.py       # Full training pipeline (HP search, calibration,
│                               #   cross-asset validation)
├── notebooks/
│   ├── 01_target_testing.ipynb         # Target definition & robustness tests
│   ├── 02_model_selection.ipynb        # HP tuning, model comparison (SPY baseline)
│   ├── 03_EEM.ipynb                    # EEM as training asset — first pivot
│   ├── 04_testsSPY.ipynb               # SPY retraining + scale_pos_weight ablation
│   └── 05_generalization_experiment.py # Full 9-asset generalization matrix
└── README.md
```

## Methodology

**Target**: Binary label — does a Golden/Death Cross occur within the next 30 days?

**Features** (12 total):
- Price/return indicators: `Return`, `Cumulated_Return_5d`
- Technical indicators: `Volatility`, `RSI14`, `ATR`, `Volume_ROC`, `VIX_spike`
- MA-based indicators: `Distance_GC`, `MA_velocity`, `MA50_slope`,
  `Distance_normalized`, `MA_cross_momentum`

**Models**: XGBoost and Random Forest classifiers.
Class imbalance (~10% positive labels) handled via `scale_pos_weight` (XGB)
and `class_weight='balanced'` (RFC).

**Validation**:
- Strict temporal split (75% train / 25% test), no random shuffling
- Hyperparameters tuned via RandomizedSearchCV with TimeSeriesSplit (k=5)
- Decision threshold calibrated on own test set (maximize F1)
- Cross-asset evaluation on 8 held-out assets without retraining

**Training assets tested**: SPY, QQQ, IWM, EEM, EWJ, EWQ, URTH, TLT, GLD

**Asset universe**: covers US equity, international equity, global equity,
bonds, and commodities — all with data available from ~2000.

> Note: USO (oil ETF, inception 2006) and HYG (high yield, inception 2007)
> were excluded due to insufficient history. GLD (inception 2004) and
> URTH (inception 2012) are included with a note on their shorter history.

## Setup

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.9+, pandas, numpy, scikit-learn, xgboost, yfinance, matplotlib, scipy

## Paper

This project is the basis of an arXiv paper (q-fin.ST / stat.ML).
The central contribution is an empirical study of how training asset selection
affects cross-asset generalizability of ML-based regime transition predictors —
an effect documented in practice but rarely studied explicitly in the literature.

*Independent Researcher — 2026*
