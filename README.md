# Golden Cross Transition Predictor

A machine learning project that predicts Golden Cross and Death Cross transitions
in equity ETFs (SPY, DIA, QQQ) up to 30 days in advance.

## Research Question

Can technical indicators anticipate MA50/MA200 crossover transitions 30 days ahead?

## Results

| Asset | Model | F1    |
|-------|-------|-------|
| SPY   | XGB   | 0.758 |
| DIA   | XGB   | 0.700 |
| QQQ   | XGB   | 0.807 |
| **Mean** | **XGB** | **0.755** |

## Project Structure

```
project/
├── src/
│   ├── data_loader.py             # yfinance data loader
│   ├── features.py                # Feature engineering pipeline
│   └── models.py                  # RFCModel and XGBModel classes
├── notebooks/
│   ├── 01_target_testing.ipynb    # Target definition & robustness tests
│   └── 02_model_selection.ipynb   # HP tuning, model comparison, cross-asset validation
└── README.md
```

## Methodology

**Target**: Binary label — does a Golden/Death Cross occur within the next 30 days?

**Features** (12 total):
- Price/return indicators: Return, Cumulated_Return_5d
- Technical indicators: Volatility, RSI14, ATR, Volume_ROC, VIX_spike
- MA-based indicators: Distance_GC, MA_velocity, MA50_slope,
  Distance_normalized, MA_cross_momentum

**Model**: XGBoost classifier with `scale_pos_weight` to handle class imbalance (~10% positive labels)

**Validation**: Strict temporal split (75% train / 25% test), cross-asset generalization
without retraining

## Setup

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.9+
- pandas, numpy
- scikit-learn
- xgboost
- yfinance
- matplotlib


## Paper

This project is the basis of an arXiv paper submitted to q-fin.ST and stat.ML. Still in progress.

*Independent Researcher — 2025*