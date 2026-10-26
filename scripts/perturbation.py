"""Sensitivity of test F1 to tiny price perturbations (same order as yfinance's call-to-call noise).

Prices are multiplied by (1 + e), e ~ N(0, EPS^2), the whole pipeline is rerun.
Writes results/perturbation.csv.

Usage: python scripts/perturbation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

from run_main import prepare_asset, XGB_PARAMS, RFC_PARAMS, EMBARGO, HORIZON
from src.data_loader import load_data
from src.features import COLUMN_ORDER
from src.models import RFCModel, XGBModel
from src.baselines import extrapolation_rule
from src.evaluation import fit_with_validation_threshold

ASSETS = ['SPY', 'DIA', 'QQQ']
EPS = 1e-6
REPS = 20
OUT = PROJECT_ROOT / 'results'


def perturb(df, rng):
    df = df.copy()
    noise = 1 + rng.normal(0, EPS, len(df))
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = df[c] * noise
    return df


def main():
    rng = np.random.default_rng(0)
    raws = {t: load_data(t) for t in ASSETS}
    rows = []
    for r in range(REPS + 1):
        data = {t: prepare_asset(t, raw if r == 0 else perturb(raw, rng)) for t, raw in raws.items()}
        train = data['SPY'][0]
        X_tr, y_tr = train[COLUMN_ORDER], train['Transition']
        for name, cls, params in [('XGB', XGBModel, XGB_PARAMS), ('RFC', RFCModel, RFC_PARAMS)]:
            model, th, _ = fit_with_validation_threshold(cls, params, X_tr, y_tr, embargo=EMBARGO)
            for t, (_, test, _, _) in data.items():
                pred = (model.predict_proba(test[COLUMN_ORDER]) >= th).astype(int)
                rows.append({'Rep': r, 'Model': name, 'Asset': t, 'Threshold': th,
                             'F1': f1_score(test['Transition'], pred, zero_division=0)})
        for t, (_, test, _, _) in data.items():
            rows.append({'Rep': r, 'Model': 'Extrapolation', 'Asset': t, 'Threshold': np.nan,
                         'F1': f1_score(test['Transition'], extrapolation_rule(test[COLUMN_ORDER], HORIZON))})
        print(f"rep {r} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'perturbation.csv', index=False)
    pert = df[df['Rep'] > 0]
    mean_us = pert.groupby(['Rep', 'Model'])['F1'].mean().unstack()
    print("\nMean F1 SPY/DIA/QQQ over perturbed runs:")
    print(mean_us.describe().loc[['mean', 'std', 'min', 'max']].round(3).to_string())
    print("\nThresholds selected:")
    print(pert[pert['Model'] != 'Extrapolation'].groupby('Model')['Threshold'].value_counts().to_string())


if __name__ == '__main__':
    main()
