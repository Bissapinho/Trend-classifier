"""Main experiment: train on SPY, evaluate on SPY/DIA/QQQ/GLD/TLT.

Thresholds are selected on a validation split inside the SPY training period.
Writes results/*.csv and results/run_info.json.

Usage: python scripts/run_main.py
"""
import sys
import json
import time
import platform
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
import yfinance
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.features import build_features, COLUMN_ORDER
from src.target import compute_transition
from src.models import RFCModel, XGBModel
from src.baselines import extrapolation_rule, stratified_expected_f1
from src.evaluation import (fit_with_validation_threshold, calibrate_threshold, iid_bootstrap_f1,
                            block_bootstrap_f1, event_recall, walk_forward)

TRAIN_ASSET = 'SPY'
ASSETS = ['SPY', 'DIA', 'QQQ', 'GLD', 'TLT']
SPLIT = 0.75
HORIZON = 30
EMBARGO = 30
B = 1000
OUT = PROJECT_ROOT / 'results'

# Hyperparameters from RandomizedSearchCV on the SPY training set (notebooks/02_model_selection.ipynb)
XGB_PARAMS = {}
RFC_PARAMS = {}


def prepare_asset(ticker, df_raw=None):
    if df_raw is None:
        df_raw = load_data(ticker)
    dates = pd.to_datetime(df_raw['Date'])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    y, crossings = compute_transition(df_raw, horizon=HORIZON)
    crossing_dates = set(dates.iloc[crossings])

    df = build_features(df_raw)
    df['Transition'] = y.values
    df['Date'] = dates.values
    df['Crossing'] = df['Date'].isin(crossing_dates)
    df = df.dropna().reset_index(drop=True)

    split = int(len(df) * SPLIT)
    return df.iloc[:split], df.iloc[split:].reset_index(drop=True), crossings, dates


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)

    data = {t: prepare_asset(t) for t in ASSETS}

    # ── Data summary ────────────────────────────────────────
    summary = []
    for t, (train, test, crossings, dates) in data.items():
        summary.append({
            'Asset': t,
            'First_raw_date': dates.iloc[0].date(), 'Last_date': dates.iloc[-1].date(),
            'Train_start': train['Date'].iloc[0].date(), 'Test_start': test['Date'].iloc[0].date(),
            'N_train': len(train), 'N_test': len(test),
            'Pos_rate_train': round(train['Transition'].mean(), 3),
            'Pos_rate_test': round(test['Transition'].mean(), 3),
            'Crossings_total': len(crossings),
            'Crossings_test': int(test['Crossing'].sum()),
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUT / 'data_summary.csv', index=False)
    print(df_summary.to_string(index=False))

    # ── Train on SPY, threshold on internal validation ──────
    train_spy = data[TRAIN_ASSET][0]
    X_tr, y_tr = train_spy[COLUMN_ORDER], train_spy['Transition']

    models, val_scores = {}, {}
    for name, cls, params in [('XGB', XGBModel, XGB_PARAMS), ('RFC', RFCModel, RFC_PARAMS)]:
        model, threshold, val_f1 = fit_with_validation_threshold(cls, params, X_tr, y_tr, embargo=EMBARGO)
        models[name] = (model, threshold)
        val_scores[name] = val_f1
        print(f"{name}: threshold={threshold:.2f} (validation F1={val_f1:.3f})")

    # The extrapolation rule's horizon is fixed to the target horizon; check what the train set prefers
    horizons = np.arange(5, 91, 5)
    f1_h = [f1_score(y_tr, extrapolation_rule(X_tr, horizon=h)) for h in horizons]
    print(f"Extrapolation rule: best horizon on SPY train = {horizons[int(np.argmax(f1_h))]} (used: {HORIZON})")

    # ── Test evaluation ─────────────────────────────────────
    rows, pred_frames, pr_auc = [], [], {}
    p_train = y_tr.mean()
    for t, (_, test, _, _) in data.items():
        X_te, y_te = test[COLUMN_ORDER], test['Transition'].values
        cross_pos = np.flatnonzero(test['Crossing'].values)

        probas = {name: m.predict_proba(X_te) for name, (m, _) in models.items()}
        preds = {name: (probas[name] >= th).astype(int) for name, (_, th) in models.items()}
        preds['Extrapolation'] = extrapolation_rule(X_te, horizon=HORIZON)
        pr_auc[t] = {name: round(average_precision_score(y_te, p), 3) for name, p in probas.items()}
        pred_frames.append(pd.DataFrame({
            'Asset': t, 'Date': test['Date'].dt.date, 'y': y_te, 'Crossing': test['Crossing'].astype(int),
            'proba_XGB': probas['XGB'], 'proba_RFC': probas['RFC'], 'pred_Extrapolation': preds['Extrapolation'],
        }))

        for name, pred in preds.items():
            iid_lo, iid_hi = iid_bootstrap_f1(y_te, pred, B=B)
            blk_lo, blk_hi = block_bootstrap_f1(y_te, pred, block=HORIZON, B=B)
            hit, n_ev = event_recall(pred, cross_pos, horizon=HORIZON)
            rows.append({
                'Asset': t, 'Model': name,
                'Precision': round(precision_score(y_te, pred, zero_division=0), 3),
                'Recall': round(recall_score(y_te, pred, zero_division=0), 3),
                'F1': round(f1_score(y_te, pred, zero_division=0), 3),
                'CI_iid_low': round(iid_lo, 3), 'CI_iid_high': round(iid_hi, 3),
                'CI_block_low': round(blk_lo, 3), 'CI_block_high': round(blk_hi, 3),
                'Events_detected': hit, 'Events': n_ev,
                'N_pred': int(pred.sum()),
            })
        rows.append({'Asset': t, 'Model': 'Stratified (expected)',
                     'F1': round(stratified_expected_f1(p_train, y_te.mean()), 3)})

    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUT / 'test_results.csv', index=False)
    pd.concat(pred_frames).to_csv(OUT / 'test_predictions.csv', index=False)
    print("\n" + df_res.to_string(index=False))

    us = df_res[df_res['Asset'].isin(['SPY', 'DIA', 'QQQ'])].groupby('Model')['F1'].mean().round(3)
    print("\nMean F1 over SPY/DIA/QQQ:\n" + us.to_string())

    # ── Naive protocol (as in notebook 02): no embargo, threshold picked on the SPY test set ──
    test_spy = data[TRAIN_ASSET][1]
    naive_rows = []
    for name, cls, params in [('XGB', XGBModel, XGB_PARAMS), ('RFC', RFCModel, RFC_PARAMS)]:
        model = cls(**params).fit(X_tr, y_tr)
        th, _ = calibrate_threshold(test_spy['Transition'], model.predict_proba(test_spy[COLUMN_ORDER]))
        for t, (_, test, _, _) in data.items():
            y_te = test['Transition'].values
            pred = (model.predict_proba(test[COLUMN_ORDER]) >= th).astype(int)
            lo, hi = iid_bootstrap_f1(y_te, pred, B=B)
            naive_rows.append({'Asset': t, 'Model': name, 'Threshold': th,
                               'F1': round(f1_score(y_te, pred, zero_division=0), 3),
                               'CI_iid_low': round(lo, 3), 'CI_iid_high': round(hi, 3)})
    df_naive = pd.DataFrame(naive_rows)
    df_naive.to_csv(OUT / 'naive_protocol.csv', index=False)
    naive_us = df_naive[df_naive['Asset'].isin(['SPY', 'DIA', 'QQQ'])].groupby('Model')['F1'].mean().round(3)
    print("\nNaive protocol:\n" + df_naive.to_string(index=False))
    print("Mean F1 over SPY/DIA/QQQ (naive):\n" + naive_us.to_string())

    # ── Walk-forward on SPY ─────────────────────────────────
    full_spy = pd.concat([data[TRAIN_ASSET][0], data[TRAIN_ASSET][1]], ignore_index=True)
    wf = {}
    for name, cls, params in [('XGB', XGBModel, XGB_PARAMS), ('RFC', RFCModel, RFC_PARAMS)]:
        df_wf, f1_global = walk_forward(full_spy[COLUMN_ORDER], full_spy['Transition'],
                                        cls, params, embargo=EMBARGO)
        df_wf.insert(0, 'Model', name)
        wf[name] = (df_wf, f1_global)
        print(f"\nWalk-forward {name}: pooled F1={f1_global:.3f}\n{df_wf.to_string(index=False)}")
    pd.concat([d for d, _ in wf.values()]).to_csv(OUT / 'walk_forward.csv', index=False)

    run_info = {
        'run_date': str(date.today()),
        'runtime_s': round(time.time() - t0),
        'python': platform.python_version(),
        'versions': {'pandas': pd.__version__, 'numpy': np.__version__, 'scikit-learn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'yfinance': yfinance.__version__},
        'thresholds': {name: th for name, (_, th) in models.items()},
        'validation_f1': {name: round(v, 3) for name, v in val_scores.items()},
        'average_precision': pr_auc,
        'walk_forward_pooled_f1': {name: round(f, 3) for name, (_, f) in wf.items()},
        'walk_forward_mean_std': {name: [round(d['F1'].mean(), 3), round(d['F1'].std(), 3)]
                                  for name, (d, _) in wf.items()},
        'mean_f1_spy_dia_qqq': us.to_dict(),
        'mean_f1_spy_dia_qqq_naive': naive_us.to_dict(),
        'settings': {'split': SPLIT, 'horizon': HORIZON, 'embargo': EMBARGO, 'bootstrap_B': B,
                     'validation_frac': 0.2, 'seed': 42},
    }
    (OUT / 'run_info.json').write_text(json.dumps(run_info, indent=2))
    print(f"\nDone in {run_info['runtime_s']}s")


if __name__ == '__main__':
    main()
