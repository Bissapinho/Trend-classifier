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
from src.baselines import extrapolation_rule, extrapolation_score, stratified_expected_f1
from src.evaluation import (fit_with_validation_threshold, calibrate_threshold, iid_bootstrap_f1,
                            block_bootstrap_f1, paired_block_bootstrap_f1_diff, event_recall,
                            event_metrics, walk_forward)

TRAIN_ASSET = 'SPY'
ASSETS = ['SPY', 'DIA', 'QQQ', 'GLD', 'TLT']
SPLIT = 0.75
HORIZON = 30
EMBARGO = 30
B = 1000
B_PAIRED = 2000
US = ['SPY', 'DIA', 'QQQ']
OUT = PROJECT_ROOT / 'results'

# Hyperparameters from RandomizedSearchCV on the SPY training set (notebooks/02_model_selection.ipynb)
XGB_PARAMS = dict(n_estimators=138, max_depth=5, learning_rate=0.0430, subsample=0.7273,
                  colsample_bytree=0.6976, gamma=0.0536, min_child_weight=1)
RFC_PARAMS = dict(n_estimators=346, max_depth=9, min_samples_split=17, min_samples_leaf=5,
                  max_features=0.5463)


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
    rows, pred_frames, pr_auc, paired_rows, event_rows = [], [], {}, [], []
    p_train = y_tr.mean()
    for t, (_, test, _, _) in data.items():
        X_te, y_te = test[COLUMN_ORDER], test['Transition'].values
        cross_pos = np.flatnonzero(test['Crossing'].values)

        probas = {name: m.predict_proba(X_te) for name, (m, _) in models.items()}
        preds = {name: (probas[name] >= th).astype(int) for name, (_, th) in models.items()}
        preds['Extrapolation'] = extrapolation_rule(X_te, horizon=HORIZON)
        scores = dict(probas, Extrapolation=extrapolation_score(X_te))
        pr_auc[t] = {name: round(average_precision_score(y_te, p), 3) for name, p in scores.items()}
        pred_frames.append(pd.DataFrame({
            'Asset': t, 'Date': test['Date'].dt.date, 'y': y_te, 'Crossing': test['Crossing'].astype(int),
            'proba_XGB': probas['XGB'], 'proba_RFC': probas['RFC'],
            'score_Extrapolation': scores['Extrapolation'], 'pred_Extrapolation': preds['Extrapolation'],
        }))

        # Paired test: same resampled days for the rule and each model
        for name in ['XGB', 'RFC']:
            diff, lo, hi, p_le0 = paired_block_bootstrap_f1_diff(
                y_te, preds['Extrapolation'], preds[name], block=HORIZON, B=B_PAIRED)
            paired_rows.append({'Asset': t, 'Comparison': f'Extrapolation - {name}',
                                'F1_diff': round(diff, 3), 'CI_low': round(lo, 3), 'CI_high': round(hi, 3),
                                'Share_diff_le_0': round(p_le0, 4), 'CI_excludes_0': bool(lo > 0 or hi < 0)})

        years = len(y_te) / 252
        for name, pred in preds.items():
            ev = event_metrics(y_te, pred, cross_pos, horizon=HORIZON)
            event_rows.append({'Asset': t, 'Model': name, 'Crossings': len(cross_pos), 'Test_years': round(years, 2),
                               'Median_lead': ev['Median_lead'], 'Min_lead': ev['Min_lead'],
                               'FA_days': ev['FA_days'], 'FA_days_per_year': round(ev['FA_days_per_year'], 1),
                               'FA_episodes': ev['FA_episodes'],
                               'FA_episodes_per_year': round(ev['FA_episodes_per_year'], 2)})

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

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(OUT / 'paired_differences.csv', index=False)
    print(f"\nPaired block bootstrap (B={B_PAIRED}), F1 rule - model:\n" + df_paired.to_string(index=False))

    df_events = pd.DataFrame(event_rows)
    df_events.to_csv(OUT / 'event_metrics.csv', index=False)
    print("\nEvent metrics:\n" + df_events.to_string(index=False))
    pooled = df_events.groupby('Model')[['FA_days', 'FA_episodes']].sum()
    fa_ratio = df_events.pivot(index='Asset', columns='Model', values='FA_days')
    fa_ratio = fa_ratio[['XGB', 'RFC']].div(fa_ratio['Extrapolation'], axis=0)
    event_summary = {
        'pooled_FA_days': pooled['FA_days'].to_dict(),
        'pooled_FA_episodes': pooled['FA_episodes'].to_dict(),
        'pooled_FA_days_ratio_to_rule': (pooled['FA_days'] / pooled.loc['Extrapolation', 'FA_days']).round(2).to_dict(),
        'per_asset_FA_days_ratio_to_rule_range': [round(fa_ratio.values.min(), 2), round(fa_ratio.values.max(), 2)],
    }

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

    # ── Protocol decomposition: change one element of the naive protocol at a time ──
    # (a) threshold on the SPY test set, model fitted without the last EMBARGO training rows
    # (b) threshold on the validation block, no embargo (inner fit up to the block, refit on all rows)
    n_tr = len(X_tr)
    decomp_rows = []
    for name, cls, params in [('XGB', XGBModel, XGB_PARAMS), ('RFC', RFCModel, RFC_PARAMS)]:
        model_a = cls(**params).fit(X_tr.iloc[:n_tr - EMBARGO], y_tr.iloc[:n_tr - EMBARGO])
        th_a, _ = calibrate_threshold(test_spy['Transition'], model_a.predict_proba(test_spy[COLUMN_ORDER]))
        model_b, th_b, _ = fit_with_validation_threshold(cls, params, X_tr, y_tr, embargo=0)
        variants = [
            ('naive', 'test', 0, None, None),
            ('a_test_threshold_embargo', 'test', EMBARGO, model_a, th_a),
            ('b_validation_threshold_no_embargo', 'validation', 0, model_b, th_b),
            ('corrected', 'validation', EMBARGO, *models[name]),
        ]
        for variant, source, emb, model, th in variants:
            for t, (_, test, _, _) in data.items():
                if variant == 'naive':
                    r = df_naive[(df_naive['Model'] == name) & (df_naive['Asset'] == t)].iloc[0]
                    th_v, f1 = r['Threshold'], r['F1']
                else:
                    pred = (model.predict_proba(test[COLUMN_ORDER]) >= th).astype(int)
                    th_v, f1 = th, round(f1_score(test['Transition'], pred, zero_division=0), 3)
                decomp_rows.append({'Variant': variant, 'Threshold_source': source, 'Embargo': emb,
                                    'Model': name, 'Asset': t, 'Threshold': th_v, 'F1': f1})
    df_decomp = pd.DataFrame(decomp_rows)
    df_decomp.to_csv(OUT / 'protocol_decomposition.csv', index=False)
    decomp_us = (df_decomp[df_decomp['Asset'].isin(US)].groupby(['Variant', 'Model'])['F1'].mean()
                 .round(3).unstack())
    decomp_th = df_decomp.groupby(['Variant', 'Model'])['Threshold'].first().unstack()
    print("\nProtocol decomposition, mean F1 SPY/DIA/QQQ:\n" + decomp_us.to_string())
    print("Thresholds:\n" + decomp_th.to_string())
    # Effect of each change, measured from both ends of the path naive -> corrected
    decomp_effects = {name: {
        'embargo_with_test_threshold': round(decomp_us.loc['a_test_threshold_embargo', name] - decomp_us.loc['naive', name], 3),
        'embargo_with_validation_threshold': round(decomp_us.loc['corrected', name] - decomp_us.loc['b_validation_threshold_no_embargo', name], 3),
        'validation_threshold_without_embargo': round(decomp_us.loc['b_validation_threshold_no_embargo', name] - decomp_us.loc['naive', name], 3),
        'validation_threshold_with_embargo': round(decomp_us.loc['corrected', name] - decomp_us.loc['a_test_threshold_embargo', name], 3),
        'total': round(decomp_us.loc['corrected', name] - decomp_us.loc['naive', name], 3),
    } for name in ['XGB', 'RFC']}
    print("Effects:\n" + json.dumps(decomp_effects, indent=2))

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
        'protocol_decomposition_mean_f1_spy_dia_qqq': decomp_us.to_dict(orient='index'),
        'protocol_decomposition_thresholds': decomp_th.to_dict(orient='index'),
        'protocol_decomposition_effects': decomp_effects,
        'event_summary': event_summary,
        'settings': {'split': SPLIT, 'horizon': HORIZON, 'embargo': EMBARGO, 'bootstrap_B': B, 'paired_bootstrap_B': B_PAIRED,
                     'validation_frac': 0.2, 'seed': 42},
    }
    (OUT / 'run_info.json').write_text(json.dumps(run_info, indent=2))
    print(f"\nDone in {run_info['runtime_s']}s")


if __name__ == '__main__':
    main()
