import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform
from xgboost import XGBClassifier

from src.features import build_features, COLUMN_ORDER
from src.data_loader import load_data


# Data loading

def import_data(ticker: str):

    df_raw = load_data(ticker)
    X_raw  = build_features(df_raw)
    df     = X_raw.copy()

    ma50      = df_raw['Close'].rolling(50).mean()
    ma200     = df_raw['Close'].rolling(200).mean()
    gc        = (ma50 > ma200).astype(int).reset_index(drop=True)

    transition = np.zeros(len(df), dtype=int)
    label = gc.iloc[0]
    for i in range(1, len(gc)):
        if gc.iloc[i] != label:
            label = gc.iloc[i]
            transition[max(0, i - 30):i] = 1

    df['Transition'] = transition
    df = df.dropna().reset_index(drop=True)

    X = df[COLUMN_ORDER]
    y = df['Transition']

    split   = int(len(df) * 0.75)
    X_train = X.iloc[:split]
    X_test  = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test  = y.iloc[split:]

    print(f"[{ticker}] Train: {len(X_train)} | pos: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"[{ticker}] Test:  {len(X_test)}  | pos: {y_test.sum()}  ({y_test.mean()*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def compute_transition(df_raw):
    ma50  = df_raw['Close'].rolling(50).mean()
    ma200 = df_raw['Close'].rolling(200).mean()
    gc    = (ma50 > ma200).astype(int)
    transition = np.zeros(len(df_raw), dtype=int)
    label = gc.iloc[0]
    for i in range(1, len(gc)):
        if gc.iloc[i] != label:
            label = gc.iloc[i]
            transition[max(0, i - 30):i] = 1
    return pd.Series(transition, name='Transition')


def prepare_asset(ticker, split=0.75):
    df_raw = load_data(ticker)
    X_raw  = build_features(df_raw)
    df     = X_raw.copy()
    df['Transition'] = compute_transition(df_raw).values
    df     = df.dropna().reset_index(drop=True)
    split_idx = int(len(df) * split)
    X = df[COLUMN_ORDER]
    y = df['Transition']
    return X.iloc[split_idx:], y.iloc[split_idx:]


# Threshold calibration

def calibrate_threshold(model, X_val, y_val, thresholds=None):
    """Returns the threshold that maximizes F1 on the validation set."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.95, 0.05)

    proba   = model.predict_proba(X_val)
    if proba.ndim == 2:
        proba = proba[:, 1]
    best_t  = 0.5
    best_f1 = 0.0

    for t in thresholds:
        pred = (proba >= t).astype(int)
        f    = f1_score(y_val, pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t  = t

    print(f"  Best threshold: {best_t:.2f}  |  F1: {best_f1:.3f}")
    return round(best_t, 2)


# Hyperparameter search

def xgbHP(X_train, y_train):
    """Random search for XGBoost. Returns fitted model."""
    f1_scorer        = make_scorer(f1_score, zero_division=0)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    tscv             = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators':     randint(200, 600),
        'max_depth':        randint(3, 8),
        'learning_rate':    uniform(0.01, 0.09),
        'subsample':        uniform(0.5, 0.4),
        'colsample_bytree': uniform(0.5, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma':            uniform(0, 0.3),
    }

    last_train_idx, last_val_idx = list(tscv.split(X_train))[-1]
    eval_set = [(X_train.iloc[last_val_idx], y_train.iloc[last_val_idx])]

    xgb_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42
    )

    rs = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=100,
        scoring=f1_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        refit=False
    )
    rs.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    print(f"  XGB best params : {rs.best_params_}")
    print(f"  XGB best CV F1  : {rs.best_score_:.3f}")

    xgb_final = XGBClassifier(
        **rs.best_params_,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42
    )
    xgb_final.fit(
        X_train.iloc[last_train_idx], y_train.iloc[last_train_idx],
        eval_set=eval_set,
        verbose=False
    )
    print(f"  XGB best n_estimators (early stopping): {xgb_final.best_iteration}")

    return xgb_final


def rfcHP(X_train, y_train):
    """Random search for RFC. Returns fitted model."""
    f1_scorer = make_scorer(f1_score, zero_division=0)
    tscv      = TimeSeriesSplit(n_splits=5)

    param_dist = {
        'n_estimators':      randint(200, 600),
        'max_depth':         randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf':  randint(1, 10),
        'max_features':      uniform(0.3, 0.5),
    }

    rfc_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rs = RandomizedSearchCV(
        estimator=rfc_base,
        param_distributions=param_dist,
        n_iter=100,
        scoring=f1_scorer,
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        random_state=42,
        refit=True
    )
    rs.fit(X_train, y_train)

    print(f"  RFC best params : {rs.best_params_}")
    print(f"  RFC best CV F1  : {rs.best_score_:.3f}")

    return rs.best_estimator_


# Cross-asset validation

def validate_cross_assets(rfc_model, xgb_model, thresh_rfc, thresh_xgb, tickers):
    results = []
    for ticker in tickers:
        X_test, y_test = prepare_asset(ticker)

        for model, thresh, name in [
            (rfc_model, thresh_rfc, 'RFC'),
            (xgb_model, thresh_xgb, 'XGB')
        ]:
            proba = model.predict_proba(X_test)
            if proba.ndim == 2:
                proba = proba[:, 1]
            pred  = (proba >= thresh).astype(int)
            results.append({
                'Asset':     ticker,
                'Model':     name,
                'Precision': round(precision_score(y_test, pred, zero_division=0), 3),
                'Recall':    round(recall_score(y_test, pred, zero_division=0), 3),
                'F1':        round(f1_score(y_test, pred, zero_division=0), 3),
                'N_pred':    int(pred.sum()),
                'N_pos':     int(y_test.sum())
            })

    return pd.DataFrame(results)


# Full pipeline for one training asset


def run_training_asset(train_ticker: str, test_tickers: list):
    """
    Full pipeline for a single training asset:
    1. Load data + build target
    2. HP search (XGB + RFC)
    3. Calibrate thresholds on own test set
    4. Cross-asset validation on all other assets
    Returns dict with models, thresholds, results DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ASSET: {train_ticker}")
    print(f"{'='*60}")

    # 1. Data
    X_train, X_test, y_train, y_test = import_data(train_ticker)

    # 2. HP search
    print("\n[XGB] Running random search...")
    xgb_model = xgbHP(X_train, y_train)

    print("\n[RFC] Running random search...")
    rfc_model = rfcHP(X_train, y_train)

    # 3. Threshold calibration
    print("\n[XGB] Calibrating threshold...")
    thresh_xgb = calibrate_threshold(xgb_model, X_test, y_test)

    print("[RFC] Calibrating threshold...")
    thresh_rfc = calibrate_threshold(rfc_model, X_test, y_test)

    # 4. Cross-asset validation
    print(f"\n[Cross-asset] Evaluating on: {test_tickers}")
    df_results = validate_cross_assets(
        rfc_model, xgb_model, thresh_rfc, thresh_xgb, test_tickers
    )

    print(df_results.to_string(index=False))
    print(f"\nMean F1 RFC : {df_results[df_results['Model']=='RFC']['F1'].mean():.3f}")
    print(f"Mean F1 XGB : {df_results[df_results['Model']=='XGB']['F1'].mean():.3f}")

    return {
        'train_ticker': train_ticker,
        'xgb_model':    xgb_model,
        'rfc_model':    rfc_model,
        'thresh_xgb':   thresh_xgb,
        'thresh_rfc':   thresh_rfc,
        'results':      df_results
    }