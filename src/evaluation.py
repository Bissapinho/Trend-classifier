import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

THRESHOLDS = np.round(np.arange(0.10, 0.951, 0.05), 2)


def calibrate_threshold(y_val, proba_val, thresholds=THRESHOLDS):
    """Threshold maximizing F1 on a validation set (never the test set)."""
    scores = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in thresholds]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def fit_with_validation_threshold(model_class, model_params, X_train, y_train,
                                  val_frac=0.2, embargo=30):
    """Pick the threshold on the last `val_frac` of the training period, then refit on all of it.

    `embargo` rows are dropped before each boundary because the label at t depends
    on prices up to t+30.
    """
    n = len(X_train)
    val_start = int(n * (1 - val_frac))
    fit_end = val_start - embargo

    inner = model_class(**model_params).fit(X_train.iloc[:fit_end], y_train.iloc[:fit_end])
    threshold, val_f1 = calibrate_threshold(y_train.iloc[val_start:],
                                            inner.predict_proba(X_train.iloc[val_start:]))

    final = model_class(**model_params).fit(X_train.iloc[:n - embargo], y_train.iloc[:n - embargo])
    return final, threshold, val_f1


def iid_bootstrap_f1(y_true, y_pred, B=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    boot = [f1_score(y_true[idx], y_pred[idx], zero_division=0)
            for idx in (rng.integers(0, n, n) for _ in range(B))]
    alpha = (100 - ci) / 2
    return np.percentile(boot, alpha), np.percentile(boot, 100 - alpha)


def block_bootstrap_f1(y_true, y_pred, block=30, B=1000, ci=95, seed=42):
    """Moving-block bootstrap: resamples contiguous blocks to respect label autocorrelation."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    n_blocks = int(np.ceil(n / block))
    boot = np.zeros(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        boot[b] = f1_score(y_true[idx], y_pred[idx], zero_division=0)
    alpha = (100 - ci) / 2
    return np.percentile(boot, alpha), np.percentile(boot, 100 - alpha)


def event_recall(y_pred, crossing_positions, horizon=30):
    """Share of crossovers with at least one alert in the `horizon` days before them.

    `crossing_positions` are positions in the test arrays (crossover day itself).
    """
    y_pred = np.asarray(y_pred)
    hits = [y_pred[max(0, c - horizon):c].any() for c in crossing_positions]
    return int(sum(hits)), len(hits)


def walk_forward(X, y, model_class, model_params, n_splits=5, min_train_size=0.5, embargo=30):
    """Expanding-window walk-forward; threshold re-calibrated inside each training window."""
    n = len(X)
    min_train = int(n * min_train_size)
    fold_size = (n - min_train) // n_splits
    rows, all_true, all_pred = [], [], []

    for k in range(n_splits):
        train_end = min_train + k * fold_size
        test_end = train_end + fold_size if k < n_splits - 1 else n
        y_te = y.iloc[train_end:test_end]
        if y_te.sum() < 2:
            rows.append({'Fold': k + 1, 'Train_size': train_end, 'N_positive': int(y_te.sum()),
                         'Threshold': np.nan, 'F1': np.nan})
            continue

        model, threshold, _ = fit_with_validation_threshold(
            model_class, model_params, X.iloc[:train_end], y.iloc[:train_end], embargo=embargo)
        pred = (model.predict_proba(X.iloc[train_end:test_end]) >= threshold).astype(int)
        all_true.extend(y_te.tolist())
        all_pred.extend(pred.tolist())
        rows.append({'Fold': k + 1, 'Train_size': train_end, 'N_positive': int(y_te.sum()),
                     'Threshold': threshold, 'F1': round(f1_score(y_te, pred, zero_division=0), 3)})

    return pd.DataFrame(rows), f1_score(all_true, all_pred, zero_division=0)


def paired_block_bootstrap_f1_diff(y_true, pred_a, pred_b, block=30, B=2000, ci=95, seed=42):
    """F1(a) - F1(b) with a moving-block bootstrap that uses the same resampled days for both.

    Returns the observed difference, the percentile interval and the share of resamples
    where the difference is <= 0.
    """
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    n = len(y_true)
    n_blocks = int(np.ceil(n / block))
    boot = np.zeros(B)
    for b in range(B):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)).ravel()[:n]
        boot[b] = (f1_score(y_true[idx], pred_a[idx], zero_division=0)
                   - f1_score(y_true[idx], pred_b[idx], zero_division=0))
    diff = f1_score(y_true, pred_a, zero_division=0) - f1_score(y_true, pred_b, zero_division=0)
    alpha = (100 - ci) / 2
    return diff, np.percentile(boot, alpha), np.percentile(boot, 100 - alpha), float((boot <= 0).mean())


def event_metrics(y_true, y_pred, crossing_positions, horizon=30, days_per_year=252):
    """Event-level view of the alerts.

    Lead time: days between the first alert in the `horizon` days before a crossover and the
    crossover itself (at most `horizon` by construction). A false-alarm day is an alert on a
    negative day; a false-alarm episode is a run of consecutive alert days that contains no
    positive day.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred).astype(int)
    leads = []
    for c in crossing_positions:
        start = max(0, c - horizon)
        hits = np.flatnonzero(y_pred[start:c])
        if len(hits):
            leads.append(c - (start + hits[0]))

    episodes = 0
    edges = np.diff(np.concatenate([[0], y_pred, [0]]))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if not y_true[s:e].any():
            episodes += 1

    years = len(y_true) / days_per_year
    fa_days = int(((y_pred == 1) & (y_true == 0)).sum())
    return {
        'Median_lead': float(np.median(leads)) if leads else np.nan,
        'Min_lead': int(min(leads)) if leads else np.nan,
        'FA_days': fa_days,
        'FA_days_per_year': fa_days / years,
        'FA_episodes': episodes,
        'FA_episodes_per_year': episodes / years,
    }
