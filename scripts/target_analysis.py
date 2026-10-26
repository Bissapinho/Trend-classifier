"""Target study on SPY: return-based target vs MA50/MA200 regime, noise robustness.

Writes results/target_analysis.json, results/noise_curves.csv and figures/target_*.pdf.

Usage: python scripts/target_analysis.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src.target import regime, return_target, noisy_regime, compute_transition

OUT = PROJECT_ROOT / 'results'
FIG = PROJECT_ROOT / 'figures'
PERIODS = [15, 30, 60, 90]
GOALS = [0.03, 0.05, 0.07, 0.10]
SIGMAS = np.linspace(0.01, 0.5, 50)
REPS = 50


def n_switches(s: pd.Series):
    s = s.dropna()
    return int((s.diff().abs() > 0).sum())


def plot_regimes(dates, close, label, title, path):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(dates, close, color='black', lw=0.8)
    ax.fill_between(dates, 0, 1, where=label.values == 1, transform=ax.get_xaxis_transform(),
                    color='tab:green', alpha=0.25, lw=0, label='1')
    ax.fill_between(dates, 0, 1, where=label.values == 0, transform=ax.get_xaxis_transform(),
                    color='tab:red', alpha=0.15, lw=0, label='0')
    ax.set_title(title)
    ax.set_ylabel('SPY close ($)')
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path); fig.savefig(path.with_suffix('.png'), dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    raw = load_data('SPY')
    dates = pd.to_datetime(raw['Date']).dt.tz_localize(None)
    close = raw['Close']

    grid = []
    for p in PERIODS:
        for g in GOALS:
            lab = return_target(close, p, g)
            valid = close.shift(-p).notna()
            grid.append({'period': p, 'goalreturn': g, 'pos_rate': round(lab[valid].mean(), 3),
                         'switches': n_switches(lab[valid])})
    df_grid = pd.DataFrame(grid)
    print(df_grid.to_string(index=False))

    ret = return_target(close, 60, 0.05)
    valid = close.shift(-60).notna()
    year = dates.dt.year
    ret_2013_17 = ret[valid & year.between(2013, 2017)].mean()

    reg = regime(close)
    ma_valid = close.rolling(200).mean().notna()
    _, crossings = compute_transition(raw)
    reg_2013_17 = reg[ma_valid & year.between(2013, 2017)].mean()

    rng = np.random.default_rng(42)
    p_clean = reg[ma_valid].mean()
    bias, flip = [], []
    for s in SIGMAS:
        b, f = [], []
        for _ in range(REPS):
            r = noisy_regime(close, s, rng)
            b.append(abs(r[ma_valid].mean() - p_clean))
            f.append((r[ma_valid] != reg[ma_valid]).mean())
        bias.append(np.mean(b))
        flip.append(np.mean(f))
    curves = pd.DataFrame({'sigma': SIGMAS, 'target_bias': bias, 'flip_rate': flip})
    curves.to_csv(OUT / 'noise_curves.csv', index=False)

    summary = {
        'return_target_grid': grid,
        'return_target_60_005': {'switches': n_switches(ret[valid]), 'pos_rate': round(ret[valid].mean(), 3),
                                 'pos_rate_2013_2017': round(ret_2013_17, 3)},
        'ma_regime': {'crossings': len(crossings), 'bullish_share': round(p_clean, 3),
                      'bullish_share_2013_2017': round(reg_2013_17, 3)},
        'noise': {'max_target_bias': round(max(bias), 3),
                  'flip_rate_min': round(min(flip), 3), 'flip_rate_max': round(max(flip), 3),
                  'sigmas': [float(SIGMAS[0]), float(SIGMAS[-1])], 'reps': REPS,
                  'noise_scale': 'sigma * std(Close) added to prices'},
    }
    (OUT / 'target_analysis.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'return_target_grid'}, indent=2))

    plot_regimes(dates[valid], close[valid], ret[valid],
                 'Return-based target (60 days, +5%)', FIG / 'target_return.pdf')
    plot_regimes(dates[ma_valid], close[ma_valid], reg[ma_valid],
                 'MA50 > MA200 regime', FIG / 'target_gc.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(SIGMAS, bias, marker='o', ms=3)
    axes[0].set_xlabel(r'$\sigma$')
    axes[0].set_ylabel(r'$|\Delta P|$')
    axes[0].set_title('Change in bullish share')
    axes[1].plot(SIGMAS, flip, marker='o', ms=3, color='tab:orange')
    axes[1].set_xlabel(r'$\sigma$')
    axes[1].set_ylabel('Flip rate')
    axes[1].set_title('Share of labels changed')
    fig.tight_layout()
    fig.savefig(FIG / 'target_noise.pdf'); fig.savefig(FIG / 'target_noise.png', dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
