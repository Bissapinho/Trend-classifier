"""Result figures from results/*.csv (run scripts/run_main.py first).

Usage: python scripts/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RES = PROJECT_ROOT / 'results'
FIG = PROJECT_ROOT / 'figures'
COLORS = {'XGB': '#2166ac', 'RFC': '#d6604d', 'Extrapolation': '#4d9221'}
LABELS = {'XGB': 'XGBoost', 'RFC': 'Random Forest', 'Extrapolation': 'Extrapolation rule'}


def pr_curve(pred, info):
    spy = pred[pred['Asset'] == 'SPY']
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in ['XGB', 'RFC']:
        p, r, _ = precision_recall_curve(spy['y'], spy[f'proba_{name}'])
        ap = info['average_precision']['SPY'][name]
        ax.plot(r, p, color=COLORS[name], lw=1.5, label=f"{LABELS[name]} (AP={ap:.3f})")
        op = (spy[f'proba_{name}'] >= info['thresholds'][name]).astype(int)
        ax.scatter(recall_score(spy['y'], op), precision_score(spy['y'], op), color=COLORS[name], marker='D', zorder=5)
    e = spy['pred_Extrapolation']
    ax.scatter(recall_score(spy['y'], e), precision_score(spy['y'], e), color=COLORS['Extrapolation'],
               marker='*', s=120, zorder=5, label=LABELS['Extrapolation'])
    ax.axhline(spy['y'].mean(), color='gray', ls='--', lw=1, label=f"Prevalence ({spy['y'].mean():.2f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'pr_curve_spy.pdf'); fig.savefig(FIG / 'pr_curve_spy.png', dpi=110)
    plt.close(fig)


def f1_ci(res):
    assets = ['SPY', 'DIA', 'QQQ', 'GLD', 'TLT']
    models = ['XGB', 'RFC', 'Extrapolation']
    x = np.arange(len(assets))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, m in enumerate(models):
        d = res[res['Model'] == m].set_index('Asset').loc[assets]
        err = [d['F1'] - d['CI_block_low'], d['CI_block_high'] - d['F1']]
        ax.bar(x + (i - 1) * w, d['F1'], w, color=COLORS[m], yerr=err, capsize=3, label=LABELS[m])
    strat = res[res['Model'] == 'Stratified (expected)'].set_index('Asset').loc[assets, 'F1']
    ax.scatter(x, strat, color='black', marker='_', s=300, zorder=5, label='Stratified random')
    ax.set_xticks(x)
    ax.set_xticklabels(assets)
    ax.set_ylabel('F1 (positive class)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, ncol=4, loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG / 'f1_block_ci.pdf'); fig.savefig(FIG / 'f1_block_ci.png', dpi=110)
    plt.close(fig)


def walk_forward(wf, info):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, marker in [('XGB', 'o'), ('RFC', 's')]:
        d = wf[(wf['Model'] == name) & wf['F1'].notna()]
        g = info['walk_forward_pooled_f1'][name]
        ax.plot(d['Fold'], d['F1'], marker=marker, color=COLORS[name], label=f"{LABELS[name]} (pooled {g:.3f})")
        ax.axhline(g, color=COLORS[name], ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1)
    ax.set_xticks(sorted(wf.loc[wf['F1'].notna(), 'Fold'].unique()))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'walk_forward.pdf'); fig.savefig(FIG / 'walk_forward.png', dpi=110)
    plt.close(fig)


def protocol_effect(res, naive):
    us = ['SPY', 'DIA', 'QQQ']
    steps = ['Threshold on test,\nno embargo', 'Threshold on validation,\n30-day embargo']
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name in ['XGB', 'RFC']:
        a = naive[(naive['Model'] == name) & naive['Asset'].isin(us)]['F1'].mean()
        b = res[(res['Model'] == name) & res['Asset'].isin(us)]['F1'].mean()
        ax.plot([0, 1], [a, b], marker='o', color=COLORS[name], label=LABELS[name])
    e = res[(res['Model'] == 'Extrapolation') & res['Asset'].isin(us)]['F1'].mean()
    ax.axhline(e, color=COLORS['Extrapolation'], ls='--', label=LABELS['Extrapolation'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(steps, fontsize=8)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel('Mean F1, SPY/DIA/QQQ')
    ax.set_ylim(0.4, 0.9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'protocol_effect.pdf'); fig.savefig(FIG / 'protocol_effect.png', dpi=110)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    res = pd.read_csv(RES / 'test_results.csv')
    pred = pd.read_csv(RES / 'test_predictions.csv')
    wf = pd.read_csv(RES / 'walk_forward.csv')
    naive = pd.read_csv(RES / 'naive_protocol.csv')
    info = json.loads((RES / 'run_info.json').read_text())
    pr_curve(pred, info)
    f1_ci(res)
    walk_forward(wf, info)
    protocol_effect(res, naive)


if __name__ == '__main__':
    main()
