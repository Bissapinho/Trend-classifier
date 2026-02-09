# Golden Cross Transition Prediction for SPY

## Overview

This quantitative finance research project aims to develop a machine learning system capable of predicting Golden Cross transitions on SPY (S&P 500 ETF) with **20-30 days advance notice**. The objective is to determine whether classical technical indicators can provide actionable predictive signals for market regime changes.

## Background and Motivation

Market regime transitions (bullish to bearish and vice versa) are rare but critical events for portfolio management. The Golden Cross (MA50 > MA200) is a widely used indicator but has a downside: it confirms a change that has already occurred. This project explores whether we can anticipate these transitions before they happen.

### Project Evolution

1. **Initial phase**: Return-based threshold regime classification - Abandoned (excessive label instability)
2. **Intermediate phase**: Golden Cross classification with contemporaneous indicators - Abandoned (no real predictive value)
3. **Current phase**: 30-day advance transition prediction - Under validation

## Methodology

### Target Definition

- **Target**: `transition_incoming` - binary (0/1)
- **Confirmation mechanism**: At least 7 positive predictions within a rolling 10-day window
- **Prediction horizon**: 30 days before actual transition
- **Class distribution**: approximately 12% positive, 88% negative

### Data

- **Primary asset**: SPY (2000-2024)
- **Planned validation assets**: QQQ, DIA, IWM, EFA
- **Number of identified transitions**: approximately 25 over 6,100 trading days
- **Source**: yfinance

### Features

Features **deliberately exclude** moving average-based indicators to prevent data leakage:

- Historical volatility
- Returns (multiple time horizons)
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Volume ROC (Rate of Change)
- Cumulative returns
- Potentially: Stochastic oscillators

### Validation

- Strict **temporal split** (no k-fold) to avoid look-ahead bias
- Robustness testing with Gaussian noise pertubation
- Multi-asset validation (if results are promising on SPY)
- Priority metric: **Precision** (false positives are costly)


## Preliminary Results

**To be completed after initial experimentation series**

## Known Limitations

1. Limited number of transitions (approximately 25) - High overfitting risk
2. Strong assumption: past patterns repeat
3. Exclusion of MA-based features - May miss important signals
4. Non-stationarity of financial markets

## Next Steps

1. Implementation of confirmation system (7/10) - In progress
2. Initial validation on SPY - In progress
3. Testing different algorithms (RF, XGBoost, etc.) - Pending
4. Important feature analysis - Pending
5. Multi-asset validation - Pending
6. arXiv paper preparation (if results are significant) - Pending

## Methodological Notes

This project explicitly accepts the possibility of a **negative result**. Demonstrating that classical technical indicators cannot predict Golden Cross transitions would be a valuable scientific contribution and would be rigorously documented for publication.

---

**Version**: 1.0  
**Last updated**: February 2026  
**Status**: Research in progess