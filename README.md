# Trend Classifier | Research-Oriented ML Project

This beginner-level machine learning project focuses on understanding and analyzing relationships between OHLC financial data and market trends, rather than attempting to produce an immediately exploitable trading signal.

The objective is not to build some sort of magical trading strategy or to produce profitability, but to explore how classical machine learning methods behave in a realistic financial time-series setting, and to highlight the inherent difficulties of trend prediction.

## Project Goals

- Import OHLC data from large, highly liquid, and dominant assets (e.g. major indices or stocks) that are representative of broader market conditions
- Perform basic feature engineering to study relationships between technical indicators and future market behavior
- Investigate how to define a **mathematically sound and interpretable target**, suitable for supervised learning in finance
- Assess the impact of **temporal leakage** and demonstrate why proper time-ordered evaluation is critical in financial ML
- Explore and compare different machine learning models to evaluate their ability to classify market trends under realistic constraints

## Philosophy

This project is closer to a **research and diagnostic exercise** than a trading system.
It aims to answer questions such as:
- Is there a stable predictive signal in classical technical indicators?
- How sensitive are results to evaluation methodology?
- What are the limits of common ML models in non-stationary financial data?

Negative or weak results are considered valid and informative outcomes, as they reflect the true complexity of financial markets.

## Disclaimer

As said before, this project is not intended for live trading, investment advice, or profit generation.
It is an educational exploration of machine learning applied to financial time series.
Any help or suggestions would be appreciated.
