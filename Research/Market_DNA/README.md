# Institutional Quantitative Research Platform - Market DNA Engine

## Overview
The **Market DNA Engine** is an institutional-grade quantitative finance framework designed to parse raw intraday (1-minute) market data and distill it into high-value behavioral features, market archetypes, statistical distributions, and automated research reports.

## Features
- **Automated Data Processing**: Cleans and normalizes intraday OHLCV datasets.
- **DNA Feature Extraction**: Computes gaps, Opening Ranges (ORB 5/10/15m), VWAP distance, MFE/MAE, ATR, ADR, and Relative Volume.
- **Institutional Market Classification**: Categorizes trading days into archetypes such as *Inside Day, Outside Day, Volatile, Gap Trend, Gap Fade, Strong Trend Up/Down,* and *Range*.
- **Multi-Format Exports**: Automatically generates CSV, Excel, HTML reports, textual summaries, and structured institutional research briefs.
- **Visual Analytics**: Compiles publication-ready distribution histograms, frequency charts, and correlation matrices.

## Project Structure
```text
Market_DNA/
│
├── config.py             # Global configurations, paths, and calculation parameters
├── loader.py             # Dataset loader and schema validator
├── feature_engine.py     # Core quantitative feature extraction engine
├── classifier.py         # Institutional market day archetype classifier
├── charts.py             # Matplotlib/Seaborn institutional visualization generator
├── report.py             # Multi-format report and summary exporter
├── utils.py              # Centralized logging framework
├── run.py                # Main execution script
└── requirements.txt      # Project dependencies