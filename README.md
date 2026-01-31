# Explainable ML Analysis of NYSE Stocks with SHAP

This project focuses on binary classification of stock growth potential (Upside/Downside) for NYSE-listed companies, with a strong emphasis on model interpretability using **SHAP (SHapley Additive exPlanations)**.

## Quick Start
To explore the full analysis interactively, open the notebook in your browser:

[![Open in NBViewer](https://img.shields.io/badge/Open%20in-NBViewer-orange?logo=jupyter&logoColor=orange)](https://nbviewer.org/github/VeronikaKolimova/shap-stock-analysis/blob/main/L_03.ipynb)


## Project Overview
- **Goal**: Predict whether a stock has Upside (>0% growth forecast) or Downside (<0% growth forecast) based on fundamental financial indicators.
- **Focus**: Not just prediction accuracy, but understanding *why* models make their decisions—using SHAP to explain feature contributions globally and locally.
- **Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Evaluation**: ROC‑AUC, F1‑score (handling class imbalance)
- **Key Analysis**: SHAP summary plots, dependence plots, and permutation importance to reveal drivers of analyst forecasts.

## Dataset
NYSE stock data with features such as:
- P/E Ratio, PEG Ratio, Beta (5y/1y)
- Quick Ratio, Current Ratio
- Debt to Equity, Debt to Assets, Net Debt to Capital

## Tech Stack
- Python, pandas, numpy
- scikit‑learn (Random Forest, Gradient Boosting, Logistic Regression)
- SHAP (Shapley values for model explanation)
- matplotlib, seaborn (visualizations)
- TensorFlow (simple baseline model)

## Key Insights
- Identified the most influential financial indicators for Upside/Downside predictions.
- Provided transparent, interpretable explanations for each prediction using SHAP.
- Handled class imbalance (≈82% Upside) with appropriate metrics and sampling.

This repository is ideal for learning how to apply explainable AI (XAI) techniques to real‑world financial data.
