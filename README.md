# High-Dimensional Macroeconomic Forecasting

This project compares dimensionality reduction and regularization methods for forecasting in high-dimensional macroeconomic and financial datasets.

The work was completed as a final project for a Machine Learning course (2025) and focuses on empirical forecasting, model comparison, and out-of-sample performance evaluation in a high-dimensional setting.

## Project Overview

The analysis includes:

- Data cleaning and stationarity transformations of macroeconomic time series
- Construction of a high-dimensional predictor dataset
- Implementation of dimensionality reduction methods (PCR, factor-based approaches)
- Estimation of regularized regression models (Ridge, LASSO, Elastic Net)
- Hyperparameter tuning via cross-validation
- Out-of-sample forecasting evaluation
- Model comparison using RMSE and MAE
- Diagnostic analysis of variable reduction techniques

## Repository Structure

Processed_Data/ Cleaned and stationary datasets

scripts/ R scripts for data processing, modeling, and diagnostics

output/ Forecast results, comparison tables, and plots

diagnostic_output/ Additional model diagnostics

Comparing_dimensionality_reduction_and_regularization_methods_for_forecasting_in_high_dimensional_macroeconomic_or_financial_data_.pdf Final report

## Data

The dataset consists of monthly macroeconomic and financial indicators, including:

- FRED-style macroeconomic variables
- Commodity prices (oil, natural gas)
- Energy and industrial production indicators

## How to Run

Run scripts in order:

1. `scripts/data_cleaning.R`
3. `scripts/methods_code.R`
4. `scripts/diagnostics_check_variable_reduction.R`

All outputs are automatically saved into `output/` and `diagnostic_output/`.

## Authors

Yuxin Zheng,

Gulnara Sadykova,

Yipeng Yan

MSc in Economics and Econometrics, University of Bologna
