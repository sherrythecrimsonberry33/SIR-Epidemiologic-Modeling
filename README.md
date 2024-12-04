# COVID-19 Epidemiological Modeling: Enhanced SIR Model with ML Predictions

COVID-19 epidemiological modeling tool combining SIR model with ML predictions. Features Runge-Kutta numerical solutions and gradient boosting forecasting.

## Table of Contents
1. [Introduction](#introduction)
2. [SIR Model Overview](#sir-model-overview)
3. [Mathematical Framework](#mathematical-framework)
4. [Numerical Methods](#numerical-methods)
5. [Machine Learning Enhancement](#machine-learning-enhancement)
6. [Implementation Details](#implementation-details)
7. [References](#references)

## Introduction

This repository contains an implementation of the SIR (Susceptible-Infected-Recovered) epidemiological model for COVID-19 spread analysis. The project features two main components:
- A classical SIR model using Runge-Kutta numerical approximation
- An enhanced version incorporating machine learning for improved predictions

## SIR Model Overview

The SIR model divides the population into three compartments:
- **S**: Susceptible population
- **I**: Infected population
- **R**: Recovered/Removed population

The model assumes:
- Fixed population size: $N = S + I + R$
- Homogeneous mixing of population
- No births, deaths, or immigration
- Recovery confers permanent immunity

## Mathematical Framework

### Basic SIR Equations

The system of differential equations governing the SIR model:

$$\frac{dS}{dt} = -\beta SI/N$$
$$\frac{dI}{dt} = \beta SI/N - \gamma I$$
$$\frac{dR}{dt} = \gamma I$$

Where:
- $\beta$: Infection rate
- $\gamma$: Recovery rate
- $R_0 = \beta/\gamma$: Basic reproduction number

### Reproduction Number

The basic reproduction number $R_0$ is calculated as:

$$R_0 = \frac{R_1 + R_2}{2}$$

where $R_1$ and $R_2$ are empirically derived reproduction numbers for each region.

## Numerical Methods

### Fourth-Order Runge-Kutta Method

The implementation uses the RK4 method for numerical integration:

$$k_1 = hf(t_n, y_n)$$
$$k_2 = hf(t_n + \frac{h}{2}, y_n + \frac{k_1}{2})$$
$$k_3 = hf(t_n + \frac{h}{2}, y_n + \frac{k_2}{2})$$
$$k_4 = hf(t_n + h, y_n + k_3)$$
$$y_{n+1} = y_n + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

where:
- $h$: step size
- $f$: the SIR system of equations
- $y_n$: the state vector $[S_n, I_n, R_n]$

## Machine Learning Enhancement

The ML-enhanced version incorporates:

### Feature Engineering
- Historical windows of S, I, R values
- Rate of change (first derivatives)
- Acceleration (second derivatives)
- Moving averages
- Population-normalized metrics

### Model Architecture
- Gradient Boosting Regressor
- Multi-output prediction
- Savitzky-Golay filtering for smoothing
- Uncertainty estimation

## Implementation Details

### Basic Model
```python
def SIR_model(u, t):
    gamma = 1/18
    beta = R0 * gamma
    S, I, R = u[0], u[1], u[2]
    dS = -beta*S*I/N
    dI = beta*S*I/N - gamma*I
    dR = gamma*I
    return [dS, dI, dR]
```

### ML Enhancement Features
- Real-time prediction updates
- Confidence intervals
- Automated parameter tuning
- Data validation and preprocessing

## References

### Epidemiological Modeling
1. Side, S., et al. (2018). "Numerical solution of SIR model for transmission of tuberculosis by Runge-Kutta method." Journal of Physics: Conference Series, 1040, 012021.

2. Bakar, S.S., & Razali, N. (2020). "Solving SEIR Model Using Symmetrized Runge Kutta Methods." International Islamic University Malaysia.

### COVID-19 Data Sources
1. Our World in Data. (2021). "Coronavirus (COVID-19) Cases." Retrieved from https://ourworldindata.org/coronavirus

2. World Health Organization. (2021). "WHO Coronavirus (COVID-19) Dashboard." Retrieved from https://data.who.int/dashboards/covid19/cases

### Reproduction Number Analysis
1. Centre for Evidence-Based Medicine. (2020). "When will it be over? An introduction to viral reproduction numbers, R0 and Re." Retrieved from https://www.cebm.net/covid-19/when-will-it-be-over-an-introduction-to-viral-reproduction-numbers-r0-and-re/
