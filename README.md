# Computational Finance - Lab Assignment 1

This repository contains the implementation for **Lab Assignment 1** of the *Computational Finance* course at the University of Amsterdam by **Kushnava Singha** and **Yoad van Praag**.

## Overview

This lab explores three key topics in computational finance:

1. **Financial Data Science**  
   Estimation of historical and implied volatility, with comparison between various volatility estimators and the CBOE-quoted VIX.

2. **Option Pricing**  
   Analytical pricing of power options using the Black–Scholes PDE and construction of replicating portfolios.

3. **Hedging Simulation**  
   Simulation of delta-hedging under both matched and mismatched volatility scenarios using the Euler method.

## File Structure

To obtain the results for each question:

- **Question 1.1** can be answered by running `1.1.ipynb`.
- **Question 1.3** is addressed in `1.3.ipynb`.

The core functionalities and helper methods used across the notebooks are implemented in the following scripts:

- `utils.py` – General utility functions for data handling and preprocessing.
- `hedging.py` – Functions for delta-hedging simulations under matched volatility.
- `mistmatched.py` – Functions for handling volatility mismatch scenarios in hedging.
