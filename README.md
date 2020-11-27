# Polyfit
Scikit learn compatible constrained one dimensional polynomial regression in Python. 

Mostly developed for educational purposes, polyfit enables fitting scikit learn compatible polynomial regression models under shape constraints.
Under the hood polynomial coefficients are optimized via [cvxpy's](https://github.com/cvxgrp/cvxpy) excellent convex optimizers. 

## Why?
Often human intuition or prior knowledge gives us an idea that relationships between variables should be monotonic or follow certain asymptotic behaviour. In this example the monotonic fit is visually much more convincing than an unconstrained fit. 

![Example fits](Example_Montonic.png)

## Installation: 

```bash
git clone https://github.com/dschmitz89/Polyfit/
cd Polyfit
pip install .
```

## Usage
Example usage can be found in polyfit/test.py which produces the above image.

API:
```python
from polyfit import PolynomRegressor, Constraints
polyestimator = PolynomRegressor(deg=3)
monotone_constraint = Constraints(monotonicity='inc')
polyestimator.fit(X, y, loss = 'l2', constraints={0: monotone_constraint})
```