# Polyfit
![alt text](https://github.com/dschmitz89/Polyfit/blob/master/Icon_new_crop.jpg "")

Scikit learn compatible constrained and robust polynomial regression in Python. 

Mostly developed for educational purposes, polyfit enables fitting scikit learn compatible polynomial regression models under shape constraints. Under the hood polynomial coefficients are optimized via [cvxpy's](https://github.com/cvxgrp/cvxpy) excellent convex optimizers. 

## Why?
Often human intuition or prior knowledge gives us an idea that relationships between variables should be monotonic or follow certain asymptotic behaviour. In this example the monotonic fit is visually much more convincing than an unconstrained fit. 

![Example fits](https://github.com/dschmitz89/Polyfit/blob/master/Example_Monotonic.png)

## Installation: 

```bash
pip install polyfit
```

## Documentation

Check the [online documentation](https://polyfit.readthedocs.io/) for an example and API reference.

## Example

Simple example to fit a polynomial of degree 3 which is monotonically increasing for the first feature:
```python
from polyfit import PolynomRegressor, Constraints
polyestimator = PolynomRegressor(deg=3, regularization = None, lam = 0)
monotone_constraint = Constraints(monotonicity='inc')
polyestimator.fit(X, y, loss = 'l2', constraints={0: monotone_constraint})
```

## Method
The constraints are enforced by imposing inequality constraints upon the polynomial coefficients. For example, if the resulting one dimensional polynomial is required to be monotonically increasing, its first derivative must be greater than 0. Enforcing this for an interval is not possible but enforcing it for a reasonable number of points within an interval (default: 20) is usually enough to guarantee the monotonicity for this interval. Given the predictor vector x, target vector y and the Vandermonde matrix V the polynomial coefficients p are then estimated by the following optimization problem:

![equation](https://latex.codecogs.com/png.latex?\underset{p}{\mathrm{argmin}}||V(x)p-y||^2=0&space;\\&space;\text{s.&space;t.&space;}\left|\frac{\partial&space;V(x)}{\partial&space;x}p\right|_{x_i}\geq&space;0\&space;\forall&space;x_i)

**Warning: by default, the polynomial is only monotonic or convex/concave for the interval of the input data!**
