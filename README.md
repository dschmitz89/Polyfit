# Polyreg
Constrained one dimensional polynomial regression in Python

Mostly developed for educational purposes, polyreg enables fitting scikit learn compatible one dimensional polynomial regression models under shape constraints.
Under the hood polynomial coefficients are optimized via cvxpy's excellent convex optimizers.

In particular, the polyreg estimator provides:
* Robust loss functions: L2, L1, Huber
* Monotony constraints
* Curvature constraints (convex/concave)
* limiting the fitted polynomial to an interval
* possibility for cross validation over the degree of the polynomial via sklearn's model selection utilities

The constraints however are only met in the interval of the training data.

Installation: 

```bash
git clone https://github.com/dschmitz89/Polyreg/
cd Polyreg
pip install .
```

Example usage can be found in the test.py file under /polyreg. It produces the following image. 

While unconstrained polynomial fitting results in a decreasing function for high x values, the constraints result in a visually more  convincing fit.

![Example fits](Example.png)
