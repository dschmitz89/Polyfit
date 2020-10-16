# -*- coding: utf-8 -*-

import numpy as np
from polyfit import load_example, PolynomRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

X, y = load_example()
x_plot = np.linspace(0, np.amax(X) + 2, 200)

DEG = 3
VERBOSE = False
datarange=[0, 1]

np_coeffs = np.polyfit(X, y, 2)
polyestimator = PolynomRegressor(deg=2)
vander = polyestimator.vander(x_plot)
pred_numpy = vander@np_coeffs[::-1]

#robust polynom fitting with L1 loss with crossvalidation to find best degree
degrees = range(1, 8)
polyestimator = PolynomRegressor()
poly_unconstrained = GridSearchCV(polyestimator,
                        param_grid={'deg': degrees},
                        scoring='neg_median_absolute_error')
poly_unconstrained.fit(X, y, groups=None, **{'loss':'l1'})
print("Best degree: ", poly_unconstrained.best_params_['deg'])
pred_unconstrained = poly_unconstrained.predict(x_plot)

#robust polynom fitting with huber loss under shape and prediction constraints
polyestimator = PolynomRegressor(deg=DEG, monotonocity='positive', curvature='concave')
polyestimator.fit(X, y, loss = 'huber', m = 0.05, yrange=datarange, verbose = VERBOSE)
pred_concave=polyestimator.predict(x_plot)

#robust polynom fitting with huber loss under monotonicity and prediction constraints
polyestimator = PolynomRegressor(deg=DEG, monotonocity='positive')
polyestimator.fit(X, y, loss = 'huber', m = 0.05, yrange=datarange, verbose = VERBOSE)
pred_mon=polyestimator.predict(x_plot)

f, ax = plt.subplots(1, figsize = (9, 5))
ax.set_xlim(-0.01, 85)
ax.set_ylim(0.2, 1.03)

ax.scatter(X, y, c='k', s=8)

ax.plot(x_plot, pred_numpy, c='g', label='Deg: 3 /Numpy Polyfit')
ax.plot(x_plot, pred_unconstrained, c='k', label='Crossvalidated Unconstrained L1 Loss')
ax.plot(x_plot, pred_mon, c='r', label='Deg: 3 /Monotonic/Bounded Huber Loss')
ax.plot(x_plot, pred_concave, c='b', label='Deg: 3 /Concave/Bounded Huber Loss')


axins = zoomed_inset_axes(ax, 2, loc='lower right', borderpad=1.5)
axins.set_xlim(55, 84) # apply the x-limits
axins.set_ylim(0.93, 1.028)

axins.yaxis.set_visible(False)
axins.xaxis.set_visible(False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0")

axins.scatter(X, y, c='k', s=8)
axins.plot(x_plot, pred_numpy, c='g')
axins.plot(x_plot, pred_unconstrained, c='k')
axins.plot(x_plot, pred_mon, c='r')
axins.plot(x_plot, pred_concave, c='b')

ax.legend(loc='upper left', frameon=False)
plt.subplots_adjust(top=0.99,
bottom=0.06,
left=0.05,
right=0.99)
plt.show()