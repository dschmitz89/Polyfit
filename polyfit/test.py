# -*- coding: utf-8 -*-

import numpy as np
from polyfit import load_example, PolynomRegressor, Constraints
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

X, y = load_example()
x_plot = np.linspace(0, np.amax(X) + 2, 200)

X = X.reshape((-1,1))

DEG = 3
VERBOSE = False

np_coeffs = np.polyfit(X.ravel(), y, DEG)
polyestimator = PolynomRegressor(deg=DEG)
vander = polyestimator.vander(x_plot)
pred_numpy = vander@np_coeffs[::-1]


polyestimator = PolynomRegressor(deg=DEG)
monotone_constraint = Constraints(monotonicity='inc')
polyestimator.fit(X, y, loss = 'l1', constraints={0: monotone_constraint})
pred_mon = polyestimator.predict(x_plot.reshape(-1, 1))

f, ax = plt.subplots(1, figsize = (9, 5))
ax.set_xlim(-0.01, 85)
ax.set_ylim(0.2, 1.03)
ax.set_title("Unconstrained polynom versus constrained polynom")
ax.scatter(X, y, c='k', s=8)

ax.plot(x_plot, pred_numpy, c='b', label='Degree=3 Unconstrained')
ax.plot(x_plot, pred_mon, c='r', label='Degree=3 Monotonic')

axins = zoomed_inset_axes(ax, 2, loc='lower right', borderpad=1.5)
axins.set_xlim(55, 84) # apply the x-limits
axins.set_ylim(0.93, 1.028)

axins.yaxis.set_visible(False)
axins.xaxis.set_visible(False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0")

axins.scatter(X, y, c='k', s=8)
axins.plot(x_plot, pred_numpy, c='b')
axins.plot(x_plot, pred_mon, c='r')

ax.legend(loc='upper left', frameon=False)
plt.subplots_adjust(top=0.92,
bottom=0.06,
left=0.05,
right=0.99)
plt.show()