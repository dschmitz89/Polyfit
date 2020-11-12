#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:25:16 2020

@author: tyrion
"""
import numpy as np
import cvxpy as cv
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from os.path import dirname

PATH = dirname(__file__)

def load_example():
    
    npzfile = np.load(PATH + '/Example_Data.npz')
    X = npzfile['X']
    y = npzfile['y']
    
    return X, y

class Constraints:

    def __init__(self, monotonicity = None, curvature = None, sign = None, \
        constraint_range = None, gridpoints = 20):

        self.monotonicity = monotonicity
        self.curvature = curvature
        self.sign = sign
        self.constraint_range = constraint_range
        self.gridpoints = gridpoints
class PolynomRegressor(BaseEstimator):
    
    def __init__(self, deg=None, monotonocity = None, curvature = None, \
                 positive_coeffs = False, negative_coeffs = False, \
                     regularization = None, lam = 0):
        
        self.deg = deg
        self.monotonocity = monotonocity
        self.curvature = curvature
        self.coeffs_ = None
        self.positive_coeffs = positive_coeffs
        self.negative_coeffs = negative_coeffs
        self.regularization = regularization
        self.lam = lam
    
    def column_norms(self, V):
        
        norms = np.sqrt(np.square(V).sum(0))
        
        norms[norms == 0] = 1
        
        return norms
    
    def vander(self, x):
        
        x = x.astype(np.float64)
        return np.fliplr(np.vander(x, N = self.deg +1))

    def vander_grad(self, x):
        
        vander = self.vander(x)
        
        red_vander = vander[:, :-1]
        
        factors = np.arange(1, self.deg+1)
        
        grad_matrix = np.zeros(shape=vander.shape)
        inner_matrix = red_vander * factors[None, :]
        grad_matrix[:, 1:] = inner_matrix
        
        return grad_matrix

    def vander_hesse(self, x):
        
        grad = self.vander_grad(x)
        
        red_grad = grad[:, :-1]
        
        factors = np.arange(1, self.deg+1)
        
        hesse_matrix = np.zeros(shape=grad.shape)
        inner = red_grad * factors[None, :]
        hesse_matrix[:, 2:] = inner[:, 1:]
        
        return hesse_matrix
        
    def predict(self, x, interactions = False):
        
        if self.coeffs_ is not None:
            
            designmatrix = self.build_designmatrix(x, interactions = interactions)
            #print("designmatrix: ", designmatrix.shape)
            return np.dot(designmatrix, self.coeffs_)
        
        else:
            
            return np.nan
    
    def build_designmatrix(self, x, interactions = False):

        n_samples, n_features = x.shape

        designmatrix = self.vander(x[:, 0])

        #loop over features and append Vandermonde matrix for each features without constant column
        for i in range(1, n_features):

            van = self.vander(x[:, i])
            #print("van shape: ", van.shape)

            designmatrix = np.hstack((designmatrix, van[:, 1:]))

        if interactions == True:

            poly = PolynomialFeatures(self.deg, interaction_only=True)
            interactions_matrix = poly.fit_transform(x)
            #print("interaction matrix: ", interactions_matrix.shape)
            interactions_matrix = interactions_matrix[:, 1 + n_features: ]
            #print("interaction matrix afetr cut: ", interactions_matrix.shape)
            designmatrix = np.hstack((designmatrix, interactions_matrix))

        return designmatrix

    def fit(self, x, y, loss = 'l2',  interactions = False, m = 1, yrange = None, constraints = None, \
            verbose = False):
        
        n_samples, n_features = x.shape
        n_coeffs = n_features * self.deg +1
        #print("number of coefficients: ", n_coeffs)
        designmatrix = self.build_designmatrix(x, interactions = interactions)
        n_coeffs = designmatrix.shape[1]
        print("design shape: ", designmatrix.shape)
        column_norms_designmatrix = self.column_norms(designmatrix)
        designmatrix = designmatrix/column_norms_designmatrix

        coeffs = cv.Variable(n_coeffs)
        
        #calculate residuals
        residuals = designmatrix @ coeffs -y
        
        #define loss function

        loss_options = {
            'l2': cv.sum_squares(residuals),
            'l1': cv.norm1(residuals),
            'huber': cv.sum(cv.huber(residuals, m))
        }
        
        data_term = loss_options[loss]

        if self.regularization is not None:

            regularization_options = {
                'l2': cv.pnorm(coeffs, 2, axis = 0)**2,
                'l1': cv.norm1(coeffs)
            }

            regularization_term = regularization_options[self.regularization]

            objective = cv.Minimize(data_term + self.lam * regularization_term)

        else:

            objective = cv.Minimize(data_term)

        #build constraints
        
        constraint_list = []

        ybound = yrange is not None
        
        if constraints is not None:

            #loop over all features

            for feature_index in constraints:

                Feature_constraints = constraints[feature_index]

                xvals_feature = x[:, feature_index]
                coefficient_index = feature_index * self.deg + 1
                feature_coefficients = coeffs[coefficient_index:coefficient_index + self.deg]

                if Feature_constraints.sign == 'positive':

                    constraint_list.append(feature_coefficients >= 0)

                elif Feature_constraints.sign == 'negative':

                    constraint_list.append(feature_coefficients <= 0)

                monotonic = Feature_constraints.monotonicity is not None
                strict_curvature = Feature_constraints.curvature is not None
                

                if monotonic or strict_curvature or ybound:

                    if Feature_constraints.constraint_range is None:

                        constraint_min = np.amin(xvals_feature)
                        constraint_max = np.amax(xvals_feature)
                        Feature_constraints.constraint_range = [constraint_min, constraint_max]

                    constraints_grid = np.linspace(Feature_constraints.constraint_range[0], \
                        Feature_constraints.constraint_range[1], num=Feature_constraints.gridpoints)

                if monotonic:

                    vander_grad = self.vander_grad(constraints_grid)[:, 1:]
                    norms = column_norms_designmatrix[coefficient_index:coefficient_index + self.deg]
                    vander_grad = vander_grad/norms

                    if Feature_constraints.monotonicity == 'inc':

                        constraint_list.append(vander_grad @ feature_coefficients >= 0)

                    elif Feature_constraints.monotonicity == 'dec':

                        constraint_list.append(vander_grad @ feature_coefficients <= 0)

                if strict_curvature:

                    vander_hesse = self.vander_hesse(constraints_grid)[:, 1:]
                    norms = column_norms_designmatrix[coefficient_index:coefficient_index + self.deg]
                    vander_hesse = vander_hesse/norms

                    if Feature_constraints.curvature == 'convex':

                        constraint_list.append(vander_hesse @ feature_coefficients >= 0)

                    elif Feature_constraints.curvature == 'concave':

                        constraint_list.append(vander_hesse @ feature_coefficients <= 0)                   

        '''
        if ybound:

            vander_constraints = self.vander(constraints_grid)
            vander_constraints = vander_constraints/column_norms_designmatrix

            constraint_list.append(vander_constraints @ coeffs >= yrange[0])
            constraint_list.append(vander_constraints @ coeffs <= yrange[1])     
        
        if yrange is not None:
            
            constraints.append(vander_constraint @ coeffs <= yrange[1])
            constraints.append(vander_constraint @ coeffs >= yrange[0])
        
        if fixed_point is not None:
            
            vander_fix = self.vander(np.array([fixed_point[0]]))
            vander_fix = vander_fix/column_norms_vander
            
            constraints.append(vander_fix @ coeffs == fixed_point[1])
        '''    
        problem = cv.Problem(objective, constraints = constraint_list)
        

            
        try:    
            
            if loss == 'l1':
            #l1 loss solved by ECOS. Lower its tolerances for convergence    
                problem.solve(abstol=1e-9, reltol=1e-9, max_iters=1000000, \
                              feastol=1e-9, abstol_inacc = 1e-7, \
                                  reltol_inacc=1e-7, verbose = verbose)            
                
            else:
                    
                #l2 and huber losses solved by OSQP. Lower its tolerances for convergence
                problem.solve(eps_abs=1e-10, eps_rel=1e-10, max_iter=10000000, \
                              eps_prim_inf = 1e-10, eps_dual_inf = 1e-10, verbose = verbose) 
                    
        #in case OSQP or ECOS fail, use SCS
        except cv.SolverError:
            
            try:
            
                problem.solve(solver=cv.SCS, max_iters=100000, eps=1e-4, verbose = verbose)
            
            except cv.SolverError:
                    
                print("cvxpy optimization failed!")
        
        #if optimal solution found, set parameters
        
        if problem.status == 'optimal':
            
            coefficients = coeffs.value/column_norms_designmatrix

            self.coeffs_ = coefficients
        
        #if not try SCS optimization
        else:
            
            try:
                
                problem.solve(solver=cv.SCS, max_iters=100000, eps=1e-6, verbose = verbose)
            
            except cv.SolverError:
                
                pass
        
        if problem.status == 'optimal':
            
            coefficients = coeffs.value/column_norms_designmatrix

            self.coeffs_ = coefficients
        
        else:
            
            print("CVXPY optimization failed")

        return self

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

coeffs = np.array([1,-5,1, 1, 1])

N = 1000
x_points = np.linspace(0,4, num = N)
y_points = np.linspace(0,5, num = N)
X_sparse = np.column_stack((x_points, y_points))
#print("X shape: ", X_sparse.shape)
poly = PolynomRegressor(deg = 2)
#D = poly.build_designmatrix(X_sparse)
#print(D.shape)

poly.coeffs_ = coeffs

z_true = poly.predict(X_sparse)
z_noisy = np.random.normal(z_true, 1)
#print(z_true)
#print("data: ", z_noisy)

poly_new = PolynomRegressor(deg = 2)#, regularization='l1', lam = 1e-1)
cons = {0: Constraints(sign='positive'), 1: Constraints(monotonicity='inc')}#, curvature='concave'
poly_new.fit(X_sparse, z_noisy, loss='l2', interactions = True)#, constraints=cons)
pred = poly_new.predict(X_sparse, interactions = True)
D = poly_new.build_designmatrix(X_sparse, interactions = True)
#print("X: ", X_sparse)
#poly = PolynomialFeatures(2, interaction_only = True)
#print("polyfeatures: ", poly.fit_transform(X_sparse))
#print("D: ", D)
#print("D shape: ", D.shape)
#print("pred: ", pred)
est_coeffs = poly_new.coeffs_
print("est. coeeffs: ", est_coeffs)


XX, YY = np.meshgrid(x_points, y_points)

ZZ = np.full_like(XX, est_coeffs[0]) + XX * est_coeffs[1] + XX * XX * est_coeffs[2] +\
    YY * est_coeffs[3] + YY * YY * est_coeffs[4] + XX * YY * est_coeffs[5]
    

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(XX, YY, ZZ, \
                       linewidth=0, antialiased=False)#, cmap=cm.coolwarm

ax.scatter(x_points, x_points, z_noisy, c = 'b', marker='o', zorder = 0)

plt.show()