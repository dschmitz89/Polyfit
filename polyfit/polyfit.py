#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 12:25:16 2020

@author: tyrion
"""
import numpy as np
import cvxpy as cv
from sklearn.base import BaseEstimator
from os.path import dirname

PATH = dirname(__file__)

def load_example():
    
    npzfile = np.load(PATH + '/Example_Data.npz')
    X = npzfile['X']
    y = npzfile['y']
    
    return X, y

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
        
    def predict(self, x):
        
        if self.coeffs_ is not None:
            
            vander = self.vander(x)
    
            return np.dot(vander, self.coeffs_)
        
        else:
            
            return np.nan
    
    def fit(self, x, y, loss = 'l2', m = 1, yrange = None, \
            constraint_range = None, gridpoints = 50, fixed_point = None, verbose = False):
        
        vander = self.vander(x)
        column_norms_vander = self.column_norms(vander)
        vander = vander/column_norms_vander
        
        #vander_grad = self.vander_grad(x)
        #vander_grad =vander_grad/column_norms_vander
        
        #vander_hesse = self.vander_hesse(x)
        #vander_hesse = vander_hesse/column_norms_vander
        
        #set up variable for coefficients to be estimated
        
        if self.positive_coeffs:
            
            coeffs = cv.Variable(self.deg +1, pos = True)
        
        elif self.negative_coeffs:
            
            coeffs = cv.Variable(self.deg +1, neg = True)
            
        else:
            
            coeffs = cv.Variable(self.deg +1)
        
        #calculate residuals
        
        residuals = vander @ coeffs -y
        
        #define loss function
        
        if self.regularization == 'l1':
            
            regularization_term = cv.norm1(coeffs)
            
        elif self.regularization == 'l2':
            
            regularization_term = cv.pnorm(coeffs, 2, axis = 0)**2
        
        else:
            
            regularization_term = 0
            
        if loss == 'l2':
            
            data_term = cv.sum_squares(residuals)
                                    
        elif loss == 'l1':
            
            data_term = cv.norm1(residuals)
        
        elif loss == 'huber':
            
            data_term = cv.sum(cv.huber(residuals, m))
        
        objective = cv.Minimize(data_term + self.lam * regularization_term)
        
        #build constraints
        
        constraints = []
        
        if constraint_range is None:
            
            constraint_range = [np.amin(x), np.amax(x)]
            
        x_grid = np.linspace(constraint_range[0], constraint_range[1], num=gridpoints)
        
        vander_constraint = self.vander(x_grid)
        vander_constraint = vander_constraint/column_norms_vander
        
        vander_grad = self.vander_grad(x_grid)
        vander_grad =vander_grad/column_norms_vander
        
        vander_hesse = self.vander_hesse(x_grid)
        vander_hesse = vander_hesse/column_norms_vander
        
        if self.monotonocity == 'positive':

            constraints.append(vander_grad@coeffs >= 0)

        elif self.monotonocity == 'negative':

            constraints.append(vander_grad@coeffs <= 0)
        
        elif self.monotonocity is not None:
            
            raise ValueError("Monotonicity constraint should be " \
                             "'positive' or 'negative'")
                
        if self.curvature == 'convex':

            constraints.append(vander_hesse@coeffs >= 0)

        elif self.curvature == 'concave':

            constraints.append(vander_hesse@coeffs <= 0)
            
        elif self.curvature is not None:
            
            raise ValueError("Curvature constraint should be " \
                             "'convex' or 'concave'")
        
        if yrange is not None:
            
            constraints.append(vander_constraint @ coeffs <= yrange[1])
            constraints.append(vander_constraint @ coeffs >= yrange[0])
        
        if fixed_point is not None:
            
            vander_fix = self.vander(np.array([fixed_point[0]]))
            vander_fix = vander_fix/column_norms_vander
            
            constraints.append(vander_fix @ coeffs == fixed_point[1])
            
        problem = cv.Problem(objective, constraints = constraints)
        

            
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
            
            coefficients = coeffs.value/column_norms_vander

            self.coeffs_ = coefficients
        
        #if not try SCS optimization
        else:
            
            try:
                
                problem.solve(solver=cv.SCS, max_iters=100000, eps=1e-6, verbose = verbose)
            
            except cv.SolverError:
                
                pass
        
        if problem.status == 'optimal':
            
            coefficients = coeffs.value/column_norms_vander

            self.coeffs_ = coefficients
        
        else:
            
            print("CVXPY optimization failed")

        return self