import numpy as np
import cvxpy as cv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures
from os.path import dirname

PATH = dirname(__file__)

def load_example():
    r'''
    Loads example data

    Arguments
    ----------

    Returns
    -------------
    X : ndarray (m, n)
        Predictor data
    y : ndarray (n, )
        Target data
    '''

    npzfile = np.load(PATH + '/Example_Data.npz')
    X = npzfile['X']
    y = npzfile['y']
    
    return X, y

class Constraints:
    '''
    Constraints class

    .. note::
        Shape constraints potentially make the model fit numerically unstable. Use at your own risk!
    
    Parameters
    -------------
    monotonicity : string, optional
        Monotonicty of the model. Should be  'inc' or 'dec'
    curvature : string, optional
        Curvature of the model . Should be  'convex' or 'concave'.
    sign : string, optional
        Sign of the polynomial coefficients . Should be  'positive' or 'negative'.
    constraint_range : list, optional
        Range over which the constraints should be enforced. Must be of the form ``[lb, ub]`` with lower bounds ``lb`` and upper bounds ``ub``
    gridpoints : int, optional, default 20
        Number of grid points on which the constraints are imposed for the optimization problem

    '''
    
    def __init__(self, monotonicity = None, curvature = None, sign = None, \
        constraint_range = None, gridpoints = 20):

        self.monotonicity = monotonicity
        self.curvature = curvature
        self.sign = sign
        self.constraint_range = constraint_range
        self.gridpoints = gridpoints

class PolynomRegressor(BaseEstimator, RegressorMixin):
    '''
    Polynomregressor class

    Fits a multivariate polynomial model to arbitrary numerical data.

    Parameters
    -------------
    deg : int 
        Degree of the polynomial
    regularization : string, optional
        Regularization to be used. Should be 'l1' or 'l2'.
    lam : float, optional, default 0
        Regularization coefficient
    interactions : bool, optional, default False
        If ``True``, also uses interaction terms of all predictor variables

    Attributes
    -------------
    coeffs_ : ndarray (n, )
        Estimated polynomial coefficients
    '''

    def __init__(self, 
                deg=None,
                regularization = None,
                lam = 0,
                interactions = False):
        
        self.deg = deg
        self.coeffs_ = None
        self.regularization = regularization
        self.lam = lam
        self.interactions = interactions
    
    def _column_norms(self, V):
        
        norms = np.sqrt(np.square(V).sum(0))
        
        norms[norms == 0] = 1
        
        return norms
    
    def _vander(self, x):
        
        x = x.astype(np.float64)
        return np.fliplr(np.vander(x, N = self.deg +1))

    def _vander_grad(self, x):
        
        vander = self._vander(x)
        
        red_vander = vander[:, :-1]
        
        factors = np.arange(1, self.deg+1)
        
        grad_matrix = np.zeros(shape=vander.shape)
        inner_matrix = red_vander * factors[None, :]
        grad_matrix[:, 1:] = inner_matrix
        
        return grad_matrix

    def _vander_hesse(self, x):
        
        grad = self._vander_grad(x)
        
        red_grad = grad[:, :-1]
        
        factors = np.arange(1, self.deg+1)
        
        hesse_matrix = np.zeros(shape=grad.shape)
        inner = red_grad * factors[None, :]
        hesse_matrix[:, 2:] = inner[:, 1:]
        
        return hesse_matrix
        
    def predict(self, x):
        r'''
        Predict the polynomial model for data x

        Parameters
        -------------
        x : ndarray (m, n) 
            Test data for which predictions should be made

        Returns
        ----------
        y : ndarray (n, )
            Model prediction
        '''

        if self.coeffs_ is not None:
            
            designmatrix = self._build_designmatrix(x)
            return np.dot(designmatrix, self.coeffs_)
        
        else:
            
            raise ValueError("Estimator needs to be trained first!")
    
    def _build_designmatrix(self, x):

        n_samples, n_features = x.shape

        designmatrix = self._vander(x[:, 0])

        #loop over features and append Vandermonde matrix for each features without constant column

        for i in range(1, n_features):

            van = self._vander(x[:, i])

            designmatrix = np.hstack((designmatrix, van[:, 1:]))

        if self.interactions == True:

            poly = PolynomialFeatures(self.deg, interaction_only=True)
            interactions_matrix = poly.fit_transform(x)
            interactions_matrix = interactions_matrix[:, 1 + n_features: ]
            designmatrix = np.hstack((designmatrix, interactions_matrix))

        return designmatrix

    def fit(self, x, y, loss = 'l2', m = 1, constraints = None, \
            verbose = False):
        '''
        Fits the polynomial model to data ``x`` and ``y`` via cvxpy

        Parameters
        --------
        x : ndarray (m, n) 
            Predictor variables of ``m`` samples and ``n`` features
        y : ndarray (n, )
            Target variable
        loss : string, optional, default 'l2'
            Loss function to use. Can be one of

                - 'l2'
                - 'l1'
                - 'huber'

        m : float, optional, default 1
            Threshold between linear and quadratic loss for huber loss
        constraints : dict, optional, default None
            Dictionary of instances of :py:class:`~Constraints`. Must be of the form ``{i: constraints_i, j: constraints_j, ...}`` where ``i`` and ``j`` are indices of the features.       
        verbose : bool, optional, default False
            If ``True``, print optimizer progress
        '''
        n_samples, n_features = x.shape
        n_coeffs = n_features * self.deg +1
        designmatrix = self._build_designmatrix(x)
        n_coeffs = designmatrix.shape[1]
        column_norms_designmatrix = self._column_norms(designmatrix)
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

                    vander_grad = self._vander_grad(constraints_grid)[:, 1:]
                    norms = column_norms_designmatrix[coefficient_index:coefficient_index + self.deg]
                    vander_grad = vander_grad/norms

                    if Feature_constraints.monotonicity == 'inc':

                        constraint_list.append(vander_grad @ feature_coefficients >= 0)

                    elif Feature_constraints.monotonicity == 'dec':

                        constraint_list.append(vander_grad @ feature_coefficients <= 0)

                if strict_curvature:

                    vander_hesse = self._vander_hesse(constraints_grid)[:, 1:]
                    norms = column_norms_designmatrix[coefficient_index:coefficient_index + self.deg]
                    vander_hesse = vander_hesse/norms

                    if Feature_constraints.curvature == 'convex':

                        constraint_list.append(vander_hesse @ feature_coefficients >= 0)

                    elif Feature_constraints.curvature == 'concave':

                        constraint_list.append(vander_hesse @ feature_coefficients <= 0)                   

        #set up cvxpy problem

        problem = cv.Problem(objective, constraints = constraint_list)
                    
        try:    
            
            if loss == 'l1' or self.regularization == 'l2':
            #l1 loss solved by ECOS. Lower its tolerances for convergence    
                problem.solve(abstol=1e-8, reltol=1e-8, max_iters=1000000, \
                              feastol=1e-8, abstol_inacc = 1e-7, \
                                  reltol_inacc=1e-7, verbose = verbose)            
                
            else:
                    
                #l2 and huber losses solved by OSQP. Lower its tolerances for convergence
                problem.solve(eps_abs=1e-8, eps_rel=1e-8, max_iter=10000000, \
                              eps_prim_inf = 1e-9, eps_dual_inf = 1e-9, verbose = verbose, \
                                  adaptive_rho = True) 
                    
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