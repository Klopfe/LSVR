import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
from .SMO_constrained import SMO_SSVR


class SSVR(BaseEstimator,RegressorMixin):
    """ Class for SSVR estimators. 
        Implementation of the Simplex Support Vector Regression using SMO algorithm.
        This uses the Nu-SVR version with nu and C parameters for the regression. 
        """

    def __init__(self, nu=0.5, C=1.0, tol=1e-3, max_iter=1000):
        self.nu=nu
        self.C=C
        self.tol=tol
        self.max_iter=max_iter
        self.warm_start = None

    def fit(self,X,y,version="greedy"):
        X, y=check_X_y(X,y)
  
        self.coef_, self.support_, self.support_vectors_, self.dual_coef_ , \
        self.intercept_, self.n_iter_,self.sub_,self.delta_, = SMO_SSVR(X, y, C=self.C, nu=self.nu, max_iter=self.max_iter, tol=self.tol, warm_start=self.warm_start)
     
        self.is_fitted_=True
        self.shape_fit_=X.shape
        self.training_=X
        self.n_support_=np.size(self.support_)
        self.warm_start = self.dual_coef_
        return self

    def predict(self,X):
        X=check_array(X, accept_sparse=True)
        check_is_fitted(self,'is_fitted_')
        return np.dot(X,self.coef_)

    def _get_coef(self):
        X=self.training_
        theta=self.dual_coef_
        l, n=self.shape_fit_
        
        sigma=theta[(2*l+n)]
        gamma=theta[(2*l):(2*l+n)]
        alpha=theta[0:l]
        alpha_star=theta[(l):(2*l)]
        
        temp=alpha*np.transpose(X)-alpha_star*np.transpose(X)
        w=gamma-np.repeat(sigma,n)-np.sum(temp,axis=1)
        return w
    
    def score(self, X, y):
        check_is_fitted(self,'is_fitted_')
        RMSE=np.sqrt(np.mean((np.dot(X,self.coef_())-y)**2))
        return RMSE
