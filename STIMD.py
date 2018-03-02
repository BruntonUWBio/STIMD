from functools import partial
import numpy as np
from numpy import pi, cos, sin
from scipy.optimize import minimize
from Decompose_MP_periodic_sym import Decompose_MP_periodic_sym
from sklearn.base import BaseEstimator,TransformerMixin

class STIMD(BaseEstimator,TransformerMixin):
    """
    This function decomposes the data matrix X into X = B S where
    S is the STIMD modes and B is the corresponding mixing matrix

    Parameters
    ----------
    X : array-like, shape (n_observations, n_samples)
        Data matrix contiaing obersved (mixed) signals
    θ_init : array-like, shape (n_modes, n_samples)
        Initial guess for the phases of the modes 
    B_init : array-like, shape (n_observations, n_modes)
        Initial guess for the mixing matrix
    verbose : boolean, optional
        Display which mode is being extracted for sanity 
        purposes
    tol : float, optional
        Function tolerance in minimization 
    disp : bool, optional
        Display minimization results


    Returns 
    -------
    X_new : array-like, shape (n_modes, n_samples)
        STIMD modes

    Attributes
    ----------
    n_modes_ : int
        number of STIMD modes
    n_samples_ : int
        number of time samples
    n_observations_ : int
        number of observations
    S_ : array-like, shape (n_modes, n_samples)
        STIMD modes
    B_ : array-like, shape (n_observations, n_modes)
        Mixing matrix
    dθ_ : array-like, shape (n_modes, n_samples)
        Instantanous frequencies of stimd modes dθ/2π = ω
    """
    def fit_transform(self,X,θ_init,B_init,verbose=False,tol=1e-6,disp=False):
        self.n_modes_, self.n_samples_ = θ_init.shape
        self.n_observations_ = X.shape[0]
        self.S_ = np.zeros((self.n_modes_,self.n_samples_))
        self.B_ = np.copy(B_init)
        self.dθ_ = np.zeros((self.n_modes_,self.n_samples_))
        
        self.__cons = list()
        self.__ϕ = np.zeros((self.n_observations_-1,self.n_modes_))

        for i in range(self.n_modes_):
            if verbose:
                print('Computing Mode',i)
            self.B_ = np.linalg.qr(self.B_)[0] # Apply Gramm-Schmidt to B
            ϕ_init = self._cartesian2spherical(self.B_[:,i]) # Convert to polar coordinates
            obj =  partial(self._objective,X=X,θ_init=θ_init[i,:]) # Construct objective function
            self.__ϕ[:,i] = minimize(obj,ϕ_init,constraints=self.__cons,tol=tol,options={'disp':disp}).x # Perform minimization
            self.S_[i,:],self.dθ_[i,:],self.B_[:,i] = self._get_params(self.__ϕ[:,i],X,θ_init[i,:]) # Get S,dθ, and B

            # Create constraint in preparation for next iteration
            self.__cons.append({'type': 'eq', 'fun': partial(self._constraint,β=self.__ϕ[:,i])})

        # Sort B and S according to specified convention
        self.B_,self._S = self.sort_modes(self.B_,self.S_)

        # Return source signals
        X_new = self.S_

        return X_new


    # If X = B S sorts B and S according to specified convention
    # Convention: First element in each column must be positive
    def sort_modes(self,B,S):

        # Get rid of sign ambiguity
        for i,col in enumerate(B.T):
            if col[0] <= 0.0:
                B[:,i] = -B[:,i]
                S[i,:] = -S[i,:]
        return B,S

    # project X in direction ϕ and with initial condition θ_init
    # Compute corresponding IMF (s), instantaneous frequency dθ and and projection vector w
    def _get_params(self,ϕ,X,θ_init):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)
        s,_,_,dθ = Decompose_MP_periodic_sym(signal,θ_init)
        return s,dθ,w

    # Objective function which we are trying to minimize
    def _objective(self,ϕ,X,θ_init):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)
        IMF = Decompose_MP_periodic_sym(signal,θ_init)[0]
        obj = np.sum((IMF-signal)**2)
        return obj

    # Constrain used to enforce polar vectors α and β to be orthogonal
    def _constraint(self,α,β):
        y = np.inner(self._spherical2cartesian(α),self._spherical2cartesian(β))
        # Soft threshold
        if y < 1e-10:
            y = 0
        return y

    # Input: x array = [x1,x2,...,xn] corresponding to unit vector in cartesian coordinates
    # Output theta array = [theta1, theta2,...thetan-1] corresponding to vector in "polar coordinates"
    # One can think of theta1 as theta, theta2 as phi, and so on
    # The formula used here can be found at https://en.wikipedia.org/wiki/N-sphere
    # Alternate implementation can be found here:
    # https://stackoverflow.com/questions/45789149/hyper-spherical-coordinates-implementation
    def _cartesian2spherical(self,x):

        n = len(x) # Number of dimensions of x
        θ = np.zeros(n-1) # Initialize θ vector
        for k in range(n-1):
            if np.linalg.norm(x[k:]) == 0.0:
                if x[k] < 0.0:
                    θ[k] = pi
                else:
                    θ[k] = 0.0
            else:
                θ[k] = np.arccos(x[k]/np.linalg.norm(x[k:]))

        # The last element
        θ[-1] = np.arccos(x[-2]/np.linalg.norm([x[-1],x[-2]]))
        if x[-1] < 0.0:
            θ[-1] = 2*pi - θ[-1]
        return θ

    # Inverse of above equation
    # Input: θ array - length n-1
    # Output: x array - unit vector length n
    # This implementation might be faster?
    # https://stackoverflow.com/questions/20133318/n-sphere-coordinate-system-to-cartesian-coordinate-system
    def _spherical2cartesian(self,θ):

        # This is to handle the case where θ was incorrectly inputted as a single number
        if type(θ) == float:
            θ = np.array([θ])

        # n is always defined to be the length of x
        n = len(θ)+1
        x = np.ones(n)

        # update elements using formula
        for k in range(n-1):
            x[k] *= cos(θ[k])
            x[k+1:] *= sin(θ[k])
        return x
