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
        self.θ_ = np.zeros((self.n_modes_,self.n_samples_))
        self.env_ = np.zeros((self.n_modes_,self.n_samples_))
        self.offset_ = np.zeros(self.n_modes_)

        self.__cons = list()
        self.__ϕ = np.zeros((self.n_observations_-1,self.n_modes_))
        self.remainder_ = np.copy(X)


        for i in range(self.n_modes_):
            if verbose:
                print('Computing Mode',i)
            #self.B_ = np.linalg.qr(self.B_)[0] # Apply Gramm-Schmidt to B
            ϕ_init = self._cartesian2spherical(self.B_[:,i]) # Convert to polar coordinates
            obj =  partial(self._objective2,X=self.remainder_,θ_init=θ_init[i,:]) # Construct objective function
  
            #self.__ϕ[:,i] = minimize(obj,ϕ_init,constraints=self.__cons,tol=tol,options={'disp':disp,'maxiter':40}).x # Perform minimization
            self.__ϕ[:,i] = minimize(obj,ϕ_init,method='Nelder-Mead',tol=tol,options={'disp':disp}).x # Perform minimization
            self.S_[i,:],self.θ_[i,:],self.dθ_[i,:],self.B_[:,i],self.env_[i,:],self.offset_[i] = self._get_params(self.__ϕ[:,i],self.remainder_,θ_init[i,:]) # Get S, θ, dθ, and B

            self.remainder_ -= np.outer(self.B_[:,i],self.S_[i,:])

            # Create constraint in preparation for next iteration
            self.__cons.append({'type': 'eq', 'fun': partial(self._constraint,β=self.__ϕ[:,i])})

        # Sort B and S according to specified convention
        self.B_,self._S = self.sort_modes2(self.B_,self.S_)

        # Return source signals
        X_new = self.S_

        print(np.linalg.norm(self.remainder_)/np.linalg.norm(X))

        return X_new


    # If X = B S sorts B and S according to specified convention
    # Here we provide two conventions

    # Convention: First element in each column must be positive
    def sort_modes(self,B,S):

        # Get rid of sign ambiguity
        for i,col in enumerate(B.T):
            if col[0] <= 0.0:
                B[:,i] = -B[:,i]
                S[i,:] = -S[i,:]
        ind = np.argsort(B[0,:])
        B = B[:,ind]
        S = S[ind,:]

        return B,S

    # Convention: First element of each mode must be positive
    def sort_modes2(self,B,S):
        for i,row in enumerate(S):
            if row[0] <= 0.0:
                B[:,i] = -B[:,i]
                S[i,:] = -S[i,:]
        return B,S

    # project X in direction ϕ and with initial condition θ_init
    # Compute corresponding IMF (s), instantaneous frequency dθ and and projection vector w
    def _get_params(self,ϕ,X,θ_init):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)
        s,θ,offset,dθ,env = Decompose_MP_periodic_sym(signal,θ_init)
        return s,θ,dθ,w,env,offset

    # Objective function which we are trying to minimize
    def _objective(self,ϕ,X,θ_init):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)

        IMF = Decompose_MP_periodic_sym(signal,θ_init)[0]
        obj = np.sum((IMF-signal)**2)
        #DEBUG
        #print('objy', np.max(np.abs(IMF)),obj,'w',w)

        return obj


    def _objective2(self,ϕ,X,θ_init):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)

        IMF = Decompose_MP_periodic_sym(signal,θ_init)[0]
        obj = np.sum((np.outer(w,IMF)-X)**2)


        return obj

    def _objective3(self,ϕ,X,θ_init):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)

        IMF = Decompose_MP_periodic_sym(signal,θ_init)[0]
        #obj = np.sum((np.outer(w,IMF)-X)**2)

        obj = np.linalg.lstsq(IMF[:,np.newaxis],X.T)[1].sum()
        print(obj)

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
            if np.linalg.norm(x[k+1:]) == 0.0:
                # DEBUG
                print('triggered',k)
                if x[k] < 0.0:
                    θ[k] = pi
                else:
                    θ[k] = 0.0
            else:
                # DEBUG
                if k < n-2:
                    θ[k] = np.arccos(x[k]/np.linalg.norm(x[k:]))
                else:
                    # The last element
                    θ[-1] = np.arccos(x[-2]/np.linalg.norm([x[-1],x[-2]]))
                    if x[-1] < 0.0:
                        θ[-1] *= -1
        return θ

    '''def _cartesian2spherical(self,x):
        n = len(x)
        θ = np.zeros(n-1)
        for k in range(n-1):
            denom = np.linalg.norm(x[k+1:])
            if denom == 0.0:
                if x[k] > 0.0:
                    θ[k] = 0.0
                else:
                    θ[k] = pi
                θ[k+1:] = 0.0
                break
            elif k < n-2:
                # acot(x) = atan(1/x)
                θ[k] = np.arctan(denom/x[k])
            else:
                θ[k] = np.arctan(x[k+1]/x[k])
        return θ'''

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
