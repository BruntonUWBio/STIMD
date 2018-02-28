from functools import partial
import numpy as np
from numpy import pi, cos, sin
from scipy.optimize import minimize
from Decompose_MP_periodic_sym import Decompose_MP_periodic_sym

class STIMD():


    # This function decomposes the data matrix X into X = B S where
    # S is the STIMD modes and B is the corresponding mixing matrix
    
    # Input: 
    # X - observed signals (nobservations x nsamples)
    # θ_ini - initiall guess for the phases of the modes (nmodes x nsamples)
    # B_0 - initial guess for the mixing matrix (nobservations x nmodes) 

    # Output:
    # None (this might change)
    # self.B - the mixing matrix
    # self.S - STIMD modes
    # self.dθ - 
    # self.ϕi - mixng matrix
    def transform(self,X,θ_ini,B0,verbose=False,minimization_tol=1e-6):
        self.nmodes, self.nsamples = θ_ini.shape
        self.nobservations = X.shape[0]
        self.S = np.zeros((self.nmodes,self.nsamples))
        self.B = np.copy(B0)
        self.dθ = np.zeros((self.nmodes,self.nsamples))
        self.cons = list()
        self.ϕi = np.zeros((self.nobservations-1,self.nmodes))

        for i in range(self.nmodes):
            if verbose:
                print('Computing Mode',i)
            self.B = np.linalg.qr(self.B)[0] # Apply Gramm-Schmidt to B
            ϕi_guess = self._cartesian2spherical(self.B[:,i]) # Convert to polar coordinates
            obj =  partial(self._objective,X=X,θ_ini=θ_ini[i,:]) # Construct objective function
            self.ϕi[:,i] = minimize(obj,ϕi_guess,constraints=self.cons,tol=minimization_tol).x # Perform minimization     
            self.S[i,:],self.dθ[i,:],self.B[:,i] = self._get_params(self.ϕi[:,i],X,θ_ini[i,:]) # Get S,dθ, and B
            
            # Create constraint in preparation for next iteration
            self.cons.append({'type': 'eq', 'fun': partial(self._constraint,ϕi=self.ϕi[:,i])}) 
        
        # Sort B and S according to specified convention
        B,S = self.sort_modes(self.B,self.S)
            
    def sort_modes(self,B,S):

        # Get rid of sign ambiguity
        for i,col in enumerate(B.T):
            if col[0] <= 0.0:
                B[:,i] = -B[:,i]
                S[i,:] = -S[i,:]
        return B,S
    
    def _get_params(self,ϕ,X,θ_ini):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)
        s,_,_,dθ = Decompose_MP_periodic_sym(signal,θ_ini)
        return s,dθ,w

    def _objective(self,ϕ,X,θ_ini):
        w = self._spherical2cartesian(ϕ)
        signal = w.dot(X)
        IMF = Decompose_MP_periodic_sym(signal,θ_ini)[0]
        obj = np.sum((IMF-signal)**2)
        return obj
      
    def _constraint(self,ϕ,ϕi):
        y = np.inner(self._spherical2cartesian(ϕ),self._spherical2cartesian(ϕi))
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