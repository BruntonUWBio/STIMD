# This code is a python translation of the NMP method discussed here 
# http://users.cms.caltech.edu/~hou/papers/Data_driven_TF_analysis_published.pdf.
# We gratefully thank Prof. Thomas Hou and Prof. Zuoqiang Shi for
# providing us with the corresponding MatLab versions of this script.

import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import scipy.io as sio

def cutoff_a1(x,a):
    y = (-np.cos((x-a)*np.pi/a)+1.)/2.
    b = (np.sign(x)+1.)/2.
    y = b*y+(1.-b)
    c = (np.sign(a-x)+1.)/2.
    y = y*c
    return y


# Computes the central difference 
# There is probably a prebuilt function in python for this
def diff_center(a1_c):
    N = len(a1_c)
    h = 1. / (N - 1)
    a1_ce = np.hstack(([2*a1_c[0]]-a1_c[1],a1_c,[2*a1_c[-1]-a1_c[-2]]))
    da1_c = (a1_ce[2:]-a1_ce[:-2])/(2*h)
    return da1_c


def smooth_theta_conv(f_fit,theta,alpha):
        N_t = len(f_fit)
        fe = np.hstack((f_fit,f_fit[1:-1][::-1]))
        fft_fit = np.fft.fft(fe)
        M = 2*np.round((theta[-1]-theta[0])/(2*np.pi))
        xf_t = 2*(theta[-1]-theta[0])
        Ne = len(fe)
        
        temp = 0
        if (Ne % 2) == 0:
            temp = Ne/2
        else:
            temp = (Ne+1)/2
        k_t = np.hstack((np.arange(0,temp),np.arange(-temp,0)))*2*np.pi/xf_t
        km = 2*np.pi/xf_t * M
        fft_a1_fit = fft_fit*cutoff_a1(np.abs(np.abs(k_t)),km*alpha)
        
        dtheta_se = np.real(np.fft.ifft(fft_a1_fit))
        dtheta_s=dtheta_se[:N_t]
        return dtheta_s


# This is a helper function used to create an array in MatLab terminology
# of the form [start:step:stop] 
# Note that subtleties appear using np.arange since the endpoint is not included
# for this function
def bracket(start,stop,step=1):
    res = []
    i = start
    while i <= stop:
        res.append(i)
        i += step
    return np.array(res)


def decompose_theta(f_fit,theta,alpha):
    N_t = len(f_fit)

    # Fourier transform
    fft_fit = np.fft.fft(f_fit)


    #print('1st elem',fft_fit[0])
    # Generate wave number
    xf_t = theta[-1] - theta[0]
    
    # Helper function to transition between arange in numpy 
    # brack notation in MatLab
    # THIS MIGHT STILL CAUSE PROBLEMS   
    #k_t = np.hstack((bracket(0,N_t / 2 -1),bracket(-N_t/2,-1)))*2*np.pi/xf_t
    k_t = np.fft.fftfreq(N_t)*2*np.pi/xf_t*N_t

    #print('km',km)
    
    # Determine the wavenumber where IMF concentrate around
    M = int(np.round((theta[-1]-theta[0])/(2*np.pi)))
    km = 2*np.pi/xf_t*M
    
    # Extract the IMF from the spectral of the signal
    #print('cut',cutoff_a1(np.abs(np.abs(k_t)-km),km*alpha))
    fft_a1_fit = fft_fit*cutoff_a1(np.abs(np.abs(k_t)-km),km*alpha)

    # Translate the spectral of IMF to the original point
    fft_env_fit = np.zeros(N_t,dtype=complex)
    
    if M > 1:
        for j in range(M):
            fft_env_fit[N_t-(M-j)] = fft_a1_fit[j]
        for j in range(M,int(np.min((np.ceil(5*M),N_t)))):
            fft_env_fit[j - M] = fft_a1_fit[j]
    elif M == 1:
        fft_env_fit[0] = fft_a1_fit[1]
    else:
        fft_env_fit[0] = fft_a1_fit[0]
    
    # Compute a(t) and b(t)
    env = np.fft.ifft(fft_env_fit)
    sio.savemat('weird.mat',{'g':f_fit,'b':theta,'a':alpha,'c':fft_env_fit})
    a1_c = 2*np.real(env)
    a1_s=-2*np.imag(env)
    
    IMF_fit = np.real(np.fft.ifft(fft_a1_fit))
    '''plt.figure('IMF')
    plt.plot(IMF_fit)
    plt.draw()
    plt.figure('a1_c')
    plt.plot(a1_c)
    plt.draw()
    plt.figure('a1_s')
    plt.plot(a1_s)
    plt.show()'''
    return IMF_fit, a1_c, a1_s

def Decompose_MP_periodic_sym(f,theta_ini,diag=0):
    
    # Initialize some working variables
    N = len(f) - 1
    xf = 1.0
    h = xf / N
    x = bracket(0.0,xf,h) # Fix so that xf is included
    N = len(x)
    c = np.floor(np.log2(N))
    N_t = 2**c
    #global theta, theta_fit, a1_c, a1_s
    
    # Initial value of theta
    theta = theta_ini
    dtheta = diff_center(theta)


    #plt.plot(theta)
    
    M = 40
    for alpha in np.arange(1/40,1/3,1/40):
        for step in range(10):
            
            # Interpolate the data to the theta space
            h_t = (theta[-1] - theta[0])/N_t
            theta_fit = np.linspace(theta[0],theta[-1]-h_t,N_t)
            #theta_fit = bracket(theta[0],theta[-1]-h_t,h_t)
            f_fit = interp1d(theta, f, kind='cubic')(theta_fit)
            

            
            # Extract the IMF and a,b by FFT in theta space
            
            
            # FIXME dtheta_add isn't quite right
            IMF_fit,a1_c,a1_s = decompose_theta(f_fit,theta,1/2)
            theta_old = theta
            
            dtheta_old = dtheta

            a1_ct = interp1d(theta_fit, a1_c, kind='cubic',fill_value='extrapolate')(theta)
            a1_st = interp1d(theta_fit, a1_s, kind='cubic',fill_value='extrapolate')(theta)
            IMF = interp1d(theta_fit, IMF_fit, kind='cubic',fill_value='extrapolate')(theta)

    
            # Calculate the envelope
            env2 = a1_ct**2 + a1_st**2
            env = np.sqrt(env2)
            env2[env2 < np.max(env2)/100] = np.max(env2)/100
            
            # Calculate the derivative by central difference
            da1_ct = diff_center(a1_ct)
            da1_st = diff_center(a1_st)
    

            # Calculate the change of theta
            dtheta_add = (a1_ct*da1_st - a1_st*da1_ct) / env2
    
            
            
            # Apply low-pass filter on the update of theta
            # The smoothness is controlled by the parameter alpha
            dtheta_add_s = smooth_theta_conv(dtheta_add,theta_old,alpha)
            
        
        
            # Calculate the parameter that makes the frequency positive
            ide = dtheta_add_s > 0
            lambda1 = 1.
            if (np.sum(ide) == 0):
                lambda1 = 1.
            else:
                lambda1 = np.min((np.min(dtheta[ide] / dtheta_add_s[ide] / 2.),1.))
    
            # update the frequency
            numCycles = np.round((theta[-1]-theta[0])/(2*np.pi))
            dtheta = 2*np.pi*numCycles/(theta[-1]-theta[0])*dtheta-lambda1*dtheta_add_s
    
            # reconstruct the phase function from the frequency
            theta = np.hstack(([0.,],cumtrapz(dtheta,dx=1./(len(dtheta)-1))))
        
            # Print the error
            error = np.linalg.norm(dtheta-dtheta_old) / np.linalg.norm(dtheta)
            
            if (error < 1e-2):
                break
            
    v_theta = theta - theta_old
    
    rc = a1_ct*np.cos(v_theta)-a1_st*np.sin(v_theta)
    rs = -a1_st*np.cos(v_theta)-a1_ct*np.sin(v_theta)
    
    cp = np.sum(env*rc) / np.sum(env**2)
    sp = np.sum(env*rs) / np.sum(env**2)

    
    phi=np.arctan(sp/cp)
    if cp < 0:
        phi += pi
    
    return IMF,theta,phi,dtheta