import matplotlib.pyplot as plt

def f(x):
    return np.exp(x)

def forward_difference(f, x, dx):
    return (f(x+dx) - f(x)) / dx

plt.figure(figsize=(8, 6))
deltasxs = [1.5, 1, 0.5, 0.2, 0.1]
nums = np.arange(1,5,0.01)

plt.plot(nums, True, label= 'True',linewidth=3)
for delt in deltasxs:
    plt.plot(nums, forward_difference(f, nums, delt), label=f'$\Delta x$ = {delt}',linewidth=2,linestyle='--')

plt.legend()
plt.ylabel("Derivative")
plt.xlabel('$x$')
plt.title('Forward Finite Difference')


def backward_difference(f, x, dx):
    return (f(x) - f(x-dx)) / dx

nums = np.arange(1,5,0.01)
true = f(nums)
plt.figure(figsize=(8, 6))
deltasxs = [1.5, 1, 0.5, 0.2, 0.1]

#plt.plot(nums, f(nums))
plt.plot(nums, true, label= 'True',linewidth=3)
for delt in deltasxs:
    plt.plot(nums, backward_difference(f, nums, delt),
             label=f'$\Delta x$ = {delt}',
             linewidth=2, linestyle='--')

plt.legend()
plt.ylabel("Derivative")
plt.xlabel('$x$')
plt.title('Backward Finite Difference')

def central_difference(f, x, dx):
    return (f(x+dx) - f(x-dx)) / (2*dx)


plt.figure(figsize=(8, 6))
deltasxs = [1.5, 1, 0.5, 0.2, 0.1]

#plt.plot(nums, f(nums))
plt.plot(nums, true, label= 'True',linewidth=3)
for delt in deltasxs:
    plt.plot(nums, central_difference(f, nums, delt), label=f'$\Delta x$ = {delt}',linewidth=2,linestyle='--')

plt.legend()
plt.ylabel("Derivative")
plt.xlabel('$x$')
plt.title('Central Finite Difference')



from scipy.stats import norm
import numpy as np

def d1(S, K, T, r, sigma):
    return (np.log(S/K) + (r + sigma**2/2)*T) /\
                     sigma*np.sqrt(T)

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma* np.sqrt(T)

def delta_call(S, K, T, r, sigma):
    N = norm.cdf
    return N(d1(S, K, T, r, sigma))
    
def delta_fdm_call(S, K, T, r, sigma, ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S+ds, K, T, r, sigma) -BS_CALL(S-ds, K, T, r, sigma))/\
                        (2*ds)
    elif method == 'forward':
        return (BS_CALL(S+ds, K, T, r, sigma) - BS_CALL(S, K, T, r, sigma))/ds
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S-ds, K, T, r, sigma))/ds
    
    
def delta_put(S, K, T, r, sigma):
    return - N(-d1(S, K, T, r, sigma))

def delta_fdm_put(S, K, T, r, sigma, ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_PUT(S+ds, K, T, r, sigma) -BS_PUT(S-ds, K, T, r, sigma))/\
                        (2*ds)
    elif method == 'forward':
        return (BS_PUT(S+ds, K, T, r, sigma) - BS_PUT(S, K, T, r, sigma))/ds
    elif method == 'backward':
        return (BS_PUT(S, K, T, r, sigma) - BS_PUT(S-ds, K, T, r, sigma))/ds



S = 100
K = 100
T = 1
r = 0.00
sigma = 0.25

prices = np.arange(1, 250,1)

deltas_c = delta_call(prices, K, T, r, sigma)
deltas_p = delta_put(prices, K, T, r, sigma)
deltas_back_c = delta_fdm_call(prices, K, T,r, sigma, ds = 0.01,method='backward')
deltas_forward_p = delta_fdm_put(prices, K, T,r, sigma, ds = 0.01,method='forward')

plt.plot(prices, deltas_c, label='Delta Call')
plt.plot(prices, deltas_p, label='Delta Put')
plt.xlabel('$S_0$')
plt.ylabel('Delta')
plt.title('Stock Price Effect on Delta for Calls/Puts' )
plt.axvline(K, color='black', linestyle='dashed', linewidth=2,label="Strike")
plt.legend()


errorc = np.array(deltas_c) - np.array(deltas_back)
errorp = np.array(deltas_p) - np.array(deltas_forward_p)

plt.plot(prices, errorc, label='FDM_CALL_ERROR')
plt.plot(prices, errorp, label='FDM_PUT_ERROR')
plt.legend()
plt.xlabel('$S_0$')
plt.ylabel('FDM Error')

def gamma(S, K, T, r, sigma):
    N_prime = norm.pdf
    return N_prime(d1(S,K, T, r, sigma))/(S*sigma*np.sqrt(T))


def gamma_fdm(S, K, T, r, sigma , ds = 1e-5, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S+ds , K, T, r, sigma) -2*BS_CALL(S, K, T, r, sigma) + 
                    BS_CALL(S-ds , K, T, r, sigma) )/ (ds)**2
    elif method == 'forward':
        return (BS_CALL(S+2*ds, K, T, r, sigma) - 2*BS_CALL(S+ds, K, T, r, sigma)+
                   BS_CALL(S, K, T, r, sigma) )/ (ds**2)
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - 2* BS_CALL(S-ds, K, T, r, sigma)
                + BS_CALL(S-2*ds, K, T, r, sigma)) /  (ds**2)  


gammas = gamma(prices, K, T, r, sigma)
gamma_forward = gamma_fdm(prices, K, T, r, sigma, ds =0.01,method='forward')

plt.plot(prices, gammas)
plt.plot(prices, gamma_forward)
plt.title('Gamma by changing $S_0$')
plt.xlabel('$S_0$')
plt.ylabel('Gamma')

plt.plot(gammas -gamma_forward)

Ts = [1,0.75,0.5,0.25]

for t in Ts:
    plt.plot(vega(prices, K, t, r, sigma), label=f'T = {t}')

plt.legend()
plt.xlabel('$S_0$')
plt.ylabel('Vega')
plt.title('Vega Decrease with Time')

def vega(S, K, T, r, sigma):
    N_prime = norm.pdf
    return S*np.sqrt(T)*N_prime(d1(S,K,T,r,sigma)) 

def vega_fdm(S, K, T, r, sigma, dv=1e-4, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S, K, T, r, sigma+dv) -BS_CALL(S, K, T, r, sigma-dv))/\
                        (2*dv)
    elif method == 'forward':
        return (BS_CALL(S, K, T, r, sigma+dv) - BS_CALL(S, K, T, r, sigma))/dv
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T, r, sigma-dv))/dv
 


def theta_call(S, K, T, r, sigma):
    p1 = - S*N_prime(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(d2(S, K, T, r, sigma)) 
    return p1 - p2

def theta_put(S, K, T, r, sigma):
    p1 = - S*N_prime(d1(S, K, T, r, sigma))*sigma / (2 * np.sqrt(T))
    p2 = r*K*np.exp(-r*T)*N(-d2(S, K, T, r, sigma)) 
    return p1 + p2

def theta_call_fdm(S, K, T, r, sigma, dt, method='central'):
    method = method.lower() 
    if method =='central':
        return -(BS_CALL(S, K, T+dt, r, sigma) -BS_CALL(S, K, T-dt, r, sigma))/\
                        (2*dt)
    elif method == 'forward':
        return -(BS_CALL(S, K, T+dt, r, sigma) - BS_CALL(S, K, T, r, sigma))/dt
    elif method == 'backward':
        return -(BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T-dt, r, sigma))/dt
    
def theta_put_fdm(S, K, T, r, sigma, dt, method='central'):
    method = method.lower() 
    if method =='central':
        return -(BS_PUT(S, K, T+dt, r, sigma) -BS_PUT(S, K, T-dt, r, sigma))/\
                        (2*dt)
    elif method == 'forward':
        return -(BS_PUT(S, K, T+dt, r, sigma) - BS_PUT(S, K, T, r, sigma))/dt
    elif method == 'backward':
        return -(BS_PUT(S, K, T, r, sigma) - BS_PUT(S, K, T-dt, r, sigma))/dt





theta_call(100,100,1,0.05,0.2, 0.1,0.05)
Ts = [1,0.75,0.5,0.25,0.1,0.05]
for t in Ts:
    plt.plot(theta_call(prices, K, t, r, sigma), label=f'T = {t}')

plt.legend()
plt.title('Theta of a call')
plt.xlabel('$S_0$')
plt.ylabel('Theta')


def rho_call(S, K, T, r, sigma):
    return K*T*np.exp(-r*T)*N(d2(S, K, T, r, sigma))

def rho_put(S, K, T, r, sigma):
    return -K*T*np.exp(-r*T)*N(-d2(S, K, T, r, sigma))


def rho_call_fdm(S, K, T, r, sigma, dr, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_CALL(S, K, T, r+dr, sigma) -BS_CALL(S, K, T, r-dr, sigma))/\
                        (2*dr)
    elif method == 'forward':
        return (BS_CALL(S, K, T, r+dr, sigma) - BS_CALL(S, K, T, r, sigma))/dr
    elif method == 'backward':
        return (BS_CALL(S, K, T, r, sigma) - BS_CALL(S, K, T, r-dr, sigma))/dr
  
def rho_put_fdm(S, K, T, r, sigma, dr, method='central'):
    method = method.lower() 
    if method =='central':
        return (BS_PUT(S, K, T, r+dr, sigma) -BS_PUT(S, K, T, r-dr, sigma))/\
                        (2*dr)
    elif method == 'forward':
        return (BS_PUT(S, K, T, r+dr, sigma) - BS_PUT(S, K, T, r, sigma))/dr
    elif method == 'backward':
        return (BS_PUT(S, K, T, r, sigma) - BS_PUT(S, K, T, r-dr, sigma))/dr


S = 100
K = 100
T = 1
r = 0.02
sigma = 0.25


print('#############DELTA###############')
print('Delta analytic =           ',delta_call(S,K,T,r,sigma))
print('Delta forward difference = ', delta_fdm_call(S, K, T, r, sigma, ds=0.01,
                                                    method='forward'))
print('Delta backward difference = ', delta_fdm_call(S, K, T, r, sigma, ds=0.01,
                                                    method='backward'))

print('Delta central difference = ', delta_fdm_call(S, K, T, r, sigma, ds=0.01,
                                                    method='central'))

print('#############GAMMA###############')
print('Gamma analytic =           ',gamma(S,K,T,r,sigma))
print('Gamma forward difference = ', gamma_fdm(S, K, T, r, sigma, ds=0.01,
                                                    method='forward'))
print('Gamma backward difference = ', gamma_fdm(S, K, T, r, sigma, ds=0.01,
                                                    method='backward'))

print('Gamma central difference = ', gamma_fdm(S, K, T, r, sigma, ds=0.01,
                                               method='central'))
                                               
print('#############VEGA###############')
print('Vega analytic =           ',vega(S,K,T,r,sigma))
print('Vega forward difference = ', vega_fdm(S, K, T, r, sigma, dv=0.001,
                                                    method='forward'))
print('Vega backward difference = ', vega_fdm(S, K, T, r, sigma, dv=0.001,
                                                    method='backward'))

print('Vega central difference = ', vega_fdm(S, K, T, r, sigma, dv=0.001,
                                               method='central'))
             
print('#############Theta###############')
print('Theta analytic =           ',theta_call(S,K,T,r,sigma))
print('Theta forward difference = ', theta_call_fdm(S, K, T, r, sigma, dt=0.01,
                                                    method='forward'))
print('Theta backward difference = ', theta_call_fdm(S, K, T, r, sigma, dt=0.01,
                                                    method='backward'))

print('Theta central difference = ', theta_call_fdm(S, K, T, r, sigma, dt=0.001,
                                                    method='central'))

print('#############RHO###############')
print('Theta analytic =           ',rho_call(S,K,T,r,sigma))
print('Theta forward difference = ', rho_call_fdm(S, K, T, r, sigma, dr=0.001,
                                                    method='forward'))
print('Theta backward difference = ', rho_call_fdm(S, K, T, r, sigma, dr=0.001,
                                                    method='backward'))

print('Theta central difference = ', rho_call_fdm(S, K, T, r, sigma, dr=0.0001,