import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar   
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)    
    

def implied_vol(opt_value, S, K, T, r, type_='call'):
    
    def call_obj(sigma):
        return abs(BS_CALL(S, K, T, r, sigma) - opt_value)
    
    def put_obj(sigma):
        return abs(BS_PUT(S, K, T, r, sigma) - opt_value)
    
    if type_ == 'call':
        res = minimize_scalar(call_obj, bounds=(0.01,6), method='bounded')
        return res.x
    elif type_ == 'put':
        res = minimize_scalar(put_obj, bounds=(0.01,6),
                              method='bounded')
        return res.x
    else:
        raise ValueError("type_ must be 'put' or 'call'")
        

###testing implied vol function####
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.45

C = BS_CALL(S, K, T, r, sigma)
iv = implied_vol(C, S, K, T, r)
print(iv)

#0.45000020780432554

tsla = web.YahooOptions('TSLA')
calls = tsla.get_call_data()

calls.reset_index(inplace=True)
calls['mid'] = (calls.Bid + calls.Ask)/2
calls['Time'] = (calls.Expiry - dt.datetime.now()).dt.days / 255


ivs = [] 

for row in calls.itertuples():
    iv = implied_vol(row.Ask, row.Underlying_Price, row.Strike, row.Time, 0.00)
    ivs.append(iv)


plt.scatter(calls.Strike, ivs, label='calculated')
plt.scatter(calls.Strike, calls.IV, label='Provided')
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.title('Implied Volatility Curve for Facebook')
plt.legend() 