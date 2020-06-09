import pandas_datareader as dr
import pandas as pd
import numpy as np
stock_list = ['GOOG','AMZN','AMT','AAPL','DIS','DOCU','FDN','GPN','HD','INTU','QQQ'
              ,'IEFA','IJR','IHI','MTUM','QUAL','IGM','IGV','LQD','USMV','IWF','IWD'
              ,'IWP','PG','CRM','SPTS','SBUX','VCR','VBK','V']
stock_list = stock_list[0:4]
print(stock_list)
API_KEY = #Enter_API_KEY
#name = 'AAPL'
df2 =pd.DataFrame()
df2 = dr.get_data_tiingo('GOOG', api_key= API_KEY) # just to give an idea of what data looks like when imported, stock choice is arbitrary
print(df2)


stock_data = pd.DataFrame()
for stock in stock_list:
    
    adj_close_data = dr.get_data_tiingo(stock, api_key=API_KEY).adjClose # we only care about adj close data, not highs and lows.
    adj_close_data.reset_index(drop = True, inplace = True)
    stock_data[stock] = adj_close_data
print(stock_data)

returns_daily = stock_data.pct_change()
#selected=list(stock_data.columns[1:])
#returns_daily = stock_data[selected].pct_change()

print(returns_daily)

expected_returns = returns_daily.mean()

#GOOG_avg_check = returns_daily['GOOG'].sum()/1256 # all this does is check the average of google to make sure averaging was done correctly
#print(GOOG_avg_check) # prints the value to check that averaging was done correctly

print(expected_returns)

cov_matrix = returns_daily.cov()

print(cov_matrix)


# from this point on I will use some code taken from here: https://gist.github.com/codecademydev/a1ff15fbdd1b13271e91727ddd05636f
# this code is used to find the efficient frontier and graph different portfolio weights on a scatter plot

import numpy
import cvxopt as opt
from cvxopt import blas, solvers

def return_portfolios(expected_returns, cov_matrix):
    port_returns = []
    port_volatility = []
    stock_weights = []
    
    selected = (expected_returns.axes)[0]
    
    num_assets = len(selected) 
    num_portfolios = 5000 # original was 5000, reduce to increase speed
    
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, expected_returns)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)
    
    portfolio = {'Returns': port_returns,
                 'Volatility': port_volatility}
    
    for counter,symbol in enumerate(selected):
        portfolio[symbol +' Weight'] = [Weight[counter] for Weight in stock_weights]
    
    df = pd.DataFrame(portfolio)
    
    column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in selected]
    
    df = df[column_order]
   
    return df
  
def optimal_portfolio(returns):
    n = returns.shape[1]
    returns = np.transpose(returns.to_numpy()) # originally as_matrix changed to values

    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)

    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x']
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


import matplotlib.pyplot as plt
from skrf import *
#from rf import *
cov_quarterly = cov_matrix # changing the name of cov_matrix to the variable name used in the codecademy scripts
single_asset_std=np.sqrt(np.diagonal(cov_quarterly))
df = return_portfolios(expected_returns, cov_quarterly) 
weights, returns, risks = optimal_portfolio(returns_daily[1:])

df.plot.scatter(x='Volatility', y='Returns', fontsize=12)
plt.plot(risks, returns, 'y-o')
plt.scatter(single_asset_std,expected_returns,marker='X',color='red',s=200)
plt.ylabel('Expected Returns',fontsize=14)
plt.xlabel('Volatility (Std. Deviation)',fontsize=14)
plt.title('Efficient Frontier', fontsize=24)
plt.show()

#### Get current portfolio info
##share_numbers = [3,3,12,16,13,34,79,16,11,12,183,177,139,40,60,195,54,25,130,183,213,120,116,37,25,309,30,
##share number = 


