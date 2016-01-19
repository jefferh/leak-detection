import numpy as np
import pandas as pd
import pandas.io.data

def tsToRegInput(D, i, L):
    # Construct the design matrix X and vector of outputs y using the time series data contained in D using the lag L
    # D is a numpy array representing a matrix. Each column of D corresponds to a time series, and each row corresponds to a time point.
    # i is the index of the target time series (starting from zero).
    # L is the (positive) lag.
    D = D.astype(float)

    T = D.shape[0] # number of time points
    J = D.shape[1] # number of time series

    N = T - L # number of data points (i.e. the number of rows in the design matrix)

    X = np.empty([N, J * L]) # initialize the design matrix
    y = np.empty(N) # initialize the output vector

    for n in range(N):
        y[n] = D[L+n, i]
        for j in range(J):
            for t in range(L):
                X[n, j*L+t] = D[L+n-t-1, j]
    return (X, y)

def get_stock_data(n, stocks): 
    dd = pd.DataFrame(np.zeros((n,len(stocks))), columns=stocks)
    for ss in stocks:
        print "getting:",ss
        try:
             df = pd.io.data.get_data_yahoo(ss, '7/1/2005')
             tt = df['Close'][-n:]
             tt.index = range(n)
             dd[ss] = tt
        except:
             print "Cant find ", ss
    return dd
