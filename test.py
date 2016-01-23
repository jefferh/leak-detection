import time
import numpy as np
import sgenRandomized as sgr
import sTests as st

start_time = time.time()

# Get stock data
sd = sgr.getStockData(1500)
# Generate 10 time series based on the stock data
(scaledStocks, genSeries, dep, funcs) = sgr.generateSeries(sd, numGenSeries=20, numDataPts=1000, maxLag=20, maxSummands=3, maxFactors=3, noiseStDev=0.25)

# Evaluate LASSO
depFitted_lasso = st.fitModel(inSeries=scaledStocks, outSeries=genSeries, maxLag=20, method='lassocv', threshold=0.2)
(relSeries, idSeries_lasso, F1_scores_lasso) = st.computeF1Score(dep, depFitted_lasso)
print "LASSO mean F1-score: " + str(np.nanmean(F1_scores_lasso))
print "LASSO standard error of mean F1-score: " + str(np.nanstd(F1_scores_lasso)/np.count_nonzero(~np.isnan(F1_scores_lasso)))

# Evaluate Elastic Net
depFitted_elasticnet = st.fitModel(inSeries=scaledStocks, outSeries=genSeries, maxLag=20, method='elasticnetcv', threshold=0.2)
(relSeries, idSeries_elasticnet, F1_scores_elasticnet) = st.computeF1Score(dep, depFitted_elasticnet)
print "Elastic Net F1-score: " + str(np.nanmean(F1_scores_elasticnet))
print "Elastic Net standard error of mean F1-score: " + str(np.nanstd(F1_scores_elasticnet)/np.count_nonzero(~np.isnan(F1_scores_elasticnet)))

# Evaluate GBM
depFitted_gbm = st.fitModel(inSeries=scaledStocks, outSeries=genSeries, maxLag=20, method='gbm', threshold=0.2)
(relSeries, idSeries_gbm, F1_scores_gbm) = st.computeF1Score(dep, depFitted_gbm)
print "GBM mean F1-score: " + str(np.nanmean(F1_scores_gbm))
print "GBM standard error of mean F1-score: " + str(np.nanstd(F1_scores_gbm)/np.count_nonzero(~np.isnan(F1_scores_gbm)))

# Evaluate GBM with Huber Loss
depFitted_gbmhuber = st.fitModel(inSeries=scaledStocks, outSeries=genSeries, maxLag=20, method='gbmhuber', threshold=0.2)
(relSeries, idSeries_gbmhuber, F1_scores_gbmhuber) = st.computeF1Score(dep, depFitted_gbmhuber)
print "GBM w/ Huber Loss mean F1-score: " + str(np.nanmean(F1_scores_gbmhuber))
print "GBM w/ Huber Loss standard error of mean F1-score: " + str(np.nanstd(F1_scores_gbmhuber)/np.count_nonzero(~np.isnan(F1_scores_gbmhuber)))

print("Time elapsed: %s seconds" % (time.time() - start_time))
