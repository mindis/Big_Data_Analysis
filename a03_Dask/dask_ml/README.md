# daskml xgboost in gcolab
```python
%%capture
# capture will not print in notebook

import os
import sys
ENV_COLAB = 'google.colab' in sys.modules

if ENV_COLAB:
    ## install modules
    !pip install dask[complete] --upgrade
    !pip install dask-ml[complete] --upgrade

    ## print
    print('Environment: Google Colaboratory.')

# NOTE: If we update modules in gcolab, we need to restart runtime.

#Imports
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split

import dask
import dask_ml
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask_ml.xgboost import XGBRegressor

print([(x.__name__,x.__version__) for x in [dask, dask_ml]])

# data
SEED = 100
X,y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                test_size=0.2,random_state=SEED)
da_Xtrain = da.from_array(X_train)
da_ytrain = da.from_array(y_train)
da_Xtest = da.from_array(X_test)
da_ytest = da.from_array(y_test)

# modelling
cluster = LocalCluster(processes=False,scheduler_port=1234)
client = Client(cluster)

est = XGBRegressor(random_state=SEED)
est.fit(da_Xtrain, da_ytrain)

da_txpreds = est.predict(da_Xtest)

# model evaluation
from sklearn import metrics

tx_preds = da_txpreds.compute()

rmse = metrics.mean_squared_error(y_test, tx_preds)**0.5
r2 = metrics.r2_score(y_test, tx_preds)

print('RMSE     : ', rmse)
print('R-Squared: ', r2)
```
