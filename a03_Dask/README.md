# daskml xgboost in gcolab
```python
%%capture
# capture will not print in notebook

import os
import sys
ENV_COLAB = 'google.colab' in sys.modules

if ENV_COLAB:
    ## install modules
    !python -m pip install dask[complete] --upgrade
    !pip install dask-ml[complete]

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


# Using dask for HPO
```python
import numpy as np
from dask.distributed import Client

import joblib
from sklearn.datasets import load_digits
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

client = Client(processes=False)             # create local cluster

digits = load_digits()

param_space = {
    'C': np.logspace(-6, 6, 13),
    'gamma': np.logspace(-8, 8, 17),
    'tol': np.logspace(-4, -1, 4),
    'class_weight': [None, 'balanced'],
}

model = SVC(kernel='rbf')
search = RandomizedSearchCV(model, param_space, cv=3, n_iter=50, verbose=10)

with joblib.parallel_backend('dask'):
    search.fit(digits.data, digits.target)
```
