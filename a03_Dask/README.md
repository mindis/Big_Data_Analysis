# Using dask for HPO
```python
from dask.distributed import Client
import joblib

client = Client()  # Connect to a Dask Cluster

with joblib.parallel_backend('dask'):
    # Your normal scikit-learn code here
    # use skearn grid.fit() here
```
