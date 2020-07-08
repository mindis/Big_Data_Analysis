# Using cudf in kaggle (30 hours/week)
- https://www.kaggle.com/cdeotte/rapids
```python
# installation
import sys
!cp ../input/rapids/rapids.0.14.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

# Usage
import cudf, io, requests
from io import StringIO
import seaborn as sns

tips_df = sns.load_dataset('tips')
tips_df = cudf.DataFrame(tips_df)
tips_df['tip_percentage'] = tips_df['tip'] / tips_df['total_bill'] * 100

# display average tip by dining party size
print(tips_df.groupby('size').tip_percentage.mean())
```

# Using cudf in colab
- https://news.developer.nvidia.com/run-rapids-on-google-colab/

Needed for Rapids: NVIDIA Tesla T4  or P4 (architecture greater than Pascal, K80 does not work)  
Needed for Xgboost (no relation to Rapids): It works with K80 architecture)  

In gcolab GPU allocation per user is restricted to 12 hours at a time.
The GPU used is the NVIDIA Tesla K80, and once the session is complete,
the user can continue using the resource by connecting to a different VM.

Try changing runtime until you get Tesla P4 or T4. I got that in Jul 8, 2020, 12:28 pm.

```python
!nvidia-smi
# if you see K80 change the runtime until you see P4.

import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
device_name = pynvml.nvmlDeviceGetName(handle)

if device_name not in [b'Tesla T4', b'Tesla P4']:
  raise Exception("""
    Unfortunately this instance does not have a T4 GPU.
    
    Please make sure you've configured Colab to request a GPU instance type.
    
    Sometimes Colab allocates a Tesla K80 instead of a T4. Resetting the instance.

    If you get a K80 GPU, try Runtime -> Reset all runtimes...
  """)
else:
  print('Woo! You got the right kind of GPU!', device_name)
  
  # install cuda
  # Install RAPIDS
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git

!bash rapidsai-csp-utils/colab/rapids-colab.sh 0.14

import sys, os

dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages')
sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:]
sys.path
exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

# test
import cudf
import pandas as pd

import pynvml
import numpy as np
import xgboost as xgb


#load data from skl, then split it into testing and training data
## Load data
from sklearn.datasets import load_boston
boston = load_boston()
pdata = pd.DataFrame(boston.data)
data = cudf.from_pandas(pdata)

## spliting training and test set
from cuml import train_test_split
X, y = data.iloc[:,:-1],data.iloc[:,12]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

# useful resources
- https://docs.rapids.ai/api/cudf/stable/10min.html
- https://github.com/rapidsai/notebooks-contrib
- https://www.deeplearningwizard.com/machine_learning/gpu/rapids_cudf/
