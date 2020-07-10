Table of Contents
=================
   * [Using cuda in colab](#using-cuda-in-colab)
   * [Using cuda in kaggle (30 hours/week)](#using-cuda-in-kaggle-30-hoursweek)
   * [useful resources](#useful-resources)

# Using cuda in colab
- https://news.developer.nvidia.com/run-rapids-on-google-colab/

- Tesla K80 does not support cuda (but supports xgboost)
- Tesla P4 , P100, T4 support cuda
- If google colab allocates K80, factory reset runtime and run all.
- Colab GPU is upto 12 hrs.

```python
# step 1: change runtime to GPU
# step 2: check gpu != K80
!nvidia-smi

# check gpu type
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

# installation of cuda on google colab (takes long time, please wait ...)

%%capture
# Install RAPIDS
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git

!bash rapidsai-csp-utils/colab/rapids-colab.sh 0.14

import sys, os

dist_package_index = sys.path.index('/usr/local/lib/python3.6/dist-packages')
sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.6/site-packages'] + sys.path[dist_package_index:]
sys.path
exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())

# checking
import cudf

gser = cudf.Series([1, 2, 3, 4])
gdf = cudf.DataFrame({'a': np.arange(0, 100), 'b': np.arange(100, 0, -1)})
print(gdf)
```

# Using cuda in kaggle (30 hours/week)
- https://www.kaggle.com/cdeotte/rapids

```python
# installation of cuda in kaggle
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

df = sns.load_dataset('tips')
gdf = cudf.DataFrame(df)
gdf['tip_percentage'] = gdf['tip'] / gdf['total_bill'] * 100

# display average tip by dining party size
print(gdf.groupby('size').tip_percentage.mean())
```

# useful resources
- https://github.com/rapidsai/notebooks-contrib
- https://www.deeplearningwizard.com/machine_learning/gpu/rapids_cudf/
