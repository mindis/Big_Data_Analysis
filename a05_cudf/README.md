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
  ```
Date: Jul 8, 2020  
It does not work.

UserWarning: You will need a GPU with NVIDIA Pascal™ or newer architecture
Detected GPU 0: Tesla K80

RuntimeError: after reduction step 1: cudaErrorInvalidDeviceFunction: invalid device function
```

# useful resources
- https://github.com/rapidsai/notebooks-contrib
- https://www.deeplearningwizard.com/machine_learning/gpu/rapids_cudf/