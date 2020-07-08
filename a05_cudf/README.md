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

All the times I have tried google colab allocated me the K80 GPU and never Tesla T4 GPU.
Rapids only works with tesla gpu (not tpu) and colab fails to run Rapids modules such as cudf.
Needed for Rapids: NVIDIA Tesla T4  
Allocated: NVIDIA Tesla K80  
Note that only Rapids wants Tesla T4  but the xgboost library can use k80 gpu and run the model on
gpu.

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
