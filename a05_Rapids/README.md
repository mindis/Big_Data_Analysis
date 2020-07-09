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


# useful resources
- https://github.com/rapidsai/notebooks-contrib
- https://www.deeplearningwizard.com/machine_learning/gpu/rapids_cudf/
