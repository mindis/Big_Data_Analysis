Table of Contents
=================
   * [map_partition](#map_partition)
   * [dask series apply on rows](#dask-series-apply-on-rows)
   * [map_partitions apply and meta](#map_partitions-apply-and-meta)
  

# WARNING
Dask is too much buggy and pre-mature, never use it. Instead of this use "PYSPARK", this is much much
stable and works for data larger than that dask can work.
For example:
```
from dask_ml.model_selection import train_test_split # FAILS
[('numpy', '1.16.4'),
 ('pandas', '0.25.0'),
 ('dask', '2.5.2'),
 ('sklearn', '0.21.2')]
 
 Its good to know dask exists, but not good to use it.
```

# dask apply
```python
import numpy as np
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                   'y': [0.2, None, 0.345, 0.40, 0.15]})
ddf = dd.from_pandas(df, npartitions=4)

# apply is always slow, but we can use for special case functions
def func(row):
    if pd.isnull(row['y']):
        return row['x'] + 100
    else:
        return row['y']
     
ddf['z'] = ddf.apply(func, axis=1, meta=float)
ddf = ddf.compute()
ddf
```

# map_partition
https://stackoverflow.com/questions/47125665/simple-dask-map-partitions-example
```python
# example 1
import dask.dataframe as dd
import pandas as pd
from dask.multiprocessing import get
import random

df = pd.DataFrame({'col_1':random.sample(range(10000), 10000), 'col_2': random.sample(range(10000), 10000) })
ddf.map_partitions(lambda df: df.assign(col_3=df.col_1 * df.col_2))

# example 2
def test_f(df, col_1, col_2):
    return df.assign(result=df[col_1] * df[col_2])


ddf_out = ddf.map_partitions(test_f, 'col_1', 'col_2')
# Here is good place to do something with BIG ddf_out dataframe before calling .compute()
result = ddf_out.compute(get=get)  # Will load the whole dataframe into memory


# example 3
ddf_out = ddf.map_partitions(lambda df: df.assign(result=df.col_1 * df.col_2))
# Here is good place to do something with BIG ddf_out dataframe before calling .compute()
result = ddf_out.compute(get=get)  # Will load the whole dataframe into memory


# example 4
def compute_date_timestamp(df,year,hr_min):
    '''
    column year = 1990-01-01  dtype = datetime
    column hr_min = 1540.  dtype = float
    '''
    hours = df[hr_min] // 100
    hours_timedelta = pd.to_timedelta(hours, unit='h')

    minutes = df[hr_min] % 100
    minutes_timedelta = pd.to_timedelta(minutes, unit='m')

    return df[year] + hours_timedelta + minutes_timedelta
    
df.map_partitions(compute_date_timestamp, 'Date', 'CRSDepTime' ).head() # will give 5 rows
```

# dask series apply on rows
https://stackoverflow.com/questions/47125665/simple-dask-map-partitions-example
```python
import numpy as np
import pandas as pd
import dask.dataframe as dd
np.random.seed(42)

df = pd.DataFrame({'col_1':np.random.randint(1,5,size=(10)),
                   'col_2': np.random.randint(1,5,size=(10)),
                   'col_3': np.random.randint(1,5,size=(10))})
ddf = dd.from_pandas(df, npartitions=4)

def test_f(dds, col_1, col_2):
    return dds[col_1] * dds[col_2]
    
ddf_out = ddf.apply(
    test_f, 
    args=('col_1', 'col_2'), 
    axis=1, 
    meta=('result', int)
).compute()


ddf['result'] = ddf_out
ddf = ddf.compute()
ddf
```

# map_partitions apply and meta
```python
import numpy as np
import pandas as pd

from dask import dataframe as dd
from dask.multiprocessing import get
from multiprocessing import cpu_count
nCores = cpu_count()


pd.set_option('display.max_colwidth',-1)

x = ({'F9_07_PZ_COMP_DIRECT': '0',
  'F9_07_PZ_DIRTRSTKEY_NAME': 'DEBRA MEALY',
  'F9_07_PZ_COMP_OTHER': '0',
  'F9_07_PZ_COMP_RELATED': '0',
  'F9_07_PZ_TITLE': 'CHAIR PERSON',
  'F9_07_PZ_AVE_HOURS_WEEK': '1.00',
  'F9_07_PC_TRUSTEE_INDIVIDUAL': 'X'},
 {'F9_07_PZ_COMP_DIRECT': '0',
  'F9_07_PZ_DIRTRSTKEY_NAME': 'HELEN GORDON',
  'F9_07_PZ_COMP_OTHER': '0',
  'F9_07_PZ_COMP_RELATED': '0',
  'F9_07_PZ_TITLE': 'VICE CHAIR',
  'F9_07_PZ_AVE_HOURS_WEEK': '1.00',
  'F9_07_PC_TRUSTEE_INDIVIDUAL': 'X'})

df = pd.DataFrame({'a': x})

cols = list(df.iloc[0,0].keys())

ddf = dd.from_pandas(df, 1)
meta = pd.DataFrame(columns=cols, dtype="O")
ans = ddf.map_partitions(lambda df: df.a.apply(pd.Series), meta=meta).compute()
ans
```
