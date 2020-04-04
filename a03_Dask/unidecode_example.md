```python
import numpy as np
import pandas as pd
import unidecode

s = pd.Series(['mañana','Ceñía']*100_000)
s.head()
0    mañana
1     Ceñía
2    mañana
3     Ceñía
4    mañana
```

# Using apply
```python
%timeit s.apply(unidecode.unidecode) 
1 loop, best of 3: 761 ms per loop
```

# Using vectorized numpy
```python
@np.vectorize
def decode(x):
  return unidecode.unidecode(x)

%timeit decode(s.values)
1 loop, best of 3: 797 ms per loop
```

# String vectorization is not supported in numba
```python
import numba

@numba.vectorize
def numba_decode(x):
  return unidecode.unidecode(x)

%timeit numba_decode(s.values)
ValueError: Unsupported array dtype: object
```

# Using dask apply
```python
import numpy as np
import pandas as pd
import unidecode
import dask


s = pd.Series(['mañana','Ceñía']*100_000)
df = s.to_frame('x')
ddf = dd.from_pandas(df, npartitions=4)
ddf.head()

def dask_decode(row):
     return unidecode.unidecode(row['x'])
%timeit ddf.apply(dask_decode, axis=1,meta = ('x', 'str')).compute()
1 loop, best of 3: 3.74 s per loop

NOTE: Dask should not be used when data fits in memory, if data fits in 
memory always use pandas. Dask is best for data not fitting in RAM.
```

# Using dask map_partitions
```python
ddf.map_partitions(lambda df: df.assign(y=  df.x.apply(unidecode.unidecode))).head()
        x       y
0  mañana  manana
1   Ceñía   Cenia
2  mañana  manana
3   Ceñía   Cenia
4  mañana  manana

%timeit ddf.map_partitions(lambda df: df.assign(y=  df.x.apply(unidecode.unidecode))).compute()
1 loop, best of 3: 825 ms per loop

%timeit df.assign(y=  df.x.apply(unidecode.unidecode))
1 loop, best of 3: 788 ms per loop

def dask_decode_args(df, col_1):
    return df.assign( y= df[col_1].apply(unidecode.unidecode)  )


ddf.map_partitions(dask_decode_args, 'x').head()
%timeit ddf.map_partitions(dask_decode_args, 'x').compute()
1 loop, best of 3: 814 ms per loop
```

# Using dask persist
```python
import numpy as np
import dask
import unidecode

@dask.delayed
def lst_decode(x):
  return [unidecode.unidecode(i) for i in x]

arr = np.array(['mañana','Ceñía']*100_000)
splits = np.array_split(arr,4)

%%time

zs = []
for i in range(4):
    z = lst_decode(splits[i])
    zs.append(z)

zs = dask.persist(*zs)  # trigger computation in the background
# Wall time: 843 ms   (Wall time: 2.45 s without @dask.delayed)

%timeit decode(arr)
#1 loop, best of 3: 740 ms per loop
```
