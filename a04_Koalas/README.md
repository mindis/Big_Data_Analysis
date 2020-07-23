# Resources
- https://databricks.com/blog/2020/03/31/10-minutes-from-pandas-to-koalas-on-apache-spark.html

# NOTES
- koalas dataframe does not support koalas series. (It needs list/array/series).
- We can use `ks.sql("select * from {kdf}")` instead of `spark.sql("select * from myTable").show()`
- By default we can not do operations in two kdfs. We need to set options.
```python
from databricks.koalas import option_context

with option_context(
        "compute.ops_on_diff_frames", True,
        "compute.default_index_type", 'distributed'):
    df = ks.range(10) + ks.range(10)
```


# Performances
- Converting `sdf -> kdf` has overhead since sdf do not have indices. 
  Use `sdf.to_koalas(index_col='A')`
