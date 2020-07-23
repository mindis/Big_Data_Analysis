# case when
- https://stackoverflow.com/questions/63061941/how-to-use-spark-agg-and-filter-for-this-example/63061997#63061997
```python
sdf.groupBy("breed").agg(F.avg('weight').alias('avg_wt'),
                         F.avg(F.when(F.col('age') > 1,F.col('weight'))).alias('avg_wt_1')
                        ).show()
```
