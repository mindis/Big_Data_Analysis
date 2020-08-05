# agggreate filter
```python
from pyspark.sql.functions import expr
sdf = spark.createDataFrame([(1,[0.2, 2.1, 3., 4., 3., 0.5]),
                             (2,[7., 0.3, 0.3, 8., 2.,])],
                             ['id','column'])
                             
(sdf
.withColumn("column<2",
  expr("""aggregate(filter(column, x -> x < 2), 0D,
  (x, acc) -> acc + x)"""))
 
.withColumn("column>2",
  expr("""aggregate(filter(column, x -> x > 2), 0D,
  (x, acc) -> acc + x)"""))
 
.withColumn("column=2",
  expr("""aggregate(filter(column, x -> x == 2), 0D,
  (x, acc) -> acc + x)"""))

).toPandas()

+---+------------------------------+--------+--------+--------+
|id |column                        |column<2|column>2|column=2|
+---+------------------------------+--------+--------+--------+
|1  |[0.2, 2.0, 3.0, 4.0, 3.0, 0.5]|0.7     |10.0    |2.0     |
|2  |[7.0, 0.3, 0.3, 8.0, 2.0]     |0.6     |15.0    |2.0     |
+---+------------------------------+--------+--------+--------+
```
