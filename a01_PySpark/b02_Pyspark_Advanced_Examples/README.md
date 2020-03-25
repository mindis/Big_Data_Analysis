# Advanced examples
- https://stackoverflow.com/questions/55958922/how-to-find-the-2nd-biggest-value-in-when-using-a-pyspark-window

# String split
```python
df = pd.DataFrame({'id': range(3),
                  'name': ['Albert Einstein', 'Mary Curie',
                          'Will H. Clington']})

df
sdf = sqlContext.createDataFrame(df)
sdf.printSchema()

# method 1
arr=int(sdf
    .select(size(split(col("name"),"\s+")).alias("size"))
    .orderBy(desc("size"))
    .collect()[0][0])
(sdf
 .withColumn('temp', split('name', '\s+'))
 .select("*",*    (coalesce(col('temp').getItem(i),lit(""))
                   .alias('product{}'.format(i+1)) 
                   for i in range(arr)))
 .drop("temp")
 .show()
 )
 
# method 2
(sdf.withColumn("tmp",split(col("name"),"\s+"))
    .withColumn("product1",col("tmp").getItem(0))
    .withColumn("product2",col("tmp").getItem(1))
    .withColumn("product3",coalesce(col("tmp").getItem(2),lit("")))
    .drop("tmp")
    .show()
 
)

# method 3
(sdf.withColumn("tmp",F.split(F.col("name"),"\s+"))
  .withColumn("product1",F.element_at(F.col("tmp"),1))
  .withColumn("product2",F.element_at(F.col("tmp"),2))
  .withColumn("product3",
               F.coalesce(F.element_at(F.col("tmp"),3),
                                           F.lit("")))
 .drop("tmp")
 .show()
)

```
