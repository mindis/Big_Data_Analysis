Table of Contents
=================
   * [Caveats](#caveats)
   * [aggregation](#aggregation)
   * [diff operation](#diff-operation)
   * [filter and average](#filter-and-average)
   * [groupby](#groupby)
   * [join two dataframes in different columns](#join-two-dataframes-in-different-columns)
   * [startswith](#startswith)

# Caveats
- pyspark sdf does not have max() min() like functions, we need to use `agg`.

# Some basics
- https://spark.apache.org/docs/latest/api/python/pyspark.sql.html
```python
# dataframes
df1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
df2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])


# columns
df.withColumn('age2', df.age + 2).collect()
df.withColumnRenamed('age', 'age2').collect()

# sorting
df.sort(df.age.desc()).collect()
df.sort("age", ascending=False).collect()
df.orderBy(df.age.desc()).collect()
df.sort(asc("age")).collect()
df.orderBy(["age", "name"], ascending=[0, 1]).collect()

# properties
df.columns
df.count()
df.distinct().count()
df.describe(['age']).show()

# mixed
df.show(truncate=3)
df.select(df.name, df.age.between(2, 4)).show()
df.select(df.age.cast("string").alias('ages')).collect()
df.sample(withReplacement=True, fraction=0.5, seed=3).count()
df.filter(df.name.contains('o')).collect() # where is alias for filter
df.filter(df.height.isNotNull()).collect()
df[df.name.isin("Bob", "Mike")].collect()
df.select(df.name, F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).show()
df.na.fill({'age': 50, 'name': 'unknown'}).show()
df.select(df.name, F.when(df.age > 3, 1).otherwise(0)).show()
df.select(df.name).orderBy(df.name.desc_nulls_last()) # asc desc desc_nulls_first()
df1.exceptAll(df2).show()
df.drop('age').collect()
df.drop_duplicates(['name', 'height']).show()
df.crosstab(col1, col2)
df.agg({"age": "max"}).collect()
df.agg(F.min(df.age)).collect()

# strings
df.filter(df.name.like('Al%')).collect()
df.filter(df.name.rlike('ice$')).collect()
df.filter(df.name.startswith('Al')).collect()
df.select(df.name.substr(1, 3).alias("col")).collect()
df.filter(df.name.endswith('ice')).collect()

df = spark.createDataFrame([("a", 1), ("b", 2), ("c",  3)], ["Col1", "Col2"])
df.select(df.colRegex("`(Col1)?+.+`")).show()

```

# aggregation
```python
# find max of a column
sdfd.agg({'budget':'max'}).show()
sdfd.agg({'budget':'max'}).collect()[0][0]
```

# apply
```python
from pyspark.sql.functions import pandas_udf, PandasUDFType
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))

@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)  
def normalize(pdf):
    v = pdf.v
    return pdf.assign(v=(v - v.mean()) / v.std())
df.groupby("id").apply(normalize).show()

```

# diff operation
```python
from pyspark.sql.window import Window
window_over_A = Window.partitionBy("A").orderBy("B")

df.withColumn("diff", F.lead("B").over(window_over_A) – df.B).show()
```

# filter and average
```python
sdfp[['price']][sdfp.manufacturer==2].agg(_avg(_col('price'))).show()
sdfp.select(_avg(_when(sdfp['manufacturer']==2, 
                       sdfp['price']))
           ).show()
```

# groupby
```python
df.groupBy().min('age', 'height').show() # min max mean count
df.groupBy("year").pivot("course", ["dotNET", "Java"]).sum("earnings").collect()
```

# intersect
```python
df1 = spark.createDataFrame([(“a”, 1), (“a”, 1), (“b”, 3), (“c”, 4)], [“C1”, “C2”])
df2 = spark.createDataFrame([(“a”, 1), (“a”, 1), (“b”, 3)], [“C1”, “C2”])
df1.intersectAll(df2).sort("C1", "C2").show()
```

# join 
```python
how = inner, cross, outer, full, full_outer, left, left_outer, right, right_outer, left_semi, and left_anti.


df.join(df2, df.name == df2.name, 'outer').select(df.name, df2.height).collect()
df.join(df2, 'name', 'outer').select('name', 'height').collect()
cond = [df.name == df3.name, df.age == df3.age]
df.join(df3, cond, 'outer').select(df.name, df3.age).collect()
df.join(df2, 'name').select(df.name, df2.height).collect()
df.join(df4, ['name', 'age']).select(df.name, df.age).collect()


# special example
(sdfp.selectExpr('code as code_proj',
                 'name as name_proj',
                 'price','manufacturer')
     .alias('A')
.join(
    sdfm.selectExpr('code as code_manu','name as manu')
        .alias('B'),
    _col('A.manufacturer') == _col('B.code_manu')
      )
.show()
)

aliter: Note: select * gives duplicates column names for code and name (POSTGESQL does not give)
spark.sql("""\
select P.code as code_proj, P.name as name_proj, price, 
       manufacturer, M.code as code_manu, M.name as name_manu
from Products P inner join Manufacturers M
on P.manufacturer = M.code
""").show()

```

# over
```python
from pyspark.sql import Window
from pyspark.sql.functions import rank, min

window = Window.partitionBy("name")\
               .orderBy("age")\
               .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.withColumn("rank", rank().over(window))\
  .withColumn("min", min('age').over(window)).show()
```

# unionByName
```python

df1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
df2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col0"])
df1.unionByName(df2).show() # union resolves columns by position as in sql, not by name
```
