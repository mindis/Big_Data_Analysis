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
- NULL and NAN are different things.
```python
Good: spark.sql("select * from MovieTheaters where isnan(Movie)").show()
Good: spark.sql("""select * from MovieTheaters where Movie = 'NaN' """).show()

Bad: sdft.filter(sdft.Movie.isNull()).show() # fails, gives empty result
Bad: sdft.where(F.isnull('Movie')).show() # fails, gives empty result
Bad: spark.sql("select * from MovieTheaters where Movie is null").show()
```

# Some basics
- https://spark.apache.org/docs/latest/api/python/pyspark.sql.html
```python
# dataframes
sdf1 = spark.createDataFrame(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), ("b",  3), ("c", 4)], ["C1", "C2"])
sdf2 = spark.createDataFrame([("a", 1), ("b", 3)], ["C1", "C2"])

# show and collect
sdf.show()
sdf.collect()
sdf.collect()[0][0]

# columns
sdf.withColumn('age2', df.age + 2)
sdf.withColumnRenamed('age', 'age2')

# sorting
sdf.sort(df.age.desc())
sdf.sort("age", ascending=False)
sdf.orderBy(df.age.desc())
sdf.sort(asc("age"))
sdf.orderBy(["age", "name"], ascending=[0, 1])

# properties
sdf.columns
sdf.count()
sdf.distinct().count()
sdf.describe(['age'])

# mixed
sdf1.exceptAll(sdf2)
sdf.show(truncate=3)
sdf.select(sdf.name, sdf.age.between(2, 4))
sdf.select(sdf.age.cast("string").alias('ages'))
sdf.sample(withReplacement=True, fraction=0.5, seed=3).count()
sdf.filter(sdf.name.contains('o')) # 'where' is alias for 'filter'
sdf.filter(sdf.height.isNotNull()) # sdf.where(F.isnull('height')).show()
sdf[sdf.name.isin("Bob", "Mike")]
sdf.select(sdf.name, F.when(sdf.age > 4, 1).when(sdf.age < 3, -1).otherwise(0))
sdf.na.fill({'age': 50, 'name': 'unknown'})
sdf.select(sdf.name, F.when(sdf.age > 3, 1).otherwise(0))
sdf.select(sdf.name).orderBy(sdf.name.desc_nulls_last()) # asc desc desc_nulls_first()
sdf.drop('age')
sdf.drop_duplicates(['name', 'height'])
sdf.crosstab(col1, col2)
sdf.agg({"age": "max"})
sdf.agg(F.min(sdf.age))

# strings
sdf.filter(sdf.name.like('Al%'))
sdf.filter(sdf.name.rlike('ice$'))
sdf.filter(sdf.name.startswith('Al'))
sdf.select(sdf.name.substr(1, 3).alias("col"))
sdf.filter(sdf.name.endswith('ice'))

df = spark.createDataFrame([("a", 1), ("b", 2), ("c",  3)], ["Col1", "Col2"])
sdf.select(sdf.colRegex("`(Col1)?+.+`"))

```

# aggregation
```python
# find max of a column
sdf.agg({'budget':'max'})
sdf.agg({'budget':'max'}).collect()[0][0]
```

# apply
```python
from pyspark.sql.functions import pandas_udf, PandasUDFType
sdf = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))

@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)  
def normalize(pdf):
    v = pdf.v
    return pdf.assign(v=(v - v.mean()) / v.std())
sdf.groupby("id").apply(normalize).show()

```

# diff operation
```python
from pyspark.sql.window import Window
window_over_A = Window.partitionBy("A").orderBy("B")

sdf.withColumn("diff", F.lead("B").over(window_over_A) – df.B).show()
```

# filter and average
```python
sdf[['price']][sdf.manufacturer==2].agg(_avg(_col('price'))).show()
sdf.select(_avg(_when(sdf['manufacturer']==2, 
                       sdf['price']))
           ).show()
```

# groupby
```python
sdf.groupBy().min('age', 'height').show() # min max mean count
sdf.groupBy("year").pivot("course", ["dotNET", "Java"]).sum("earnings").collect()
```

# intersect
```python
sdf1 = spark.createDataFrame([(“a”, 1), (“a”, 1), (“b”, 3), (“c”, 4)], [“C1”, “C2”])
sdf2 = spark.createDataFrame([(“a”, 1), (“a”, 1), (“b”, 3)], [“C1”, “C2”])
sdf1.intersectAll(sdf2).sort("C1", "C2").show()
```

# join 
```python
how = inner, cross, outer, full, full_outer, left, left_outer, right, right_outer, left_semi, and left_anti.


sdf.join(sdf2, sdf.name == sdf2.name, 'outer').select(sdf.name, sdf2.height)

sdf.join(sdf2, 'name', 'outer').select('name', 'height')

cond = [sdf.name == sdf3.name, sdf.age == df3.age]
sdf.join(sdf3, cond, 'outer').select(sdf.name, df3.age)

sdf.join(sdf2, 'name').select(sdf.name, sdf2.height)
sdf.join(sdf4, ['name', 'age']).select(sdf.name, sdf.age)


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

sdf.withColumn("rank", rank().over(window))\
  .withColumn("min", min('age').over(window)).show()
```

# unionByName
```python

sdf1 = spark.createDataFrame([[1, 2, 3]], ["col0", "col1", "col2"])
sdf2 = spark.createDataFrame([[4, 5, 6]], ["col1", "col2", "col0"])
sdf1.unionByName(sdf2).show() # union resolves columns by position as in sql, not by name
```

# when otherwise
```python
m = sdf.select(F.mean('price')).collect()[0][0]
sdf.withColumn('discounted_price',
                F.when( F.col('price') > m,
                        F.col('price')*0.8
                      )
                .otherwise(F.col('price'))
               ).show()
```
