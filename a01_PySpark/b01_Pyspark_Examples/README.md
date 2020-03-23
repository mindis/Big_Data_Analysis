Table of Contents
=================
   * [Imports](#imports)
   * [Convert pandas df to spark df](#convert-pandas-df-to-spark-df)
   * [Rename columns](#rename-columns)
   * [Get StringIndexers for multiple columns](#get-stringindexers-for-multiple-columns)
   * [Multiple conditions](#multiple-conditions)

# Imports
```python
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf # @udf("integer") def myfunc(x,y): return x - y
from pyspark.sql import functions as F # stddev format_number date_format, dayofyear, when
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

print([(x.__name__,x.__version__) for x in [np, pd, pyspark]])

spark = pyspark.sql.SparkSession.builder.appName('bhishan').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc) # spark_df = sqlContext.createDataFrame(pandas_df)
sc.setLogLevel("INFO")
```

# Convert pandas df to spark df
```python
df = pd.DataFrame({'PERSONID':0,
                   'LASTNAME':'Doe',
                   'FIRSTNAME':'John',
                   'ADDRESS':'Museumplein',
                   'CITY':'Amsterdam',
                   'RESULT': True,
                  },
                  index=[0])
                  
 print(df)
    PERSONID LASTNAME FIRSTNAME      ADDRESS       CITY  RESULT
0         0      Doe      John  Museumplein  Amsterdam    True

df.dtypes
PERSONID      int64
LASTNAME     object
FIRSTNAME    object
ADDRESS      object
CITY         object
RESULT         bool

#Create PySpark DataFrame Schema
schema = StructType([
    StructField('PERSONID',IntegerType(),True),
    StructField('LASTNAME',StringType(),True),
    StructField('FIRSTNAME',StringType(),True),
    StructField('ADDRESS',StringType(),True),
    StructField('CITY',StringType(),True),
    StructField('RESULT',BooleanType(),True)
    ])



#Create Spark DataFrame from Pandas
sdf = sqlContext.createDataFrame(df, schema)
sdf.show()
```

# Rename columns
```python
data = sqlContext.createDataFrame([("Alberto", 2), ("Dakota", 2)], 
                                  ["Name", "myage"])
# using selectExpr
df = data.selectExpr("Name as name", "myage as age")

# with column renamed
df = data.withColumnRenamed("Name", "name")\
       .withColumnRenamed("myage", "age")
       
# alias
df = data.select(col("Name").alias("name"), col("myage").alias("age"))

# toDF
newcols = ['name','age']
df = data.toDF(*newcols)

# using sql table
sqlContext.registerDataFrameAsTable(data, "myTable")
df = sqlContext.sql("SELECT Name AS name, myage as age from myTable")
```

# Get StringIndexers for multiple columns
```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in mycols ]


pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)

df_r.show()
```

# Multiple conditions
```python
# multiple when
cond1 = F.col("fruit1").isNull() | F.col("fruit2").isNull()
cond2 = F.col("fruit1") == F.col("fruit2")
new_column_1 = (F.when(cond1, 3).when(cond2, 1).otherwise(0))

# expr
myexpr = """IF(fruit1 IS NULL OR fruit2 IS NULL, 3,
            IF(fruit1 = fruit2, 1, 0))"""
new_column_2 = F.expr(myexpr)

# udf
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf

def func(fruit1, fruit2):
    if fruit1 == None or fruit2 == None:
        return 3
    if fruit1 == fruit2:
        return 1
    return 0

func_udf = udf(func, IntegerType())
df2 = df.withColumn('new_column',func_udf(df['fruit1'], df['fruit2']))

# checking dataframe
df = sc.parallelize([
    ("orange", "apple"), ("kiwi", None), (None, "banana"), 
    ("mango", "mango"), (None, None)
]).toDF(["fruit1", "fruit2"])
```
