# imports
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


# groupby transform
```python
df = pd.DataFrame({'category':['a','a','b','b','b'],
                   'value':   [10,12,100,120,130],
                  })

print(df)
schema = StructType([
    StructField('category',StringType(),True),
    StructField('value',IntegerType(),True)
    ])

sdf = sqlContext.createDataFrame(df, schema)
sdf.show()

df['category_mean'] = df.groupby("category")["value"].transform('mean')


#===================== join with another means_df
sdf_means = sdf.groupBy("category").mean("value").alias("means")
sdf3 = sdf.alias("sdf").join(sdf_means,
                      _col("sdf.category") ==  _col("means.category"))
                      
#===================== broadcast data to all partitions to speed up calculation
sdf_means = sdf.groupBy("category").mean("value").alias("means")
sdf2 = sdf.alias("sdf").join(F.broadcast(sdf_means),
                             _col("sdf.category") == _col("means.category"))

#====================== using Window partitionBy
from pyspark.sql.window import Window

window_var = Window().partitionBy('category')
sdf4 = sdf.withColumn('category_mean', F.mean('value').over(window_var))
                      
#====================== sql over partition by
sdf.registerTempTable('sdf')

sdf5 = spark.sql("""
select *, mean(value)
OVER (PARTITION BY category) as category_mean
from sdf
""")

```
