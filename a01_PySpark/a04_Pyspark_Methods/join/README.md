# Imports
```python
import numpy as np
import pandas as pd
from pyspark.sql.types import *

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

# Join two dataframes
```python
sdf1 = spark.createDataFrame([
    (1,2,3,4),(10,20,30,40),(100,200,300,400),                
    ], ("col1","col2","col3","col4"))

sdf1.show()
sdf2 = spark.createDataFrame([
    (5,6,7,8),(50,60,70,80),(500,600,700,800),              
    ], ("col5","col6","col7","col8"))
sdf2.show()
sdf1 = sdf1.withColumn("id", F.monotonically_increasing_id())
sdf1.show() # first id is 8589934592 (all ids are same for both dataframes)
sdf2 = sdf2.withColumn("id", F.monotonically_increasing_id())
sdf2.show()
sdf = sdf1.join(sdf2, "id", "inner").drop("id")
sdf.show()
+----+----+----+----+----+----+----+----+
|col1|col2|col3|col4|col5|col6|col7|col8|
+----+----+----+----+----+----+----+----+
|   1|   2|   3|   4|   5|   6|   7|   8|
| 100| 200| 300| 400| 500| 600| 700| 800|
|  10|  20|  30|  40|  50|  60|  70|  80|
+----+----+----+----+----+----+----+----+
Note: rows order are chaged here. (sdf1.col1 has 1,10,100 not 1,100,10)

Note: row orders might chage when joining two dataframes.
```

# join two dataframes at different columns
- Just rename the column and make same column name
- pdf.merge is sdf.join
```python
sProduct.select(['CategoryId'])\
.join(sCategory.withColumnRenamed('Id','CategoryId')\
               .select('CategoryId','CategoryName'),
       on='CategoryId')\
.groupby('CategoryName')\
.count()\
.orderBy('count',ascending=False)\
.show()


equivalent in pandas:
Product[['CategoryId']]\
.merge(Category[['Id','CategoryName']],
       left_on='CategoryId',right_on='Id')\
['CategoryName'].value_counts(ascending=False)
```
