Table of Contents
=================
   * [Useful Resources](#useful-resources)
   * [PySpark Introduction](#pyspark-introduction)
   * [Pyspark Imports](#pyspark-imports)
   * [Date Time Manipulation](#date-time-manipulation)
   * [Important Functions](#important-functions)
   * [Useful Alfred Commands](#useful-alfred-commands)

# Useful Resources
- [pyspark.sql module](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html)
- [Koalas: pandas API on Apache Spark](https://koalas.readthedocs.io/en/latest/)
- [Join and Aggregate PySpark DataFrames](https://hackersandslackers.com/join-aggregate-pyspark-dataframes/)
- [Analytics Vidhya: Complete Guide on DataFrame Operations in PySpark](https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/)
- [Medium post github notebook: 6 Differences Between Pandas And Spark DataFrames](https://github.com/christophebourguignat/notebooks/blob/master/Spark-Pandas-Differences.ipynb)
- [From Pandas to Apache Sparks Dataframe](https://ogirardot.wordpress.com/2015/07/31/from-pandas-to-apache-sparks-dataframe/)
- [Github: Learning Apache Spark](https://github.com/runawayhorse001/LearningApacheSpark)

# PySpark Introduction
> Apache Spark is a lightning fast real-time processing framework.
  It does in-memory computations to analyze data in real-time. 
  It came into picture as Apache Hadoop MapReduce was performing batch processing
  only and lacked a real-time processing feature.
  Hence, Apache Spark was introduced as it can perform stream processing in
  real-time and can also take care of batch processing.

> Apart from real-time and batch processing, Apache Spark supports interactive queries and iterative algorithms also.
  Apache Spark has its own cluster manager, where it can host its application.
  It leverages Apache Hadoop for both storage and processing.
  It uses HDFS (Hadoop Distributed File system) for storage and it can run Spark applications on YARN as well.

> Apache Spark is written in Scala programming language.
  To support Python with Spark, Apache Spark Community released a tool, PySpark.
  Using PySpark, you can work with RDDs in Python programming language also.
  It is because of a library called Py4j that they are able to achieve this.

> PySpark offers PySpark Shell which links the Python API to the spark core and initializes the Spark context.
Majority of data scientists and analytics experts today use Python because of its rich library set.
Integrating Python with Spark is a boon to them.

# Pyspark Imports
```python
# python
import numpy as np
import pandas as pd

# pyspark
import pyspark
spark = pyspark.sql.SparkSession.builder.appName('bhishan').getOrCreate()
print([(x.__name__,x.__version__) for x in [np, pd, pyspark]])

# sql
from pyspark.sql.functions import col
from pyspark.sql.functions import udf # @udf("integer") def myfunc(x,y): return x - y
from pyspark.sql import functions as F # stddev format_number date_format, dayofyear, when
from pyspark.sql.types import StructField, StringType, IntegerType,  FloatType, StructType

# ml feature
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.feature import OneHotEncoder,OneHotEncoderEstimator
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

# classifiers
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

# cross validation
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import CrossValidatorModel

# model evaluation regression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

# model evaluation classification
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import ClusteringEvaluator
```

# Basics
```python
1. distinct
sdf['col'].drop_duplicates() does not work
sdf.select('col').distinct().show()

2. mean
sdf.select('a').mean() does not work
sdf.select(F.mean('a')).show()

3. groupby mean
pandas : df.groupby('a')['b'].mean()
pandas : df.groupby('a').agg(mean_b=('b','mean')) # Named Aggregation
pyspark: sdf.groupby('a').agg({'b':'mean'}).show()

4. groupby mean then select
pd : df.groupby('a')['b'].mean()[lambda x: x>150]
spk: sdf.groupby('a').agg({'b':'mean'}).filter('avg(b)>150').show()
```

# Date Time Manipulation
```python

#--------------------------------------------------
# Create Date and Hour columns from Seconds column
from pyspark.sql.functions import expr, hour

df_time_hour = (df.select('Time').withColumn('Date',
    expr("timestamp(unix_timestamp('2019-01-01 00:00:00') + Time)"))
  .withColumn('hour', hour('Date')))
 
df_time_hour.show(n=5,truncate=False)
  
grp = df_time_hour.groupBy('hour').count().sort('hour',ascending=False)
grp.toPandas().plot.bar()
```

# Important Functions
```python
def spark_df_from_pandas(pandas_df):
    df_dtype = pandas_df.dtypes.astype(str).reset_index()
    df_dtype.columns = ['column','dtype']

    mapping = {'int64'  : 'IntegerType()',
               'float64': 'DoubleType()',
               'bool'   : 'BooleanType()',
               'object' : 'StringType()',
               'datetime64[ns]': 'DateType()',
              }

    df_dtype['dtype'] = df_dtype['dtype'].map(mapping)
    df_dtype['schema'] = "    StructField('" +\
                         df_dtype['column'] + "'," +\
                         df_dtype['dtype'] + ",True),"

    head = 'StructType([\n'
    body = '\n'.join(df_dtype['schema'])
    tail = '\n    ])'

    schema = head + body + tail
    spark_df = sqlContext.createDataFrame(pandas_df, eval(schema))
    return spark_df
```

# Useful Alfred Commands
```
From   : Employee = pd.read_sql("""SELECT * from Employee""", conn)
To     : sEmployee = spark_df_from_pandas(Employee)
Command: fr (.*)(\s+=)(.*) s\1 = spark_df_from_pandas(\1)\ns\1.createOrReplaceTempView("\1")\n

Note: first go to empty cell in jupyter notebook, paste command in Alfred, then copy Employee data and run last alfred command.

```
