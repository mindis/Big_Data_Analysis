#!python
# -*- coding: utf-8 -*-#
"""
* File Name : test_pyspark.py

* Purpose : Test the installation of pyspark

* Creation Date : Jun 30, 2020 Tue

* Last Modified : Tue Jun 30 18:37:37 2020

* Created By :  Bhishan Poudel

* Usage:

conda env list # make sure you have spk environment
source activate spk
which python # make sure your python is spark env python
python test_pyspark.py

"""
# Imports
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf # @udf("integer") def myfunc(x,y): return x - y
from pyspark.sql import functions as F # stddev format_number date_format, dayofyear, when
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

print([(x.__name__,x.__version__) for x in [np, pd, pyspark]])

# setup pyspark
spark = pyspark.sql.SparkSession.builder.appName('bhishan').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc) # spark_df = sqlContext.createDataFrame(pandas_df)
sc.setLogLevel("INFO")

# data
data = sqlContext.createDataFrame([("Alberto", 2), ("Dakota", 2)],
                                  ["Name", "myage"])

# using selectExpr
df = data.selectExpr("Name as name", "myage as age")
print('\n\n') # spark prints various loggin info, add some new lines.
print(df)
print('\n\n')

# close the session
spark.stop()
print("Congratulations! Your pyspark installation is successful.")
