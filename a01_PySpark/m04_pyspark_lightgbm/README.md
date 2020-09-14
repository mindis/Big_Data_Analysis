# Lightbgm using pyspark
Github link: https://github.com/Azure/mmlspark

# Example
```python
# Load the libraries
import numpy as np
import pandas as pd
import os
HOME = os.path.expanduser('~')

import findspark
# findspark.init(HOME + "/Softwares/Spark/spark-3.0.0-bin-hadoop2.7")

# I need to use spark 2.4.6 to use lgbm
findspark.init(HOME + "/Softwares/Spark/spark-2.4.6-bin-hadoop2.7")

import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

spark = (pyspark.sql.SparkSession.builder.appName("MyApp")
         
    # config for mmlspark
    .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc2") 
    .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
         
    # usual
    .getOrCreate()
    )
import mmlspark
SEED = 100

df_eval = pd.DataFrame({
    "Model": [],
    "Description": [],
    "Accuracy": [],
    "Precision": [],
    "AUC": []
})

from pyspark.ml.feature import VectorAssembler

from pyspark.ml.classification import RandomForestClassifier
from mmlspark.lightgbm import LightGBMClassifier

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#=====================================================================
# Load the data
sdf = spark.read.csv('affairs.csv',inferSchema=True,header=True)
print((sdf.count(),len(sdf.columns)))

print(sdf.printSchema())
sdf.show(5)

#=====================================================================
# Data Preparation
from pyspark.ml.feature import VectorAssembler

inputCols = ['rate_marriage', 'age', 'yrs_married', 'children', 'religious']
assembler = VectorAssembler(inputCols=inputCols, outputCol="features")

train,test = sdf.select(['features','affairs']).randomSplit([0.75,0.25],seed=SEED)
sdf = assembler.transform(sdf)

#========================= Modelling ==================================
# Random Forest
from pyspark.ml.classification import RandomForestClassifier

model = RandomForestClassifier(labelCol='affairs',seed=SEED)

# model.save('lgb.pkl')
# model = LightGBMClassifier.load('lgb.pkl')

model = model.fit(train)
test_preds = model.transform(test)

acc = MulticlassClassificationEvaluator(
    labelCol='affairs',
    metricName='accuracy'
    ).evaluate(test_preds)

precision = MulticlassClassificationEvaluator(
    labelCol='affairs',
    metricName='weightedPrecision'
    ).evaluate(test_preds)

auc = BinaryClassificationEvaluator(
    labelCol='affairs'
   ).evaluate(test_preds)


row = ["rf",'default',acc,precision,auc]

df_eval.loc[len(df_eval)] = row
df_eval = df_eval.drop_duplicates()
df_eval

#==================== Lightgbm
from mmlspark.lightgbm import LightGBMClassifier

model = LightGBMClassifier(labelCol='affairs')

# model.save('lgb.pkl')
# model = LightGBMClassifier.load('lgb.pkl')

model = model.fit(train)
test_preds = model.transform(test)

acc = MulticlassClassificationEvaluator(
    labelCol='affairs',
    metricName='accuracy'
    ).evaluate(test_preds)

precision = MulticlassClassificationEvaluator(
    labelCol='affairs',
    metricName='weightedPrecision'
    ).evaluate(test_preds)

auc = BinaryClassificationEvaluator(
    labelCol='affairs'
   ).evaluate(test_preds)


row = ["lgb",'default',acc,precision,auc]
df_eval.loc[len(df_eval)] = row
df_eval = df_eval.drop_duplicates()
df_eval
```
