# Description
- https://docs.microsoft.com/en-us/archive/blogs/cindygross/understanding-wasb-and-hadoop-storage-in-azure

# Example
```python
# Load the libraries
import numpy as np
import pandas as pd
import os
HOME = os.path.expanduser('~')

import findspark
# findspark.init(HOME + "/Softwares/Spark/spark-3.0.0-bin-hadoop2.7")

# We need to use spark 2.4.6 to use lgbm
findspark.init(HOME + "/Softwares/Spark/spark-2.4.6-bin-hadoop2.7")

import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

spark = (pyspark.sql.SparkSession.builder.appName("MyApp")

    # config for microsoft ml spark
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

print(f'pyspark version: {pyspark.__version__}')

#================= Load the data =================================
sdf = spark.read.csv('affairs.csv',inferSchema=True,header=True)
print((sdf.count(),len(sdf.columns)))

print(sdf.printSchema())
sdf.show(5)

from mmlspark.stages import SummarizeData
summary = SummarizeData().transform(sdf)
summary.toPandas()

#==================== Data Preparation =============================
from mmlspark.featurize import CleanMissingData
from pyspark.ml.feature import VectorAssembler

cols = ['rate_marriage', 'yrs_married', 'children', 'religious']
removeNansMedian = (CleanMissingData()
              .setCleaningMode("Median")                                                 
              .setInputCols(cols)
              .setOutputCols(cols))

sdf = removeNansMedian.fit(sdf).transform(sdf)

cols = ['age']
removeNansMean = (CleanMissingData()
              .setCleaningMode("Mean")                                                 
              .setInputCols(cols)
              .setOutputCols(cols))
sdf = removeNansMean.fit(sdf).transform(sdf)

inputCols = ['rate_marriage', 'age', 'yrs_married', 'children', 'religious']
labelCol = "affairs"
assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
sdf = assembler.transform(sdf)


train,test,validation = sdf.select(['features',labelCol]).randomSplit([0.6,0.2,0.2],seed=SEED)

#====================== Modelling =============================
# Logistic Regression
from mmlspark.train import TrainClassifier
from pyspark.ml.classification import LogisticRegression

model = TrainClassifier(model=LogisticRegression(), labelCol=labelCol, numFeatures=256)
model_name = "logreg.pkl"
model.write().overwrite().save(model_name)

model = TrainClassifier.load(model_name)
model = model.fit(train)

from mmlspark.train import ComputeModelStatistics, TrainedClassifierModel

prediction = model.transform(test)
metrics = ComputeModelStatistics().transform(prediction)
df_metrics = metrics.toPandas()
df_metrics

# Different models
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from mmlspark.train import TrainClassifier
import itertools

lrHyperParams       = [0.05, 0.2]
logisticRegressions = [LogisticRegression(regParam = hyperParam)
                       for hyperParam in lrHyperParams]
lrmodels            = [TrainClassifier(model=lrm, labelCol=labelCol).fit(train)
                       for lrm in logisticRegressions]

rfHyperParams       = itertools.product([5, 10], [2, 3]) # [(5,2),(5,3) etc]
randomForests       = [RandomForestClassifier(numTrees=hyperParam[0], maxDepth=hyperParam[1])
                       for hyperParam in rfHyperParams]
rfmodels            = [TrainClassifier(model=rfm, labelCol=labelCol).fit(train)
                       for rfm in randomForests]

gbtHyperParams      = itertools.product([8, 16], [2, 3])
gbtclassifiers      = [GBTClassifier(maxBins=hyperParam[0], maxDepth=hyperParam[1])
                       for hyperParam in gbtHyperParams]
gbtmodels           = [TrainClassifier(model=gbt, labelCol=labelCol).fit(train)
                       for gbt in gbtclassifiers]

trainedModels       = lrmodels + rfmodels + gbtmodels

from mmlspark.automl import FindBestModel
bestModel = FindBestModel(evaluationMetric="AUC", models=trainedModels).fit(test)

display(bestModel.getEvaluationResults().limit(5).toPandas())
display(bestModel.getBestModelMetrics().toPandas())
display(bestModel.getAllModelMetrics().toPandas())

from mmlspark.train import ComputeModelStatistics

predictions = bestModel.transform(validation)
metrics = ComputeModelStatistics().transform(predictions)
print("Best model's accuracy on validation set = "
      + "{0:.2f}%".format(metrics.first()["accuracy"] * 100))
print("Best model's AUC on validation set = "
      + "{0:.2f}%".format(metrics.first()["AUC"] * 100))
      
df_metrics = metrics.toPandas()
cm = df_metrics['confusion_matrix'][0]
print(cm)
df_cm = pd.DataFrame(cm.toArray())
df_cm
```
