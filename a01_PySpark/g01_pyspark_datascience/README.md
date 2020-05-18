# Cross validation
```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="label",
                                             metricName="accuracy")


# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.1, 0.3, 0.5]) # regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2]) # Elastic Net Parameter (Ridge = 0)
             .build())

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, \
                    estimatorParamMaps=paramGrid, \
                    evaluator=evaluator, \
                    numFolds=5)

cvModel = cv.fit(train)

predictions = cvModel.transform(test)

# Evaluate best model
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)
```
