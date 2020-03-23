# Create schema
- pyspark schmema
```
From:
code       int64
name      object
budget     int64

To:
schema = StructType([
    StructField('code',IntegerType(),True)
    StructField('name',StringType(),True)
    StructField('budget',IntegerType(),True)
    ])

sdf = sqlContext.createDataFrame(df, schema)
sdf.show()

```
