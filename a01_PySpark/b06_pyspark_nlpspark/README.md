# Description
github: 

# Test installation
```python
import sparknlp


spark = sparknlp.start()
data = [
  ("New York is the greatest city in the world", 0),
  ("The beauty of Paris is vast", 1),
  ("The Centre Pompidou is in Paris", 1)
]

sdf = spark.createDataFrame(data, ["text","label"])
sdf.show()
```
