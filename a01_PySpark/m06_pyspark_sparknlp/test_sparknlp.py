import sys

sys.path.append("/Users/poudel/opt/miniconda3/envs/mysparknlp/lib/python3.7/site-packages")

import sparknlp


spark = sparknlp.start()
data = [
  ("New York is the greatest city in the world", 0),
  ("The beauty of Paris is vast", 1),
  ("The Centre Pompidou is in Paris", 1)
]

sdf = spark.createDataFrame(data, ["text","label"])
print(sdf.show())
