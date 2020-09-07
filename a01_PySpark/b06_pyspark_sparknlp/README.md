# Description
github: https://github.com/JohnSnowLabs/spark-nlp

# Installation
Download pyspark 2.4.4 and put in `~/Softwares/Spark`: https://archive.apache.org/dist/spark/spark-2.4.4/

```bash
conda create -n mysparknlp python=3.6 -y
conda install -n mysparknlp -c johnsnowlabs spark-nlp

source activate mysparknlp
conda install ipykernel
python -m ipykernel install --user --name mysparknlp --display-name "Python36(myspark)"

# DO NOT use ! when installing from terminal
# this takes about an hour to install, please wait

pip install --ignore-installed --default-timeout=100  pyspark==2.4.4  
conda install -n mysparknlp -c numpy pandas
```



# Test installation
```python
#=============== setup sparknlp
import os
import sys

sys.path.append("/Users/poudel/opt/miniconda3/envs/mysparknlp/lib/python3.7/site-packages")
os.environ["SPARK_HOME"] = "/Users/poudel/Softwares/Spark/spark-2.4.4-bin-hadoop2.7"
os.environ["PYSPARK_PYTHON"] = "/Users/poudel/opt/miniconda3/envs/mysparknlp/bin/python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "jupyter"
os.environ["PYSPARK_DRIVER_PYTHON_OPTS"] = "notebook"
#================ setup sparknlp end

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
