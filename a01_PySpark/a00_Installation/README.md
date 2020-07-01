Table of Contents
=================
   * [Installation for MacOS 10.15 Catalina (July 2020)](#installation-for-macos-1015-catalina-july-2020)
      * [Install Java in MacOS](#install-java-in-macos)
      * [Install spark-hadoop](#install-spark-hadoop)
   * [Installation for Colab](#installation-for-colab)
   * [Test installation script](#test-installation-script)
   * [Test installation notebook](#test-installation-notebook)

# Installation for MacOS 10.15 Catalina (July 2020)
- Install java8
- Install spark-hadoop
- References
  + https://gi

## Install Java in MacOS
```bash
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk9

# test java home
echo $JAVA_HOME
/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home

ls /Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
java -version
```

## Install spark-hadoop
- Download the binary and put it in ~/Softwares
  + http://spark.apache.org/downloads.html
  + For July 2020 this is spark3.0.0 and hadoop2.7
- Then add the path in bashrc

```bash
## Date: June 30, 2020 for pyspark
## for Java (required by pyspark) June 30, 2020
export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
export JAVA_HOME="$(/usr/libexec/java_home -v 1.8)"
export PATH=$SPARK_HOME/bin:$PATH
export PYSPARK_PYTHON=/Users/poudel/miniconda3/envs/spk/bin/python
```

- Source the updated bashrc
```bash
source ~/.bash_profile
```

- If you already have old spk environment, then remove it
```bash
conda env list
conda remove --name spk --all
conda env list # make sure there is no spk
```

- Create new miniconda env called spk
```bash
conda create -n spk python=3.7
source activate spk
conda install ipykernel
python -m ipykernel install --user --name spk --display-name "Spark3.0.0"
conda install -n spk -c conda-forge autopep8  yapf black

/Users/poudel/opt/miniconda3/envs/spk/bin/pip install py4j
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install pyspark
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install pandasql


conda install -n spk -c conda-forge scikit-learn
conda install -n spk -c conda-forge pandas pandasql pandas-profiling
conda install -n spk -c conda-forge dask

```
- Go to new terminal tab (base of conda, not spk env)
- Open a jupyter notebook, select spark environment and run a test example.

# Installation for Colab
- Download and mount the required folder of spark-hadoop
```
%%bash
# We dont need to downloa spark-hadoop binary, pip install works fine.
!pip install pyspark
!pip install koalas
```


# Test installation script
- Note: Never forget to stop the spark session in the end.
```python
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

```

# Test installation notebook
```python
%%capture
# capture will not print in notebook

import os
import sys
ENV_COLAB = 'google.colab' in sys.modules

if ENV_COLAB:
    !pip install pyspark
    !pip install koalas

    #### print
    print('Environment: Google Colaboratory.')

# NOTE: If we update modules in gcolab, we need to restart runtime.

#=================== next cell ========================
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
spark = pyspark.sql.SparkSession.builder.appName('myApp').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc) # spark_df = sqlContext.createDataFrame(pandas_df)
sc.setLogLevel("INFO")

# data
data = sqlContext.createDataFrame([("Alberto", 2), ("Dakota", 2)],
                                  ["Name", "myage"])

# using selectExpr
df = data.selectExpr("Name as name", "myage as age")
df

```
