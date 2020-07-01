

# Installation for MacOS 10.15 Catalina (July 2020)
We need to install java. We dont need to download spark-hadoop file
and put in the PATH. pip install pyspark will take care it itself.


## Install Java in MacOS
```bash

# First check if you already have java or not
java -version

# if we do not have java, install it.
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk9

# test java home
echo $JAVA_HOME
/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home

ls /Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
java -version

source ~/.bash_profile
```

## Create new conda environment
- If you already have old spk environment, then remove it
```bash
conda env list
conda remove --name spk --all
conda env list # make sure there is no spk
```

- Create new miniconda env called spk
```bash
# Create new env 
conda create -n spk python=3.7
source activate spk

# adding new kernel to ipython
# Dont use sudo unless required.
conda install ipykernel  # NOTE there are TWO spk below
python -m ipykernel install --user --name spk --display-name "Spark3"

# install required packages
conda install -n spk -c conda-forge autopep8  yapf black # needed for jupyter linting


# use pip for pyspark
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install py4j
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install pyspark
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install pandasql
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install natsort
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install 
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install 
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install 
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install 
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install 
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install 


# Install in given order
conda install -n spk -c conda-forge numpy scipy matplotlib  seaborn
conda install -n spk -c conda-forge openpyxl pytables pyarrow  pandas 
conda install -n spk -c conda-forge pandas-datareader  pandas-profiling
conda install -n spk -c conda-forge scikit-learn numba psutil 
conda install -n spk -c conda-forge plotly plotly_express 
conda install -n spk -c conda-forge shap  # shap needs psutil and numba
conda install -n spk -c conda-forge unidecode 
conda install -n spk -c conda-forge hyperopt
conda install -n spk -c conda-forge optuna
conda install -n spk -c conda-forge koalas
conda install -n spk -c conda-forge nose pytest
conda install -n spk -c conda-forge swifter 
conda install -n spk -c conda-forge statsmodels
conda install -n spk -c conda-forge 
conda install -n spk -c conda-forge 

# imblearn needs matplotlib, pandas, requests, pillow, tqdm
 /Users/poudel/opt/miniconda3/envs/spk/bin/pip install imblearn

# special installations
/Users/poudel/opt/miniconda3/envs/spk/bin/pip install --pre nbdime
nbdime config-git --enable --global

# test it
jupyter-notebook  a.ipynb # Kernel choose spk

import pyspark
pyspark.__version__

# to remove
conda info --envs
conda env list
conda remove --name spk --all


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
