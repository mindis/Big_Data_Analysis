Table of Contents
=================
   * [installation for gcolab](#installation-for-gcolab)
   * [Installation for simplici](#installation-for-simplici)
   * [Installation for MacOS Mojave](#installation-for-macos-mojave)
   * [Test installation](#test-installation)
   * [Testing](#testing)


# installation for gcolab
Make sure you have downloaded the correcty spark-2.4.3 file in Google drive and mounted the drive.
```
%%bash

export SPARK_PATH="drive/My Drive/Colab Notebooks/spark-2.4.3-bin-hadoop2.7"
export PYSPARK_DRIVER_PYTHON="jupyter" 
export PYSPARK_DRIVER_PYTHON_OPTS="notebook" 

!pip install pyspark
!pip install koalas
```

# Installation for simplici
- download latest version of apache spark
http://spark.apache.org/downloads.html

- Move this tar file to ~/Softwares and untar it.
- Add path to your .bash_profile
```bash
export SPARK_PATH=~/Softwares/spark-2.4.3-bin-hadoop2.7
export PYSPARK_DRIVER_PYTHON="jupyter" 
export PYSPARK_DRIVER_PYTHON_OPTS="notebook" 

#For python 3, You have to add the line below or you will get an error
# export PYSPARK_PYTHON=python3
alias snotebook='$SPARK_PATH/bin/pyspark --master local[2]'
```
- I already had Java installed (so it works without installing hadoop)
- Install the python module pyspark
```python
# for enthought canopy environment called deeplr (computer pisces)
$ alias deeplr
alias deeplr='edm shell --environment deeplr'

$ which pip
/Users/poudel/Library/Enthought/Canopy/edm/envs/deeplr/bin/pip

/Users/poudel/Library/Enthought/Canopy/edm/envs/deeplr/bin/pip install pyspark

# Optional install hadoop
https://hadoop.apache.org/releases.html
export HADOOP_HOME=~/Softwares/hadoop-3.2.0
```

# Installation for MacOS Mojave
In my mac I had to install java8 and spark-hadoop

https://github.com/jupyter/jupyter/issues/248
https://stackoverflow.com/questions/54566362/how-to-install-java-9-10-on-mac-with-homebrew/54566494

```bash
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk9

# test java home
echo $JAVA_HOME
/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
```


Bash profile settings:
```bash
## Date: Aug 27, 2019 for pyspark
export SPARK_PATH=~/Softwares/spark-2.4.3-bin-hadoop2.7
export PYSPARK_DRIVER_PYTHON="jupyter"
export PYSPARK_DRIVER_PYTHON_OPTS="notebook"

alias snotebook='$SPARK_PATH/bin/pyspark --master local[2]'
 
# for hadoop
export HADOOP_HOME=~/Softwares/hadoop-3.2.0
export JAVA_HOME=/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
export JAVA_HOME="$(/usr/libexec/java_home -v 1.8)"
export PATH=$SPARK_HOME/bin:$PATH
export PYSPARK_PYTHON=/Users/poudel/miniconda3/envs/spk/bin/python
```

Then
`source ~/.bash_profile`


New environment for pyspark:
```bash
conda create -n spk python=3
source activate spk
conda install ipykernel
python -m ipykernel install --user --name spk --display-name "Python (spk)"
conda install -n spk -c conda-forge autopep8  yapf
conda install -n spk -c conda-forge pyspark

conda install -n spk -c conda-forge scikit-learn # installs scipy
conda install -n spk -c conda-forge pandas
conda install -n spk -c conda-forge dask # installs bokeh
conda install -n spk -c conda-forge pandas pandasql pandas-profiling
conda install -n spk -c conda-forge nltk
conda install -n spk -c conda-forge gensim
conda install -n spk -c conda-forge spacy
```

# Test installation
https://www.tutorialspoint.com/pyspark/pyspark_sparkcontext.htm

```python
import random
from pyspark import SparkContext

sc = SparkContext("local", "app1")
NUM_SAMPLES = 100000000

def inside(p):
    x, y = random.random(), random.random()
    return x*x + y*y < 1


count = sc.parallelize(range(0, NUM_SAMPLES)).filter(inside).count()
pi = 4 * count / NUM_SAMPLES
print(pi)


### Example 2
# create data
with open('hello.txt','w') as fo:
    fo.write('there is an apple\n')
    fo.write('banana\n')
    fo.write('an orange\n')

# imports
from pyspark import SparkContext

# run this cell only once
sc = SparkContext("local", "First App")

# now to some work
logFile = 'hello.txt'
logData = sc.textFile(logFile).cache()
numAs = logData.filter(lambda s: 'a' in s).count()
numBs = logData.filter(lambda s: 'b' in s).count()

print ("Lines with a: %i, lines with b: %i" % (numAs, numBs))
# Lines with a: 3, lines with b: 1
```

# Testing
```python
# pyspark
import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf # @udf("integer") def myfunc(x,y): return x - y
from pyspark.sql import functions as F # stddev format_number date_format, dayofyear, when
from pyspark.sql.types import StructField, StringType, IntegerType, StructType

print([(x.__name__,x.__version__) for x in [np, pd, pyspark]])

spark = pyspark.sql.SparkSession.builder.appName('bhishan').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc) # spark_df = sqlContext.createDataFrame(pandas_df)
sc.setLogLevel("INFO")
data = sqlContext.createDataFrame([("Alberto", 2), ("Dakota", 2)], 
                                  ["Name", "myage"])
# using selectExpr
df = data.selectExpr("Name as name", "myage as age")

# with column renamed
df = data.withColumnRenamed("Name", "name")\
       .withColumnRenamed("myage", "age")
       
# alias
df = data.select(col("Name").alias("name"), col("myage").alias("age"))

# toDF
newcols = ['name','age']
df = data.toDF(*newcols)

# using sql table
sqlContext.registerDataFrameAsTable(data, "myTable")
df = sqlContext.sql("SELECT Name AS name, myage as age from myTable")

# koalas
import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession

s = ks.Series([1, 3, 5, np.nan, 6, 8])
s
```
