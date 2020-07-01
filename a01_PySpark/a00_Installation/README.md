

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

```
- Go to new terminal tab (base of conda, not spk env)
- Open a jupyter notebook, select spark environment and run a test example.

# Installation for Colab
- Download and mount the required folder of spark-hadoop
```bash
%%bash
# We dont need to downloa spark-hadoop binary, pip install works fine.
!pip install pyspark
!pip install koalas
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
import pyspark
spark = pyspark.sql.SparkSession.builder.appName('myApp').getOrCreate()
sdf = spark.createDataFrame([("a", 1), ("b", 3)],
                            ["C1", "C2"]
                           )

sdf.show()
```
