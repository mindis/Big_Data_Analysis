# Installation
https://vaex.readthedocs.io/en/latest/installing.html
```python
# xx is my conda environment
conda install -n xx -c conda-forge vaex

# Note: we may also need PyQt5
# some modules are not available in conda, install them using pip
/Users/poudel/miniconda3/envs/xx/bin/pip install vaex-ml
/Users/poudel/miniconda3/envs/xx/bin/pip install vaex-graphql


# For machine learning
conda install -n xx -c conda-forge  lightgbm xgboost catboost
```

# Articles
- [Jupyter notebook of Medium Ariticle](https://nbviewer.jupyter.org/github/vaexio/vaex-examples/blob/master/medium-nyc-taxi-data-ml/vaex-taxi-ml-article.ipynb)
- [Towards DS: ML impossible: Train 1 billion samples in 5 minutes on your laptop using Vaex and Scikit-Learn](https://towardsdatascience.com/ml-impossible-train-a-1-billion-sample-model-in-20-minutes-with-vaex-and-scikit-learn-on-your-9e2968e6f385)
