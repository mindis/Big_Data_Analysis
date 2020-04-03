# Databricks notebook source exported at Mon, 19 Sep 2016 14:58:03 UTC
# MAGIC %md
# MAGIC # [Predicting Red Hat Business Value](https://www.kaggle.com/c/predicting-red-hat-business-value)
# MAGIC Hosted by [Kaggle](https://www.kaggle.com/)
# MAGIC 
# MAGIC ## In short we will ...
# MAGIC 
# MAGIC 1. Configurations
# MAGIC    * Mount an *s3 bucket*
# MAGIC    * Set the default *data* & *submission* path
# MAGIC    * Set default number of partitions
# MAGIC 2. Import the competition data sets
# MAGIC   * Download the competition data sets directly
# MAGIC 3. Create an *Estimator*
# MAGIC   * Estimator
# MAGIC   * Model
# MAGIC 4. Preprocess the raw data sets
# MAGIC   * Load *act_train* and *people* csv file as a dataframe
# MAGIC   * Parse the dates into different bins
# MAGIC   * Join *people* and *act_train* dataframes
# MAGIC     * Rename the column names of *people* and *train*
# MAGIC     * Join and cache
# MAGIC   * Create *Training*, *validation* and *Test* sets from the joined dataframe
# MAGIC 5. Make predictions
# MAGIC   * Hyper-parameter optimization
# MAGIC     * Logistic regression
# MAGIC     * Random forest classifier
# MAGIC   * Ensemble
# MAGIC 6. Create submission

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1: Configurations

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1: Mount an s3 bucket
# MAGIC To export the submission dataframe at the end, we need to mount an _[AWS s3](https://aws.amazon.com/s3/) bucket_. For detailed instructions visit the following article by _Databricks_.
# MAGIC 
# MAGIC > [Data Import How-To Guide](https://databricks.com/wp-content/uploads/2015/08/Databricks-how-to-data-import.pdf) by _Databricks_

# COMMAND ----------

import urllib

ACCESS_KEY = ""
SECRET_KEY = ""
ENCODED_SECRET_KEY = urllib.quote(SECRET_KEY, "")
AWS_BUCKET_NAME = ""
MOUNT_NAME = "s3"

# comment out the following line after you have successfully mounted your bucket
# dbutils.fs.mount("s3n://%s:%s@%s" % (ACCESS_KEY, ENCODED_SECRET_KEY, AWS_BUCKET_NAME), "/mnt/%s" % MOUNT_NAME)

display(dbutils.fs.ls("/mnt/%s" % MOUNT_NAME))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2: Set the default *data* & *submission* path
# MAGIC If you prefer to import the competition data sets by pushing them to your s3 bucket first:
# MAGIC  - Make sure to set the *data_root* to point to the directory containing the raw data sets in your s3 bucket

# COMMAND ----------

data_root = '/tmp/redhat_data/'
submission_path = '/mnt/{}/submission.txt'.format(MOUNT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3: Set default number of partitions
# MAGIC 
# MAGIC Following the configuration from [UC Berkeleyx cs120x](https://www.edx.org/course/distributed-machine-learning-apache-uc-berkeleyx-cs120x) lab 3.

# COMMAND ----------

sqlContext.setConf('spark.sql.shuffle.partitions', '6')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Import the competition data sets
# MAGIC 
# MAGIC Since we have mounted an s3 bucket we can simply upload the competition data to the s3 bucket and access them in this notebook. In this case, make sure to update the **data_root** to point to the directory containing the raw competition data sets. (skip section 2.1 if you choose to follow this approach).
# MAGIC 
# MAGIC Alternatively you can download all the competition data directly, and this what section 2.1 is about.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1: Download the competition data sets directly
# MAGIC 
# MAGIC Here we adapt the script posted by [John Ramey](http://ramhiser.com/) and the comment by [Ole Henrik Skogstrøm](https://disqus.com/by/ole_henrik_skogstr_m/) to download all the competition data sets.
# MAGIC 
# MAGIC > [How to Download Kaggle Data with Python and requests.py](http://ramhiser.com/2012/11/23/how-to-download-kaggle-data-with-python-and-requests-dot-py/)
# MAGIC 
# MAGIC * Due to the Requests API [changes](http://www.python-requests.org/en/master/api/#api-changes) we will change the _prefetch_ flag to _stream_

# COMMAND ----------

from requests import get, post
from os import mkdir, remove
from os.path import exists
from shutil import rmtree
import zipfile

def purge_all_downloads(db_full_path):
  # Removes all the downloaded datasets
  if exists(db_full_path): rmtree(db_full_path)

def datasets_are_available_locally(db_full_path, datasets):
  # Returns True only if all the competition datasets are available locally in Databricks CE
  if not exists(db_full_path): return False
  for df in datasets:
    # Assumes all the datasets end with '.csv' extention
    if not exists(db_full_path + df + '.csv'): return False
  return True

def remove_zip_files(db_full_path, datasets):
  for df in datasets:
    remove(db_full_path + df + '.csv.zip')
    
def unzip(db_full_path, datasets):
  for df in datasets:
    with zipfile.ZipFile(db_full_path + df + '.csv.zip', 'r') as zf:
      zf.extractall(db_full_path)
  remove_zip_files(db_full_path, datasets)

def download_datasets(competition, db_full_path, datasets, username, password):
  # Downloads the competition datasets if not availible locally  
  if datasets_are_available_locally(db_full_path, datasets):
    print 'All the competition datasets have been downloaded, extraced and are ready for you !'
    return
  
  purge_all_downloads(db_full_path)
  mkdir(db_full_path)
  kaggle_info = {'UserName': username, 'Password': password}
  
  for df in datasets:
    url = (
      'https://www.kaggle.com/account/login?ReturnUrl=' +
      '/c/' + competition + '/download/'+ df + '.csv.zip'
    )
    request = post(url, data=kaggle_info, stream=True)
    
    # write data to local file
    with open(db_full_path + df + '.csv.zip', "w") as f:
      for chunk in request.iter_content(chunk_size = 512 * 1024):
        if chunk: f.write(chunk)
  
  # extract competition data 
  unzip(db_full_path, datasets)
  print('done !')

# COMMAND ----------

# KAGGLE_USERNAME = ''
# KAGGLE_PASSWORD = ''

# download_datasets(
#   competition='predicting-red-hat-business-value',
#   db_full_path= ('/dbfs' + data_root), # here we need the full path
#   datasets=['act_train', 'act_test', 'sample_submission', 'people'],
#   username=KAGGLE_USERNAME,
#   password=KAGGLE_PASSWORD
# )

# COMMAND ----------

display(dbutils.fs.ls(data_root))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3: Create an Estimator
# MAGIC 
# MAGIC To transform the raw dataframes within a pipeline, here we create a custom estimator that:
# MAGIC 
# MAGIC - Extracts one-hot-encoding (OHE) features from categorical columns
# MAGIC - Adds the continuous columns to the feature vector
# MAGIC - Extracts OHE features from boolean columns
# MAGIC - Creates a sparse feature vector
# MAGIC - Carries over other columns
# MAGIC 
# MAGIC We will use the answer by [zero323](http://stackoverflow.com/users/1560062/zero323) on Stackoverflow in designing a one-hot-encoding __estimator__.
# MAGIC 
# MAGIC > [how to roll a custom estimator in pyspark mllib](http://stackoverflow.com/questions/37270446/how-to-roll-a-custom-estimator-in-pyspark-mllib)
# MAGIC 
# MAGIC The OHE dictionary was inspired by UC Berkeleyx cs120x lab-3.
# MAGIC 
# MAGIC > [UC Berkeleyx cs120x](https://www.edx.org/course/distributed-machine-learning-apache-uc-berkeleyx-cs120x)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1: Estimator
# MAGIC 
# MAGIC The Encoder creates a dictionary mapping (key, value) tuples (where each key is a __column name__ and each values is a __category__) to a unique integer. This dictionary is then used to create a __Model__.

# COMMAND ----------

from pyspark.ml import Estimator
from collections import defaultdict
from pyspark.sql.functions import explode

class Encoder(Estimator):
  # INIT
  _defaultParamMap = {}
  _paramMap = {}
  _params = {}
  
  def __init__(self, label_column_name='label', id_column_name='id', weight_column_name='weight'):
    self.label_column_name = label_column_name
    self.id_column_name = id_column_name
    self.weight_column_name = weight_column_name
  def set_label_col(self, name):
    self.label_column_name = name
  def set_id_col(self, name):
    self.id_column_name = name
  def set_weight_column_name(name):
    self.weight_column_name = name
  
  # HELPERS
  def create_key_value_rdd(self, input_df):
    # returns a key value rdd where each key is the column name and each value is the value of the attribute
    return(
      input_df.rdd
      .map(lambda r: r.asDict())
      .map(lambda h: [(k, v) for k, v in h.iteritems()])
    )
  
  def create_ohe_dict(self, input_rdd):
    # creates a dict to mapping each key value pair to a unique number
    return (
      input_rdd
      .flatMap(lambda l: l)
      .distinct()
      .zipWithIndex()
      .collectAsMap()
    )
  
  def group_columns_by_type(self, schema):
    column_name_type_tuples = map(lambda col: col.simpleString().split(':'), schema)
    # exclude ID, Label, and Weight columns
    feature_columns = [
      col for col in column_name_type_tuples if col[0] not in [self.label_column_name, self.id_column_name, self.weight_column_name]
    ]

    
    boolean_column_names = [col[0] for col in feature_columns if col[1] == 'boolean']
    string_column_names = [col[0] for col in feature_columns if col[1] == 'string']
    integer_column_names = [col[0] for col in feature_columns if col[1] == 'int']
    
    return (string_column_names, boolean_column_names, integer_column_names)
    
  # FITTING MODEL
  def _fit(self, df):
    # Group columns
    string_column_names, boolean_column_names, integer_column_names = self.group_columns_by_type(df.schema)
    
    # Create an OHE dictionary for columns of String type
    kv_rdd = self.create_key_value_rdd(df.select(*string_column_names))
    ohe_dict = self.create_ohe_dict(kv_rdd)
    
    return EncoderModel(
      ohe_dict, self.id_column_name, self.label_column_name, self.weight_column_name,
      boolean_column_names, string_column_names, integer_column_names
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2: Model

# COMMAND ----------

from pyspark.ml import Model
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

class EncoderModel(Model):
  # INIT
  _defaultParamMap = {}
  _paramMap = {}
  _params = {}
  
  def __init__(
    self, one_hot_encoding_dictionary, id_col_name, label_col_name, weight_column_name,
    boolean_column_names, string_column_names, integer_column_names
  ):
    self.one_hot_encoding_dictionary = one_hot_encoding_dictionary
    self.id_column_name = id_col_name
    self.label_column_name = label_col_name
    self.weight_column_name = weight_column_name
    self.boolean_column_names = boolean_column_names
    self.string_column_names = string_column_names
    self.integer_column_names = integer_column_names
    
  # HELPERS
  def create_key_value_rdd(self, input_df):
    # returns a key value rdd where each key is the column name and each value is the value of the attribute
    return(
      input_df.rdd
      .map(lambda r: r.asDict())
      .map(lambda h: [(k, v) for k, v in h.iteritems()])
    )
  
  def gen_feats_rdd(self, input_rdd, ohe_dict_broadcast, has_id, has_weight, has_label):
    # HELPERS
    def get_value(raw_feats, column_name):
      for observation in raw_feats:
        if observation[0] == column_name: return observation[1]
      return None
    def get_values(raw_feats, column_names):
      args = []
      for observation in raw_feats:
        if observation[0] in column_names:
          args.append(observation[1])
      return args
    def get_bool_feats(raw_feats, boolean_column_names, index):
      values = get_values(raw_feats, boolean_column_names)
      return zip(range(index, index + len(values)), map(lambda b: int(b), values))
    def get_int_feats(raw_feats, integer_column_names, index):
      values = get_values(raw_feats, integer_column_names)
      return zip(range(index, index + len(values)), values)
    def get_ohe_feats(raw_feats, ohe_dict_broadcast):
      args = []
      for observation in raw_feats:
        if observation in ohe_dict_broadcast.value:
          index = ohe_dict_broadcast.value[observation]
          args.append((index, 1))
      return args
    
    # FEATURE VECTOR
    boolean_column_names = self.boolean_column_names
    integer_column_names = self.integer_column_names
    def join_feats(raw_feats, ohe_dict_broadcast): 
      ohe_feats = get_ohe_feats(raw_feats, ohe_dict_broadcast)
      next_index = len(ohe_dict_broadcast.value)
      bool_feats = get_bool_feats(raw_feats, boolean_column_names, next_index)
      next_index += len(boolean_column_names)
      int_feats = get_int_feats(raw_feats, integer_column_names, next_index)
      next_index += len(integer_column_names)
      return SparseVector(next_index, ohe_feats + bool_feats + int_feats)
    
    # return the new dataframe
    id_column_name = self.id_column_name
    weight_column_name = self.weight_column_name
    label_column_name = self.label_column_name
    def transform_row(f):
      new_row = []
      if has_id:
        new_row.append(get_value(f, id_column_name))
      if has_weight:
        new_row.append(get_value(f, weight_column_name))
      new_row.append(join_feats(f, ohe_dict_broadcast))
      if has_label:
        new_row.append(get_value(f, label_column_name))
      return new_row
    
    return input_rdd.map(lambda f: transform_row(f))
    
  # TRANSFORM DATAFRAME
  def _transform(self, input_df):
    # creates a ohe dataframe with id, label and features columns
    has_label = self.label_column_name in input_df.schema.names
    has_id = self.id_column_name in input_df.schema.names
    has_weight = self.weight_column_name in input_df.schema.names
    
    ohe_dict_broadcast = sc.broadcast(self.one_hot_encoding_dictionary)
    
    input_kv_rdd = self.create_key_value_rdd(input_df)
    input_ohe_rdd = self.gen_feats_rdd(input_kv_rdd, ohe_dict_broadcast, has_id, has_weight, has_label)
    
    # for some reason I couldn't cast the Neumerical columns in toDF. I had to first create them as StringType columns and then cast them to DoubleType
    fields = []
    if has_id:
      fields.append(StructField("id", StringType(), True))
    if has_weight:
      fields.append(StructField("weight_st", StringType(), True))
    fields.append(StructField("features", VectorUDT(), True))
    if has_label:
      fields.append(StructField("label_st", StringType(), True))

    schema = StructType(fields)
    ohe_df = input_ohe_rdd.toDF(schema)
    
    if has_label:
      ohe_df = ohe_df.withColumn('label', ohe_df['label_st'].cast(DoubleType())).drop("label_st")
    if has_weight:
      ohe_df = ohe_df.withColumn('weight', ohe_df['weight_st'].cast(DoubleType())).drop("weight_st")
    return ohe_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 4: Preprocess the raw data sets

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1: Load *act_train* and *people* csv file as a dataframe

# COMMAND ----------

raw_train_df = (
  sqlContext.read.format('com.databricks.spark.csv')
  .options(header=True, delimiter=',', inferschema=True)
  .load(data_root+'/act_train.csv')
).drop('activity_id')
print raw_train_df.count()
print raw_train_df.printSchema()

raw_people_df = (
  sqlContext.read.format('com.databricks.spark.csv')
  .options(header=True, delimiter=',', inferschema=True)
  .load(data_root+'/people.csv')
)
print raw_people_df.count()
print raw_people_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 4.2: Parse the dates into different bins

# COMMAND ----------

from pyspark.sql.types import StringType

def parse_dates(input_df):
  udf_date_to_day = udf(lambda date: date.day, StringType())
  udf_date_to_month = udf(lambda date: date.month, StringType())
  udf_date_to_weekday = udf(lambda date: date.weekday(), StringType())
  udf_date_to_year = udf(lambda date: date.year, StringType())
  
  return (
    input_df.select(
      '*',
      udf_date_to_day(input_df['date']).alias('d_day'),
      udf_date_to_month(input_df['date']).alias('d_mon'),
      udf_date_to_weekday(input_df['date']).alias('d_wd'),
      udf_date_to_year(input_df['date']).alias('d_yr')
    ).drop('date')
  )

# COMMAND ----------

raw_train_with_time_feat_df = parse_dates(raw_train_df)
raw_people_with_time_feat_df = parse_dates(raw_people_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### K-Means (still looking into it)
# MAGIC 
# MAGIC Here I tried to group users together with KMeans and feed these groups as new categorical features to the optimizer.
# MAGIC 
# MAGIC This is an expensive operation for large Ks and did not improve the AUC score much.
# MAGIC 
# MAGIC If you want to try this out, just uncomment the following lines and comment the code where I assign *people_renamed_df* in section 4.3.1

# COMMAND ----------

# from pyspark.ml.clustering import KMeans
# kmeans = KMeans(maxIter=80)
# people_vec_df = (
#   Encoder(id_column_name='people_id')
#   .fit(raw_people_with_time_feat_df)
#   .transform(raw_people_with_time_feat_df)
#   .withColumnRenamed('id', 'people_id')
# )

# people_acc_df = people_vec_df
# for i in range(2, 4):
#   col_name = 'p_c_{}'.format(i)
#   kmeans.setK(i).setPredictionCol(col_name)
#   model = kmeans.fit(people_vec_df)
#   temp = model.transform(people_acc_df)
#   people_acc_df = temp.withColumn(col_name+'s', temp[col_name].cast(StringType())).drop(col_name)
# people_acc_df = people_acc_df.drop('features')

# people_prep_df = raw_people_with_time_feat_df.join(people_acc_df, on='people_id', how='inner')

# people_renamed_df = people_prep_df.toDF(
#   *map(lambda col_name: 'p_' + col_name, people_prep_df.schema.names)
# ).withColumnRenamed('p_people_id', 'p_id')

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3: Join people and act_train dataframes

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3.1: Rename the column names of *people* and *train*
# MAGIC 
# MAGIC The column names in the people dataframe are in conflict with the train dataframe.
# MAGIC * We will rename the columns before joining the two dataframes.
# MAGIC 
# MAGIC __This name conflict will not prevent the join operation but I want to differentiate between columns from train and people dataframe. This becomes useful when we construct the *one-hot-encoding* dictionary__

# COMMAND ----------

people_renamed_df = raw_people_with_time_feat_df.toDF(
  *map(lambda col_name: 'p_' + col_name, raw_people_with_time_feat_df.schema.names)
).withColumnRenamed('p_people_id', 'p_id')

train_renamed_df = (
  raw_train_with_time_feat_df
  .toDF(*map(lambda col_name: 't_' + col_name, raw_train_with_time_feat_df.schema.names))
  .withColumnRenamed('t_people_id', 'p_id').withColumnRenamed('t_outcome', 'label')
)

print people_renamed_df.schema.names
print train_renamed_df.schema.names

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3.2: Join, drop-duplicates and cache

# COMMAND ----------

joined_train_people_df = (
  train_renamed_df
  .join(people_renamed_df, on='p_id', how='inner')
  .dropDuplicates()
  .cache()
)

print joined_train_people_df.count()
print joined_train_people_df.schema.names

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 4.4: Create *Training*, *validation* and *Test* sets from the *joined* dataframe
# MAGIC 
# MAGIC I have tried two different splitting strategy:
# MAGIC * Random split
# MAGIC   * It is fast; however, it does not provide a validation set representative of the leader board
# MAGIC * Random split by people_id
# MAGIC   * It better represents the leader board; however, it is excruciatingly slow !
# MAGIC   
# MAGIC We will use the answer by [zero323](http://stackoverflow.com/users/1560062/zero323) on Stackoverflow to create three dataframes with disjoined set of people_ids.
# MAGIC 
# MAGIC > [pyspark split filter dataframe by columns values](http://stackoverflow.com/questions/35190109/pyspark-split-filter-dataframe-by-columns-values)

# COMMAND ----------

# a simple random split

# weights = [0.8, 0.2] # training and test weights
# fraction = 0.01 # about 40K
# seed = 110

# train_df, test_df = joined_train_people_df.sample(False, fraction, seed).randomSplit(weights, seed)

# print train_df.cache().count()
# print test_df.cache().count()

# COMMAND ----------

weights = [0.8, 0.1, 0.1]
seed = 110

train_p_id, valid_p_id, test_p_id = map(
  lambda df: df.rdd.flatMap(lambda id: id).collect(),
  (
    joined_train_people_df
    .select('p_id')
    .distinct()
    .sample(False, 0.1)
    .randomSplit(weights=weights, seed=seed)
  )
)

# COMMAND ----------

train_dub_df = (
  joined_train_people_df
  .where(joined_train_people_df['p_id'].isin(train_p_id))
  .drop('p_id')
)
train_df = train_dub_df.groupBy(train_dub_df.schema.names).count().withColumnRenamed('count', 'weight')

valid_df = joined_train_people_df.where(joined_train_people_df['p_id'].isin(valid_p_id)).drop('p_id').dropDuplicates()
test_df = joined_train_people_df.where(joined_train_people_df['p_id'].isin(test_p_id)).drop('p_id').dropDuplicates()

print train_df.cache().count()
print valid_df.cache().count()
print test_df.cache().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 5: Make predictions

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1: Hyper-parameter optimization
# MAGIC 
# MAGIC If a validation set with unseen people_id is not important, we can simply use a provided TrainValidationSplit (I have commented out the code for this).
# MAGIC 
# MAGIC > [ML Tuning](https://spark.apache.org/docs/latest/ml-tuning.html)
# MAGIC 
# MAGIC Since we want to use a specific validation set, we should write our own optimizer (basically a couple of nested **for loops**).
# MAGIC 
# MAGIC Running a notebook on the **Community Edition** of Databricks has a number of implications:
# MAGIC 
# MAGIC 1. We only have 6GB of RAM to play with
# MAGIC 
# MAGIC 2. Our cluster gets terminated about every hour if the notebook is inactive.
# MAGIC 
# MAGIC As a result finetuning random forests on the complete dataset leads to memory errors.
# MAGIC Also our cluster gets terminated if we pick a large set of hyper parameters. This is due to the second point (the cell runs for more than an hour and our cluster gets terminated).

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml import Pipeline

encoder = Encoder()
roc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")

# COMMAND ----------

encoder_model = encoder.fit(train_df)
train_vec_df = encoder_model.transform(train_df)
valid_vec_df = encoder_model.transform(valid_df)
test_vec_df = encoder_model.transform(test_df)

# COMMAND ----------

# this is a helper for visualizing a heatmap of validation scores

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def display_heat_map(s, a, b):
#   x, y = len(a), len(b) 
#   heat_map_matrix = np.array(s).reshape(x, y)
#   f = plt.figure(figsize=(y + 1, x))
#   sns.heatmap(heat_map_matrix, square=True)
#   display(f)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1.1: Logistic regression

# COMMAND ----------

# from pyspark.ml.classification import LogisticRegression

# lr_reg_param = [1e-6]
# lr_max_iter = [80]
# lr = LogisticRegression(
#   standardization=False, threshold=0.5,
#   predictionCol='LR_P', probabilityCol='LR_Prob', rawPredictionCol='LR_rawProb'
# )
# roc_eval.setRawPredictionCol('LR_rawProb')

# lr_pip = Pipeline(stages=[encoder, lr])

# lr_param_grid = (
#   ParamGridBuilder()
#   .addGrid(lr.regParam, lr_reg_param)
#   .addGrid(lr.maxIter, lr_max_iter)
#   .build()
# )

# lr_tv = TrainValidationSplit(
#   estimator=lr_pip,
#   estimatorParamMaps=lr_param_grid,
#   evaluator=roc_eval,
#   trainRatio=0.8
# )

# lr_model = lr_tv.fit(train_df)
# lr_test_pred = lr_model.transform(test_df)

# print roc_eval.evaluate(lr_test_pred)

# print zip(lr_model.getEstimatorParamMaps(), lr_model.validationMetrics)

# display_heat_map(lr_model.validationMetrics, lr_max_iter, lr_reg_param)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
  standardization=False, threshold=0.5,
  predictionCol='LR_P', probabilityCol='LR_Prob', rawPredictionCol='LR_rawProb',
  maxIter=80, elasticNetParam=0.0, weightCol='weight'
)
roc_eval.setRawPredictionCol('LR_rawProb')

lr_models = []
lr_best_score = 0
lr_best_model = None

lr_reg_param = [1e-6]
for r in lr_reg_param:
  lr.setRegParam(r)
  model = lr.fit(train_vec_df)
  valid_pred_df = model.transform(valid_vec_df)
  valid_roc_score = roc_eval.evaluate(valid_pred_df)
  
  lr_models.append(('regParam: {}'.format(r), valid_roc_score))
  
  if valid_roc_score > lr_best_score:
    lr_best_score = valid_roc_score
    lr_best_model = model

lr_test_pred_df = lr_best_model.transform(test_vec_df)
lr_test_roc_score = roc_eval.evaluate(lr_test_pred_df)

print "Best model scores: {}".format(lr_test_roc_score)
print lr_models

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1.2: Random forest classifier

# COMMAND ----------

# from pyspark.ml.classification import RandomForestClassifier

# rf_num_trees = [40, 80, 160]
# rf_max_depth = [4, 8, 16, 30]

# rf = RandomForestClassifier(
#   predictionCol='RF_P', probabilityCol='RF_Prob', rawPredictionCol='RF_rawProb'
# )
# roc_eval.setRawPredictionCol('RF_rawProb')

# rf_pip = Pipeline(stages=[encoder, rf])

# rf_param_grid = (
#   ParamGridBuilder()
#   .addGrid(rf.numTrees, rf_num_trees)
#   .addGrid(rf.maxDepth, rf_max_depth)
#   .build()
# )

# rf_tv = TrainValidationSplit(
#   estimator=rf_pip,
#   estimatorParamMaps=rf_param_grid,
#   evaluator=roc_eval,
#   trainRatio=0.8
#   )

# rf_model = rf_tv.fit(train_df.drop('p_id'))
# rf_test_pred = rf_model.transform(test_df)

# print roc_eval.evaluate(rf_test_pred)

# print zip(rf_model.getEstimatorParamMaps(), rf_model.validationMetrics)

# display_heat_map(rf_model.validationMetrics, rf_num_trees, rf_max_depth)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
  predictionCol='RF_P', probabilityCol='RF_Prob', rawPredictionCol='RF_rawProb',
  featureSubsetStrategy='sqrt',
  numTrees=80
)
roc_eval.setRawPredictionCol('RF_rawProb')

rf_models = []
rf_best_score = 0
rf_best_model = None

rf_max_depth = [16]
for md in rf_max_depth:
  rf.setMaxDepth(md)
  model = rf.fit(train_vec_df)
  valid_pred_df = model.transform(valid_vec_df)
  valid_roc_score = roc_eval.evaluate(valid_pred_df)
  
  rf_models.append(('maxDepth: {}'.format(md), valid_roc_score))
  
  if valid_roc_score > rf_best_score:
    rf_best_score = valid_roc_score
    rf_best_model = model

rf_test_pred_df = rf_best_model.transform(test_vec_df)
rf_test_roc_score = roc_eval.evaluate(rf_test_pred_df)

print "Best model scores: {}".format(rf_test_roc_score)
print rf_models

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 5.2: Ensemble
# MAGIC 
# MAGIC This section is to help the *one hour time limit* and *limited RAM*. Instead of using a complete training set, we fit multiple models to random subsets of the training set and average them together. This helps us avoid the limitations of the Community Edition.
# MAGIC 
# MAGIC _the following did not imporve my score much_

# COMMAND ----------

def extract_probabilities(input_df, column_names):
  input_df_column_names = input_df.schema.names
  num_col = len(input_df_column_names)
  column_indices = [input_df_column_names.index(name) for name in column_names]
  
  def ep(point):
    arr = []
    for i in xrange(num_col):
      if i in column_indices:
        arr.append(float(point[i][1]))
      else:
        arr.append(point[i])
        
    return arr
    
  return (
    input_df
    .rdd
    .map(lambda row: ep(row))
    .toDF(input_df_column_names)
  )

def ensemble_fit(classifier, t_df, n=40, s=None, prefix=''):
  if not s:
    s = 1.0 / n
  
  models = []
  for i in xrange(n):
    (
      classifier
      .setPredictionCol('{}_o_{}'.format(prefix, i))
      .setProbabilityCol('{}_p_{}'.format(prefix, i))
      .setRawPredictionCol('{}_rp_{}'.format(prefix, i))
    )
    model = classifier.fit(train_vec_df.sample(False, s))
    models.append(model)
    
  return models

def ensemble_transform(models, t_df):
  t_acc_df = t_df
  for model in models:
    t_acc_df = model.transform(t_acc_df)
  return t_acc_df

def ensemble_evaluate(evaluator, t_df, p_col_names):
  scores = []
  for col in p_col_names:
    evaluator.setRawPredictionCol(col)
    s = evaluator.evaluate(t_df)
    scores.append(s)
  return scores

def avg_p(input_df, o_col_names):
  return input_df.withColumn('avg', sum(input_df[col] for col in o_col_names)/ len(o_col_names))

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
# from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col

rf = RandomForestClassifier(featureSubsetStrategy='sqrt', numTrees=40)
# lr = LogisticRegression(standardization=False, threshold=0.5, maxIter=80, elasticNetParam=0.0, weightCol='weight')

n = 10
s = 0.01
prefix = 'rf'

models = ensemble_fit(rf, train_vec_df, n=n, s=s, prefix=prefix)
test_ens_pred_df = ensemble_transform(models, test_vec_df)
rp_col_n = ['{}_rp_{}'.format(prefix, i) for i in xrange(0, n)]
scores = ensemble_evaluate(roc_eval, test_ens_pred_df, rp_col_n)
p_col_n = ['{}_p_{}'.format(prefix, i) for i in xrange(0, n)]
test_ens_ext_p_df = extract_probabilities(test_ens_pred_df, p_col_n)
print(scores)

# COMMAND ----------

ens_df = avg_p(test_ens_ext_p_df, p_col_n).select(col('avg').cast(DoubleType()), 'label')
roc_eval.setRawPredictionCol('avg')
print roc_eval.evaluate(ens_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6: Create a submission
# MAGIC 
# MAGIC Here we will use the parameters for the simplest single model that resulted in a decent performance on all the training set to make a submission.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
  standardization=False, threshold=0.5, regParam=4e-7, maxIter=80,
  predictionCol='LR_P', probabilityCol='LR_Prob', rawPredictionCol='LR_rawProb'
)

lr_pip = Pipeline(stages=[encoder, lr])

# COMMAND ----------

lr_model = lr_pip.fit(joined_train_people_df.drop('p_id').dropDuplicates())

# COMMAND ----------

raw_test_df = (
  sqlContext.read.format('com.databricks.spark.csv')
  .options(header=True, delimiter=',', inferschema=True)
  .load(data_root+'/act_test.csv')
)

raw_test_with_time_feat_df = parse_dates(raw_test_df)

raw_test_renamed_df = (
  raw_test_with_time_feat_df
  .toDF(*map(lambda col_name: 't_' + col_name, raw_test_with_time_feat_df.schema.names))
  .withColumnRenamed('t_people_id', 'p_id').withColumnRenamed('t_outcome', 'label')
)

joined_test_people_df = (
  raw_test_renamed_df
  .join(people_renamed_df, on='p_id', how='inner')
  .withColumnRenamed('t_activity_id', 'id')
)

# COMMAND ----------

sub_transformed_df = lr_model.transform(joined_test_people_df)
sub_prob_df = extract_probabilities(sub_transformed_df, ['LR_Prob'])

# COMMAND ----------

(
  sub_prob_df
  .select(
    sub_prob_df['id'].alias('activity_id'),
    sub_prob_df['LR_Prob'].alias('outcome')
  )
  .coalesce(1)
  .write
  .csv(submission_path, header=True)
)
