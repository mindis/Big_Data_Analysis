# Setup
1. [setup google cloud client](https://cloud.google.com/bigquery/docs/reference/libraries) add add PATH to bash profile.
1. Install python module `pip install --upgrade google-cloud-bigquery`

# Example code
```python
from google.cloud import bigquery

# Construct a BigQuery client object.
client = bigquery.Client()

query = """
    SELECT name, SUM(number) as total_people
    FROM `bigquery-public-data.usa_names.usa_1910_2013`
    WHERE state = 'TX'
    GROUP BY name, state
    ORDER BY total_people DESC
    LIMIT 20
"""
query_job = client.query(query)  # Make an API request.

print("The query data:")
for row in query_job:
    # Row values can be accessed by field name or index.
    print("name={}, count={}".format(row[0], row["total_people"]))
```

# Third party Modules
## pandas-gbq
- [pandas-gbq](https://pandas-gbq.readthedocs.io/en/latest/install.html#conda)
```python
# this also installs google-cloud-bigquery module itself
# but, we need to have credential json file in PATH.
conda install -n dataSc -c conda-forge pandas-gbq
```

Example
```python
import pandas_gbq
import functools

@functools.lru_cache(maxsize=1024)
def get_data():
    sql = """
    SELECT country_name, alpha_2_code
    FROM [bigquery-public-data:utility_us.country_code_iso]
    WHERE alpha_2_code LIKE 'Z%'
    """
    df = pandas_gbq.read_gbq(sql,dialect="legacy")

    return df
# data is downloaded only first time we call the function.
df = get_data()
df
```

## bigquery_helper
- https://github.com/SohierDane/BigQuery_Helper
```python
pip install -e git+https://github.com/SohierDane/BigQuery_Helper#egg=bq_helpe
```

Example code
```python
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "openaq")
bq_assistant.list_tables()
bq_assistant.head("global_air_quality", num_rows=3)
bq_assistant.table_schema("global_air_quality")
QUERY = "SELECT location, timestamp, pollutant FROM `bigquery-public-data.openaq.global_air_quality`"
bq_assistant.estimate_query_size(QUERY)
df = bq_assistant.query_to_pandas(QUERY)
df.head(2)
```

# Jupyter magic
- https://googleapis.dev/python/bigquery/latest/magics.html

```python
%load_ext google.cloud.bigquery

params = {"num": 5}
%%bigquery --params $params

SELECT name, SUM(number) as count
FROM `bigquery-public-data.usa_names.usa_1910_current`
GROUP BY name
ORDER BY count DESC
LIMIT @num
```

# Upload new dataset
- https://www.youtube.com/watch?v=MH5M2Crn6Ag
- http://eforexcel.com/wp/downloads-18-sample-csv-files-data-sets-for-testing-sales/

```
google: console google cloud
go to: https://console.cloud.google.com/
From the top left three lines menu, go to BigQuery menu
Sandbox account is free, but if we want to upgrade, we need credit card account.

I got default project-id "analog-signal-250501"
select that project
create dataset > dataset id "demo" (note data expires in 60 days from sandbox)
demo > create empty table > table name: "salesRecords" > Schema "tick on schema"

query: select * from demo.SalesRecords limit 3;
More > format query (this will formats the SQL code)

====================
To select few columns

click on demo.table it shows SalesRecords Schema
click on column name, it will be inserted in query.

SELECT 
region,
sum(total_profit)
FROM `analog-signal-250501.demo.SalesRecords`
group by region

Save query > Profit_by_Region > Save view > Profit_by_Region

```