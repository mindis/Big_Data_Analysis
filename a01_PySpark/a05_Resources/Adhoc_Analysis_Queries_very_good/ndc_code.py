import pyspark 
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName('ndc test').getOrCreate()
class S3Contructor:
    def __init__(self, filename):
        self.filename = filename 


    def s3_string(self):
        return os.path.join('s3a://ndcprod-ndc-data/data/base-internal/global', self.filename)






os = ['ch=mib;appver=8.11.0;os=Android 5.1;model=ZTE BLADE A110;jai',
'mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Tri',
'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTM',
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:47.0) Gecko',
'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML']


def extract_os(stringname):
    if 'iOS' in stringname:
        return 'iOS'
    elif
        'Android' in stringname:
        return 'Android'
    else:
         return 'others'


