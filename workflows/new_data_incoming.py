# Databricks notebook source
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinHashLSH
from pyspark.sql.functions import col, count, when, isnan, split, explode, sum
from pyspark.sql import SparkSession
import string
import random
import zipfile
import tempfile
import os
dbutils.widgets.text("incoming_dataset_zip_filename", "1683045200930",
                     "Incoming dataset includes users, review, business")

# COMMAND ----------

experiment_id = "593719883075035"
s3_path = "/mnt/ique-processing"

# AWS S3 bucket name
AWS_S3_BUCKET = "ique-processing-s3"
AWS_S3_AK = "[hidden]"
AWS_S3_SK = "[hidden]"
# Mount name for the bucket
MOUNT_NAME = "/mnt/ique-processing"

# COMMAND ----------

try:
    # Source url
    SOURCE_URL = "s3n://{0}:{1}@{2}".format(AWS_S3_AK,
                                            AWS_S3_SK, AWS_S3_BUCKET)
    # Mount the drive
    dbutils.fs.mount(SOURCE_URL, MOUNT_NAME)
except:
    print("already mounted")

# COMMAND ----------


temp_dir_name = ''.join(random.choice(
    string.ascii_letters + string.digits) for i in range(10))
file_path = '/dbfs/databricks/driver/'

# 打开zip文件并解压到临时目录
with zipfile.ZipFile('/dbfs' + s3_path + '/' + dbutils.widgets.get("incoming_dataset_zip_filename"), 'r') as zip_ref:
    zip_ref.extractall(temp_dir_name)

extracted_files = os.listdir()

# COMMAND ----------

extracted_files_path = 'file:/databricks/driver/' + temp_dir_name
dbutils.fs.mv(extracted_files_path, "/databricks/driver/" +
              temp_dir_name, recurse=True)

# COMMAND ----------

dbfs_extracted_files_path = "/databricks/driver/" + temp_dir_name

# COMMAND ----------


business_schema = StructType([
    StructField("address", StringType(), True),
    StructField("attributes", StructType([
        StructField("ByAppointmentOnly", StringType(), True)
    ])),
    StructField("business_id", StringType(), False),
    StructField("categories", StringType(), True),
    StructField("city", StringType(), True),
    StructField("hours", StringType(), True),
    StructField("is_open", LongType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("name", StringType(), True),
    StructField("postal_code", StringType(), True),
    StructField("review_count", LongType(), True),
    StructField("stars", DoubleType(), True),
    StructField("state", StringType(), True)
])


incoming_business_df = spark.read.schema(business_schema).json(
    dbfs_extracted_files_path + '/business.json')
incoming_review_df = spark.read.json(
    dbfs_extracted_files_path + '/review.json')
incoming_user_df = spark.read.json(dbfs_extracted_files_path + '/user.json')

# COMMAND ----------


# 将传入数据写入到数据表
incoming_business_df.write.mode("append").format(
    "delta").saveAsTable("business")
incoming_review_df.write.mode("append").format("delta").saveAsTable("review")
incoming_user_df.write.mode("append").format("delta").saveAsTable("user")


# COMMAND ----------
