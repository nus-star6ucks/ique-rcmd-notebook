# Databricks notebook source
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinHashLSH
from pyspark.sql.functions import col, count, when, isnan, split, explode, sum
from pyspark.sql import SparkSession
!pip3 install kaggle

# COMMAND ----------

!mkdir ~/.kaggle
!touch ~/.kaggle/kaggle.json & & echo '{"username":"unissn7","key":"[hidden]"}' > ~/.kaggle/kaggle.json
!kaggle datasets download yelp-dataset/yelp-dataset

!unzip yelp-dataset.zip

# COMMAND ----------

# 导入 dbutils 模块
# import pyspark.dbutils

# 创建 SparkSession 对象
spark = SparkSession.builder.appName("Batch Import to DBFS").getOrCreate()

# 待导入文件列表
file_list = ["/databricks/driver/yelp_academic_dataset_user.json",
             "/databricks/driver/yelp_academic_dataset_business.json",
             "/databricks/driver/yelp_academic_dataset_review.json"]

# 批量导入文件到 DBFS
for file_path in file_list:
    dbutils.fs.mv("file:" + file_path, "/FileStore/datasets/" +
                  file_path.split("/")[-1])

# COMMAND ----------


# 读取 JSON 文件
# dataframe
# json_data = spark.read.json("dbfs:/FileStore/datasets/yelp_academic_dataset_business.json")
# json_data.write.format("delta").mode("overwrite").save("/FileStore/tables/dataset_business")
# sql
# spark.sql("CREATE TABLE dataset_business USING delta LOCATION '/FileStore/tables/dataset_business'")

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("Feature Engineering") \
    .getOrCreate()

# 加载 Yelp 数据集
pre_path = "dbfs:/FileStore/datasets/"
business_path = pre_path + "yelp_academic_dataset_business.json"
review_path = pre_path + "yelp_academic_dataset_review.json"
user_path = pre_path + "yelp_academic_dataset_user.json"

# spark.sql("CREATE OR REPLACE TEMPORARY VIEW business USING json OPTIONS" + " (path
# 'dbfs:/FileStore/datasets/yelp_academic_dataset_business.json')") spark.sql("select * from business").show()

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
business_df = spark.read.schema(business_schema).json(business_path)
review_df = spark.read.json(review_path)
user_df = spark.read.json(user_path)

# Json文件存储为table
business_df.write.mode("overwrite").format("delta").saveAsTable("business")
review_df.write.mode("overwrite").format("delta").saveAsTable("review")
user_df.write.mode("overwrite").format("delta").saveAsTable("user")

# COMMAND ----------

dbutil.fs.ls('dbfs:/FileStore/datasets/')

# COMMAND ----------
