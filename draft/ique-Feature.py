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

!ls

# COMMAND ----------

!ls
# !pwd

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/tables/"))

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

# 去重
# business_df = business_df_1.dropDuplicates()
# review_df = review_df_1.dropDuplicates()
# user_df = user_df_1.dropDuplicates()

# Json文件存储为table
business_df.write.mode("overwrite").format("delta").saveAsTable("business")
review_df.write.mode("overwrite").format("delta").saveAsTable("review")
user_df.write.mode("overwrite").format("delta").saveAsTable("user")

# print(business_df.head(10))
# display(business_df)
# display(review_df)
# display(user_df)
# print(business_df.count(),len(business_df.columns))
# business_df.printSchema()

'''
特征工程
'''
# 1. 商家特征
# 类别特征
category_split = split(business_df.categories, ", ")
business_categories = business_df.select(
    "business_id", explode(category_split).alias("category"))
# business_categories.show()

# 数值特征
business_features = business_df.select(
    "business_id", "stars", "review_count", "latitude", "longitude")
# business_features.show()

# 2. 评论特征
review_features = review_df.groupBy("business_id").agg(
    count(when(review_df.stars >= 2.5, True)).alias("positive_review_count"),
    count(when(review_df.stars < 2.5, True)).alias("negative_review_count")
)
# review_features.show()

# 3. 用户特征
user_features = user_df.select(
    "user_id", "average_stars", "review_count", "fans", "elite")
# user_features.show()

# 转换ID特征
# 将string类型的id字段转换为int类型
business_indexer = StringIndexer(
    inputCol="business_id", outputCol="businessIndex").setHandleInvalid("skip")
user_indexer = StringIndexer(
    inputCol="user_id", outputCol="userIndex").setHandleInvalid("skip")

pipeline = Pipeline(stages=[business_indexer, user_indexer])
model = pipeline.fit(review_df)
review_df_transformed = model.transform(review_df)

# 合并特征
business_features = business_features.join(
    review_features, on="business_id", how="left")
business_features = business_features.join(
    business_categories, on="business_id", how="left")

# 转换类别特征
category_indexer = StringIndexer(
    inputCol="category", outputCol="categoryIndex").setHandleInvalid("skip")
category_encoder = OneHotEncoder(
    inputCol="categoryIndex", outputCol="categoryVec")

pipeline = Pipeline(stages=[category_indexer, category_encoder])
model = pipeline.fit(business_features)
business_features_transformed = model.transform(business_features)

# 创建特征向量
# 跳过所有null值
vector_assembler = VectorAssembler(
    inputCols=["stars", "review_count", "latitude", "longitude", "positive_review_count", "negative_review_count",
               "categoryVec"],
    outputCol="features",
    handleInvalid="skip"
)
business_features_vectorized = vector_assembler.transform(
    business_features_transformed)

# 使用局部敏感哈希 (LSH) 筛选出最不相关的商家
minhash_lsh = MinHashLSH(
    inputCol="features", outputCol="hashes", numHashTables=5)
lsh_model = minhash_lsh.fit(business_features_vectorized)
business_hashed = lsh_model.transform(business_features_vectorized)

# display(business_features_transformed)
# display(business_features_vectorized)
# display(business_hashed)

# 保存预处理后的数据
business_hashed.write.mode("overwrite").parquet(
    "dbfs:/tables/processed_data.parquet")
review_df_transformed.write.mode("overwrite").format(
    "delta").saveAsTable("transformed_data")


# COMMAND ----------

review_df_transformed.write.mode("overwrite").format(
    "delta").saveAsTable("transformed_data")

# COMMAND ----------

print(review_df_1.count())
print(review_df.count())

# COMMAND ----------

# business_hashed.write.mode("overwrite").format("delta").saveAsTable("NewBusiness")


# dbutils.fs.rm("dbfs:/tables/processed_data.parquet/", True)
# display(dbutils.fs.ls("dbfs:/tables/processed_data.parquet/"))
# print(spark.version)
# print(f"Hadoop version = {sc._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}")

# COMMAND ----------

# display(business_features_transformed)
# display(business_features_vectorized)
# display(business_hashed)
# review_df_transformed.show(50)

# COMMAND ----------

# review_df_transformed.filter(review_df_transformed.user_id == 'EBa-0-6AKoy6jziNexDJtg').show()
