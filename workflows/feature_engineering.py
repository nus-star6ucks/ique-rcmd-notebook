# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, split, explode, sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinHashLSH
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("Feature Engineering") \
    .getOrCreate()

business_df = spark.table('business')
review_df = spark.table('review')
user_df = spark.table('user')


# COMMAND ----------


'''
特征工程
'''
# 1. 商家特征

# 类别特征
category_split = split(business_df.categories, ", ")
business_categories = business_df.select("business_id", explode(category_split).alias("category"))
# business_categories.show()

# 数值特征
business_features = business_df.select("business_id", "stars", "review_count", "latitude", "longitude")
# business_features.show()

# 2. 评论特征
review_features = review_df.groupBy("business_id").agg(
    count(when(review_df.stars >= 2.5, True)).alias("positive_review_count"),
    count(when(review_df.stars < 2.5, True)).alias("negative_review_count")
)
# review_features.show()

# 3. 用户特征
user_features = user_df.select("user_id", "average_stars", "review_count", "fans", "elite")
# user_features.show()

# 转换ID特征
# 将string类型的id字段转换为int类型
business_indexer = StringIndexer(inputCol="business_id", outputCol="businessIndex").setHandleInvalid("skip")
user_indexer = StringIndexer(inputCol="user_id", outputCol="userIndex").setHandleInvalid("skip")

pipeline = Pipeline(stages=[business_indexer, user_indexer])
model = pipeline.fit(review_df)
review_df_transformed = model.transform(review_df)

# 合并特征
business_features = business_features.join(review_features, on="business_id", how="left")
business_features = business_features.join(business_categories, on="business_id", how="left")

# 转换类别特征
category_indexer = StringIndexer(inputCol="category", outputCol="categoryIndex").setHandleInvalid("skip")
category_encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")

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
business_features_vectorized = vector_assembler.transform(business_features_transformed)


# 使用局部敏感哈希 (LSH) 筛选出最不相关的商家
minhash_lsh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
lsh_model = minhash_lsh.fit(business_features_vectorized)
business_hashed = lsh_model.transform(business_features_vectorized)

# COMMAND ----------

# 更新 parquet
business_hashed.write.mode('overwrite').parquet("dbfs:/tables/business_hashed.parquet")
review_df_transformed.write.mode('overwrite').parquet("dbfs:/tables/review_transformed.parquet")

# COMMAND ----------

review_df_transformed.write.mode("overwrite").format("delta").saveAsTable("transformed_data")
business_df.write.mode('overwrite').parquet("dbfs:/tables/business.parquet")

# COMMAND ----------

review_df_transformed.select("user_id", "userIndex").distinct().write.mode('overwrite').parquet("dbfs:/tables/user_map.parquet")

# COMMAND ----------

review_df_transformed.select("business_id", "businessIndex").distinct().write.mode('overwrite').parquet("dbfs:/tables/business_map.parquet") 

# COMMAND ----------

