# Databricks notebook source
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf
from sklearn.metrics import mean_squared_error
import os
from sklearn.linear_model import SGDRegressor
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, split, explode, sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinHashLSH
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import *

# COMMAND ----------

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("Machine Learning Engineering") \
    .getOrCreate()

# 读取预处理后的数据
business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

# # 计算商家之间的相似度
# approx_similarity = lsh_model.approxSimilarityJoin(business_hashed, business_hashed, threshold=0.8,
#                                                    distCol="JaccardDistance")

# 训练 SGDRegressor 模型
train_data, test_data = business_hashed.randomSplit([0.8, 0.2])

x_train = np.array(train_data.select("features").rdd.map(lambda x: x[0].toArray()).collect())
y_train = np.array(train_data.select("stars").rdd.map(lambda x: x[0]).collect())
# print(x_train)
sgd_model = SGDRegressor()
sgd_model.fit(x_train, y_train)

# 评估 SGDRegressor 模型
x_test = np.array(test_data.select("features").rdd.map(lambda x: x[0].toArray()).collect())
y_test = np.array(test_data.select("stars").rdd.map(lambda x: x[0]).collect())

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# COMMAND ----------

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("Machine Learning Engineering") \
    .getOrCreate()

# 读取预处理后的数据
business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

# # 计算商家之间的相似度
# approx_similarity = lsh_model.approxSimilarityJoin(business_hashed, business_hashed, threshold=0.8,
#                                                    distCol="JaccardDistance")

# 训练 SGDRegressor 模型
train_data, test_data = business_hashed.randomSplit([0.8, 0.2])

x_train = np.array(train_data.select("features").rdd.map(lambda x: x[0].toArray()).collect())
y_train = np.array(train_data.select("stars").rdd.map(lambda x: x[0]).collect())
# print(x_train)
sgd_model = SGDRegressor()
sgd_model.fit(x_train, y_train)

# example
# 给定的 business_id 列表
business_ids = ["NAqJ1ZEU2nGYqUafvdxXJg", "_IEpk-HKGA7ceQfxfCncRw", "1oMUtLLzQcI3qFym7LYoCg", "5E49HDnDjgEXeWP-8GKPZQ", "rt4xczK0fV6o7gPg9FNZGQ"]
business_features_selected = business_hashed.filter(business_hashed.business_id.isin(business_ids))
# display(business_features_selected)
x_test = np.array(business_features_selected.select("features").rdd.map(lambda x: x[0].toArray()).collect())
y_pred = sgd_model.predict(x_test)

# COMMAND ----------

display(y_pred)

# COMMAND ----------

