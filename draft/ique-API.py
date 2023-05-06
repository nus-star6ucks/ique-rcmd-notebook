# Databricks notebook source
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from pyspark.sql.types import * 
from pyspark.sql.functions import *
from pyspark.sql import SQLContext

spark = SparkSession.builder.appName("Recommendaion Engineering").getOrCreate()
als = spark.read.parquet("dbfs:/tables/als_user_recommendation.parquet")


# COMMAND ----------

als.first()

# COMMAND ----------

# 读取预处理后的数据
business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

# 读取转换后数据
review_table = spark.read.table(tableName="hive_metastore.default.transformed_data")
# review_table.show()

df = spark.read.table("hive_metastore.default.transformed_data").sample(withReplacement=False, fraction=0.01)

# 基于用户对商店评分的推荐模型
# user对business的评分Rating：表示user对business的喜好程度
als_business = ALS(userCol="userIndex", itemCol="businessIndex", ratingCol="stars", coldStartStrategy="drop", rank=10, maxIter=5,
                    nonnegative=True, implicitPrefs=True)
als_business_model = als_business.fit(df)


# COMMAND ----------

# 将dafaframe转换为numpy array
def transform_to_array(df_result):
    pandas_result = df_result.toPandas()
    array_result = pandas_result.values
    return array_result

# COMMAND ----------

spark = SparkSession.builder.appName("Recommendaion Engineering").getOrCreate()

# 基于地理位置进行推荐
def recommend_based_on_coordinates(latitude, longitude, n_recommendations=5):
    """
    @param latitude, longitude: user location info
    @param n_recommendations: number of recommended business
    @param model_uri: optional, if specified the model will be loaded from the given URI
    @return: array of recommended business info
    """
    business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

    dist_udf = udf(lambda x, y: np.sqrt((x - latitude) ** 2 + (y - longitude) ** 2).tolist(), DoubleType())
    business_distinct = business_hashed.select(business_hashed.business_id,business_hashed.latitude,business_hashed.longitude).distinct()
    # display(business_distinct)
    nearby_business = business_distinct.withColumn("distance", dist_udf(business_distinct.latitude, business_distinct.longitude))
    # display(nearby_business)
    business_result = nearby_business.sort(col("distance")).limit(n_recommendations).select(col("business_id"))
    # display(business_result)
    recom_result = transform_to_array(business_result)
    return recom_result

res = recommend_based_on_coordinates(50.0, 100.0)
# print(res)

# COMMAND ----------

# 基于用户对商店评分的推荐模型
def recommend_based_on_user_id(user_ids: list, all_user=False, n_recommendations=5):
    """
    @param user_ids: list of user ids
    @param n_recommendations: number of recommended business
    @param all_user: recommend for all users or not
    @return: array of recommendation results [userIndex,array[business_Index,Rating]]
    """
    if all_user:
        business_recom = als_business_model.recommendForAllUsers(n_recommendations)
    else:
        users = review_table.filter((review_table.user_id).isin(user_ids)).select("userIndex").distinct()
        business_recom = als_business_model.recommendForUserSubset(users, n_recommendations)
    # display(business_recom)
    recom_result = transform_to_array(business_recom)
    # display(recom_result)
    return recom_result

# recommend_based_on_user_id(all_user=True)
# display(res)
userlist = ["qVc8ODYU5SZjKXVBgXdI7w"]
res = recommend_based_on_user_id(userlist)

# COMMAND ----------

# 将businessIndex转换为business_id
def transform_business_id(businessIndices):
    """
    @param businessIndexs: list of business Indexs
    @return: array of business origin ids
    """
    business_id_result = review_table.filter((review_table.businessIndex).isin(businessIndices)).select("business_id").distinct()
    pandas_business_id = business_id_result.toPandas()
    pandas_business_id.apply(str)
    business_ids = pandas_business_id.values
    return business_ids

# 将userIndex转换为user_id
def transform_user_id(userIndexs):
    """
    @param userIndexs: list of user Indexs
    @return: array of user origin ids
    """
    user_id_result = review_table.filter((review_table.userIndex).isin(userIndices)).select("user_id").distinct()
    # display(user_id_result)
    pandas_user_id = user_id_result.toPandas()
    pandas_user_id.user_id.apply(str)
    user_ids = pandas_user_id.values
    return user_ids
    

userIndices = [row[0] for row in res]
print(userIndices)
userIDs = transform_user_id(userIndices)
print(userIDs)
for row in res:
    business = row[1]
    businessIndices = [row[0] for row in business]
    print(businessIndices)
    businessIDs = transform_business_id(businessIndices)
    print(businessIDs)

# COMMAND ----------

# 根据business_id返回business信息
def get_business_info(business_ids):
    """
    @param business_ids: list of business id
    @return: array of corresponding business information
    """
    business_table = spark.read.table(tableName="hive_metastore.default.business")
    business_info = business_table.filter(business_table.business_id.isin(business_ids))
    info_result = transform_to_array(business_info)
    return info_result

ids = ['GXFMD0Z4jEVZBCsbPf4CTQ','1b5mnK8bMnnju_cvU65GqQ','yPSejq3_erxo9zdVYTBnZA','VAeEXLbEcI9Emt9KGYq9aA','C9K3579SJgLPp0oAOM29wg']
res = get_business_info(ids)
print(res)

# COMMAND ----------


# # 基于地理位置进行推荐
# def recommend_based_on_coordinates(latitude, longitude, n_recommendations=5, model_uri=None):
#     """
#     :param latitude, longitude: user location info
#     :param n_recommendations: number of recommended business
#     :param model_uri: optional, if specified the model will be loaded from the given URI
#     :return: array of recommended business info
#     """
#     business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

#     dist_udf = udf(lambda x, y: np.sqrt((x - latitude) ** 2 + (y - longitude) ** 2).tolist(), DoubleType())
#     business_distinct = business_hashed.select(business_hashed.business_id,business_hashed.latitude,business_hashed.longitude).distinct()
#     # display(business_distinct)
#     nearby_business = business_distinct.withColumn("distance", dist_udf(business_distinct.latitude, business_distinct.longitude))
#     # display(nearby_business)
#     business_result = nearby_business.sort(col("distance")).limit(n_recommendations).select(col("business_id"))
#     # display(business_result)
#     recom_result = transform_to_array(business_result)
#     # display(recom_result)
#     return recom_result


# # 保存模型到本地文件
# mlflow.spark.save_model(udf_model)

# COMMAND ----------

# from pyspark.sql.functions import col
# from pyspark.sql.functions import udf
# from pyspark.sql.types import DoubleType
# import numpy as np
# import mlflow

# # log the pipeline as a model in mlflow
# # with mlflow.start_run(experiment_id=3909609827766338, run_name="recommend_based_on_coordinates"):
# #     mlflow.spark.log_model(trained_pipeline, "gps-model-zoe")

# with mlflow.start_run(experiment_id=3909609827766338, run_name="recommend_based_on_coordinates"):
#     recommendation_model = RecommendationModel()
#     recommendation_model.fit(spark, 50.0, 100.0, 10)