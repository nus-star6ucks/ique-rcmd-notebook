# Databricks notebook source
import mlflow
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.sql.functions import explode, col, collect_list, struct
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinHashLSH
from pyspark.sql.functions import col, count, when, isnan, split, explode, sum
from pyspark.sql import SparkSession
from mlflow.models.signature import infer_signature
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
als_model_experiment_id = "[hidden]"

# COMMAND ----------


spark = SparkSession.builder.appName("Train ALS Model").getOrCreate()

# ALS
# with mlflow.start_run(experiment_id=als_model_experiment_id):
review_df_transformed = spark.read.table("transformed_data")

user_business_ratings = review_df_transformed.select(
    "userIndex", "businessIndex", "stars")

train_ratings, test_ratings = user_business_ratings.randomSplit([
                                                                0.8, 0.2], seed=42)
als = ALS(
    rank=20,
    maxIter=20,
    regParam=0.3,
    userCol="userIndex",
    itemCol="businessIndex",
    ratingCol="stars",
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42,
)

als_model = als.fit(train_ratings)
predictions = als_model.transform(test_ratings)

evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="stars", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

mlflow.log_metric("rmse", rmse)

# join business_id, user_id
business_map = spark.read.parquet("dbfs:/tables/business_map.parquet")
user_map = spark.read.parquet("dbfs:/tables/user_map.parquet")

als_user_data = als_model.recommendForAllUsers(20)
flat_df = als_user_data.select("userIndex", explode(
    "recommendations").alias("recommendation"))
joined_df = flat_df.join(
    business_map, flat_df.recommendation.businessIndex == business_map.businessIndex)
joined_df = joined_df.select(
    "userIndex",
    struct(
        "recommendation.businessIndex",
        "business_id",
        "recommendation.rating"
    ).alias("recommendation")
)
result_df = joined_df.groupBy("userIndex").agg(
    collect_list("recommendation").alias("recommendations")
)

als_user_data = result_df.join(
    user_map, result_df.userIndex == user_map.userIndex).drop(user_map.userIndex)
als_user_data.write.mode('overwrite').parquet(
    "dbfs:/tables/als_user_recommendation.parquet")

# COMMAND ----------
