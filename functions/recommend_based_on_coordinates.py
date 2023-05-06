# Databricks notebook source
from pyspark.sql.types import *
import shutil
import os
import tempfile
import mlflow
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from pyspark.sql.functions import udf
from pyspark.ml.recommendation import ALS
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName(
    "recommend_based_on_coordinates").getOrCreate()

dbutils.widgets.text("latitude", "28.1", "Latitude")
dbutils.widgets.text("longitude", "-39.1", "Longitude")
dbutils.widgets.text("n_recommendations", "5", "n_recommendations")
dbutils.widgets.text("run_id", "0", "run_id")

run_id = dbutils.widgets.get("run_id")

# COMMAND ----------


# 基于地理位置进行推荐


def recommend_based_on_coordinates(latitude, longitude, n_recommendations=5):
    """
    :param latitude, longitude: user location info
    :param n_recommendations: number of recommended business
    :param model_uri: optional, if specified the model will be loaded from the given URI
    :return: array of recommended business info
    """
    business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

    dist_udf = udf(lambda x, y: np.sqrt((x - latitude) ** 2 +
                   (y - longitude) ** 2).tolist(), DoubleType())
    business_distinct = business_hashed.select(
        business_hashed.business_id, business_hashed.latitude, business_hashed.longitude).distinct()
    # display(business_distinct)
    nearby_business = business_distinct.withColumn("distance", dist_udf(
        business_distinct.latitude, business_distinct.longitude))
    # display(nearby_business)
    business_result = nearby_business.sort(col("distance")).limit(
        n_recommendations).select(col("business_id"))
    return business_result


# result = recommend_based_on_coordinates(float(dbutils.widgets.get("latitude")), float(dbutils.widgets.get("longitude")), int(dbutils.widgets.get("n_recommendations")))
# result.write.format("delta").mode("overwrite").save("/results/" + run_id)

# COMMAND ----------

# Define the model class
class GeoLocationRecommendation(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.spark = SparkSession.builder \
            .appName("Geolocation based recommendations") \
            .getOrCreate()
       # Load parquet file from artifacts
        parquet_path = context.artifacts["business_hashed"]
        # ("/tables/processed_data.parquet") # parquet_path
        self.business_data = self.spark.read.parquet(parquet_path)

    def predict(self, context, model_input):
        latitude = model_input["latitude"][0]
        longitude = model_input["longitude"][0]
        n_recommendations = 10

        dist_udf = udf(lambda x, y: np.sqrt((x - latitude) **
                       2 + (y - longitude) ** 2).tolist(), DoubleType())
        business_distinct = self.business_data.select(
            self.business_data.business_id, self.business_data.latitude, self.business_data.longitude).distinct()
        nearby_business = business_distinct.withColumn("distance", dist_udf(
            business_distinct.latitude, business_distinct.longitude))
        business_result = nearby_business.sort(col("distance")).limit(
            n_recommendations).select(col("business_id"))

        return list(business_result.rdd.map(lambda x: x[0]).collect())


# COMMAND ----------


# 将 Conda 环境依赖关系写入一个临时文件
conda_env_str = """
name: geolocation_recommendation_env
channels:
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - pyspark
  - scikit-learn
  - mlflow
  - pip
  - pip:
    - mlflow
"""

temp_dir = tempfile.mkdtemp()
conda_env_path = os.path.join(temp_dir, "conda.yaml")
with open(conda_env_path, "w") as f:
    f.write(conda_env_str)


with mlflow.start_run(experiment_id="[hidden]"):
    run_id = mlflow.active_run().info.run_id
    mlflow.log_artifact("/dbfs/tables/business_hashed.parquet",
                        "recommend_based_on_coordinates/artifacts")

    # 使用临时文件路径记录模型
    mlflow.pyfunc.log_model(
        artifact_path="recommend_based_on_coordinates",
        python_model=GeoLocationRecommendation(),
        artifacts={
            "business_hashed": "runs:/{run_id}/recommend_based_on_coordinates/artifacts/business_hashed.parquet".format(run_id=run_id)},
        conda_env=conda_env_path
    )

# 在完成记录模型后，您可以删除临时目录
shutil.rmtree(temp_dir)


# COMMAND ----------
