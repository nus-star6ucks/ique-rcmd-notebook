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
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.recommendation import ALS
import pandas as pd
import numpy as np
als_model_experiment_id = "[hidden]"
als_model_experiment_serve_id = "[hidden]"

# COMMAND ----------


latest_run_id = spark.read.format(
    "mlflow-experiment").load(als_model_experiment_id).limit(1).head(1)[0]['run_id']


class ALSBusinessRateRecommendation(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.spark = SparkSession.builder \
            .appName("Serve ALS Model") \
            .getOrCreate()
        parquet_path = context.artifacts["als_user_recommendation"]
        self.als_user_data = self.spark.read.parquet(parquet_path)

    def predict(self, context, model_input):
        user_id = str(model_input["user_id"][0])
        data = self.als_user_data.filter(col("user_id") == user_id).collect()

        if len(data) == 0:
            return []

        business_id_list = [row.business_id for row in data[0].recommendations]
        return business_id_list


# COMMAND ----------


# 将 Conda 环境依赖关系写入一个临时文件
conda_env_str = """
name: sgd_business_ids_env
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
with mlflow.start_run(experiment_id=als_model_experiment_serve_id):
    run_id = mlflow.active_run().info.run_id
    mlflow.log_artifact("/dbfs/tables/als_user_recommendation.parquet",
                        "als_business_rate_recommendation/artifacts")

    # 使用临时文件路径记录模型
    mlflow.pyfunc.log_model(
        artifact_path="als_business_rate_recommendation",
        python_model=ALSBusinessRateRecommendation(),
        artifacts={
            "als_user_recommendation": "runs:/{run_id}/als_business_rate_recommendation/artifacts/als_user_recommendation.parquet".format(run_id=run_id)},
        conda_env=conda_env_path
    )
# 在完成记录模型后，您可以删除临时目录
shutil.rmtree(temp_dir)


# COMMAND ----------
