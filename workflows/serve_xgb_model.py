# Databricks notebook source
from pyspark.sql.types import *
import shutil
import os
import tempfile
import mlflow
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from sklearn.metrics import mean_squared_error
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.ml.recommendation import ALS
import pandas as pd
import numpy as np
xgb_model_experiment_id = "[hidden]"
xgb_model_serve_experiment_id = "[hidden]"

# COMMAND ----------


latest_run_id = spark.read.format(
    "mlflow-experiment").load(xgb_model_experiment_id).limit(1).head(1)[0]['run_id']


class XGBBusinessRateRecommendation(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.spark = SparkSession.builder \
            .appName("Serve XGB Model") \
            .getOrCreate()
        self.business_data = self.spark.read.parquet(
            "/tables/business_hashed.parquet")
        self.xgb_model = mlflow.pyfunc.load_model(
            'runs:/' + latest_run_id + '/xgb-model')

    def predict(self, context, model_input):
        business_ids = model_input["business_ids"].to_list()
        business_features = business_hashed.filter(
            col("business_id").isin(business_ids))
        business_features_df = business_features.select("business_id", "features").rdd.map(
            lambda x: (x[0], Vectors.dense(x[1]))).toDF(["business_id", "features"])

        # Predict for the given business_ids
        predictions = xgb_model.transform(business_features_df)
        sorted_predictions = predictions.select(
            "business_id", "prediction").distinct().orderBy("prediction", ascending=False)

        return [r.business_id for r in sorted_predictions.select('business_id').collect()]


# COMMAND ----------


# 将 Conda 环境依赖关系写入一个临时文件
conda_env_str = """
name: xgb_business_ids_env
channels:
  - defaults
dependencies:
  - python=3.9
  - numpy
  - pandas
  - pyspark
  - scikit-learn
  - mlflow
  - xgboost
  - pip
  - pip:
    - mlflow
"""

temp_dir = tempfile.mkdtemp()
conda_env_path = os.path.join(temp_dir, "conda.yaml")
with open(conda_env_path, "w") as f:
    f.write(conda_env_str)


with mlflow.start_run(experiment_id=xgb_model_serve_experiment_id):
    run_id = mlflow.active_run().info.run_id

    # 使用临时文件路径记录模型
    mlflow.pyfunc.log_model(
        artifact_path="xgb_business_rate_recommendation",
        python_model=XGBBusinessRateRecommendation(),
        conda_env=conda_env_path
    )

# 在完成记录模型后，您可以删除临时目录
shutil.rmtree(temp_dir)


# COMMAND ----------
