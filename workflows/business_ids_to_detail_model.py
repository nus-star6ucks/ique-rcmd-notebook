# Databricks notebook source
business_ids_to_detail_model_exp_id = "[hidden]"

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("business_ids_to_detail_model").getOrCreate()

# COMMAND ----------

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
import mlflow

class BusinessDataModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.spark = SparkSession.builder \
            .appName("Serve Production Model") \
            .getOrCreate()
        parquet_path = context.artifacts["business_ids_to_detail"]
        self.business_table = self.spark.read.parquet(parquet_path)

    def to_business_infos(self, business_ids):
        """
        @param business_ids: list of business id
        @return: array of corresponding business information
        """
        business_info = self.business_table.filter(self.business_table.business_id.isin(business_ids))
        return business_info
    
    def predict(self, context, model_input):
        business_ids = list(model_input["business_ids"])
        return self.to_business_infos(business_ids).collect()


# COMMAND ----------

import tempfile
import os

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


with mlflow.start_run(experiment_id=business_ids_to_detail_model_exp_id):
    run_id = mlflow.active_run().info.run_id
    mlflow.log_artifact("/dbfs/tables/business.parquet", "business_ids_to_detail/artifacts")

    # 使用临时文件路径记录模型
    mlflow.pyfunc.log_model(
        artifact_path="business_ids_to_detail",
        python_model=BusinessDataModel(),
        artifacts={"business_ids_to_detail": "runs:/{run_id}/business_ids_to_detail/artifacts/business.parquet".format(run_id=run_id)},
        conda_env=conda_env_path
    )

# 在完成记录模型后，您可以删除临时目录
import shutil
shutil.rmtree(temp_dir)


# COMMAND ----------

