# Databricks notebook source
# 训练并保存模型至 Databricks

import numpy as np
import pandas as pd
import mlflow
import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import udf
from sklearn.metrics import mean_squared_error
import os
from xgboost import XGBRegressor

with mlflow.start_run(experiment_id="[hidden]"):

    business_hashed = spark.read.parquet("dbfs:/tables/processed_data.parquet")

    # 训练 XGBoostRegressor 模型
    train_data, test_data = business_hashed.randomSplit([0.8, 0.2])

    x_train = np.array(train_data.select("features").rdd.map(
        lambda x: x[0].toArray()).collect())
    y_train = np.array(train_data.select(
        "stars").rdd.map(lambda x: x[0]).collect())
    xgb_model = XGBRegressor()
    xgb_model.fit(x_train, y_train)

    # 评估 XGBoostRegressor 模型
    x_test = np.array(test_data.select("features").rdd.map(
        lambda x: x[0].toArray()).collect())
    y_test = np.array(test_data.select(
        "stars").rdd.map(lambda x: x[0]).collect())

    mlflow.log_param("predict", x_test)
    y_pred = xgb_model.predict(x_test)

    mlflow.sklearn.log_model(xgb_model, "xgb-model")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse", rmse)
