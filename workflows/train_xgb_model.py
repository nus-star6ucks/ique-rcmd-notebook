# Databricks notebook source
from xgboost.spark import SparkXGBRegressor
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinHashLSH
from pyspark.sql.functions import col, count, when, isnan, split, explode, sum
from pyspark.sql import SparkSession
from sklearn.linear_model import SGDRegressor
import os
from sklearn.metrics import mean_squared_error
from pyspark.sql.functions import udf
from pyspark.ml.recommendation import ALS
import mlflow
import pandas as pd
import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator
sgd_model_experiment_id = "[hidden]"
sgd_model_serve_experiment_id = "[hidden]"

# COMMAND ----------


latest_run_id = spark.read.format(
    "mlflow-experiment").load(sgd_model_experiment_id).limit(1).head(1)[0]['run_id']
with mlflow.start_run(experiment_id=sgd_model_experiment_id):
    spark = SparkSession.builder \
        .appName("Train SGD Model") \
        .getOrCreate()
    business_hashed = spark.read.parquet("/tables/business_hashed.parquet")

    train_data, test_data = business_hashed.randomSplit([0.8, 0.2])
    train_data_df = train_data.selectExpr(
        "cast(stars as double) as label", "features")
    xgb_regressor = SparkXGBRegressor(
        objective='reg:squarederror', maxDepth=5, numRound=100)

    xgb_model = xgb_regressor.fit(train_data_df)

    test_data_df = test_data.selectExpr(
        "cast(stars as double) as label", "features")
    mlflow.log_param("predict", test_data_df)
    predictions = xgb_model.transform(test_data_df)

    evaluator = RegressionEvaluator(
        predictionCol="prediction", labelCol="label", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    mlflow.xgboost.log_model(xgb_model.get_booster(), "xgb-model")
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------


# 训练并保存模型至 Databricks

# SGD
latest_run_id = spark.read.format(
    "mlflow-experiment").load(sgd_model_experiment_id).limit(1).head(1)[0]['run_id']
with mlflow.start_run(experiment_id=sgd_model_experiment_id):
    spark = SparkSession.builder \
        .appName("Train SGD Model") \
        .getOrCreate()
    business_hashed = spark.read.parquet("/tables/business_hashed.parquet")

    # logged_model = 'runs:/' + latest_run_id + '/sgd-model'
    # sgd_model = mlflow.sklearn.load_model(model_uri=logged_model)

    # 训练 SGDRegressor 模型
    train_data, test_data = business_hashed.randomSplit([0.8, 0.2])

    x_train = np.array(train_data.select("features").rdd.map(
        lambda x: x[0].toArray()).collect())
    y_train = np.array(train_data.select(
        "stars").rdd.map(lambda x: x[0]).collect())
    sgd_model = SGDRegressor()
    sgd_model.fit(x_train, y_train)
    # sgd_model.partial_fit(x_train, y_train)

    # 评估 SGDRegressor 模型
    x_test = np.array(test_data.select("features").rdd.map(
        lambda x: x[0].toArray()).collect())
    y_test = np.array(test_data.select(
        "stars").rdd.map(lambda x: x[0]).collect())

    mlflow.log_param("predict", x_test)
    y_pred = sgd_model.predict(x_test)

    mlflow.sklearn.log_model(sgd_model, "sgd-model")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mlflow.log_metric("rmse", rmse)

# COMMAND ----------

logged_model = 'runs:/[hidden]/sgd-model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------


# COMMAND ----------


business_hashed = spark.read.parquet("/tables/business_hashed.parquet")

train_data, test_data = business_hashed.randomSplit([0.8, 0.2])

# COMMAND ----------

train_data_df = train_data.selectExpr(
    "cast(stars as double) as label", "features")
train_data_df.display()

# COMMAND ----------

xgb_regressor = SparkXGBRegressor(
    objective='reg:squarederror', maxDepth=5, numRound=100)

xgb_model = xgb_regressor.fit(train_data_df)

# COMMAND ----------

test_data_df = test_data.selectExpr(
    "cast(stars as double) as label", "features")
predictions = xgb_model.transform(test_data_df)

# COMMAND ----------

predictions.describe()

# COMMAND ----------


# Define the evaluator
evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="label", metricName="rmse")

# Evaluate the model on the test data
rmse = evaluator.evaluate(predictions)

print("RMSE:", rmse)

# COMMAND ----------

# 给定的 business_id 列表
business_ids = ["Pns2l4eNsfO8kk83dixA6A",
                "CF33F8-E6oudUQ46HnavjQ", "n_0UpQx1hsNbnPUSlodU8w"]

business_features = business_hashed.filter(
    col("business_id").isin(business_ids))
business_features_df = business_features.select("business_id", "features").rdd.map(
    lambda x: (x[0], Vectors.dense(x[1]))).toDF(["business_id", "features"])

# Predict for the given business_ids
predictions = xgb_model.transform(business_features_df)
sorted_predictions = predictions.select(
    "business_id", "prediction").distinct().orderBy("prediction", ascending=False)


# COMMAND ----------

business_ids = [r.business_id for r in sorted_predictions.select(
    'business_id').collect()]

# COMMAND ----------


# COMMAND ----------
