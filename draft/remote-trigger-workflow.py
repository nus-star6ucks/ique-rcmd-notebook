# Databricks notebook source
import json
import requests

# COMMAND ----------


url = "https://[hidden]/api/2.0/jobs/run-now"
job_id = "[hidden]"
payload = {
    "job_id": job_id,
    "notebook_params": {
        "business_path": "yelp_academic_dataset_business_output_8.json",
        "user_path": "yelp_academic_dataset_user_8.json",
        "review_path": "yelp_academic_dataset_review_8.json"
    }
}
headers = {
    "Authorization": "Bearer [hidden]",
    "Content-Type": "application/json"
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.json())


# COMMAND ----------

!ls

# COMMAND ----------
