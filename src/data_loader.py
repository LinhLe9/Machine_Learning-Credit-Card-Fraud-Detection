import os
import functions_framework
from google.cloud import bigquery
import kagglehub
import pandas as pd

# @functions_framework.http cho phép trigger Cloud Function thông qua HTTP request (hoặc Cloud Scheduler)
@functions_framework.http
def run_fraud_pipeline(request):
    try:
        print("Start loading data from Kaggle pipeline...")
        
        # 1. Tải dataset từ Kaggle
        path = kagglehub.dataset_download("chetanmittal033/credit-card-fraud-data")
        file_path = os.path.join(path, "fraudTest.csv")
        credit_df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset: {credit_df.shape}")

        # 2. Initial Big Query
        project_id = "frauddetection-507910"
        dataset_id = "fraud_detection_dataset"
        table_id = "raw_transactions"
        table_ref = f"{project_id}.{dataset_id}.{table_id}"

        client = bigquery.Client(project=project_id)

        # Create dataset if not exist
        dataset_ref = f"{project_id}.{dataset_id}"
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = "EU"
        client.create_dataset(dataset, exists_ok=True)

        # Configure pushing job, overwrite if exist
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE", 
        )

        # Push data into BigQuery
        print("Đang đẩy dữ liệu lên Google BigQuery...")
        job = client.load_table_from_dataframe(credit_df, table_ref, job_config=job_config)
        job.result()  

        return f"Succeed! Data now live on table: {table_ref}", 200

    except Exception as e:
        print(f"Error while running pipeline: {str(e)}")
        return f"Error: {str(e)}", 500