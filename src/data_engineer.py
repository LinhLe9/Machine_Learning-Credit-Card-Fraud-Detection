import os
from google.cloud import bigquery
from pathlib import Path
import numpy as np
import pandas as pd
from google.cloud.exceptions import NotFound
import joblib
from utils import sin_transformer, cos_transformer 
from google.cloud import bigquery

root_dir = Path(__file__).resolve().parent.parent
project_id = "frauddetection-507910"
client = bigquery.Client(project=project_id)

datasets = list(client.list_datasets())
print("Connect successfully! Available dataset in the project:")
for dataset in datasets:
    print(f"- {dataset.dataset_id}")

try:
    # Lấy max(unix_time) trừ đi 3600 giây (1 tiếng) để tạo biên độ an toàn
    query_max_time = f"""
        SELECT IFNULL(MAX(unix_time) - 3600, 0) 
        FROM `{project_id}.fraud_detection_dataset.cleaned_transaction`
    """
    max_unix = client.query(query_max_time).to_dataframe().iloc[0, 0]
    max_unix = 0 if pd.isna(max_unix) else max_unix
except Exception:
    max_unix = 0

# Sau đó dùng mốc này để query View lấy dữ liệu có độ trễ an toàn 1 tiếng
query = f"""
    SELECT * 
    FROM `frauddetection-507910.fraud_detection_dataset.v_rolling_features`
    WHERE unix_time > {max_unix}
"""

print("Loading data from BigQuery View Table 'v_rolling_feature'...")
df = client.query(query).to_dataframe()
print(f"Successfully loaded {len(df)} data rows.")

# ==========================================
# Def to calculate distance using [Haversine formula] 
# ==========================================
def calculate_haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculate the distance using longtitude and latitude by Haversine fomula.
    """
    radius = 6371.0 # Bán kính Trái Đất (km)

    # Chuyển đổi độ thập phân sang radian
    rad_lat1 = np.radians(df[lat1])
    rad_long1 = np.radians(df[long1])
    rad_lat2 = np.radians(df[lat2])
    rad_long2 = np.radians(df[long2])

    delta_lat = rad_lat2 - rad_lat1
    delta_long = rad_long2 - rad_long1

    a = np.sin(delta_lat / 2)**2 + np.cos(rad_lat1) * np.cos(rad_lat2) * np.sin(delta_long / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance_km = radius * c
    return distance_km

# ==========================================
# Data Engineer
# 1. Extract hour from trans_date
# 2. Calculate speed, time_diff_sec
# ==========================================
# sort the transaction by each card then sort by time
df = df.sort_values(by=['cc_num', 'unix_time']).reset_index(drop=True)
df['hour'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour 

df['distance_km'] = calculate_haversine_distance(df, lat1='lat', long1='long', lat2='merch_lat', long2='merch_long')

# calculate the distance between the last transaction location (prev_merch_long/lat) and this transaction location (merch_long/lat)
# calculate the time difference between the last transaction --> then calculate speed by distance/time
# group each transaction by each card (cc_num) using group by and shift
df['prev_merch_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
df['prev_merch_long'] = df.groupby('cc_num')['merch_long'].shift(1)
df['prev_unix'] = df.groupby('cc_num')['unix_time'].shift(1)


# create a new boolean column to flag the first transaction
df['is_first_txn'] = df['prev_unix'].isna().astype(int)
# Deal with the error lat,long --> flag the invalid latitude and longtitude
df['is_invalid_latlong'] = (
    (df['merch_lat'] == 0) | (df['merch_long'] == 0) |
    (df['prev_merch_lat'] == 0) | (df['prev_merch_long'] == 0)
).astype(int)

# Calculate the distance from last merch location
df['dist_from_prev_txn'] = calculate_haversine_distance(
    df, 
    lat1 ='merch_lat', long1='merch_long', 
    lat2='prev_merch_lat', long2='prev_merch_long'
)

# Calculate the time differenct from the last transaction
df['time_diff_sec'] = df['unix_time'] - df['prev_unix']

# Calculate speed 
df['speed'] = 0.0
mask_normal = (df['time_diff_sec'] > 0) & (df['is_invalid_latlong'] == 0)
df.loc[mask_normal, 'speed'] = (df['dist_from_prev_txn'] / df['time_diff_sec']) * 3600
# Case: time diff = 0 & dist from prev > 0 & is_invalid_latlong == 0 --> instant moving
mask_teleport = (df['time_diff_sec'] == 0) & (df['dist_from_prev_txn'] > 0) & (df['is_invalid_latlong'] == 0)
df.loc[mask_teleport, 'speed'] = 9999.0

# Fill the NaN data points with 0 
df[['time_diff_sec', 'speed','dist_from_prev_txn']] = df[['time_diff_sec', 'speed','dist_from_prev_txn']].fillna(0)

# drop the redundant column
df = df.drop(columns=['prev_merch_lat', 'prev_merch_long','prev_unix'])

print("Loading trained model pipeline...")
model_path = root_dir / "models" / "fraud_detection_pipeline.joblib"
model_pipeline = joblib.load(model_path)

# 2. Lấy lại dữ liệu mới nhất từ bảng production (hoặc dùng trực tiếp DataFrame df hiện tại)
print("Running fraud prediction on current data...")
df["predicted_fraud"] = model_pipeline.predict(df)
df["fraud_probability"] = model_pipeline.predict_proba(df)[:, 1]
fraud_count = df["predicted_fraud"].sum()
print(f"Prediction completed! Detected {fraud_count} potential frauds out of {len(df)} transactions.")

# 1. Định nghĩa tên bảng staging và bảng production chính
staging_table = "frauddetection-507910.fraud_detection_dataset.staging_transaction"
production_table = "frauddetection-507910.fraud_detection_dataset.cleaned_transaction"

# 2. Ghi DataFrame hiện tại vào bảng staging (dùng 'replace' để làm mới bảng staging mỗi lần chạy)
print(f"Writing data into staging table: {staging_table}...")
job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE") # Tương đương replace
job = client.load_table_from_dataframe(df, staging_table, job_config=job_config)
job.result() # Đợi load xong
print("Staging table updated successfully via BigQuery client.")

try:
    client.get_table(production_table)
    table_exists = True
except NotFound:
    table_exists = False

if not table_exists:
    # Trường hợp 1: Chạy lần đầu (Bảng chưa có) -> Tạo mới bảng production từ bảng staging
    print(f"Production table not found. Creating {production_table} for the first run...")
    create_query = f"""
        CREATE TABLE `{production_table}` AS 
        SELECT * FROM `{staging_table}`
    """
    client.query(create_query).result()
    print("Production table created and initialized successfully.")
else:
    # Trường hợp 2: Các lần chạy sau (Bảng đã có) -> Dùng lệnh MERGE chống trùng lặp
    print(f"Executing MERGE into existing production table: {production_table}...")
    merge_query = f"""
        MERGE `{production_table}` T
        USING `{staging_table}` S
        ON T.cc_num = S.cc_num AND T.unix_time = S.unix_time AND T.amt = S.amt
        WHEN NOT MATCHED THEN
          INSERT ROW;
    """
    client.query(merge_query).result()
    print("Finished! Data successfully merged into production.")