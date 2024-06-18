import sys
import subprocess

def install_requirements():
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            "/opt/ml/processing/input/requirements.txt"
        ])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages from requirements.txt: {e}")
        sys.exit(1)

# Install the dependencies
install_requirements()

import json
import pathlib
import pickle
import tarfile
import joblib
import numpy as np
import pandas as pd
from time import gmtime, strftime
import time 
import uuid
import argparse
import boto3 
import pandas as pd 
import sagemaker

from sagemaker.feature_store.feature_definition import FeatureDefinition
from sagemaker.feature_store.feature_definition import FeatureTypeEnum
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

import os 
os.environ['AWS_DEFAULT_REGION'] = 'ap-south-1'
s3_client = boto3.resource('s3') 
region_name = 'ap-south-1'
boto_session = boto3.Session(region_name=region_name)

sagemaker_client = boto_session.client(service_name='sagemaker', region_name=region_name)
featurestore_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime', region_name=region_name)
feature_store_session = Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client,
    sagemaker_featurestore_runtime_client=featurestore_runtime
)

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role() 
default_bucket = sagemaker_session.default_bucket() 

feature_store_offline_s3_uri = 's3://' + default_bucket
record_identifier_feature_name = "identifyrecord"
event_time_feature_name = "Eventime"



if __name__ == "__main__":
    # train_path = "/opt/ml/processing/train_process/train.csv"
    # df = pd.read_csv(train_path, header=None)
    
    # column_schemas  = [{'name': col, 'type': dtype} for col, dtype in pd.read_csv(df).dtypes.items()]
    # feature_group_name = f"telemetry-feature-group-{str(uuid.uuid4())[:8]}"
    # current_time_sec = int(round(time.time()))

    # df["Eventime"] = pd.Series([current_time_sec]*len(df), dtype="float64")
    # feature_group = FeatureGroup(
    #     name=feature_group_name, sagemaker_session=feature_store_session
    # )
    # feature_group.load_feature_definitions(data_frame=df)
    # feature_group.create(
    #     s3_uri=feature_store_offline_s3_uri,
    #     record_identifier_name=record_identifier_feature_name,
    #     event_time_feature_name=event_time_feature_name,
    #     role_arn=role,
    #     enable_online_store=True
    # )
    # feature_group.ingest(
    #         data_frame=df, max_workers=3, wait=True
    # )
    # parser = argparse.ArgumentParser()
    # parser.add_argument('fgn', type=str)
    # args = parser.parse_args()

    columns = ['speed', 'engine_status', 'fuel_level',
       'battery_voltage', 'tire_pressure', 'current_gear', 'odometer_reading', 'coolant_level']
    train_path = "/opt/ml/processing/train/"
    x_train = pd.DataFrame(np.load(os.path.join(train_path, "x_train.npy")),columns = columns)
    y_train = pd.DataFrame(np.load(os.path.join(train_path, "y_train.npy")))
    #y_train = np.load(os.path.join(train_path, "y_train.npy"))
    #print(pd.DataFrame(x_train).columns.values)
    # Generate synthetic data
    num_records = len(x_train)
    np.random.seed(42)
    # print(args.fgn)
    # Record identifier column
    record_identifier = np.arange(1, num_records + 1)
    
    # Event time column (with date and hours)
    start_date = pd.to_datetime('2022-01-01')
    end_date = pd.to_datetime('2023-12-31')
    event_time = pd.date_range(start=start_date, end=end_date, periods=num_records)
    event_time_formatted = event_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    x_train["identifyrecord"] = record_identifier
    x_train["Eventime"] = event_time_formatted
    x_train["engine_temperature"]= y_train
    
    # Define your feature group
    feature_group_name = "tensorflow-telemetry-featuregroup-TTF2"
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    
    # Ingest the data
    feature_group.ingest(data_frame=x_train, max_workers=3, wait=True)
    print("data ingested")


    
    print("=======DATA FETCH STARTED..using.. ATHENA QUERY FORM OFFLINE FEATURE STORE=======")
    feature_group_name = "tensorflow-telemetry-featuregroup-TTF2"
    feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=sagemaker_session)
    identity_query = feature_group.athena_query()
    identity_table = identity_query.table_name
    
    query_string = 'SELECT speed, engine_status, fuel_level, battery_voltage, tire_pressure, current_gear, odometer_reading, coolant_level  FROM "'+identity_table+'" '
    
    # run Athena query. The output is loaded to a Pandas dataframe.
    dataset = pd.DataFrame()
    identity_query.run(query_string=query_string, output_location='s3://'+default_bucket+'/query_results/')
    identity_query.wait()
    x_train_fs = identity_query.as_dataframe()

    query_string = 'SELECT engine_temperature  FROM "'+identity_table+'" '
    identity_query.run(query_string=query_string, output_location = f's3://{default_bucket}/query_results/')
    identity_query.wait()
    y_train_fs = identity_query.as_dataframe()


    feature_train_path = "/opt/ml/processing/train_fs"
    np.save(os.path.join(feature_train_path, "x_train.npy"), x_train_fs.to_numpy())
    np.save(os.path.join(feature_train_path, "y_train.npy"), y_train_fs.to_numpy())
    print("Fetched data using Athena Query and saved to folder")
