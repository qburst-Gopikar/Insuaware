# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from pyspark.sql import SparkSession
from pyspark import SparkContext
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import json
import joblib

sc = SparkContext("local", "insuaware prediction")
spark = SparkSession.builder.appName('insuaware_prediction_model').getOrCreate()


def create_batch_prediction_job_sample(
    project= 'fr-inference-engine',
    display_name= 'insuaware prediction',
    model_name= 'insuaware_pred_model',
    instances_format= 'csv',
    gcs_source_uri= 'gs://logistic_regression_testing/log_reg_model/data/insuraware_test_data.csv',
    predictions_format= 'jsonl',
    gcs_destination_output_uri_prefix= 'gs://logistic_regression_testing/gender_pred_model/insuaware/output1/',
    location = "us-central1",
    api_endpoint = "us-central1-aiplatform.googleapis.com"):
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    
    # Initialize client that will be used to create and send requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    model_parameters_dict = {}
    model_parameters = json_format.ParseDict(model_parameters_dict, Value())

    batch_prediction_job = {
        "display_name": display_name,
        # Format: 'projects/{project}/locations/{location}/models/{model_id}'
        "model": 'projects/fr-inference-engine/locations/us-central1/models/2210559331550625792',
        "model_parameters": model_parameters,
        "input_config": {
            "instances_format": instances_format,
            "gcs_source": {"uris": [gcs_source_uri]},
        },
        "output_config": {
            "predictions_format": predictions_format,
            "gcs_destination": {"output_uri_prefix": gcs_destination_output_uri_prefix},
        },
        "dedicated_resources": {
            "machine_spec": {
                "machine_type": "n1-standard-2"
            },
            "starting_replica_count": 1,
            "max_replica_count": 1,
        },
    }
    parent = f"projects/{project}/locations/{location}"
    response = client.create_batch_prediction_job(parent=parent, batch_prediction_job=batch_prediction_job)
    print("response:", response)
    
create_batch_prediction_job_sample()

