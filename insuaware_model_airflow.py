import datetime

from airflow import models
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocInstantiateWorkflowTemplateOperator,
)
from airflow.utils.dates import days_ago

project_id = models.Variable.get("project_id")


default_args = {
    # Tell airflow to start one day ago, so that it runs as soon as you upload it
    "start_date": days_ago(1),
    "project_id": project_id,
}

# Define a DAG (directed acyclic graph) of tasks.
# Any task you create within the context manager is automatically added to the
# DAG object.
with models.DAG(
    # The id you will see in the DAG airflow page
    "insuaware_model_DAG01",
    default_args=default_args,
    # The interval with which to schedule the DAG
    schedule_interval=datetime.timedelta(days=1),  # Override to match your needs
) as dag:

    start_template_job = DataprocInstantiateWorkflowTemplateOperator(
        # The task id of your job
        task_id="insuaware_prediction_DAG",
        # The template id of your workflow
        template_id="insuaware_model_template1",
        project_id=project_id,
        # The region for the template
        region="us-central1",
    )