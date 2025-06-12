from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.spark_kubernetes import SparkKubernetesOperator

dag = DAG(
    "dag2_spark_preprocess",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual trigger
    catchup=False,
)


# Data Pipeline - Real Spark Job
preprocess_data = SparkKubernetesOperator(
    task_id="preprocess_data",
    namespace="default",
    application_file="C2S4-spark-preprocess.yaml",
    dag=dag,
)

# Training Pipeline
train_models = BashOperator(
    task_id="train_models",
    bash_command='echo "TODO: Train model variants with Ray Tune..."',
    dag=dag,
)

# Model Selection
select_best = BashOperator(
    task_id="select_best_model",
    bash_command='echo "TODO: Select best model from MLflow..."',
    dag=dag,
)

# Deployment
deploy_model = BashOperator(
    task_id="deploy_model",
    bash_command='echo "TODO: Deploy model with Ray Serve..."',
    dag=dag,
)

# Dependencies
preprocess_data >> train_models >> select_best >> deploy_model
