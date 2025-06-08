from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator

dag = DAG(
    'ml_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual trigger
    catchup=False
)

# Data Pipeline
preprocess_data = BashOperator(
    task_id='preprocess_data',
    bash_command='echo "Processing images with Spark..."',
    dag=dag
)

# Training Pipeline
train_models = BashOperator(
    task_id='train_models',
    bash_command='echo "Training 6 model variants with Ray Tune..."',
    dag=dag
)

# Model Selection
select_best = BashOperator(
    task_id='select_best_model',
    bash_command='echo "Selecting best model from MLflow..."',
    dag=dag
)

# Deployment
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='echo "Deploying model with Ray Serve..."',
    dag=dag
)

# Dependencies
preprocess_data >> train_models >> select_best >> deploy_model
