from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG(
    "dag1_all_stubs",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual trigger
    catchup=False,
)

# Data Pipeline
preprocess_data = BashOperator(
    task_id="preprocess_data",
    bash_command='echo "TODO: Process images with Spark..."',
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
