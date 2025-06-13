from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import V1Volume, V1VolumeMount

dag = DAG(
    "dag3_mlflow_tracking",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,  # Manual trigger
    catchup=False,
)

# Data Pipeline - back to fake to save time
preprocess_data = BashOperator(
    task_id="preprocess_data",
    bash_command='echo "TODO: Process images with Spark..."',
    dag=dag,
)

# Training Pipeline - Log to MLflow
train_models = KubernetesPodOperator(
    task_id="train_models",
    name="train-models",
    namespace="default",
    on_finish_action="delete_succeeded_pod",
    image="docker.io/komikomodor/mlops-workshop:latest",
    image_pull_policy="IfNotPresent",
    cmds=["python"],
    arguments=["/scripts/C3S2-fake-train.py"],
    volumes=[
        V1Volume(
            name="data-volume",
            persistent_volume_claim={"claimName": "workshop-pvc"},
        ),
    ],
    volume_mounts=[
        V1VolumeMount(name="data-volume", sub_path="scripts", mount_path="/scripts"),
    ],
    dag=dag,
)

# Model Selection - Query MLflow for best model
select_best = KubernetesPodOperator(
    task_id="select_best_model",
    name="select-best",
    namespace="default",
    on_finish_action="delete_succeeded_pod",
    image="docker.io/komikomodor/mlops-workshop:latest",
    image_pull_policy="IfNotPresent",
    cmds=["python"],
    arguments=["/scripts/C3S3-mlflow-chooser.py", "finger-counting-fake"],
    volumes=[
        V1Volume(
            name="data-volume",
            persistent_volume_claim={"claimName": "workshop-pvc"},
        ),
    ],
    volume_mounts=[
        V1VolumeMount(name="data-volume", sub_path="scripts", mount_path="/scripts"),
        V1VolumeMount(name="data-volume", sub_path="data", mount_path="/data"),
    ],
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
