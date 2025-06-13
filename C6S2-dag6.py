from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import V1EnvVar
from kubernetes.client.models.v1_volume_mount import V1VolumeMount
from kubernetes.client.models.v1_volume import V1Volume

dag = DAG(
    "dag6_ray_serve",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
)

# Data Pipeline - back to fake to save time
preprocess_data = BashOperator(
    task_id="preprocess_data",
    bash_command='echo "TODO: Process images with Spark..."',
    dag=dag,
)

# Training Pipeline - back to fake to save time
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
deploy_model = KubernetesPodOperator(
    task_id="deploy_model",
    name="ray-serve-deploy",
    namespace="default",
    on_finish_action="delete_succeeded_pod",
    image="docker.io/komikomodor/mlops-workshop:latest",
    image_pull_policy="IfNotPresent",
    cmds=["python"],
    arguments=["/scripts/C6S1-serve.py"],
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
    env_vars=[
        V1EnvVar(name="RAY_SERVE_DEFAULT_HTTP_HOST", value="0.0.0.0"),
    ],
    dag=dag,
)
# Dependencies
preprocess_data >> train_models >> select_best >> deploy_model
