from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import V1EnvVar
from kubernetes.client.models.v1_volume_mount import V1VolumeMount
from kubernetes.client.models.v1_volume import V1Volume

dag = DAG(
    "dag5_ray_tune",
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

train_models = KubernetesPodOperator(
    on_finish_action="delete_succeeded_pod",
    task_id="train_models",
    name="ray-tune-training",
    namespace="default",
    image_pull_policy="IfNotPresent",
    image="docker.io/komikomodor/mlops-workshop:latest",
    cmds=["python"],
    arguments=["/scripts/C5S5-ray-tune.py"],
    env_vars=[V1EnvVar(name="PYTHONPATH", value="/scripts")],
    volumes=[
        V1Volume(
            name="data-volume",
            persistent_volume_claim={
                "claimName": "workshop-pvc",
            },
        ),
    ],
    volume_mounts=[
        V1VolumeMount(name="data-volume", sub_path="scripts", mount_path="/scripts"),
        V1VolumeMount(name="data-volume", sub_path="data", mount_path="/data"),
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
    arguments=["/scripts/C3S3-mlflow-chooser.py", "finger-counting"],
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
