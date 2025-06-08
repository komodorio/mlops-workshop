# MLOps Workshop

Prerequisite: k8s kIND installed

Starting from clean Kind cluster:
```shell
kind delete cluster
kind create cluster
```

Alternatively: configure kubeconfig (kubectl) to use some existing cluster.

## Chapter 1: Top-level orchestrator

```shell
kubectl create configmap ml-pipeline-dag --from-file=dag1.py
```


```shell
helm repo add apache-airflow https://airflow.apache.org/
helm upgrade --install airflow apache-airflow/airflow -f airflow-values.yaml
```

```shell
while true; do kubectl port-forward svc/airflow-webserver 8080:8080; sleep 1; done 
```


