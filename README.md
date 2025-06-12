# MLOps Workshop
![](https://storage.googleapis.com/kaggle-datasets-images/98056/230124/e022946c52ffb2150ea1550285d1b8c9/dataset-cover.jpg)

## Chapter 0: Preparations

Prerequisite: k8s kIND installed, helm installed, k9s installed

Starting from clean Kind cluster:

```shell
kind delete cluster
kind create cluster
```

Alternatively: configure kubeconfig (kubectl) to use some existing cluster.

Optional: connect Komodor

```shell
xdg-open https://app.komodor.com
helm repo add komodorio https://helm-charts.komodor.io 
helm upgrade --install --create-namespace --namespace komodor komodor-agent komodorio/komodor-agent --set apiKey=`cat komodor-key.txt` --set clusterName=local-kind-`date +%s` 
```

The skill of IT engineer is not to deploy flawlessly (because nobody does), but to troubleshoot quickly
How do we eat elephant? Piece by piece
Main skill of data engineer is slow drinking coffee
In short, the real production setup is very complex, we simplified it.
Explain workshop setting: bypass auth, everything in single cluster and same namespace.
Using PVCs and configmaps for storage - workshop setup.
All has to go into same namespace because of using PVCs to share the data. Real world would use Git, S3, NFS etc.
Creating dedicated images would decrease the startup overhead.
Training on CPU is not recommended to children and pregnant women.

```shell
docker build . -t docker.io/komodorio/mlops-workshop:latest
kind load docker-image docker.io/komodorio/mlops-workshop:latest
```

```shell
kubectl apply -f C0S1-workshop-pvc.yaml
kubectl apply -f C0S2-workshop-helper-pod.yaml
```

## Chapter 1: Airflow as top-level orchestrator

```shell
helm repo add apache-airflow https://airflow.apache.org/
helm upgrade --install airflow apache-airflow/airflow -f C1S2-airflow-values.yaml
```

Wait for Airflow pods to become ready.

```shell
xdg-open http://localhost:8082
kubectl port-forward svc/airflow-webserver 8082:8080 
```

See empty Airflow UI.

```shell
kubectl cp C1S3-dag1.py workshop-helper-pod:/data/dags/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag1_all_stubs && airflow dags trigger dag1_all_stubs"
```

Go to Airflow UI, see first DAG passing all green. Start it again manually, if you wish.

## Chapter 2: Spark for data preparation

```shell
helm repo add spark-operator https://kubeflow.github.io/spark-operator
helm upgrade --install spark-operator spark-operator/spark-operator
kubectl apply -f C2S1-airflow-spark-crb.yaml
```

Wait for operator deploy to complete.

Loading dataset: [koryakinp/fingers](https://www.kaggle.com/datasets/koryakinp/fingers) from Kaggle. Taking 200 images
as "small" dataset, full images

```shell
kubectl exec workshop-helper-pod -- mkdir -p /data/scripts
kubectl cp C2S2-preprocess_fingers.py workshop-helper-pod:/data/scripts/
kubectl cp C2S3-dag2.py workshop-helper-pod:/data/dags/
kubectl cp C2S4-spark-preprocess.yaml workshop-helper-pod:/data/dags/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag2_spark_preprocess && airflow dags trigger dag2_spark_preprocess"
```

Go to Airflow UI and second DAG, look at logs of preprocess_data step. Start it second time, notice is goes faster (
cached download).

Note that we'll disable this step of DAG until the end, to make DAGs run faster. The dataset is already in storage
volume.

## Chapter 3: MLFlow to track experiments

```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm upgrade --install mlflow bitnami/mlflow --set artifactRoot.s3.enabled=false --set tracking.auth.enabled=false --set artifactRoot.filesystem.enabled=true
```

Wait for its pods to become ready

```shell
xdg-open http://localhost:8083
kubectl port-forward svc/mlflow-tracking 8083:80 
```

See its empty UI of MLFlow.

Start tracking experiments, initially with fake metrics from fake training.

```shell
kubectl cp C3S1-dag3.py workshop-helper-pod:/data/dags/
kubectl cp C3S2-fake-train.py workshop-helper-pod:/data/scripts/
kubectl cp C3S3-mlflow-chooser.py workshop-helper-pod:/data/scripts/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag3_mlflow_tracking && airflow dags trigger dag3_mlflow_tracking"
```

Go to Airflow UI to see third DAG running and completing.
After its completion, go to MLFlow UI, refresh list of experiments, see "finger-counting-fake", browse into it: runs,
metrics etc. Add "val_accuracy" column to table.

## Chapter 4: PyTorch and real training

TODO: illustrate and explain 4 models we are working with. It's just like "Miss Universe" we want to choose.

We'll run direct training of single model variant. The training script is parameterized to have all our combinations
inside.
We're skipping realistic long data loading (eg from S3) by mounting volume to Pod.

```shell
kubectl cp C4S1_train.py workshop-helper-pod:/data/scripts/
kubectl cp C4S2-dag4.py workshop-helper-pod:/data/dags/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag4_real_training && airflow dags trigger dag4_real_training"
```

In Airflow, open fourth DAG and see it succeeds.
After its completion, go to MLFlow UI, refresh list of experiments, see "finger-counting-debug", look at the model
metrics graphs.
Best model chooser would choose that run, can be validated in data volume /data/best_model_info.txt file.

## Chapter 5: Ray Tune Experiment Scaling

Ray is very sensitive to version mismatch of its client and Python. Also needs access to all python sources, both on
head and workers. Our Docker image is critical here.
We mount volume into containers of Ray to skip data loading and script fetching.

```shell
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm upgrade --install kuberay-operator kuberay/kuberay-operator -f C5S1-kuberay-operator-values.yaml
helm upgrade --install ray-cluster kuberay/ray-cluster -f C5S2-ray-values.yaml
```
Wait for its pods to become ready.
```shell
xdg-open http://localhost:8265
kubectl port-forward service/ray-cluster-kuberay-head-svc 8265:8265 
```
See empty Ray interface. 
Let's run Ray-scaled experiment. 4 models with 2 learning rates = 8 experiments total.
Ray will decide how to run experiments according to their resource requests.
```shell
kubectl cp C5S4-dag5.py workshop-helper-pod:/data/dags/
kubectl cp C5S5-ray-tune.py workshop-helper-pod:/data/scripts/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag5_ray_tune && airflow dags trigger dag5_ray_tune"
```
At this stage your cooling fan should go WE-E-E-E-E-E-E. Can get OutOfMemory problems, too. That's quite alright.

Look at running DAG in Airflow and multiple experiment runs in MLFlow. It starts to take a lot of time, we'll skip this step in the next chapter, results are already in MLFlow, ready for querying.

Wait for Airflow DAG to complete (can also proceed further, we have best model written in previous steps).

## Chapter 6: Ray Serve the best model

Let's deploy the best model for serving.
For speed, we stubbed other steps of DAG.
```shell
kubectl cp C6S1-serve.py workshop-helper-pod:/data/scripts/
kubectl cp C6S2-dag6.py workshop-helper-pod:/data/dags/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag6_ray_serve && airflow dags trigger dag6_ray_serve"
```
Wait for Airflow DAG to complete.
Validate the Ray Serve started to function inside cluster.

```shell
kubectl exec workshop-helper-pod -- /bin/bash -c "echo '{\"image\":\"'\$(base64 -w 0 /data/data/fingers_data/test/ffb81e76-93e8-4e7f-87cf-7a91a62c8f25_3R.png)'\"}' | curl -X POST -H 'Content-Type: application/json' -d @- http://ray-cluster-kuberay-head-svc:8000/"
```


## Chapter 7: Everything together, at once

Skip this step if you trust it will work, because it will run long.
It would be good to delete all runs of small dataset from "finger-counting" experiment, to not compete for best model chooser.

```shell
kubectl cp C7S1-ray-tune-large.py workshop-helper-pod:/data/scripts/
kubectl cp C7S2-dag7.py workshop-helper-pod:/data/dags/
kubectl exec deployment/airflow-scheduler -- /bin/bash -c "airflow dags reserialize && airflow dags unpause dag7_complete && airflow dags trigger dag7_complete"
```

After this, your machine will be busy for long. This is where you want to switch to GPU training.


## Chapter 8: Consuming the results

Let's have a web UI to interact with our model inference.
```shell
kubectl cp C8S1-index.html workshop-helper-pod:/data/scripts/
kubectl cp C8S2-server.py workshop-helper-pod:/data/scripts/
kubectl exec workshop-helper-pod -- /bin/bash -c "cd /data/scripts && python C8S2-server.py"
```
It will hang running that command.

Open in parallel terminal.
```shell
xdg-open http://localhost:8080
kubectl port-forward workshop-helper-pod 8080:8080
```

Have fun uploading images from disk or even capturing via camera.

Also you can try a local app using camera live.
```shell
python3 C8S3-camera.py
```


