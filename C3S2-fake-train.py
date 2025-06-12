import mlflow
import random

mlflow.set_tracking_uri("http://mlflow-tracking.default")
mlflow.set_experiment("finger-counting-fake")

# Simulate 3 model training runs
models = ["simple_cnn", "mobilenet_v2", "resnet18"]
for model in models:
    with mlflow.start_run(run_name=f"{model}_training"):
        # Simulate training metrics
        accuracy = random.uniform(0.85, 0.95)
        loss = random.uniform(0.1, 0.3)

        mlflow.log_param("model_type", model)
        mlflow.log_param("dataset_size", "large")
        mlflow.log_metric("train_accuracy", accuracy)
        mlflow.log_metric("train_loss", loss)
        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_loss", loss)

        # Log dummy model artifact
        with open(f"{model}.txt", "w") as f:
            f.write(f"Model: {model}, Accuracy: {accuracy}")
        mlflow.log_artifact(f"{model}.txt")

        print(f"Logged {model} with accuracy: {accuracy}")
