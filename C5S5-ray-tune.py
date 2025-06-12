import ray
from ray import tune
from C4S1_train import train_model

# Connect to existing Ray cluster
ray.init(address="ray://ray-cluster-kuberay-head-svc:10001")  # Adjust service name/port as needed

# Define parameter grid
param_space = {
    "model_type": tune.grid_search(
        [
            "cnn_deep",
            "cnn_wide",
            "mobilenet_v2",
            "resnet18",
        ]
    ),
    "dataset_size": tune.grid_search(
        [
            "small",
            # "large",
        ]
    ),
    "lr": tune.grid_search(
        [
            0.001,
            0.0001,
        ]
    ),
    "epochs": 5,
}

if __name__ == "__main__":
    print("Connected to Ray cluster")
    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Run Ray Tune on the cluster
    analysis = tune.run(
        train_model,
        config=param_space,
        resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust based on your cluster
        num_samples=1,  # grid_search handles combinations
        verbose=1,
        metric="val_accuracy",
        mode="max",
    )

    print("Ray Tune completed!")
    print(f"Best config: {analysis.best_config}")
    print(f'Best accuracy: {analysis.best_result["val_accuracy"]}')

    ray.shutdown()
