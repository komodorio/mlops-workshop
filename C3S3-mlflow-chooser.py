import sys
import mlflow

mlflow.set_tracking_uri("http://mlflow-tracking.default")
experiment_name = sys.argv[1]
experiment = mlflow.get_experiment_by_name(experiment_name)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_accuracy DESC, metrics.train_accuracy DESC"],
    max_results=1,
)

if not runs.empty:
    best_run = runs.iloc[0]
    run_id = best_run["run_id"]
    model_type = best_run["params.model_type"]
    accuracy = best_run["metrics.val_accuracy"]

    print(f"Best model: {model_type} with accuracy: {accuracy}")

    # Save model info for Ray Serve
    with open("/data/best_model_info_%s.txt" % experiment_name, "w") as f:
        f.write(f"{run_id}\n{model_type}\n{accuracy}")

    print(f"Model info saved")
else:
    raise Exception("No runs found")
