import json

import pandas as pd

try:
    from .inference_pipeline import run_inference_pipeline
    from .runtime import ensure_paths_exist, load_config, project_root
except ImportError:
    from inference_pipeline import run_inference_pipeline
    from runtime import ensure_paths_exist, load_config, project_root

from ml_from_scratch.utils import load_json


def predict(df: pd.DataFrame):
    """This function makes a prediction with input from client. Used in app.py"""
    training_config_path = project_root / "api" / "config" / "breast_cancer_training.yaml"
    runtime_config_path = project_root / "api" / "config" / "runtime_config.yaml"

    # check if config files exist
    ensure_paths_exist({"training_config": training_config_path, "runtime_config": runtime_config_path})
    
    # load config and setup paths
    training_config = load_config(training_config_path)
    runtime_config = load_config(runtime_config_path)

    paths_cfg = training_config["paths"]
    runtime_inference_cfg = runtime_config["inference"]

    model_path = project_root / paths_cfg["model_path"]
    artifact_manifest_path = project_root / paths_cfg["artifact_manifest_path"]
    max_batch_items = runtime_inference_cfg["max_batch_items"]

    # check if artifacts exist
    ensure_paths_exist(
        {
            "model": model_path,
            "artifact_manifest": artifact_manifest_path,
        }
    )

    # load and run inference
    artifact_manifest = load_json(artifact_manifest_path)
    results = run_inference_pipeline(model_path, artifact_manifest, df, max_batch_items)
    return results


def main():
    sample_df = pd.read_csv(project_root / "api" / "artifacts" / "inference_data.csv")
    print(json.dumps(predict(sample_df)))


if __name__ == "__main__":
    main()
