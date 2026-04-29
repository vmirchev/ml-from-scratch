import json

import numpy as np
import pandas as pd

try:
    from .runtime import ensure_paths_exist, load_config, project_root
    from .train_pipeline import run_training_pipeline
except ImportError:
    from runtime import ensure_paths_exist, load_config, project_root
    from train_pipeline import run_training_pipeline

from ml_from_scratch.utils import save_json, set_seeds


def main():
  config_path = project_root / "api" / "config" / "breast_cancer_training.yaml"
  ensure_paths_exist({"config": config_path})
  config = load_config(config_path)

  model_cfg = config["model"]
  training_section = config["training"]
  seed = training_section["seed"]
  epochs = training_section["epochs"]

  if epochs <= 0:
      raise ValueError(f"Training epochs must be > 0, got {epochs}.")

  set_seeds(seed)

  paths_cfg = config["paths"]
  model_path = project_root / paths_cfg["model_path"]
  artifact_manifest_path = project_root / paths_cfg["artifact_manifest_path"]
  data_path = project_root / paths_cfg["training_data_path"]

  ensure_paths_exist({"data": data_path})
  
  df = pd.read_csv(data_path)
  if "target" not in df.columns:
      raise ValueError("Training data must include a 'target' column.")

  feature_names = df.drop("target", axis=1).columns.tolist()
  features = df[feature_names].to_numpy(dtype=np.float64)
  targets = df["target"].to_numpy(dtype=np.int64)

  unique_targets = np.unique(targets)
  if not np.all(np.isin(unique_targets, [0, 1])):
      raise ValueError(
          f"Training targets must be binary 0/1 labels. Found: {unique_targets.tolist()}"
      )

  training_yaml_cfg = config["training"]
  pipeline_results = run_training_pipeline(model_cfg, training_yaml_cfg, features, targets)

  model = pipeline_results["model"]
  model.save(model_path)

  scaler = pipeline_results["scaler"]
  history = pipeline_results["history"]
  selected_threshold = pipeline_results["selected_threshold"]
  validation_threshold_metrics = pipeline_results["validation_threshold_metrics"]
  test_metrics_at_default_threshold = pipeline_results["test_metrics_at_default_threshold"]
  test_metrics_at_selected_threshold = pipeline_results["test_metrics_at_selected_threshold"]

  model_architecture = {
      "weight_init": model_cfg["weight_init"],
      "input_features": len(feature_names),
      "l2_lambda": training_yaml_cfg["l2_lambda"],
      "layers": model_cfg["layers"],
  }

  artifacts_manifest_json = {
      "model_architecture": model_architecture,
      "feature_names" : feature_names,
      "scaler": "ZScoreNormalization",
      "scaler_mean": scaler.mean.tolist(),
      "scaler_std": scaler.std.tolist(),
      "threshold": selected_threshold,
      "validation_threshold_metrics": validation_threshold_metrics,
      "test_metrics": test_metrics_at_selected_threshold,
      "seed": training_yaml_cfg["seed"]
      # dataset_version
      # train_date
      # model version
  }

  save_json(artifact_manifest_path, artifacts_manifest_json)

  results = {
      "train_loss": history["train_losses"][-1],
      "val_loss": history["val_losses"][-1],
      "selected_threshold": selected_threshold,
      "validation_threshold_metrics": validation_threshold_metrics,
      "test_metrics": test_metrics_at_selected_threshold,
      "test_metrics_at_default_threshold": test_metrics_at_default_threshold,
      "model_path": str(model_path),
      "artifact_manifest_path": str(artifact_manifest_path),
  }

  print("Training complete with results")
  print(json.dumps(results))


if __name__ == "__main__":
    main()
