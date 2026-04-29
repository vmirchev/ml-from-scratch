import numpy as np
import pandas as pd

try:
    from .runtime import build_model
except ImportError:
    from runtime import build_model

from ml_from_scratch.activations import sigmoid
from ml_from_scratch.metrics import binary_predictions_from_probs
from ml_from_scratch.preprocessing import ZScoreNormalization


def build_scaler_from_manifest(artifact_manifest):
    scaler = ZScoreNormalization()
    scaler.mean = np.array(artifact_manifest["scaler_mean"])
    scaler.std = np.array(artifact_manifest["scaler_std"])
    return scaler


def load_model_from_manifest(model_path, artifact_manifest):
    model_architecture = artifact_manifest["model_architecture"]
    model = build_model(
        model_architecture,
        model_architecture["input_features"],
        model_architecture["l2_lambda"],
    )
    model.load(model_path)
    model.eval()
    return model


def validate_inference_dataframe(df, feature_names, max_batch_items):
    if df.shape[0] > max_batch_items:
        raise ValueError(f"Too many items in single batch: {df.shape[0]} > {max_batch_items}")

    missing = set(feature_names) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    extra = set(df.columns) - set(feature_names)
    if extra:
        raise ValueError(f"Unexpected extra features: {extra}")

    df = df[feature_names]
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"All features must be numeric. Non-numeric columns: {non_numeric_columns}"
        )

    return df


def run_inference_pipeline(model_path, artifact_manifest, inference_df, max_batch_items):
    model = load_model_from_manifest(model_path, artifact_manifest)

    return run_inference_with_model(model, artifact_manifest, inference_df, max_batch_items)

def run_inference_with_model(model, artifact_manifest, inference_df, max_batch_items):
    feature_names = artifact_manifest["feature_names"]
    threshold = artifact_manifest["threshold"]

    validated_df = validate_inference_dataframe(inference_df, feature_names, max_batch_items)
    scaler = build_scaler_from_manifest(artifact_manifest)

    X = scaler.transform(validated_df.to_numpy(dtype=np.float64))
    logits = model.forward(X)
    probabilities = sigmoid(logits).reshape(-1)
    predictions = binary_predictions_from_probs(probabilities, threshold=threshold)
    items = [
        {"prediction": int(prediction), "probability": float(probability)}
        for prediction, probability in zip(predictions, probabilities)
    ]

    return {
        "total": len(items),
        "items": items,
    }
