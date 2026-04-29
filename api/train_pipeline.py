from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .runtime import build_model
except ImportError:
    from runtime import build_model

from ml_from_scratch.activations import sigmoid
from ml_from_scratch.losses import BinaryCrossEntropyLoss
from ml_from_scratch.metrics import binary_predictions_from_probs
from ml_from_scratch.preprocessing import ZScoreNormalization
from ml_from_scratch.utils import create_batches, evaluate_binary_model_with_metrics


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    lr: float
    threshold: float = 0.5


@dataclass
class TrainDataContainer:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def normalize_data(X_train, X_val, X_test):
    scaler = ZScoreNormalization()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return scaler, X_train, X_val, X_test


def train_single_batch(model, loss_fn, X_batch, y_batch, cfg: TrainingConfig):
    logits = model.forward(X_batch)
    batch_loss = loss_fn.forward(logits, y_batch)

    predictions = binary_predictions_from_probs(sigmoid(logits), threshold=cfg.threshold)
    correct_in_batch = np.sum(predictions == y_batch)
    total_in_batch = len(y_batch)

    grad_logits = loss_fn.backward()
    model.backward(grad_logits)
    model.step(cfg.lr)

    return {
        "batch_loss": batch_loss,
        "correct_in_batch": correct_in_batch,
        "total_in_batch": total_in_batch,
    }


def evaluate_binary_classification_model(model, loss_fn, cfg: TrainingConfig, X_eval, y_eval):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in create_batches(X_eval, y_eval, cfg.batch_size, shuffle=False):
        logits = model.forward(X_batch)
        batch_loss = loss_fn.forward(logits, y_batch)
        loss_sum += batch_loss * len(y_batch)

        predictions = binary_predictions_from_probs(sigmoid(logits), threshold=cfg.threshold)
        correct += np.sum(predictions == y_batch)
        total += len(y_batch)

    return {"loss": loss_sum / total, "accuracy": correct / total}


def train_model(model, loss_fn, cfg: TrainingConfig, data: TrainDataContainer):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for _ in range(cfg.epochs):
        model.train()
        epoch_train_loss_sum = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for X_batch, y_batch in create_batches(data.X_train, data.y_train, cfg.batch_size):
            batch_train_results = train_single_batch(model, loss_fn, X_batch, y_batch, cfg)
            epoch_train_loss_sum += (
                batch_train_results["batch_loss"] * batch_train_results["total_in_batch"]
            )
            epoch_train_correct += batch_train_results["correct_in_batch"]
            epoch_train_total += batch_train_results["total_in_batch"]

        train_losses.append(epoch_train_loss_sum / epoch_train_total)
        train_accuracies.append(epoch_train_correct / epoch_train_total)

        val_metrics = evaluate_binary_classification_model(
            model, loss_fn, cfg, data.X_val, data.y_val
        )
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
    }


def threshold_evaluation(model, X_val, y_val, loss_fn):
    thresholds = np.linspace(0.00, 1.00, 51)
    results_by_threshold = []

    for threshold in thresholds:
        metrics = evaluate_binary_model_with_metrics(
            model, X_val, y_val, loss_fn, threshold=threshold
        )
        results_by_threshold.append(
            {
                "threshold": threshold,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
            }
        )

    results_df = pd.DataFrame(results_by_threshold)
    best_idx = results_df["f1"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]

    return best_threshold, results_df.loc[best_idx].to_dict()


def run_training_pipeline(model_cfg, training_yaml_cfg, features, targets):
    seed = training_yaml_cfg["seed"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        features,
        targets,
        test_size=training_yaml_cfg["test_size"],
        random_state=seed,
        stratify=targets,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=training_yaml_cfg["val_size_from_test"],
        random_state=seed,
        stratify=y_temp,
    )

    scaler, X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    model = build_model(model_cfg, X_train.shape[1], training_yaml_cfg["l2_lambda"])
    loss_fn = BinaryCrossEntropyLoss()
    training_cfg = TrainingConfig(
        epochs=training_yaml_cfg["epochs"],
        batch_size=training_yaml_cfg["batch_size"],
        lr=training_yaml_cfg["lr"],
        threshold=0.5,
    )

    history = train_model(
        model,
        loss_fn,
        training_cfg,
        TrainDataContainer(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val),
    )
    test_metrics_at_default_threshold = evaluate_binary_classification_model(
        model, loss_fn, training_cfg, X_test, y_test
    )
    selected_threshold, validation_threshold_metrics = threshold_evaluation(
        model, X_val, y_val, loss_fn
    )
    test_metrics_at_selected_threshold = evaluate_binary_model_with_metrics(
        model, X_test, y_test, loss_fn, threshold=selected_threshold
    )
    test_metrics_at_selected_threshold["threshold"] = selected_threshold

    return {
        "model": model,
        "scaler": scaler,
        "history": history,
        "selected_threshold": selected_threshold,
        "validation_threshold_metrics": validation_threshold_metrics,
        "test_metrics_at_default_threshold": test_metrics_at_default_threshold,
        "test_metrics_at_selected_threshold": test_metrics_at_selected_threshold,
    }
