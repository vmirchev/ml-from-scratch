from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parents[1]
from ml_from_scratch.activations import ReLU
from ml_from_scratch.layers import LinearLayer
from ml_from_scratch.model import Model


def ensure_paths_exist(paths: dict[str, Path]):
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict, in_features:int, l2_lambda) -> Model:
    layers = []
    
    for layer_cfg in config["layers"]:
        layer_type = layer_cfg["type"]

        if layer_type == "linear":
            out_features = layer_cfg["out_features"]
            layers.append(
                LinearLayer(
                    in_features=in_features,
                    out_features=out_features,
                    initialization=config["weight_init"],
                    l2_lambda=l2_lambda,
                )
            )
            in_features = out_features
            continue

        if layer_type == "relu":
            layers.append(ReLU())
            continue

        raise ValueError(f"Unknown layer type: {layer_type}")

    return Model(layers)
