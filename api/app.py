import logging
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

try:
    from .predict import predict
except ImportError:
    from predict import predict

app = FastAPI(title="NN From Scratch Inference API Demo")
logger = logging.getLogger(__name__)


@app.post("/inference")
def inference(payload: dict[str, Any]):
    try:
        df = pd.DataFrame([payload])
        results = predict(df)
    except (FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail="Unexpected error occurred. Run training again...") from exc
    except (ValueError) as exc:
        raise HTTPException(status_code=400, detail="Unexpected error occurred.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during single-item inference")
        raise HTTPException(status_code=500, detail="Internal server error") from exc

    return results["items"][0]


@app.post("/batch_inference")
def batch_inference(payload: list[dict[str, Any]]):
    if not payload:
        raise HTTPException(status_code=400, detail="Request body must contain at least one item.")

    try:
        df = pd.DataFrame(payload)
        return predict(df)
    except (FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail="Unexpected error occurred. Run training again...") from exc
    except (ValueError) as exc:
        raise HTTPException(status_code=400, detail="Unexpected error occurred.") from exc
    except Exception as exc:
        logger.exception("Unexpected error during batch inference")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
