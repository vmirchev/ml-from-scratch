# API

This folder contains the FastAPI inference service and the supporting inference pipeline.

## Run

Install the project dependencies from the repository root, then start the server:

```bash
pip install -e .
```

```bash
uvicorn api.app:app --reload
```

## Endpoints

### `POST /inference`

Accepts a single JSON object with the model feature fields.

Example request:

```json
{
  "mean radius": 14.1,
  "mean texture": 20.2,
  "mean perimeter": 91.6,
  "mean area": 600.4,
  "mean smoothness": 0.1,
  "mean compactness": 0.13,
  "mean concavity": 0.1,
  "mean concave points": 0.06,
  "mean symmetry": 0.18,
  "mean fractal dimension": 0.06,
  "radius error": 0.4,
  "texture error": 1.2,
  "perimeter error": 2.8,
  "area error": 35.0,
  "smoothness error": 0.005,
  "compactness error": 0.02,
  "concavity error": 0.03,
  "concave points error": 0.01,
  "symmetry error": 0.02,
  "fractal dimension error": 0.003,
  "worst radius": 16.5,
  "worst texture": 25.1,
  "worst perimeter": 110.2,
  "worst area": 800.7,
  "worst smoothness": 0.14,
  "worst compactness": 0.25,
  "worst concavity": 0.3,
  "worst concave points": 0.12,
  "worst symmetry": 0.28,
  "worst fractal dimension": 0.08
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.9321
}
```

### `POST /batch_inference`

Accepts a list of JSON objects using the same feature schema as `/inference`.

Example response:

```json
{
  "total": 2,
  "items": [
    {
      "prediction": 1,
      "probability": 0.9321
    },
    {
      "prediction": 0,
      "probability": 0.1042
    }
  ]
}
```
