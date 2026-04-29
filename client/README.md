# Client

This folder contains a simple PHP form that uploads JSON files and sends them as POST requests to the FastAPI inference API.

## Files

- `index.php`: browser-based test client
- `single_item.json`: sample payload for `/inference`
- `batch_items.json`: sample payload for `/batch_inference`

## Run

Install the project dependencies from the repository root:

```bash
pip install -e .
```

Start the PHP built-in server from the repository root:

```bash
php -S localhost:8080 -t client
```

Then open:

```text
http://localhost:8080/index.php
```

Make sure the FastAPI app is also running, for example:

```bash
uvicorn api.app:app --reload
```
