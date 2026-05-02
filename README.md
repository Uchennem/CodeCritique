# CodeCritique

CodeCritique is an AI-assisted code review prototype built for the CSE 499 senior project. The application accepts a pasted code sample in a web UI, forwards it to a Python analysis service, and returns software quality feedback.

## Current Architecture

- `index.js` starts the Express app and serves the EJS frontend.
- `routes/index.js` handles code submission and forwards analysis requests to the Python service.
- `python_service/main.py` extracts metrics, generates findings, computes a rule-based risk score, and optionally compares that result with an ML model.
- `python_service/train_model.py` retrains the saved model from the project dataset.

## Extracted ML Features

- `loc`: total lines of code
- `max_nesting`: deepest control-flow nesting
- `branch_points`: approximate number of decision points
- `avg_fn_len`: average function length
- `max_fn_len`: longest function length
- `num_errors`: number of error-severity findings
- `num_warnings`: number of warning-severity findings
- `comment_ratio`: comments divided by total lines

## Run the App

### 1) Install Node dependencies

```bash
pnpm install
```

### 2) Start the Python analysis service

From the project root:

```bash
pnpm run start:py
```

This starts FastAPI on `http://127.0.0.1:8000`.

### 3) Start the web app

From the project root:

```bash
pnpm run start:web
```

This starts Express on `http://127.0.0.1:4000`.

### 4) Open the app

Open `http://127.0.0.1:4000` in your browser, paste code, and run analysis.

## Environment Notes

- The web app port is set by `PORT` in `.env` (default: `4000`).
- The Python API URL is set by `PY_SERVICE_URL` in `.env` (default: `http://127.0.0.1:8010`).
- If you see `WinError 10013` on a port, pick another open port and update both:
  - Python startup port
  - `PY_SERVICE_URL` in `.env`

## Retrain the Model

From `python_service`:

```bash
..\.venv\Scripts\python.exe train_model.py
```

The training script reads `training_data.csv`, retrains the classifier, and writes `risk_model.joblib`.
