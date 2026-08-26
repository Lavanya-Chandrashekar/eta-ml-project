# Week 3 (M4) packaging: containerize the trained model + serving API.
#
# Note: this Dockerfile is written for correctness and was validated by
# manual review, but was NOT built inside the sandbox this project was
# developed in (no internet access there to pull the base image / packages,
# and no Docker daemon available). Build and run it on your own machine:
#
#   docker build -t eta-prediction-api .
#   docker run -p 5000:5000 eta-prediction-api
#
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching: only reinstalls when
# requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what's needed to serve: the API, the feature-engineering code it
# imports, the trained model artifacts, and the requirements file already
# copied above. Deliberately NOT copying raw/processed data, notebooks, or
# tests into the image -- keeps it small and avoids shipping training data
# inside a production serving container.
COPY api/ ./api/
COPY src/features/ ./src/features/
COPY src/data/generate_synthetic_data.py ./src/data/generate_synthetic_data.py
COPY models/best_model.joblib ./models/best_model.joblib
COPY models/best_model_feature_columns.json ./models/best_model_feature_columns.json

# predictions.jsonl gets written here at runtime; mount a volume in
# production so logs survive container restarts, e.g.:
#   docker run -p 5000:5000 -v $(pwd)/logs:/app/logs eta-prediction-api
RUN mkdir -p /app/logs

EXPOSE 5000

# Flask's built-in dev server is used here to match api/app.py directly and
# keep the assignment's scope focused on ML packaging rather than WSGI
# server tuning. For real production traffic, swap this CMD for a
# production WSGI server, e.g.:
#   RUN pip install gunicorn
#   CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "4", "api.app:app"]
CMD ["python", "api/app.py"]
