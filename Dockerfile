# Small, official Python base image — keeps the final image lean.
FROM python:3.12-slim

WORKDIR /app

# Copy requirements first (before the rest of the code).
# Docker caches layers: if only your code changes (not dependencies),
# this layer is reused and rebuilds are much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest: model artifacts, api code, data folder (for the SQLite log file)
COPY models/ ./models/
COPY api/ ./api/
RUN mkdir -p ./data

EXPOSE 8000

# 0.0.0.0 (not 127.0.0.1) is required so the server accepts connections
# from outside the container, not just from within it.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
