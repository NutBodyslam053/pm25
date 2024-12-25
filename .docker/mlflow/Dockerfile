# FROM python:3.11-slim-buster
FROM ghcr.io/mlflow/mlflow:v2.18.0

# Install python package
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Expose the default MLflow tracking port
EXPOSE 5000

# Define entrypoint for the container
# CMD ["mlflow", "server", "--host", "0.0.0.0"]
CMD ["mlflow", "server", "--host", "0.0.0.0", "--serve-artifacts", "--artifacts-destination", "s3://mlflow"]