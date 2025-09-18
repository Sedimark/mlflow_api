<h1 align="center">Welcome to mlflow_api</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://opensource.org/license/MIT" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> SEDIMARK Rest API for interacting with the MLFlow in the toolbox

## Usage

- uv
  
  ### Requirements
  
  - uv >= 0.5.0 
  - python >= 3.11.0
    
  ### Running
    
    - First run

      ```bash
      uv run src/mlflow_api/main.py
      ```
    - All the other runs

      ```bash
      uv run mlflow_api
      ```
- Docker
  ### Environment Variables
  - MLFLOW_TRACKING_USERNAME - The username for the local MLFlow instance
  - MLFLOW_TRACKING_PASSWORD - The password for the local MLFlow instance
  - AWS_ACCESS_KEY_ID - The access key for the local MINIO/remote S3 instance
  - AWS_SECRET_ACCESS_KEY - The secret key for the local MINIO/remote S3 instance
  - MLFLOW_S3_ENDPOINT_URL - The url for the local MINIO/remote S3 instance
  - MLFLOW_TRACKING_INSECURE_TLS - The type of connection for the local MLFlow instance (true/false)
  - MLFLOW_TRACKING_URI - The url for the local MLFlow instance
  
  ### Building the image
  ```bash
  docker build -t mlflow_api .
  ```

  ### Running
  ```bash
    docker run -itd -p 8000:8000 \
  -e MLFLOW_TRACKING_USERNAME=admin \
  -e MLFLOW_TRACKING_PASSWORD=password \
  -e AWS_ACCESS_KEY_ID=<key> \
  -e AWS_SECRET_ACCESS_KEY=<secret> \ 
  -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9001 \
  -e MLFLOW_TRACKING_INSECURE_TLS=true \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  mlflow_api
  ```
  <div align="center">

# 🚀 MLflow API

*A powerful REST API for seamless MLflow integration*

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000)](https://github.com/yourusername/mlflow-api)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-3.1+-orange.svg)](https://mlflow.org)

</div>

## 📋 Overview

The MLflow API is a comprehensive REST API service that provides seamless integration with MLflow for machine learning model management. Built with FastAPI, it offers endpoints for model registration, versioning, deployment, and various ML operations including support for multiple frameworks like PyTorch, TensorFlow, Keras, and scikit-learn.

## ✨ Features

- 🔍 **Model Discovery**: Browse and search registered models
- 📊 **Metrics & Parameters**: Access model parameters, metrics, and metadata
- 🗂️ **Version Management**: Handle model versions and lifecycle stages
- 📈 **Dataset Integration**: Access training datasets and artifacts
- 🖼️ **Image Artifacts**: Retrieve and display model-related images
- 🔄 **Model Import/Export**: Seamlessly transfer models between environments
- 🎯 **Predictions**: Make predictions using registered models
- 🧠 **Multi-Framework Support**: Works with PyTorch, TensorFlow, Keras, scikit-learn
- 🔧 **Framework Tools**: Get optimizers and loss functions for different ML frameworks
- 📦 **Model Packaging**: Package models for deployment

## 🛠️ Supported ML Frameworks

| Framework | Import | Export | Predictions | Packaging |
|-----------|--------|--------|-------------|-----------|
| PyTorch | ✅ | ✅ | ✅ | ✅ |
| TensorFlow | ✅ | ✅ | ✅ | ✅ |
| Keras | ✅ | ✅ | ✅ | ✅ |
| scikit-learn | ✅ | ✅ | ✅ | ✅ |
| PyFunc | ✅ | ✅ | ✅ | ✅ |

## 🚀 Quick Start

### Prerequisites

- Python >= 3.12.0
- uv >= 0.5.0 (recommended) or pip
- Access to an MLflow tracking server
- S3-compatible storage (MinIO or AWS S3)

### 🐍 Using uv (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd mlflow_api

# First run - install dependencies and run
uv run src/mlflow_api/main.py

# Subsequent runs
uv run mlflow_api
```

### 🐳 Using Docker

#### Environment Variables

Create a `.env` file with the following variables:

```env
MLFLOW_TRACKING_USERNAME=admin
MLFLOW_TRACKING_PASSWORD=password
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MLFLOW_S3_ENDPOINT_URL=http://localhost:9001
MLFLOW_TRACKING_INSECURE_TLS=true
MLFLOW_TRACKING_URI=http://localhost:5000
```

#### Build and Run

```bash
# Build the Docker image
docker build -t mlflow-api .

# Run the container
docker run -itd -p 8000:8000 \
  -e MLFLOW_TRACKING_USERNAME=admin \
  -e MLFLOW_TRACKING_PASSWORD=password \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9001 \
  -e MLFLOW_TRACKING_INSECURE_TLS=true \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  mlflow-api
```

## 📚 API Documentation

Once the service is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔧 API Endpoints

### 🏠 Health Check
- `GET /` - Server health check

### 🤖 Models
- `GET /models` - List all registered models
- `GET /model/parameters` - Get model parameters
- `GET /model/metrics` - Get model metrics
- `GET /model/dataset` - Download training dataset
- `GET /model/images` - Get model artifacts (images)
- `GET /model/versions` - Get model versions
- `GET /model/package` - Package model for deployment
- `GET /model/export` - Export model as ZIP
- `POST /model/import` - Import model from ZIP
- `POST /model/predict` - Make predictions
- `POST /model/register` - Register a new model

### 🧠 Framework Tools
- `GET /optimizers/{framework}` - Get available optimizers
- `GET /losses/{framework}` - Get available loss functions

*Supported frameworks: `torch`, `keras`*

## 🏗️ Project Structure

```
mlflow_api/
├── src/mlflow_api/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI application and endpoints
│   ├── mlflow_client.py     # MLflow client wrapper
│   ├── models.py            # Model handlers for different frameworks
│   └── schemas.py           # Pydantic schemas for API responses
├── .github/workflows/       # CI/CD workflows
├── Dockerfile              # Docker configuration
├── pyproject.toml          # Project configuration and dependencies
└── README.md               # This file
```

## 🧪 Example Usage

### Register a Model
```bash
curl -X POST "http://localhost:8000/model/register" \
     -H "Content-Type: application/json" \
     -d '{"run_id": "experiment_id/run_id", "model_name": "my_model"}'
```

### Get Model Parameters
```bash
curl "http://localhost:8000/model/parameters?name=my_model&version=1"
```

### Make Predictions
```bash
curl -X POST "http://localhost:8000/model/predict?name=my_model" \
     -F "file=@data.csv"
```

### Export a Model
```bash
curl "http://localhost:8000/model/export?name=my_model&version=1" \
     --output my_model_v1.zip
```

## 🔒 Security & Configuration

The API uses environment variables for configuration. Ensure secure storage of credentials:

- Use strong passwords for MLflow authentication
- Secure S3/MinIO access keys
- Consider using Docker secrets or Kubernetes secrets in production
- Enable TLS in production environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [MLflow](https://mlflow.org/)
- Supports multiple ML frameworks
- Created by SIEMENS SRL

---

<div align="center">

**Made with ❤️ for the ML community**

</div>
