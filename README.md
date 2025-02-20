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
