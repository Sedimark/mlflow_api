# SEDIMARK MLFlow API

Rest API for interating with a local MLFlow instance

## Usage

- uv
  
  ### Requirements
  
  - uv >= 0.5.0 
  - python >= 3.11.0
    
  ### Running
    
    ```bash
    uv build .
    cd dist
    pip install *.whl
    mlflow_api
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
  
  ```bash
  docker build -t mlflow_api .
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
