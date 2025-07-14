import os
import yaml
import json
import mlflow
import base64
import shutil
import zipfile
import tempfile
import pandas as pd
from datetime import datetime
from json import JSONDecodeError
from io import StringIO, BytesIO
from typing import Any, Dict, List
from mlflow_api.models import model_handlers
from mlflow.artifacts import download_artifacts
from mlflow import MlflowClient, MlflowException

def flatten_dict(d: dict, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Client:
    def __init__(self):
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv('MLFLOW_S3_ENDPOINT_URL')
        os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = os.getenv('MLFLOW_TRACKING_INSECURE_TLS')
        os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        self.client = MlflowClient()

    def models(self):
        returns = []
        try:
            models_list = self.client.search_registered_models()
        except Exception:
            return None
    
        if len(models_list) == 0:
            return []

        for model in models_list:
            timestamp_s = model.creation_timestamp / 1000
            model_info = {
                "name": model.name,
                "framework": model.tags.get("framework", ""),
                "description": model.description,
                "tags": model.tags,
                "versions": []
            }
 
            for version in model.latest_versions:
                timestamp_s = version.creation_timestamp / 1000
                version_info = {
                    "version": version.version,
                    "creation_date": datetime.utcfromtimestamp(timestamp_s).strftime('%Y-%m-%d %H:%M:%S'),
                    "input_examples": None,
                    "output_examples": None,
                }

                model_uri = self.client.get_model_version_download_uri(model.name, version.version)
                remote_model_info = mlflow.models.get_model_info(model_uri)
                signature =remote_model_info.signature

                if signature:
                    version_info["input_examples"] = json.loads(signature.to_dict()["inputs"])
                    version_info["output_examples"] = json.loads(signature.to_dict()["outputs"])
                    
                model_info["versions"].append(version_info)

            returns.append(model_info)

        return returns

    def model_parameters(self, name: str, version: str) -> Dict[str, Any] | None:
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id if version is None \
            else self.client.get_model_version(name, version).run_id
        if not run_id:
            return None
        parameters = self.client.get_run(run_id).data.params
        try:
            parameters = {k: json.loads(v.replace("'", '"')) for k, v in parameters.items()}
        except JSONDecodeError:
            pass
        return flatten_dict(parameters)

    def model_versions(self, name: str) -> List[Dict[str, str]] | None:
        versions = self.client.search_model_versions(f"name='{name}'")

        return [{"version": version.version,
                 "type": "" if version.tags.get("model_type") is None else version.tags["model_type"]}
                for version in versions]

    def model_metrics(self, name: str, version: str) -> Dict[str, Any] | None:
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id if version is None \
            else self.client.get_model_version(name, version).run_id
        metrics = self.client.get_run(run_id).data.metrics
        return metrics

    def model_dataset(self, name: str, version: str):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id if version is None \
            else self.client.get_model_version(name, version).run_id
        if not run_id:
            return None

        try:
            dataset = mlflow.artifacts.load_text(f"runs:/{run_id}/dataset")
            return pd.read_csv(StringIO(dataset))
        except MlflowException:
            return None

    def model_images(self, name: str, version: str):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id if version is None \
            else self.client.get_model_version(name, version).run_id

        if not run_id:
            return None
        try:
            artifacts = self.client.list_artifacts(run_id, path="figures")
            images = {}
            for artifact in artifacts:
                print(artifact.path)
                if ".png" in artifact.path:
                    image = mlflow.artifacts.load_image(f"runs:/{run_id}/{artifact.path}")
                    buffered = BytesIO()
                    image_format = image.format if image.format else 'PNG'
                    image.save(buffered, format=image_format)
                    images[str(artifact.path).split("/")[-1]] = (f"data:image/png;base64,"
                                                                 f"{(base64.b64encode(buffered.getvalue()).decode('utf-8'))}")
        except MlflowException:
            return None

        return images

    def model_predict(self, name: str, df: pd.DataFrame):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id
        if not run_id:
            return None

        artifacts = self.client.list_artifacts(run_id)

        artifacts = [artifact.path for artifact in artifacts if "model" in artifact.path and artifact.is_dir]

        content = mlflow.artifacts.load_text(f"runs:/{run_id}/{artifacts[0]}/MLmodel")

        content = yaml.safe_load(StringIO(content))

        model_type = content["flavors"]["python_function"]["loader_module"].split(".")[1]

        if model_type in list(model_handlers.keys()):
            handler = model_handlers[model_type]
            handler.load_model(model_uri=f"runs:/{run_id}/{artifacts[0]}")
            return handler.predict(df)

        return None

    def model_package(self, name: str):
        run_id = self.client.get_registered_model(name).latest_versions[0].run_id

        if not run_id:
            return None
        
        artifacts = self.client.list_artifacts(run_id)

        artifacts = [artifact.path for artifact in artifacts if "model" in artifact.path and artifact.is_dir]

        content = mlflow.artifacts.load_text(f"runs:/{run_id}/{artifacts[0]}/MLmodel")
               
        content = yaml.safe_load(StringIO(content))

        model_type = content["flavors"]["python_function"]["loader_module"].split(".")[1]

        if model_type in list(model_handlers.keys()):
            handler = model_handlers[model_type]
            return handler.save_model(f"runs:/{run_id}/{artifacts[0]}")

        return None

    def model_export(self, model_name: str, model_version: str) -> BytesIO:
        temp_dir = tempfile.mkdtemp(prefix="mlflow_export_")
        zip_buffer = BytesIO()

        try:
            model_version_details = self.client.get_model_version(model_name, model_version)
            run_id = model_version_details.run_id
        
            model_metadata = {
                "name": model_name,
                "version": model_version,
                "run_id": run_id,
                "description": model_version_details.description or "",
                "status": model_version_details.status,
                "stage": model_version_details.current_stage,
                "tags": model_version_details.tags
            }
        
            model_dir = os.path.join(temp_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(model_metadata, f, indent=2)
        
            run_dir = os.path.join(temp_dir, "runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
        
            run = self.client.get_run(run_id)
            run_data = {
                "info": {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "experiment_name": self.client.get_experiment(run.info.experiment_id).name,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "lifecycle_stage": run.info.lifecycle_stage,
                },
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags
            }
        
            with open(os.path.join(run_dir, "metadata.json"), "w") as f:
                json.dump(run_data, f, indent=2)
        
            artifacts_dir = os.path.join(run_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            download_artifacts(artifact_uri=model_version_details.source, dst_path=artifacts_dir, tracking_uri=os.getenv("MLFLOW_TRACKING_URI"))
        
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
        
            zip_buffer.seek(0)
            return zip_buffer
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def model_import(self, zip_path: str, target_experiment_name: str="Default", register_model: bool=True):
        temp_dir = tempfile.mkdtemp(prefix="mlflow_import_")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
    
            with open(os.path.join(temp_dir, "model", "metadata.json"), "r") as f:
                model_metadata = json.load(f)
    
            model_name = model_metadata["name"]
            original_run_id = model_metadata["run_id"]
    
            run_dir = os.path.join(temp_dir, "runs", original_run_id)
            with open(os.path.join(run_dir, "metadata.json"), "r") as f:
                run_data = json.load(f)
    
            if target_experiment_name:
                experiment = self.client.get_experiment_by_name(target_experiment_name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(target_experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            else:
                experiment_name = run_data["info"].get("experiment_name", "Imported Model")
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = self.client.create_experiment(experiment_name)
                else:
                    experiment_id = experiment.experiment_id
    
            with mlflow.start_run(experiment_id=experiment_id) as run:
                new_run_id = run.info.run_id
        
                for key, value in run_data["params"].items():
                    mlflow.log_param(key, value)
        
                for key, value in run_data["metrics"].items():
                    mlflow.log_metric(key, float(value))
        
                system_tags = ["mlflow.runName", "mlflow.parentRunId", "mlflow.user", 
                              "mlflow.source.name", "mlflow.source.type"]
                for key, value in run_data["tags"].items():
                    if key not in system_tags:
                        mlflow.set_tag(key, value)
        
                mlflow.set_tag("mlflow.imported", "true")
                mlflow.set_tag("mlflow.original_run_id", original_run_id)
        
                artifacts_dir = os.path.join(run_dir, "artifacts")
                mlflow.log_artifacts(artifacts_dir, run_id=new_run_id)
        
                if register_model:
                    model_uri = f"runs:/{new_run_id}/model"
                
                    try:
                        self.client.get_registered_model(model_name)
                    except Exception:
                        self.client.create_registered_model(
                            name=model_name, 
                            description=f"Imported model from {os.path.basename(zip_path)}",
                            tags=model_metadata.get("tags", {})
                        )
                
                    mv = self.client.create_model_version(
                        name=model_name,
                        source=model_uri,
                        run_id=new_run_id,
                        description=model_metadata.get("description", "Imported model version")
                    )
                
                    if model_metadata.get("tags"):
                        for key, value in model_metadata.get("tags", {}).items():
                            self.client.set_model_version_tag(model_name, mv.version, key, value)
                
                    if model_metadata.get("stage") not in ["None", None]:
                        self.client.transition_model_version_stage(
                            name=model_name,
                            version=mv.version,
                            stage=model_metadata.get("stage")
                        )
                
                    result = {
                        "original_model_name": model_metadata["name"],
                        "original_model_version": model_metadata["version"],
                        "original_run_id": original_run_id,
                        "new_run_id": new_run_id,
                        "experiment_id": experiment_id,
                        "model_registered": register_model,
                        "new_model_name": model_name,
                        "new_model_version": mv.version
                    }
                else:
                    result = {
                        "original_model_name": model_metadata["name"],
                        "original_model_version": model_metadata["version"],
                        "original_run_id": original_run_id,
                        "new_run_id": new_run_id,
                        "experiment_id": experiment_id,
                        "model_registered": register_model,
                        "new_model_name": None,
                        "new_model_version": None
                    }
        
                return result
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    @staticmethod
    def model_register(run_id: str, model_name: str):
        return mlflow.register_model(f"runs:/{run_id}", model_name)
