# MLflow ONNX Rust

A comparison of performance between serving models trained via popular python machine learning libraries. Models will be benchmarked via Flask and FastAPI. Then we test whether packaing these as MLflow artifacts has any significance. The same tests will be executed after exporting these models to the standard [ONNX](https://github.com/onnx/onnx) format. Lastly, I will serve these ONNX models in a Rust webserver.

My hypothesis is that data scientists can train models in python using the library of their choice, registering them in MLflow to inherit package dependency capturing and a standardized model packaging format. The MLflow SDK can then be leveraged to export these models to ONNX, and then deployed as Rust microservices for performance gains.


## Getting Started

### Starting MLflow

```bash

mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root /home/username/Projects/mlflow-onnx-rust/mlruns

```

### Starting a Server

```bash

usage: main.py [-h] --model_name MODEL_NAME --model_version MODEL_VERSION [--mlflow_tracking_uri MLFLOW_TRACKING_URI] [--temporary_model_file TEMPORARY_MODEL_FILE]

CLI to switch which model to serve.

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
  --model_version MODEL_VERSION
  --mlflow_tracking_uri MLFLOW_TRACKING_URI
  --temporary_model_file TEMPORARY_MODEL_FILE
```

```bash

python fastapi/main.py --model_name binary-classifier --model_version 4

```

### Load Testing

```bash



```