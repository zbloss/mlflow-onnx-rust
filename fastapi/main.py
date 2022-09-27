import argparse
import json

import torch
import mlflow
import numpy as np
import onnxruntime
import uvicorn
from onnx_converstion import OnnxConversion
from onnx_request import OnnxRequest

from fastapi import FastAPI, Request

parser = argparse.ArgumentParser(description="CLI to switch which model to serve.")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--model_version", type=int, required=True)
parser.add_argument(
    "--mlflow_tracking_uri", type=str, required=False, default="http://localhost:5000"
)
parser.add_argument(
    "--temporary_model_file", type=str, required=False, default="model.onnx"
)

app = FastAPI(
    description="API Service that loads models from the MLflow Model Registry, then exports and serves them as ONNX Models"
)


@app.on_event("startup")
def startup():
    args, uargs = parser.parse_known_args()
    
    global session
    global input_name
    global label_name
    global model
    global model_predict

    onnx_converter = OnnxConversion(**vars(args))
    session = onnxruntime.InferenceSession(onnx_converter.convert_model_to_onnx())
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    
    print('onnx_converter.ml_flavor', onnx_converter.ml_flavor)

    if onnx_converter.ml_flavor == 'mlflow.sklearn':
        model = mlflow.sklearn.load_model(onnx_converter.model_uri)
        def model_predict(x):
            return model.predict(x)

    if onnx_converter.ml_flavor == 'mlflow.pytorch':
        model = mlflow.pytorch.load_model(onnx_converter.model_uri)
        def model_predict(x):
            return model(torch.tensor(x).float())       

    if onnx_converter.ml_flavor == 'mlflow.xgboost':
        model = mlflow.xgboost.load_model(onnx_converter.model_uri)
        def model_predict(x):
            return model.predict(x) 

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/healthcheck")
async def healthcheck():
    return 200


@app.post("/predict")
def predict(features: OnnxRequest):
    data = features.dict()["data"]

    prediction = session.run(
        [label_name], {input_name: np.array(data).astype(np.float32)}
    )
    return {"prediction": prediction[0].tolist()}


@app.post("/predict-without-onnx")
def predict(features: OnnxRequest):
    data = features.dict()["data"]
    prediction = model_predict(data)
    return {"prediction": prediction[0].tolist()}



if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True, debug="true")
