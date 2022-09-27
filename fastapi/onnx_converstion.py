import json

import mlflow
import numpy as np
import onnx
import torch

import onnxmltools
from onnxconverter_common.data_types import FloatTensorType

from skl2onnx import convert_sklearn

class OnnxConversion:
    def __init__(
        self,
        model_name,
        model_version,
        mlflow_tracking_uri: str = "http://localhost:5000",
        temporary_model_file: str = "model.onnx",
    ):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.model_name = model_name
        self.model_version = model_version
        self.temporary_model_file = temporary_model_file
        self.model_uri = f"models:/{self.model_name}/{self.model_version}"

        try:
            model = mlflow.pyfunc.load_model(self.model_uri)
            self.ml_flavor = model.metadata.to_dict()["flavors"]["python_function"][
                "loader_module"
            ]

            input_example_artifact_path = model.metadata.saved_input_example_info[
                "artifact_path"
            ]
            input_example = mlflow.artifacts.load_dict(
                f"{self.model_uri}/{input_example_artifact_path}"
            )["data"]
            self.number_of_features = np.array(input_example).shape[-1]
            self.initial_type = [
            ("float_input", FloatTensorType([None, self.number_of_features]))
        ]

        except Exception as e:
            msg = f"OnnxConversion currently only supports Python Models: {e}"
            ValueError(msg)
            print(msg)

    def __write_model_to_file(self, onnx_model):

        with open(self.temporary_model_file, 'wb') as f:
            f.write(onnx_model.SerializeToString())
            f.close()
        
        return True

    def _convert_sklearn_model(self):

        model = mlflow.sklearn.load_model(self.model_uri)
        onx = onnxmltools.convert_sklearn(
            model, 
            initial_types=self.initial_type
        )
        success = self.__write_model_to_file(onx)

        return success

    def _convert_xgboost_model(self):

        model = mlflow.xgboost.load_model(self.model_uri)
        onx = onnxmltools.convert_xgboost(model, initial_types=self.initial_type)
        success = self.__write_model_to_file(onx)
        return success

    def _convert_h20_model(self):

        model = mlflow.h2o.load_model(self.model_uri)
        onx = onnxmltools.convert_h20(model, initial_types=self.initial_type)
        success = self.__write_model_to_file(onx)
        return success

    def _convert_keras_model(self):

        model = mlflow.keras.load_model(self.model_uri)
        onx = onnxmltools.convert_keras(model, initial_types=self.initial_type)
        success = self.__write_model_to_file(onx)
        return success

    def _convert_lightgbm_model(self):

        model = mlflow.lightgbm.load_model(self.model_uri)
        onx = onnxmltools.convert_lightgbm(model, initial_types=self.initial_type)
        success = self.__write_model_to_file(onx)
        return success

    def _convert_spark_model(self):

        model = mlflow.spark.load_model(self.model_uri)
        onx = onnxmltools.convert_sparkml(model, initial_types=self.initial_type)
        success = self.__write_model_to_file(onx)
        return success

    def _convert_tensorflow_model(self):

        model = mlflow.tensorflow.load_model(self.model_uri)
        onx = onnxmltools.convert_tensorflow(model)
        success = self.__write_model_to_file(onx)
        return success

    def _convert_pytorch_model(self):

        model = mlflow.pytorch.load_model(self.model_uri)

        torch.onnx.export(
            model,
            torch.rand((1, self.number_of_features), dtype=torch.float32),
            self.temporary_model_file,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["float_input"],  # the model's input names
            output_names=["output_label"],  # the model's output names
            dynamic_axes={
                "float_input": {0: "batch_size"},  # variable length axes
                "output_label": {0: "batch_size"},
            },
        )
        return True

    def convert_model_to_onnx(self):

        completed = False

        if self.ml_flavor == "mlflow.sklearn":
            completed = self._convert_sklearn_model()

        if self.ml_flavor == "mlflow.pytorch":
            completed = self._convert_pytorch_model()

        if self.ml_flavor == 'mlflow.xgboost':
            completed = self._convert_xgboost_model()
        
        if self.ml_flavor == 'mlflow.spark':
            completed = self._convert_spark_model()

        if self.ml_flavor == 'mlflow.keras':
            completed = self._convert_keras_model()

        if self.ml_flavor == 'mlflow.tensorflow':
            completed = self._convert_tensorflow_model()

        if self.ml_flavor == 'mlflow.h20':
            completed = self._convert_h20_model()

        if self.ml_flavor == 'mlflow.lightgbm':
            completed = self._convert_lightgbm_model()

        assert completed == True, f"Unable to convert model to ONNX."
        onnx_model = onnx.load(self.temporary_model_file)
        onnx.checker.check_model(onnx_model)
        return self.temporary_model_file
