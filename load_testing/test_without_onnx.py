import json
from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    @task
    def predict(self):
        payload = json.dumps(
            {
                "data": [
                    [
                        1.0,
                        28.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                ]
            }
        )
        self.client.post("/predict-without-onnx", payload)
