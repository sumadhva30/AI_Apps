import json
from langchain.llms.azureml_endpoint import ContentFormatterBase


class MLFlowFormatter(ContentFormatterBase):
    """Content handler for MLFlow models on Azure ML"""

    def format_request_payload(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(
            {"input_data": {"columns": [0], "index": [0], "data": [prompt]}}
        )
        return str.encode(input_str)

    def format_response_payload(self, output: bytes) -> str:
        response_json = json.loads(output)
        return str(response_json[0][0])
