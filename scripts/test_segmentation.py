from inference_sdk import InferenceHTTPClient
import supervision as sv
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com", # use local inference server
    api_key="qryUFbPf3rlnsSdDBYAR"
)

result = client.run_workflow(
    workspace_name="dinhai",
    workflow_id="custom-workflow",
    images={
        "image": "scripts/realsense_20250416_163647.jpg"
    }
)

detections = sv.Detections.from_inference(result)