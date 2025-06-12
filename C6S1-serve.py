import base64
from io import BytesIO

import mlflow.pytorch
import ray
import torch
import torchvision.transforms as transforms
from PIL import Image
from ray import serve

ray.init("ray://ray-cluster-kuberay-head-svc:10001")

# Read best model info
with open("/data/best_model_info.txt", "r") as f:
    lines = f.read().strip().split("\n")
    run_id = lines[0]
    model_type = lines[1]

# Load model from MLflow
mlflow.set_tracking_uri("http://mlflow-tracking.default")
model_uri = f"runs:/{run_id}/model"
print("Downloading model from %s" % model_uri)
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Same transforms as training
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
        transforms.Lambda(lambda x: transforms.functional.invert(x)),  # Invert brightness
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@serve.deployment(num_replicas=1)
class FingerCountingModel:
    def __init__(self):
        self.model = model
        self.transform = transform

    async def __call__(self, request):
        # Handle different input formats
        if hasattr(request, "json"):
            data = await request.json()
        else:
            data = request

        try:
            # Decode base64 image
            if "image" in data:
                image_data = base64.b64decode(data["image"])
                image = Image.open(BytesIO(image_data)).convert("RGB")
            else:
                return {"error": "No image provided"}

            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "model_type": model_type,
                "all_probabilities": probabilities[0].tolist(),
            }

        except Exception as e:
            return {"error": str(e)}


serve.run(FingerCountingModel.bind())
print("Model deployed successfully!")
