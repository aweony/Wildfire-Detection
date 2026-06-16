import logging
from pathlib import Path
import cv2
import torch
from PIL import Image
from multiclass.multi_model import MultiClassModel
from preprocessing import test_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CLASS_NAMES = ["fire", "no fire", "smoke", "active fire"]
DEFAULT_MODEL_PATH = Path(__file__).parent / "model.pth"
DEFAULT_IMAGE_PATH = Path(__file__).parent / "fire.jpg"

def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> MultiClassModel:
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'.\n"
            "Train the model first by running:\n"
            "  python src/multiclass/train_multi.py"
        )
    # Create a MultiClassModel object
    model = MultiClassModel(num_classes=len(CLASS_NAMES), freeze_backbone=False)
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    # Put model into evaluation mode
    model.eval()
    return model

def predict_image(image_path: Path = DEFAULT_IMAGE_PATH, model_path: Path = DEFAULT_MODEL_PATH) -> str:
    model = load_model(model_path)

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    tensor = test_transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

    prediction = CLASS_NAMES[pred_idx]
    logging.info(f"Prediction for '{image_path}': {prediction}")
    return prediction

def predict_frame(frame, model: MultiClassModel) -> str:
    # Convert BGR frame from OpenCV to RGB PIL Image and apply transformations
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = test_transform(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

    prediction = CLASS_NAMES[pred_idx]
    logging.info(f"Prediction for current frame: {prediction}")
    return prediction


if __name__ == "__main__":
    result = predict_image(DEFAULT_IMAGE_PATH, DEFAULT_MODEL_PATH)
    print(f"Result: {result}")
