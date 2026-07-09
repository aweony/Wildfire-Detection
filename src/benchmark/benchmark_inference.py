from time import perf_counter
from src.multiclass.multi_model import MultiClassModel
from src.inference import predict_frame, load_model, DEFAULT_MODEL_PATH
import cv2
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
def load_model_with_timing(model_path: str = DEFAULT_MODEL_PATH) -> MultiClassModel:
    start = perf_counter()
    # Load the model once before the loop
    model = load_model(DEFAULT_MODEL_PATH)
    load_time = perf_counter() - start
    logging.info(f"Model loaded in {load_time:.4f} seconds")
    return model

def predict_frame_with_timing(frame, model) -> None:
    times = []
    for _ in range(100):
        start = perf_counter()
        prediction = predict_frame(frame, model)
        times.append(perf_counter() - start)
    average_time = sum(times) / len(times)
    fps = 1 / average_time
    logging.info(f"Average inference time: {average_time:.4f} seconds/frame")
    logging.info(f"Estimated speed: {fps:.1f} FPS")

if __name__ == "__main__":
    model = load_model_with_timing()
    frame = cv2.imread("data/multi_raw/Forect Fire/Forest Fire_Dataset/test/nofire/nofire_test_1011.jpg")  # Replace with the path to your test image
    if frame is None:
        logging.error("Benchmark frame couldn't be read")
    else:
        predict_frame_with_timing(frame, model)
