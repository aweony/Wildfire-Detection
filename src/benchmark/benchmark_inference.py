from time import perf_counter
from src.multiclass.multi_model import MultiClassModel
from src.inference import predict_frame, load_model, DEFAULT_MODEL_PATH
import cv2
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
LOAD_MODEL = []
FPS = []
AVERAGE_TIME = []

def load_model_with_timing(model_path: str = DEFAULT_MODEL_PATH) -> MultiClassModel:
    start = perf_counter()
    # Load the model once before the loop
    model = load_model(DEFAULT_MODEL_PATH)
    load_time = perf_counter() - start
    LOAD_MODEL.append(load_time)
    return model

def predict_frame_with_timing(frame, model) -> None:
    times = []
    for _ in range(100):
        start = perf_counter()
        prediction = predict_frame(frame, model)
        times.append(perf_counter() - start)
    average_time = sum(times) / len(times)
    fps = 1 / average_time
    FPS.append(fps)
    AVERAGE_TIME.append(average_time)

if __name__ == "__main__":
    for i in range(10):
        model = load_model_with_timing()
        frame = cv2.imread("data/multi_raw/Forect Fire/Forest Fire_Dataset/test/nofire/nofire_test_1011.jpg")  # Replace with the path to your test image
        if frame is None:
            logging.error("Benchmark frame couldn't be read")
        else:
            predict_frame_with_timing(frame, model)
    average_load_time = sum(LOAD_MODEL) / len(LOAD_MODEL)
    average_fps = sum(FPS) / len(FPS)
    average_inference_time = sum(AVERAGE_TIME) / len(AVERAGE_TIME)
    logging.info(f"Average model load time over 10 runs: {average_load_time:.4f} seconds")
    logging.info(f"Average FPS over 10 runs: {average_fps:.1f} FPS")
    logging.info(f"Average inference time over 10 runs: {average_inference_time:.4f} seconds/frame")