import cv2
import logging
from src.inference import predict_frame, load_model, DEFAULT_MODEL_PATH
from src.detection.person_animal_detect import detect_person_animal
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
VIDEO_PATH = Path(__file__).resolve().parent / "drone.mp4"
camera_index = str(VIDEO_PATH)# Change this index based on your drone camera feed
camera = cv2.VideoCapture(camera_index) 

if not camera.isOpened():
    logging.error("Drone camera can't be opened")
    exit()  # Use exit() instead of return since this is not inside a function

logging.info("Drone camera loaded successfully")

            # Load the model once before the loop
model = load_model(DEFAULT_MODEL_PATH)

while True:
    success, frame = camera.read()

    if not success:
        logging.error("Frame couldn't be read from drone camera")
        break

    # Predict the class of the current frame
    prediction = predict_frame(frame, model)
    dect_frame = detect_person_animal(frame, prediction)

    # Overlay the prediction on the frame
    cv2.putText(dect_frame, prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("WildFire Detection Drone Camera", dect_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        logging.info("Drone camera turned off")
        break

camera.release()
cv2.destroyAllWindows()