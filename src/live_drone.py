import cv2
import logging
from inference import predict_frame, load_model, DEFAULT_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def drone_camera_loop(camera_index=0):
    camera = cv2.VideoCapture(camera_index)  # Placeholder for drone camera feed

    if not camera.isOpened():
        logging.error("Drone camera can't be opened")
        return

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

        # Overlay the prediction on the frame
        cv2.putText(frame, prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("WildFire Detection Drone Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Drone camera turned off")
            break

    camera.release()
    cv2.destroyAllWindows()