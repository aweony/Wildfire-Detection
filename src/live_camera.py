import cv2
import logging
from inference import predict_frame, load_model, DEFAULT_MODEL_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    logging.error("Camera can't be open")
    exit()

logging.info("Camera loaded successfully")

# load the model once before the loop
model = load_model(DEFAULT_MODEL_PATH)

while True:
    success, frame = camera.read()

    if not success:
        logging.error("Frame couldn't be read")
        break

    # predict the class of the current frame
    prediction = predict_frame(frame, model)

    # overlay the prediction on the frame
    cv2.putText(frame, prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("WildFire Detection Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Camera turned off")
        break

camera.release()
cv2.destroyAllWindows()
