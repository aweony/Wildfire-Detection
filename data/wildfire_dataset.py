import kagglehub
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
logging.info("Path to dataset files: %s", path)
# create the path
destination = Path(__file__).parent / "raw"
destination.mkdir(parents=True, exist_ok=True)
# move path to the raw folder
for item in Path(path).iterdir():
    shutil.move(str(item), destination)
logging.info("Dataset moved to data/raw/")