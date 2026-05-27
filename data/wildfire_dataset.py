import kagglehub
from pathlib import Path
import shutil

path = kagglehub.dataset_download("elmadafri/the-wildfire-dataset")
print("Path to dataset files:", path)
# create the path
destination = Path(__file__).parent / "raw"
destination.mkdir(parents=True, exist_ok=True)
# move path to the raw folder
for item in Path(path).iterdir():
    shutil.move(str(item), destination)
print("Dataset moved to data/raw/")