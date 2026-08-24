from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

Image.MAX_IMAGE_PIXELS = None

DATA_DIR = Path(__file__).parent.parent / "data" / "multi_raw" / "Forect Fire" / "Forest Fire_Dataset"

# transform the data to tensors for train and test 
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # teaches the model to use texture/structure, not just warm colors
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# gets the image from the data folder
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=test_transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)

# The raw dataset has 4 folders (fire, nofire, smoke, smokefire) but "fire" and
# "smokefire" (active fire) are too visually similar to separate reliably, so
# they're merged into a single "fire" class here. ImageFolder always assigns
# indices alphabetically, so class_to_idx is identical across the train/val/test
# splits (same folder names) and this mapping applies to all three.
_ACTIVE_FIRE_INDEX = train_dataset.class_to_idx["smokefire"]
_FIRE_INDEX = train_dataset.class_to_idx["fire"]


def _merge_active_fire_into_fire(raw_index: int) -> int:
    return _FIRE_INDEX if raw_index == _ACTIVE_FIRE_INDEX else raw_index


for dataset in (train_dataset, val_dataset, test_dataset):
    dataset.target_transform = _merge_active_fire_into_fire

CLASS_NAMES = ["fire", "no fire", "smoke"]

# Loads the dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
