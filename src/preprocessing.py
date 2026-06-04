from pathlib import Path
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

Image.MAX_IMAGE_PIXELS = None

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "the_wildfire_dataset_2n_version"
# transform the data to tensors for train and test 
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
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
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)

# Loads the dataset
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
