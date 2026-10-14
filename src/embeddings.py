import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_CSV = os.path.join(DATA_DIR, "splits", "train.csv")

IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2")
]

OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBEDDING_FILE = os.path.join(
    OUTPUT_DIR,
    "train_embeddings.npy"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "train_embedding_metadata.csv"
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n========== EMBEDDING GENERATION ==========\n")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# IMAGE PATH LOOKUP
# ============================================================

print("\nBuilding image path lookup...")

image_lookup = {}

for folder in IMAGE_DIRS:

    for filename in os.listdir(folder):

        if filename.lower().endswith(".jpg"):

            image_id = os.path.splitext(filename)[0]

            image_lookup[image_id] = os.path.join(
                folder,
                filename
            )

print(f"Images indexed: {len(image_lookup):,}")

# ============================================================
# LOAD TRAINING METADATA
# ============================================================

df = pd.read_csv(TRAIN_CSV)

print(f"Reference images: {len(df):,}")

# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class HAM10000Dataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_id = row["image_id"]

        image_path = image_lookup[image_id]

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        return image, image_id


# ============================================================
# DATALOADER
# ============================================================

dataset = HAM10000Dataset(df)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# ============================================================
# LOAD RESNET-50
# ============================================================

print("\nLoading pretrained ResNet-50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(weights=weights)

# Remove final classification layer
model.fc = torch.nn.Identity()

model = model.to(device)

model.eval()

print("ResNet-50 loaded.")

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

all_embeddings = []
all_image_ids = []

print("\nGenerating embeddings...\n")

with torch.no_grad():

    for images, image_ids in tqdm(
        loader,
        desc="Processing images"
    ):

        images = images.to(
            device,
            non_blocking=True
        )

        embeddings = model(images)

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(
            embeddings,
            p=2,
            dim=1
        )

        embeddings = embeddings.cpu().numpy()

        all_embeddings.append(embeddings)

        all_image_ids.extend(image_ids)

# ============================================================
# COMBINE
# ============================================================

embeddings = np.vstack(all_embeddings)

print("\n========== RESULTS ==========\n")

print(f"Embedding shape: {embeddings.shape}")

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

np.save(
    EMBEDDING_FILE,
    embeddings
)

metadata = df.copy()

metadata.to_csv(
    METADATA_FILE,
    index=False
)

print(f"\nEmbeddings saved to:")
print(EMBEDDING_FILE)

print("\nMetadata saved to:")
print(METADATA_FILE)

print("\n========== COMPLETE ==========\n")