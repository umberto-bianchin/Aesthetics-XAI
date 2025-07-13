import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision.models.inception import InceptionOutputs

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Path configuration ===
# Replace with the right folder path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/datasets/top50/content/EVA_together'
    VOTES_CSV = '/content/drive/MyDrive/datasets/eva-dataset-master/data/votes_filtered.csv'
    CHECKPOINT_DIR = '/content/drive/MyDrive/models'
    print("Mounted Google Drive and set dataset paths.")
except ImportError:
    # Local paths (not running in Colab)
    BASE_DIR = 'datasets/top50/content/EVA_together'
    VOTES_CSV = 'datasets/eva-dataset-master/data/votes_filtered.csv'
    CHECKPOINT_DIR = 'checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Load votes and compute means ===
cols = ['image_id','user_id','score','difficulty','visual','composition','quality','semantic','vote_time','1','2','3','4']
votes = pd.read_csv(VOTES_CSV, sep='=', header=0, names=cols, engine='python')
means = (
    votes
    .groupby('image_id')[['score', 'difficulty', 'visual', 'composition', 'quality', 'semantic']]
    .mean()
    .reset_index()
)
means['image_id'] = means['image_id'].astype(str)
means.rename(columns={'score':'total'}, inplace=True)

# === Create labels dataframe ===
filepaths, totals, diffs, viss, comps, quals, sems = [], [], [], [], [], [], []
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    image_id = os.path.splitext(folder)[0]
    row = means[means['image_id'] == image_id]
    if row.empty:
        continue
    t, d, v, c, q, s = row[['total','difficulty','visual','composition','quality','semantic']].iloc[0]
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            fp = os.path.join(folder_path, fname)
            filepaths.append(fp)
            totals.append(t)
            diffs.append(d)
            viss.append(v)
            comps.append(c)
            quals.append(q)
            sems.append(s)

labels_df = pd.DataFrame({
    'filepath': filepaths,
    'total': totals,
    'difficulty': diffs,
    'visual': viss,
    'composition': comps,
    'quality': quals,
    'semantic': sems
})

# === Split dataset: train/val/test ===
df_train_val, df_test = train_test_split(labels_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(df_train_val, test_size=0.25, random_state=42)

# === Dataset class ===
class EVADataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels = self.df[['total','difficulty','visual','composition','quality','semantic']].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label

# === Transforms and DataLoaders ===
target_size = (299, 299)
batch_size = 32

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(target_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
transform_eval = transforms.Compose([
    transforms.Resize(target_size),
    transforms.CenterCrop(target_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = EVADataset(train_df, transform=transform_train)
val_dataset = EVADataset(val_df, transform=transform_eval)
test_dataset = EVADataset(df_test, transform=transform_eval)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# === Model definition ===
class InceptionMultiOut(nn.Module):
    def __init__(self):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT
        base = models.inception_v3(weights=weights, aux_logits=True)
        base.AuxLogits = None
        base.fc = nn.Identity()
        self.base = base
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.base(x)
        if isinstance(x, InceptionOutputs):
            x = x.logits
        return self.head(x)

# === Training function ===
def train_model(model, train_loader, val_loader, epochs, lr, checkpoint_path, unfreeze=False):
    if unfreeze:
        for param in model.base.parameters():
            param.requires_grad = True
    else:
        for param in model.base.parameters():
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                val_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

# === Evaluation function ===
def evaluate(model, loader):
    model.eval()
    criterion = nn.MSELoss()
    total_loss, total_mae = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            total_loss += criterion(preds, labels).item() * images.size(0)
            total_mae += torch.abs(preds - labels).mean().item() * images.size(0)
    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)

# === Main Execution ===
if __name__ == "__main__":
    model = InceptionMultiOut().to(device)

    train_model(model, train_loader, val_loader, epochs=20, lr=1e-4,
                checkpoint_path=os.path.join(CHECKPOINT_DIR, 'inception_stage1.pth'), unfreeze=False)

    for name, param in list(model.base.named_parameters())[-50:]:
        param.requires_grad = True

    train_model(model, train_loader, val_loader, epochs=10, lr=1e-5,
                checkpoint_path=os.path.join(CHECKPOINT_DIR, 'inception_stage2.pth'), unfreeze=True)

    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'inception_stage2.pth')))
    val_loss, val_mae = evaluate(model, val_loader)
    test_loss, test_mae = evaluate(model, test_loader)
    print(f"Validation: Loss = {val_loss:.4f}, MAE = {val_mae:.4f}")
    print(f"Test: Loss = {test_loss:.4f}, MAE = {test_mae:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'inception_multiout_final.pth'))