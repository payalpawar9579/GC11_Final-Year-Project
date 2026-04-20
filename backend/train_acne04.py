import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# =========================
# PATHS
# =========================
DATASET_DIR = Path(r"..\dataset")
OUT_DIR = Path(r".\models")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "acne_model.pt"

# =========================
# SETTINGS
# =========================
NUM_CLASSES = 4
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# DATA
# =========================
def check_dataset_structure():
    required_dirs = [
        DATASET_DIR / "train",
        DATASET_DIR / "val",
        DATASET_DIR / "test",
    ]

    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing dataset folder: {d.resolve()}")

    for split in ["train", "val", "test"]:
        for cls in ["0", "1", "2", "3"]:
            cls_dir = DATASET_DIR / split / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir.resolve()}")

def build_loaders():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATASET_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(DATASET_DIR / "test", transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

# =========================
# MODEL
# =========================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model

# =========================
# EVALUATION
# =========================
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc

# =========================
# TRAIN
# =========================
def train():
    print("Checking dataset...")
    check_dataset_structure()

    print("Device:", DEVICE)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders()

    print("Classes mapping:", train_ds.class_to_idx)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={time.time() - start:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "img_size": IMG_SIZE,
                },
                MODEL_PATH
            )
            print(f"Saved best model to {MODEL_PATH} (val_acc={best_val_acc:.4f})")

    print("\n=== Final Test Evaluation ===")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model file not found: {MODEL_PATH.resolve()}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nTraining complete.")
    print(f"Best model saved at: {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    train()