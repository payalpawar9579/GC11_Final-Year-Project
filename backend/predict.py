import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

MODEL_PATH = Path("models/acne_model.pt")

def build_model(num_classes):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def main(img_path: str):
    device = torch.device("cpu")

    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict = ckpt["model_state"]
    class_to_idx = ckpt.get("class_to_idx", None)
    img_size = int(ckpt.get("img_size", 224))

    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = ["0", "1", "2", "3"]

    model = build_model(len(class_names))
    model.load_state_dict(state_dict)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    print("Image:", img_path)
    print("Pred idx:", pred, "Class:", class_names[pred])
    print("Probs:", [round(float(p), 4) for p in probs])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path_to_image.jpg")
        sys.exit(1)
    main(sys.argv[1])