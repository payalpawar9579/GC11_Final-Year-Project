import sys
import json
from pathlib import Path
import traceback
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "acne_model.pt"

HUMAN_LABELS = ["Normal", "Mild", "Moderate", "Severe"]


def build_model(num_classes: int):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def to_human_label(name):
    if isinstance(name, str) and name.isdigit():
        idx = int(name)
        if 0 <= idx < len(HUMAN_LABELS):
            return HUMAN_LABELS[idx]
    return str(name)


def severity_from_idx(idx: int):
    mapping = {
        0: "Low",
        1: "Mild",
        2: "Moderate",
        3: "High"
    }
    return mapping.get(idx, "Unknown")


def build_explanation(pred_label, confidence, temp_c, ph_value):
    parts = []
    parts.append(f"Model predicted {pred_label} with confidence {round(confidence * 100, 1)}%.")

    if temp_c > 0:
        if temp_c >= 37.5:
            parts.append("Temperature is high, which may indicate stronger inflammation.")
        elif temp_c >= 35.0:
            parts.append("Temperature is slightly elevated.")
        else:
            parts.append("Temperature is within a typical skin-surface range.")

    if ph_value > 0:
        if ph_value >= 7.5:
            parts.append("Skin pH is alkaline, which may increase irritation or acne risk.")
        elif ph_value >= 6.5:
            parts.append("Skin pH is slightly imbalanced.")
        else:
            parts.append("Skin pH is in a relatively safer range.")

    return " ".join(parts)


def build_recommendation(pred_label, temp_c, ph_value):
    if pred_label.lower() == "severe" or temp_c >= 37.5 or ph_value >= 7.5:
        return "High-risk indication. Dermatologist consultation is recommended."
    if pred_label.lower() == "moderate" or temp_c >= 35.0 or ph_value >= 6.8:
        return "Moderate-risk indication. Maintain skin hygiene and monitor symptoms closely."
    if pred_label.lower() == "mild":
        return "Mild condition indicated. Use gentle skincare and avoid irritants."
    return "No major issue detected. Continue routine skin care and observation."


def main(img_path: str, temp_c: float = 0.0, ph_value: float = 0.0):
    try:
        device = torch.device("cpu")

        if not MODEL_PATH.exists():
            print(json.dumps({
                "success": False,
                "error": f"Model file not found at {MODEL_PATH}"
            }))
            sys.exit(1)

        image_path = Path(img_path)
        if not image_path.exists():
            print(json.dumps({
                "success": False,
                "error": f"Image file not found at {image_path}"
            }))
            sys.exit(1)

        ckpt = torch.load(MODEL_PATH, map_location=device)

        if "model_state" not in ckpt:
            print(json.dumps({
                "success": False,
                "error": "Invalid checkpoint format. 'model_state' key not found."
            }))
            sys.exit(1)

        state_dict = ckpt["model_state"]
        class_to_idx = ckpt.get("class_to_idx", None)
        img_size = int(ckpt.get("img_size", 224))

        if class_to_idx:
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        else:
            class_names = ["0", "1", "2", "3"]

        num_classes = len(class_names)

        model = build_model(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        tfm = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        img = Image.open(image_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()
            pred_idx = int(torch.argmax(probs).item())

        raw_class = class_names[pred_idx]
        pred_label = to_human_label(raw_class)
        confidence = float(probs[pred_idx])
        all_probs = [round(float(p), 4) for p in probs]

        severity = severity_from_idx(pred_idx)
        explain = build_explanation(pred_label, confidence, temp_c, ph_value)
        recommendation = build_recommendation(pred_label, temp_c, ph_value)

        result = {
            "success": True,
            "pred_idx": pred_idx,
            "pred_label": pred_label,
            "prediction": pred_label,
            "severity": severity,
            "confidence": round(confidence, 4),
            "probs": all_probs,
            "tempC": round(float(temp_c), 2),
            "phValue": round(float(ph_value), 2),
            "explain": explain,
            "recommendation": recommendation,
            "imagePath": str(image_path).replace("\\", "/")
        }

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": "Prediction failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }))
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
      print(json.dumps({
          "success": False,
          "error": "Usage: python predict_one.py path_to_image.jpg [tempC] [phValue]"
      }))
      sys.exit(1)

    image_arg = sys.argv[1]
    temp_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    ph_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    main(image_arg, temp_arg, ph_arg)