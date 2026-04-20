from flask import Flask, request, jsonify, Response, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path
import traceback
import requests
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image

# ==================================
# APP
# ==================================

app = Flask(__name__, template_folder="templates")
CORS(app)

# ==================================
# PATHS
# ==================================

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
MODEL_PATH = ROOT / "models" / "acne_model.pt"

UPLOAD_DIR.mkdir(exist_ok=True)

# ==================================
# SETTINGS
# ==================================

HUMAN_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ESP32_TIMEOUT = 10

device = torch.device("cpu")
model = None
class_names = HUMAN_LABELS
img_size = 224

# ==================================
# MODEL BUILD
# ==================================

def build_model(num_classes):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ==================================
# HELPERS
# ==================================

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def to_human_label(name):
    if isinstance(name, str) and name.isdigit():
        i = int(name)
        if 0 <= i < len(HUMAN_LABELS):
            return HUMAN_LABELS[i]
    return str(name)

def explain_label(lbl, conf):
    if conf < 0.55:
        return f"Low confidence ({conf:.2f}). Try better lighting."
    if lbl == "Normal":
        return "Looks normal / low acne signs."
    if lbl == "Mild":
        return "Mild acne signs detected."
    if lbl == "Moderate":
        return "Moderate acne inflammation detected."
    if lbl == "Severe":
        return "Severe acne detected."
    return "Prediction generated."

def get_ip():
    ip = request.args.get("ip", "").strip()
    if not ip:
        raise Exception("Missing ESP32 IP")
    return ip

# ==================================
# LOAD MODEL
# ==================================

try:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=device)

    state_dict = ckpt["model_state"]
    img_size = int(ckpt.get("img_size", 224))
    class_to_idx = ckpt.get("class_to_idx", None)

    if class_to_idx:
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        class_names = HUMAN_LABELS

    model = build_model(len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("✅ Model loaded")
    print("Classes:", class_names)

except Exception:
    print("❌ Model load failed")
    traceback.print_exc()

# ==================================
# IMAGE TRANSFORM
# ==================================

tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    ),
])

# ==================================
# HOME PAGE
# ==================================

@app.get("/")
def home():
    return render_template("index.html")

# ==================================
# BACKEND HEALTH
# ==================================

@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "model_loaded": model is not None,
        "img_size": img_size,
        "class_names": class_names
    })

# ==================================
# IMAGE PREDICTION
# ==================================

@app.post("/api/analyze-image")
def analyze_image():

    try:

        if model is None:
            return jsonify({"error":"Model not loaded"}),500

        if "image" not in request.files:
            return jsonify({"error":"No image file"}),400

        f = request.files["image"]

        if not allowed_file(f.filename):
            return jsonify({"error":"Unsupported file"}),400

        filename = secure_filename(f.filename)

        save_path = UPLOAD_DIR / filename
        f.save(save_path)

        img = Image.open(save_path).convert("RGB")

        x = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():

            logits = model(x)

            probs_t = torch.softmax(logits,dim=1)[0].cpu()

            probs = probs_t.tolist()

            pred_idx = int(torch.argmax(probs_t))

        raw_label = class_names[pred_idx]

        pred_label = to_human_label(raw_label)

        confidence = float(probs[pred_idx])

        return jsonify({
            "pred_idx": pred_idx,
            "raw_label": raw_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "probs": probs,
            "explain": explain_label(pred_label,confidence)
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error":"Prediction failed",
            "details":str(e)
        }),500

# ==================================
# SIMPLE CHAT
# ==================================

@app.post("/api/chat")
def chat():

    try:

        data = request.get_json()

        question = data.get("question","")

        live = data.get("live",{})

        temp = live.get("tempC")
        ph = live.get("phValue")
        redness = live.get("image",{}).get("rednessPct")

        msg = "AI Explanation:\n"

        if temp:
            msg += f"Temperature: {temp} °C\n"

        if ph:
            msg += f"Skin pH: {ph}\n"

        if redness:
            msg += f"Redness: {redness}%\n"

        msg += f"\nQuestion: {question}"

        return jsonify({"reply":msg})

    except Exception as e:

        return jsonify({"error":str(e)}),500

# ==================================
# ESP32 HEALTH
# ==================================

@app.get("/api/esp32/health")
def esp32_health():

    try:

        ip = get_ip()

        # try /health
        try:
            r = requests.get(f"http://{ip}/health",timeout=5)
            if r.status_code == 200:
                return jsonify({"ok":True,"endpoint":"/health"})
        except:
            pass

        # try /capture
        try:
            r = requests.get(f"http://{ip}/capture",timeout=5)
            if r.status_code == 200:
                return jsonify({"ok":True,"endpoint":"/capture"})
        except:
            pass

        # try /stream
        try:
            r = requests.get(f"http://{ip}/stream",timeout=5)
            if r.status_code == 200:
                return jsonify({"ok":True,"endpoint":"/stream"})
        except:
            pass

        # try /data
        try:
            r = requests.get(f"http://{ip}/data",timeout=5)
            if r.status_code == 200:
                return jsonify({"ok":True,"endpoint":"/data"})
        except:
            pass

        return jsonify({"ok":False,"error":"ESP32 not responding"}),500

    except Exception as e:

        return jsonify({"error":str(e)}),500

# ==================================
# ESP32 DATA
# ==================================

@app.get("/api/esp32/data")
def esp32_data():

    try:

        ip = get_ip()

        r = requests.get(f"http://{ip}/data",timeout=ESP32_TIMEOUT)

        return Response(
            r.content,
            content_type="application/json"
        )

    except Exception as e:

        return jsonify({"error":str(e)}),500

# ==================================
# ESP32 CAPTURE
# ==================================

@app.get("/api/esp32/capture")
def esp32_capture():

    try:

        ip = get_ip()

        r = requests.get(f"http://{ip}/capture",timeout=ESP32_TIMEOUT)

        return Response(
            r.content,
            content_type="image/jpeg"
        )

    except Exception as e:

        return jsonify({"error":str(e)}),500

# ==================================
# ESP32 STREAM
# ==================================

@app.get("/api/esp32/stream")
def esp32_stream():

    try:

        ip = get_ip()

        r = requests.get(
            f"http://{ip}/stream",
            stream=True,
            timeout=ESP32_TIMEOUT
        )

        def generate():

            for chunk in r.iter_content(chunk_size=1024):

                if chunk:
                    yield chunk

        return Response(
            generate(),
            content_type="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:

        return jsonify({"error":str(e)}),500

# ==================================
# MAIN
# ==================================

if __name__ == "__main__":

    print("🚀 Flask backend running")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )