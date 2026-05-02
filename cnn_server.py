"""
CNN Inference Server — Olive Leaf Classifier
Model: MobileNetV3-Large fine-tuned on 3 classes
Classes loaded from class_info.json (same file used during training)

Endpoint: POST /classify  → { label, label_ar, confidence, all_scores }
Runs on  : http://localhost:8003
"""

import io
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("./models/best_model.pt")
CLASS_INFO_PATH = Path("./models/class_info.json")
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load class_info.json ──────────────────────────────────────────────────────
with open(CLASS_INFO_PATH, encoding="utf-8") as f:
    class_info = json.load(f)

CLASS_NAMES   = class_info["class_names"]        # e.g. ["aculus_olearius", "healthy", "peacock_spot"]
DARIJA_LABELS = class_info["darija_labels"]       # e.g. {"healthy": "ورقة سليمة", ...}
MEAN          = class_info["mean"]
STD           = class_info["std"]

# ── Image transform (same as training) ───────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ── Load model ────────────────────────────────────────────────────────────────
def load_model() -> nn.Module:
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier = nn.Sequential(
        nn.Linear(960, 512),
        nn.Dropout(0.2),
        nn.Hardswish(),
        nn.Linear(512, 128),
        nn.Dropout(0.2),
        nn.Hardswish(),
        nn.Linear(128, len(CLASS_NAMES)),
    )
    model.load_state_dict(checkpoint, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"✓ Model ready — classes: {CLASS_NAMES}")
    return model

model = load_model()

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Olive Leaf CNN Classifier")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES}

@app.post("/classify")
async def classify(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]

        pred_idx   = int(probs.argmax())
        pred_name  = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        all_scores = {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)}

        print(f"Classified: {pred_name} ({confidence:.1%}) | {all_scores}")

        return {
            "label":      pred_name,
            "label_ar":   DARIJA_LABELS.get(pred_name, ""),
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🫒 Olive CNN Classifier — http://localhost:8003")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8003)
