# 🚀 Phase 2: Backend Setup & Launch

## 📋 Prerequisites

### 1. Download Model Files from Kaggle
After CNN training completes:
1. Go to your Kaggle notebook
2. Look for "Output" section
3. Download:
   - `best_model.pt` → Save to `d:\hackathon\models\best_model.pt`
   - `class_info.json` → Save to `d:\hackathon\models\class_info.json`

### 2. Create Models Directory
```bash
mkdir d:\hackathon\models
```

## 🔧 Install Dependencies

```bash
pip install fastapi uvicorn python-multipart
pip install torch torchvision  # If not already installed
pip install anthropic
pip install edge-tts
pip install pydub librosa
pip install pillow opencv-python
```

## 🔑 Set API Key

You need a Claude API key from Anthropic.

**On Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**On Windows (Permanent in Environment Variables):**
1. Press `Win+X` → System
2. Advanced system settings → Environment Variables
3. New User variable:
   - Name: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...` (your key)

**Test it:**
```bash
echo $env:ANTHROPIC_API_KEY
```

## 🏃 Run Backend

```bash
cd d:\hackathon
python backend.py
```

**Expected output:**
```
======================================================================
[*] Starting Olive Leaf Disease Detection Backend
======================================================================
[OK] Classes: ['healthy', 'aculus_olearius', 'olive_peacock_spot']
[OK] Model loaded from ./models/best_model.pt
[*] API: http://localhost:8000
[*] Docs: http://localhost:8000/docs
```

If you see this, backend is running! ✅

## 🧪 Test the API

### Option 1: Interactive Docs
Go to: http://localhost:8000/docs

You'll see all endpoints with a "Try it out" button.

### Option 2: Python Test
```python
import requests
from pathlib import Path

# Assume you have a test image
image_path = Path('test_leaf.jpg')

with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)

print(response.json())
```

**Expected response:**
```json
{
  "status": "ACCEPT",
  "disease": "healthy",
  "confidence": 0.92,
  "darija_response": "الورقة سليمة، ما فيها حتى مرض.",
  "all_probs": {
    "healthy": 0.92,
    "aculus_olearius": 0.06,
    "olive_peacock_spot": 0.02
  },
  "guardrail_reason": "PASSED"
}
```

### Option 3: Curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_leaf.jpg"
```

## 📊 API Endpoints

### `GET /health`
Check if backend is running
```
Response: {status: "OK", model_loaded: true, classes: [...]}
```

### `POST /predict`
Predict disease from image
```
Input: image file
Output: {status, disease, confidence, darija_response, all_probs}
```

### `POST /predict-with-voice`
Full pipeline: image + voice question → response + audio
```
Input: image file, audio file (optional)
Output: response text + base64 audio
```

## 🎯 What Happens Inside

1. **Image upload** → Preprocessing (224x224, normalize)
2. **CNN inference** → Get disease prediction + confidence
3. **Guardrails check**:
   - If CNN confidence < 70% → REFUSE
   - If KB similarity < 60% → REFUSE
   - Else → PROCEED
4. **Claude prompt** → Build grounded prompt with ONLY KB context
5. **Claude response** → Get Darija answer (no hallucination)
6. **TTS synthesis** → Convert to audio (if needed)

## 🆘 Troubleshooting

### Error: "Model not found at ./models/best_model.pt"
**Fix:** Make sure you downloaded the files from Kaggle to the right location
```bash
ls d:\hackathon\models\
```

### Error: "ModuleNotFoundError: No module named 'anthropic'"
**Fix:** Install dependencies
```bash
pip install anthropic
```

### Error: "Failed to initialize CUDA"
**Fix:** GPU issues - backend will use CPU instead (slower but works)

### Error: "ANTHROPIC_API_KEY not set"
**Fix:** Set environment variable (see section above)

## ⏱️ Time Check

- H0-H2.5: ✅ CNN training done
- H2.5-H3.5: ⏳ Backend setup + test (15 min)
- H3.5-H4: Voice integration (15 min)
- H4-H6: Voice pipeline (2 hours)
- H6-H7.5: PWA frontend (1.5 hours)
- H7.5-H9: Demo + hardening (1.5 hours)

## ✅ Next Step

Once backend is running and `/predict` returns responses:
1. Install voice dependencies (`openai`, `edge-tts`)
2. Add voice transcription (Whisper)
3. Build PWA frontend

Ready? 🚀
