import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
from transformers import pipeline

import torch

print("Loading Whisper model (TuniSpeech-AI/whisper-tunisian-dialect)...")
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "automatic-speech-recognition",
    model="TuniSpeech-AI/whisper-tunisian-dialect",
    tokenizer="openai/whisper-small",
    feature_extractor="openai/whisper-small",
    device=device,
    torch_dtype=torch.float16 if device == 0 else torch.float32,
    generate_kwargs={
        "language": "arabic",
        "task": "transcribe"
    }
)
print("Model loaded successfully.")

app = FastAPI(title="Tunisian ASR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name
        
    try:
        # Run inference
        print(f"Transcribing {temp_audio_path}...")
        result = pipe(temp_audio_path)
        text = result["text"]
        print(f"Result: {text}")
    except Exception as e:
        print(f"Error during transcription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # Clean up
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    return {"text": text}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting ASR Server on http://localhost:8001")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8001)
