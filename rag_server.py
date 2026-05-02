"""
Olive Voice Assistant - RAG + LLM + TTS Server
Steps 5-8: Hybrid retrieval, anti-hallucination guard, LLM reformulation in darija, TTS

Endpoints:
  POST /query      - text query → retrieval → LLM → TTS audio
  POST /full_query - ASR text + CNN label → combined → response
  GET  /health     - health check
"""

import os
import re
import pickle
import asyncio
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ── Embeddings ────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

# ── TTS ────────────────────────────────────────────────────────────────────────
import edge_tts  # pip install edge-tts

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Anti-hallucination threshold: if top similarity < this → refuse to answer
RELEVANCE_THRESHOLD = 0.60   # Step 6: guard-rail
TOP_K = 5                     # number of chunks to retrieve

# TTS voice for Arabic (edge-tts Microsoft voices)
TTS_VOICE = "ar-TN-HediNeural"  # Tunisian Arabic voice
TTS_VOICE_FALLBACK = "ar-SA-HamedNeural"  # fallback MSA

AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

# ── Load index ────────────────────────────────────────────────────────────────
print("Loading FAISS index and corpus...")
try:
    index = faiss.read_index(str(INDEX_DIR / "olive.index"))
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(INDEX_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    print(f"✓ Loaded {index.ntotal} vectors, {len(chunks)} chunks")
except Exception as e:
    print(f"✗ Could not load index: {e}")
    print("  Run build_corpus.py first!")
    index, chunks, metadata = None, [], []

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embed_model = SentenceTransformer(EMBEDDING_MODEL)
print("✓ Embedding model loaded")

from deep_translator import GoogleTranslator

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Olive Voice Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    text: str                          # user question (in darija/arabic)
    cnn_label: Optional[str] = None    # disease label from CNN (e.g. "Peacock Spot")
    cnn_confidence: Optional[float] = None

class QueryResponse(BaseModel):
    answer_text: str
    audio_url: Optional[str]
    retrieved_chunks: List[Dict]
    top_score: float
    refused: bool
    source_citations: List[str]


# ── Step 5: Hybrid retrieval ──────────────────────────────────────────────────
def retrieve(query: str, cnn_label: Optional[str] = None, top_k: int = TOP_K) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Semantic search on corpus, optionally boosted by CNN disease label.
    Returns (chunks, metadata, scores)
    """
    if index is None or index.ntotal == 0:
        return [], [], []

    # Augment query with CNN result for better retrieval
    augmented_query = query
    if cnn_label:
        augmented_query = f"{query} {cnn_label} olive disease symptoms treatment"

    # Embed query
    q_emb = embed_model.encode([augmented_query], normalize_embeddings=True).astype("float32")

    # FAISS search
    scores, indices = index.search(q_emb, top_k)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    result_chunks = []
    result_meta = []
    result_scores = []

    for score, idx in zip(scores, indices):
        if idx >= 0 and idx < len(chunks):
            result_chunks.append(chunks[idx])
            result_meta.append(metadata[idx])
            result_scores.append(float(score))

    return result_chunks, result_meta, result_scores


# ── Step 6: Anti-hallucination guard ─────────────────────────────────────────
def is_relevant(top_score: float) -> bool:
    return top_score >= RELEVANCE_THRESHOLD


def build_refusal_response() -> str:
    """Polite refusal in Tunisian Arabic when no relevant info found."""
    return (
        "معلوماتي ما تنجمش تعاونك في هذا الموضوع. "
        "نصيحتي متوجه لعند مرشد فلاحي متخصص باش يعطيك المعلومة الصحيحة. "
        "يمكنك تتصل بالإرشاد الفلاحي في منطقتك."
    )


# ── Step 7: Translation to Arabic (Free Alternative to LLM) ────────────────────
def translate_and_format(question: str, retrieved_chunks: List[str], meta: List[Dict], cnn_label: Optional[str] = None) -> Tuple[str, List[str]]:
    """Translate chunks to Arabic and format a direct response without LLM API costs."""
    sources = set()
    translator = GoogleTranslator(source='auto', target='ar')
    
    translated_parts = []
    
    if cnn_label:
        try:
            translated_label = translator.translate(cnn_label)
            translated_parts.append(f"بناءً على تحليل الصورة، وجدنا: {translated_label}.")
        except:
            translated_parts.append(f"بناءً على تحليل الصورة، وجدنا: {cnn_label}.")
            
    for i, (chunk, m) in enumerate(zip(retrieved_chunks, meta)):
        sources.add(m['source'])
        try:
            # Extract just the first 1-2 sentences to keep it very brief
            sentences = [s.strip() for s in chunk.split('.') if len(s.strip()) > 10]
            short_chunk = '. '.join(sentences[:2]) + "." if sentences else chunk[:150] + "..."
            
            translated_text = translator.translate(short_chunk)
            translated_parts.append(translated_text)
        except Exception as e:
            print(f"Translation error: {e}")
            
    # Combine translated parts
    if not translated_parts:
        return "صار خطأ في معالجة الطلب. عاود لحقًا.", list(sources)
        
    answer = " حسب موسوعة الزيتون: " + " ".join(translated_parts)
    return answer, list(sources)


# ── Step 8: TTS synthesis ─────────────────────────────────────────────────────
async def text_to_speech(text: str) -> Optional[str]:
    """Convert Arabic text to speech using edge-tts, return audio file path."""
    # Cache by content hash
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    audio_path = AUDIO_DIR / f"{text_hash}.mp3"

    if audio_path.exists():
        return str(audio_path)

    try:
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(str(audio_path))
        print(f"✓ TTS audio saved: {audio_path}")
        return str(audio_path)
    except Exception as e:
        print(f"TTS error with {TTS_VOICE}: {e}")
        try:
            communicate = edge_tts.Communicate(text, TTS_VOICE_FALLBACK)
            await communicate.save(str(audio_path))
            return str(audio_path)
        except Exception as e2:
            print(f"TTS fallback error: {e2}")
            return None


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "corpus_size": len(chunks),
        "index_loaded": index is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint:
    1. Retrieve relevant chunks (Step 5)
    2. Check relevance threshold (Step 6)
    3. Call LLM for darija reformulation (Step 7)
    4. Generate TTS audio (Step 8)
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # Step 5: Retrieve
    retrieved, meta, scores = retrieve(req.text, req.cnn_label, TOP_K)

    top_score = scores[0] if scores else 0.0
    print(f"Query: {req.text[:80]}...")
    print(f"Top similarity score: {top_score:.3f} (threshold: {RELEVANCE_THRESHOLD})")

    # Step 6: Anti-hallucination guard
    if not is_relevant(top_score) or not retrieved:
        refusal = build_refusal_response()
        audio_path = await text_to_speech(refusal)
        audio_url = f"/audio/{Path(audio_path).name}" if audio_path else None
        return QueryResponse(
            answer_text=refusal,
            audio_url=audio_url,
            retrieved_chunks=[],
            top_score=top_score,
            refused=True,
            source_citations=[]
        )

    # Step 7: Translation mapping (Free alternative)
    # Passing only the single most relevant chunk [:1] to keep the response very short
    answer, sources = translate_and_format(req.text, retrieved[:1], meta[:1], req.cnn_label)

    # Step 8: TTS
    audio_path = await text_to_speech(answer)
    audio_url = f"/audio/{Path(audio_path).name}" if audio_path else None

    return QueryResponse(
        answer_text=answer,
        audio_url=audio_url,
        retrieved_chunks=[
            {"text": c[:200] + "...", "source": m["source"], "score": round(s, 3)}
            for c, m, s in zip(retrieved[:3], meta[:3], scores[:3])
        ],
        top_score=top_score,
        refused=False,
        source_citations=sources
    )


@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve TTS audio files."""
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(str(audio_path), media_type="audio/mpeg")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🫒 Olive Voice Assistant - RAG + LLM + TTS Server")
    print("  Running on http://localhost:8002")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8002)
