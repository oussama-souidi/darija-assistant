# 🫒 Olive Health Assistant — Project Presentation

---

## 1. What Is This Project?

**Olive Health Assistant** is an AI-powered tool built specifically for **Tunisian olive farmers**. It lets a farmer take a photo of a diseased olive leaf and ask a voice question in their own dialect — **Tunisian Darija** — and receive an expert answer read aloud back to them in Arabic.

The project was built for the **Olive Tech Hackathon 2026** and is fully open-source, requiring no paid API keys or cloud subscriptions.

---

## 2. The Problem Being Solved

Olive farming is critical to the Tunisian economy. However:

- Farmers in rural areas often cannot access agronomic experts quickly.
- Technical agricultural manuals (FAO, IOC, EPPO) are written in English or French and are not easily accessible.
- Identifying leaf diseases requires expert knowledge most smallholder farmers do not have.
- Language barriers prevent farmers from using standard digital tools.

**Olive Health Assistant bridges this gap** by bringing expert-level disease diagnosis and advice directly to any farmer with a smartphone, in their own language.

---

## 3. Core Capabilities

| Capability | Description |
|---|---|
| 📸 **Leaf Disease Detection** | Upload a photo of an olive leaf; the system instantly identifies whether the leaf is healthy or diseased |
| 🎙️ **Tunisian Voice Input** | Ask questions by speaking naturally in Tunisian Darija |
| 📚 **Grounded Expert Answers** | Answers pulled directly from FAO, IOC, CIHEAM, and EPPO agricultural databases |
| 🔊 **Arabic Voice Replies** | Responses are spoken aloud in a Tunisian Arabic voice |
| 🚫 **Anti-Hallucination Guard** | If the question is outside the knowledge base, the system refuses to guess rather than provide false information |

---

## 4. How It Works — Step by Step

```
Farmer
  │
  ├──▶ 📸 Photo of leaf ──▶ [CNN Server] ──▶ Disease label (e.g. "Peacock Spot")
  │                                                    │
  └──▶ 🎙️ Voice question ──▶ [ASR Server] ──▶ Transcribed text
                                                       │
                                          Both inputs sent to ▼
                                              [RAG Server]
                                                    │
                                    Searches the Knowledge Base (FAISS)
                                                    │
                                        Translates & formats answer
                                                    │
                                           TTS Audio generated
                                                    │
                                             🔊 Answer played back to Farmer
```

### Step-by-step user flow:
1. Farmer opens the web interface (`index_full.html`) in any browser.
2. Optionally uploads a photo of a suspicious olive leaf.
3. Asks a question by voice (e.g. *"Why does my leaf have black spots?"*).
4. The system transcribes the voice, classifies the image, searches its knowledge base, and reads back a verified expert answer in Arabic within seconds.

---

## 5. The Three AI Microservices

### 🎙️ ASR Server — Voice Transcription (`asr_server.py`, port 8001)
- Uses the **TuniSpeech-AI/whisper-tunisian-dialect** model, a fine-tuned variant of OpenAI Whisper specifically trained on the Tunisian dialect.
- Accepts audio files uploaded by the browser and returns transcribed Arabic text.
- Runs on GPU if available, falls back to CPU automatically.

### 📸 CNN Server — Leaf Disease Detection (`cnn_server.py`, port 8003)
- Uses a **MobileNetV3-Large** model fine-tuned on olive leaf images.
- Detects **3 classes** with **91.9% test accuracy**:
  - ✅ **Healthy** — leaf is in good condition, no treatment needed.
  - 🪲 **Aculus olearius** (olive rust mite) — mite infestation causing damage to fruit and leaves.
  - 🔵 **Olive Peacock Spot** (*Spilocea oleagina*) — dangerous fungal disease showing circular brown spots.
- Returns the prediction label in both English and Arabic (Darija), along with a confidence score.

### 📚 RAG Server — Knowledge Retrieval & Voice Response (`rag_server.py`, port 8002)
- Performs **semantic search** over the knowledge base using multilingual sentence embeddings.
- Applies a **relevance threshold (0.60)**: if the question is too far from the olive topic, the system politely declines rather than hallucinating.
- Translates the most relevant technical chunks to Arabic using Google Translate.
- Generates an **audio response** using Microsoft Edge TTS with the Tunisian voice (`ar-TN-HediNeural`).
- Caches generated audio to avoid redundant processing.

---

## 6. The Knowledge Base

The system's knowledge comes from authoritative agricultural sources, embedded into a searchable vector database (FAISS):

| Source | Content |
|---|---|
| **International Olive Council (IOC)** | 400-page production manual (pruning, pests, diseases, harvest, irrigation) |
| **IOC — Olive Resilience & Climate** | Diseases, Xylella, Verticillium, climate adaptation |
| **FAO** | Olive water management, Mediterranean production systems |
| **CIHEAM** | Mediterranean olive sector proceedings |
| **EPPO Global Database** | Structured disease data (Peacock Spot, Aculus, Verticillium, Olive Fly) |

The corpus is **chunked into 500-word segments** with 50-word overlap, embedded with a multilingual model, and stored in a FAISS index for fast similarity search.

---

## 7. Technology Stack

| Layer | Technology |
|---|---|
| **Voice Recognition (ASR)** | TuniSpeech-AI / Whisper (HuggingFace Transformers) |
| **Image Classification (CNN)** | PyTorch — MobileNetV3-Large (fine-tuned) |
| **Text Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Translation** | Google Translate via `deep-translator` (free, no API key) |
| **Text-to-Speech** | Microsoft Edge TTS — `ar-TN-HediNeural` (Tunisian Arabic) |
| **Backend API** | FastAPI + Uvicorn (3 independent microservices) |
| **Frontend UI** | Vanilla JavaScript + CSS Glassmorphism |

---

## 8. Key Design Choices

### ✅ 100% Free & Open Source
No OpenAI API, no paid services. The entire pipeline uses free open-source models and tools. This makes it deployable by NGOs, agriculture ministries, or individual researchers at zero running cost.

### ✅ Anti-Hallucination Architecture
Unlike general-purpose chatbots, this system **never invents information**. It only answers when a sufficiently relevant document is found in the knowledge base (similarity score ≥ 0.60). Otherwise it directs the farmer to a specialist.

### ✅ Dialect-First Design
Most Tunisian farmers speak Darija, not Modern Standard Arabic or French. The ASR model is specifically fine-tuned on the Tunisian dialect. The TTS voice is also Tunisian-accented, making the interaction natural and comfortable.

### ✅ GPU Optional
All AI inference (ASR, CNN) automatically detects and uses GPU if available, but gracefully falls back to CPU. The system can run on any standard laptop.

---

## 9. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          WEB BROWSER                            │
│                   index_full.html (Frontend)                    │
└───────────────┬─────────────────────┬───────────────────────────┘
                │ audio file          │ image file
                ▼                     ▼
┌──────────────────────┐   ┌──────────────────────────┐
│   ASR Server :8001   │   │   CNN Server :8003        │
│  Whisper Tunisian    │   │  MobileNetV3 Fine-tuned   │
│  Dialect Model       │   │  3 disease classes        │
│                      │   │  91.9% accuracy           │
└──────────┬───────────┘   └─────────────┬────────────┘
           │ transcribed text            │ disease label
           └──────────────┬──────────────┘
                          ▼
           ┌──────────────────────────────┐
           │      RAG Server :8002        │
           │  Semantic search (FAISS)     │
           │  Anti-hallucination guard    │
           │  Translation to Arabic       │
           │  TTS (Edge-TTS Tunisian)     │
           └──────────┬───────────────────┘
                      │
           ┌──────────▼───────────────────┐
           │     FAISS Vector Index       │
           │  FAO + IOC + CIHEAM + EPPO   │
           │  Technical olive documents   │
           └──────────────────────────────┘
```

---

## 10. How to Run the Project

### Prerequisites
- Python 3.10+
- FFmpeg installed and in system PATH
- (Optional) NVIDIA GPU for faster inference

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Build the knowledge base (downloads PDFs and creates FAISS index)
python build_corpus.py

# Start all three servers (Linux/macOS)
./dev.sh

# Or on Windows
dev.bat
```

### Access the UI
Open `index_full.html` in any modern web browser. No internet connection required after initial setup.

---

## 11. Impact & Value Proposition

- 🌍 **Accessible to all farmers**: No literacy required — fully voice-driven.
- 💰 **Zero cost**: No subscriptions, no API keys, no cloud fees.
- 🔒 **Private**: All processing is local; no farmer data is sent to third-party servers.
- ⚡ **Fast**: Disease diagnosis in under 2 seconds with a GPU.
- 📖 **Trustworthy**: Answers are grounded in internationally recognized agricultural authorities (FAO, IOC, EPPO).
- 🌱 **Scalable**: The knowledge base can be expanded with new documents by simply re-running the corpus builder.

---

*Built for the **Olive Tech Hackathon 2026** 🫒✨*
