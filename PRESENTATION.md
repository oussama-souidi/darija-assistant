# 🫒 Olive Health Assistant — Vue d'ensemble du projet

> **Présenté au Olive Tech Hackathon 2026**

---

## 📌 Contexte et problématique

Les agriculteurs tunisiens spécialisés dans la culture de l'olivier font face à un défi majeur : identifier rapidement les maladies qui frappent leurs oliveraies et obtenir des conseils agronomiques fiables. L'accès à l'expertise est souvent limité dans les zones rurales, et les manuels techniques disponibles sont rédigés en langues étrangères, inaccessibles à la majorité des producteurs locaux.

**Olive Health Assistant** répond à ce besoin en combinant l'intelligence artificielle de pointe avec l'accessibilité linguistique : l'agriculteur interagit en **dialecte tunisien (Darija)**, reçoit un diagnostic de maladie à partir d'une simple photo de feuille, et obtient une réponse vocale grounded dans des sources scientifiques reconnues.

---

## 🎯 Objectifs du projet

- **Diagnostiquer** automatiquement les maladies des feuilles d'olivier à partir d'une photographie.
- **Comprendre** les questions posées oralement en dialecte tunisien.
- **Répondre** avec des conseils agronomiques vérifiés, traduits en arabe et restitués vocalement.
- **Éviter** toute hallucination de l'IA grâce à un garde-fou anti-invention strict.
- **Fonctionner** entièrement sans clés API payantes — 100 % open source et gratuit.

---

## 🏗️ Architecture générale

Le projet est organisé en **trois microservices indépendants** qui collaborent pour traiter la requête de l'agriculteur, plus une **interface web** légère.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agriculteur                              │
│                (Photo de feuille + Question vocale)             │
└──────────────┬──────────────────────────┬───────────────────────┘
               │                          │
               ▼                          ▼
  ┌─────────────────────┐    ┌─────────────────────────┐
  │   Serveur CNN        │    │   Serveur ASR            │
  │  (Port 8003)         │    │   (Port 8001)            │
  │  Classification de   │    │   Transcription vocale   │
  │  maladies foliaires  │    │   Darija → Texte         │
  └──────────┬──────────┘    └────────────┬────────────┘
             │  Label maladie             │  Texte transcrit
             └──────────────┬────────────┘
                            ▼
               ┌─────────────────────────┐
               │    Serveur RAG           │
               │    (Port 8002)           │
               │  Recherche sémantique    │
               │  Anti-hallucination      │
               │  Traduction → Arabe      │
               │  Synthèse vocale (TTS)   │
               └────────────┬────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │  Réponse à l'agriculteur │
               │  (Texte arabe + Audio)   │
               └─────────────────────────┘
```

---

## 🔄 Flux de l'application (étape par étape)

### Étape 1 — Saisie utilisateur (Frontend)
L'agriculteur ouvre `index_full.html` dans son navigateur. Il peut :
- **Uploader une photo** d'une feuille d'olivier suspecte.
- **Enregistrer sa question** oralement en Darija tunisien (ex. : « شنوا هذا المرض وكيفاش نعالجو ? »).

### Étape 2 — Reconnaissance vocale (Serveur ASR · Port 8001)
L'audio enregistré est envoyé au serveur ASR. Le modèle **Whisper fine-tuné sur le dialecte tunisien** (`TuniSpeech-AI/whisper-tunisian-dialect`) transcrit la parole en texte arabe/darija. Le texte est renvoyé au frontend.

### Étape 3 — Classification de la maladie (Serveur CNN · Port 8003)
La photo de feuille est envoyée au serveur CNN. Un modèle **MobileNetV3-Large fine-tuné** (précision de test : 91,9 %) classe la feuille parmi trois catégories :

| Classe | Description |
|--------|-------------|
| `healthy` | Feuille saine, aucun traitement nécessaire |
| `aculus_olearius` | Infestation par l'acarien de l'olivier |
| `olive_peacock_spot` | Œil de paon — maladie fongique (Spilocea oleagina) |

Le label de la maladie (en anglais et en arabe dialectal) est retourné avec son score de confiance.

### Étape 4 — Recherche sémantique dans la base de connaissances (Serveur RAG · Port 8002)
Le texte transcrit et le label CNN sont fusionnés en une requête enrichie, puis encodés avec le modèle d'embeddings multilingue. Une **recherche par similarité cosine** dans l'index FAISS identifie les passages les plus pertinents de la base documentaire.

**Base de connaissances** (manuels techniques indexés) :

| Source | Contenu |
|--------|---------|
| FAO | Techniques de production oléicole, gestion de l'eau |
| IOC (Conseil Oléicole International) | Manuel de culture, résilience climatique |
| CIHEAM | Filière oléicole méditerranéenne |
| EPPO | Fiches maladies : Œil de paon, Verticillium, Acariens |

### Étape 5 — Garde-fou anti-hallucination
Si le score de similarité maximal est **inférieur à 0,60**, le système refuse de répondre et invite l'agriculteur à consulter un conseiller agricole local — plutôt que d'inventer une information potentiellement dangereuse.

### Étape 6 — Traduction et formulation de la réponse
Les passages récupérés (en anglais) sont traduits en arabe via **Google Translate** (`deep-translator`), puis formatés en une réponse concise et compréhensible, citant la source documentaire.

### Étape 7 — Synthèse vocale (TTS)
La réponse arabe est convertie en audio MP3 par le moteur **Microsoft Edge TTS** avec la voix tunisienne `ar-TN-HediNeural`. Un mécanisme de cache évite de re-synthétiser des réponses identiques.

### Étape 8 — Restitution à l'agriculteur
Le frontend reçoit la réponse textuelle arabe et l'URL de l'audio. L'interface affiche le texte et joue automatiquement la réponse vocale.

---

## 🛠️ Stack technologique

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| **ASR** | `TuniSpeech-AI/whisper-tunisian-dialect` (Hugging Face Transformers) | Reconnaissance vocale en Darija tunisien |
| **Vision / CNN** | MobileNetV3-Large fine-tuné (PyTorch) | Classification des maladies foliaires |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Vectorisation multilingue des textes |
| **Base vectorielle** | FAISS (Facebook AI Similarity Search) | Recherche sémantique ultra-rapide |
| **Extraction PDF** | PyMuPDF (`fitz`) | Parsing des manuels techniques en PDF |
| **Traduction** | Google Translate via `deep-translator` | Traduction EN → AR sans API payante |
| **TTS** | Microsoft Edge TTS (`edge-tts`) — voix `ar-TN-HediNeural` | Synthèse vocale en arabe tunisien |
| **Backend** | FastAPI + Uvicorn | API REST haute performance (3 microservices) |
| **Frontend** | Vanilla JS + CSS Glassmorphism | Interface web légère, sans framework |

---

## 🌿 Construction de la base de connaissances (`build_corpus.py`)

1. **Téléchargement** des PDFs depuis les sources officielles (FAO, IOC, CIHEAM) + intégration des fiches EPPO embarquées directement dans le code.
2. **Extraction du texte** page par page avec PyMuPDF.
3. **Découpage en chunks** de 500 mots avec un chevauchement de 50 mots pour maintenir le contexte.
4. **Encodage vectoriel** de chaque chunk avec le modèle sentence-transformers.
5. **Indexation FAISS** : les vecteurs sont sauvegardés dans `faiss_index/olive.index` avec les métadonnées associées (`chunks.pkl`, `metadata.pkl`).

---

## 🛡️ Points forts du projet

- **Fiabilité** : Zéro hallucination grâce au seuil de pertinence (`RELEVANCE_THRESHOLD = 0.60`).
- **Accessibilité** : Interface et réponses en dialecte tunisien, vocales, pour les agriculteurs peu lettrés.
- **Souveraineté** : Fonctionne entièrement en local ou avec des services gratuits — aucune dépendance à des APIs payantes.
- **Précision** : 91,9 % de précision sur le jeu de test pour le modèle CNN de classification.
- **Scalabilité** : Architecture microservices découplée, chaque composant peut être amélioré indépendamment.

---

## 📁 Structure du projet

```
darija-assistant/
├── asr_server.py          # Microservice ASR (Whisper) — Port 8001
├── cnn_server.py          # Microservice CNN (MobileNetV3) — Port 8003
├── rag_server.py          # Microservice RAG + TTS — Port 8002
├── build_corpus.py        # Construction de l'index FAISS
├── index_full.html        # Interface web (Frontend)
├── requirements.txt       # Dépendances Python
├── corpus_data/           # PDFs des manuels agricoles
├── faiss_index/           # Index vectoriel FAISS
│   ├── olive.index
│   ├── chunks.pkl
│   └── metadata.pkl
├── models/                # Modèle CNN entraîné
│   ├── best_model.pt
│   └── class_info.json
├── dev.sh                 # Lancement en un seul clic (Linux/macOS)
└── dev.bat                # Lancement en un seul clic (Windows)
```

---

## 🚀 Lancement rapide

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Construire la base de connaissances (à faire une seule fois)
python build_corpus.py

# 3. Lancer les trois serveurs simultanément
./dev.sh          # Linux / macOS
dev.bat           # Windows

# 4. Ouvrir l'interface dans le navigateur
# → Ouvrir index_full.html
```

---

## 🌍 Impact attendu

Ce projet vise à **démocratiser l'accès au savoir agronomique** en Tunisie, en particulier pour les petits producteurs d'olives qui n'ont pas les moyens d'embaucher des consultants spécialisés. En combinant la reconnaissance d'image, le traitement du langage naturel et la synthèse vocale dans la langue de l'agriculteur, Olive Health Assistant représente une application concrète et inclusive de l'IA au service du monde rural.

---

*Projet développé dans le cadre du **Olive Tech Hackathon 2026** 🫒✨*
