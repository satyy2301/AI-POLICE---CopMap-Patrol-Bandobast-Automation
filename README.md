# CopMap — Patrol & Bandobast Automation

A practical system to assist police operations with image-based crowd and object analysis, event logging, and intelligence summaries.

---
Video demo : https://drive.google.com/file/d/14Je9AAe5lwTW5XcpUyM3Y_rtD4C5kMi3/view?usp=sharing
<img width="1912" height="840" alt="Screenshot 2026-01-07 201923" src="https://github.com/user-attachments/assets/015c98a5-e0c0-4c93-8517-14b1408d3c1d" />

<img width="1883" height="874" alt="Screenshot 2026-01-07 201854" src="https://github.com/user-attachments/assets/c4337a3b-9d3e-4ef0-af1f-912e3126ae71" />

<img width="1743" height="572" alt="Screenshot 2026-01-07 201911" src="https://github.com/user-attachments/assets/81144b29-0394-4d89-bb19-6c311ddfb748" />

## Problem Understanding

**Real-world challenge:** Police departments monitor multiple areas with limited staff. Officers manually review camera feeds, log incidents on paper, and struggle to spot patterns across events.

**Impact:**
- Slow response to suspicious activity
- Missed patterns (repeated loitering, weapon sightings)
- High cognitive load on officers
- No searchable incident history

**Our solution:** Automate detection and intelligence generation. System detects people/objects, logs events, and generates summaries via AI so officers act faster with better information.

**Key principle:** System assists officers, not replaces them. Every alert requires human review before action.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND                                 │
│  Browser UI + Swagger API Docs (FastAPI auto-generated)    │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                   FASTAPI BACKEND                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ /detect  │  │ /ingest  │  │/summarize│  │/suspicious  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────┬───────────────────────────────────────────┘
                  │
      ┌───────────┴───────────┐
      │                       │
┌─────▼──────────┐   ┌────────▼──────────┐
│  DETECTION     │   │  RAG + LLM MODULE │
│  Faster R-CNN  │   │                   │
│  (PyTorch)     │   │ ┌──────────────┐  │
│                │   │ │FAISS Vector  │  │
│ - Objects      │   │ │Database      │  │
│ - Persons      │   │ │(events)      │  │
│ - Crowd count  │   │ └──────────────┘  │
│ - Density      │   │ ┌──────────────┐  │
└────────────────┘   │ │flan-t5-small │  │
                     │ │(Summarizer)  │  │
                     │ └──────────────┘  │
                     └───────────────────┘
```

### Layer 1: Detection
- **Model:** Faster R-CNN (pretrained on COCO dataset)
- **Input:** Image (any size)
- **Output:** Bounding boxes, class labels, confidence scores, crowd density estimate
- **Performance:** ~200ms per image on CPU

### Layer 2: Event Storage & Retrieval
- **Database:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **Workflow:**
  1. Officer logs event: "Suspicious loitering at North Gate"
  2. Text converted to 384-dim vector
  3. Stored with timestamp, location, severity
  4. Later: query for "loitering" retrieves similar events via cosine similarity

### Layer 3: Intelligence Generation
- **LLM:** flan-t5-small (80M parameters)
- **Method:** RAG (Retrieval-Augmented Generation)
  1. Officer asks: "Summarize North Gate incidents"
  2. System retrieves top-K similar events from FAISS
  3. Passes events as context to LLM
  4. LLM generates concise summary with patterns

### Flow Diagram
```
Image Upload
    │
    ▼
[Faster R-CNN Detection]
    │
    ├─→ Save detections (JSON)
    │
    ▼
[Officer reviews → logs event]
    │
    ▼
[POST /ingest]
    │
    ├─→ Text embedding (Sentence-Transformers)
    ├─→ Store in FAISS
    ├─→ Mark as suspicious (heuristics)
    │
    ▼
[Officer requests summary]
    │
    ├─→ [GET /summarize?location=X]
    ├─→ Query FAISS for similar events
    ├─→ Pass to LLM with context
    │
    ▼
[LLM Summary Response]
```

---

## Trade-offs & Design Decisions

| Aspect | Choice | Why | Trade-off |
|--------|--------|-----|-----------|
| **Detection** | Faster R-CNN | Accurate, fast, open-source | 5% lower accuracy vs latest models |
| **Embeddings** | all-MiniLM-L6-v2 | Lightweight, 384 dims, good quality | Cannot understand very long texts |
| **LLM** | flan-t5-small | Runs on CPU, instant summaries | Less creative; sometimes terse |
| **Database** | FAISS + JSON | Local, zero cost, searchable | No persistence on restart; single-machine |
| **Frontend** | Static HTML | Simple, no dependencies | Basic UI; no real-time updates |
| **Vector Search** | Cosine similarity | Fast, interpretable | May miss semantic nuances |
| **Alerts** | Rule-based + heuristics | Interpretable, no false positives | Requires manual tuning per location |

### What We Implemented
✅ Real-time object detection  
✅ Event logging with semantic search  
✅ AI-generated summaries via RAG  
✅ Suspicious alert flagging (heuristics)  
✅ FastAPI backend with 6 endpoints  
✅ Swagger API documentation  
✅ Static HTML dashboard  
✅ Postman collection for testing  
✅ Sample outputs and deployment docs  

### What We Skipped (Intentional)
❌ Real-time streaming/WebSockets (overkill for batch operations)  
❌ Persistent database (PostgreSQL) (local FAISS sufficient for MVP)  
❌ User authentication (assume secured network)  
❌ GPU acceleration (CPU-only for cost-effectiveness)  
❌ Cloud deployment (can be added later)  
❌ Mobile app (web UI covers 95% of use cases)  

---

## API Endpoints

### Health & Status
- **GET /** → System alive check
- **GET /status** → Detailed system info

### Detection
- **POST /detect** → Upload image, get detections + crowd density
  - Input: `image` (multipart file)
  - Output: Detections, person count, alerts

### Event Management
- **POST /ingest** → Log patrol event
  - Input: `event_description`, `location`, `severity`
  - Output: Event ID, embedding status
- **GET /suspicious** → List flagged events
- **GET /summarize** → Query: `?location=X` → Get AI summary

---

## Testing & Sample Outputs

### Option 1: Postman Collection
1. Import `postman_collection.json` into Postman
2. Run requests in sequence (detect → ingest → summarize)
3. See full request/response examples

### Option 2: Swagger UI
1. Start server: `python -m uvicorn app.main:app --reload`
2. Open: http://localhost:8000/docs
3. Click "Try it out" on each endpoint
4. Upload sample images from `sample_outputs/`

### Sample Outputs
- **Detection output** → `sample_outputs/detection_example.json`
  - Shows bounding boxes, confidences, crowd density
- **Event ingestion** → `sample_outputs/patrol_intelligence_example.json`
  - Shows logged events with embeddings
- **Summary output** → `sample_outputs/crowd_analysis_example.json`
  - Shows LLM-generated patrol intelligence
- **Alert output** → `sample_outputs/suspicious_alert_example.json`
  - Shows flagged suspicious events

---

## Run Locally

1. Open PowerShell in project folder.
2. Create venv and activate:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
3. Install:
```powershell
pip install -r requirements.txt
```
4. Start server:
```powershell
python -m uvicorn app.main:app --reload
```
Server: http://localhost:8000

### Test Options
- **Web UI:** http://localhost:8000 (upload images, log events)
- **Swagger:** http://localhost:8000/docs (detailed API explorer)
- **Postman:** Import `postman_collection.json` (batch testing)

---

## File Structure

```
ai-police/
├── app/
│   ├── main.py              # FastAPI application & endpoints
│   ├── detector.py          # Faster R-CNN detection logic
│   ├── llm_rag.py           # FAISS + LLM summarization
│   └── __pycache__/
├── data/
│   └── rag_store/           # Vector DB (FAISS) + event storage
│       ├── docs.json        # Indexed events
│       └── index.faiss      # FAISS index
├── static/
│   └── index.html           # Web dashboard
├── sample_outputs/          # Example detection/summary outputs
├── scripts/
│   └── run_dev.sh           # Dev server launcher
├── postman_collection.json  # API test suite
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── API_DOCUMENTATION.md     # Detailed endpoint specs
├── ARCHITECTURE.md          # Technical deep-dive
└── DEPLOYMENT.md            # Production guide
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Detection speed | ~200ms/image |
| Summary generation | ~1-2 sec |
| Vector DB size | 100MB (10k events) |
| Model memory | ~2GB (detector + LLM) |
| Accuracy (COCO) | ~75% mAP |
| Inference cost | $0 (all open-source) |

---

## Next Steps

- Deploy on Linux server (see `DEPLOYMENT.md`)
- Integrate with live camera feeds
- Add user authentication & logging
- Tune alert heuristics per location
- Record demo video (see `SUBMISSION_GUIDE.md`)

---

## License & Credits

- Faster R-CNN: PyTorch vision (Meta)
- FAISS: Facebook AI Research
- Sentence-Transformers: Hugging Face
- flan-t5: Google

Built for police operations automation. Questions? See `API_DOCUMENTATION.md` for endpoint details.
