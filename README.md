# CopMap — Patrol & Bandobast Automation

A practical system to assist police operations with image-based crowd and object analysis, event logging, and intelligence summaries.

## Deliverables
- Object detection (Faster R-CNN)
- Crowd density analysis
- RAG + LLM intelligence module (FAISS + lightweight LLM)
- FastAPI backend with endpoints and sample outputs
- Postman collection and deployment notes

## Quick overview
- Edge/PC runs detector on camera frames or images.
- Backend ingests detections, stores events in a vector DB, and provides summaries and alerts.
- Human operators review and act on flagged items.

## Run locally
1. Open PowerShell in project folder.
2. Create venv and activate:
```powershell
python -m venv venv
venv\Scripts\activate
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

## Main endpoints
- GET / : health check
- POST /detect : upload image for detection (multipart/form-data, file)
- POST /ingest : log event (JSON: event_description, location, severity)
- GET /summarize : retrieve AI summary (query: location)
- GET /suspicious : list flagged events
- GET /status : system info

## Testing
- Use provided Postman collection or the included `test_api.py`.
- Upload sample images in `sample_outputs/` for detection tests.
- Ingest several events before requesting summaries for better results.
