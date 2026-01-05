# Architecture (compact)

Components
- Edge/Camera agent: captures frames, runs local detector or forwards images.
- Detector: Faster R-CNN (edge or server) → detections + crowd metrics.
- FastAPI backend: ingests detections/events, exposes APIs.
- Vector DB: FAISS used for event embeddings and RAG retrieval.
- LLM: lightweight model (e.g., flan-t5-small) for summaries via retrieved context.
- Alerting: simple rule engine to generate alerts from detections/events.
- Storage: local filesystem for images, SQLite/JSON for metadata (adjustable).

Data flow
1. Frame/image → Detector → detection result.
2. Result → POST /detect → backend stores event + vector embedding.
3. User requests /summarize → backend retrieves relevant vectors → LLM produces summary.
4. Alerts produced on ingest or detection (pushed via webhook/DB entry).

Trade-offs
- Edge detection reduces bandwidth but increases deployment complexity.
- FAISS is low-cost and local; not horizontally scalable without orchestration.
- flan-t5-small reduces cost but limits linguistic richness; suitable for concise summaries.

Security & risks
- False positives require human review before action.
- Data retention and access control must be enforced before deployment.