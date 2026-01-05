
# API Documentation

Base URL: http://localhost:8000

Endpoints

1. GET /
- Description: Health check
- Response: { "status": "ok" }

2. POST /detect
- Description: Run object detection on uploaded image
- Request: multipart/form-data, field `file` (image)
- Success response (200):
{
  "detections": [
    {"label":"person","score":0.95,"bbox":[x1,y1,x2,y2]},
    ...
  ],
  "crowd_density": "low|medium|high",
  "alerts": ["crowd_alert"]
}

3. POST /ingest
- Description: Log an event to vector DB
- Request JSON:
{
  "event_description": "text",
  "location": "string",
  "severity": "low|medium|high",
  "timestamp": "ISO8601 (optional)"
}
- Response: { "status":"ok", "id":"event_id" }

4. GET /summarize
- Description: Return LLM summary for a location/time-range
- Query params:
  - location (required)
  - limit (optional, default 20)
- Response:
{
  "location":"North Gate",
  "summary_text":"...",
  "key_insights":[ "...", "..." ]
}

5. GET /suspicious
- Description: List flagged events
- Query params: location (optional)
- Response: [{ "id":"", "description":"", "reason":"heuristic", "severity":"" }]

6. GET /status
- Description: System info and model status
- Response: { "models_loaded": true, "vector_db": "ok" }

Examples (curl)

Health:
curl http://localhost:8000/

Detect:
curl -X POST http://localhost:8000/detect -F "file=@sample_outputs/sample.jpg"

Ingest:
curl -X POST http://localhost:8000/ingest -H "Content-Type:application/json" -d "{\"event_description\":\"loitering\",\"location\":\"Gate\",\"severity\":\"medium\"}"

Summarize:
curl "http://localhost:8000/summarize?location=Gate"