# Deployment Notes

Local (dev)
1. venv:
   python -m venv venv
   venv\Scripts\activate
2. Install:
   pip install -r requirements.txt
3. Run:
   python -m uvicorn app.main:app --reload

Docker (recommended for consistency)
- Dockerfile (simple)
- Build:
  docker build -t ai-police .
- Run:
  docker run -p 8000:8000 -e ENV=prod -v ./data:/app/data ai-police

Environment variables
- VECTOR_DB_PATH: ./data/faiss.index
- MODEL_PATH: path/to/local/model (optional)
- LOG_LEVEL: info/debug

Production considerations
- Serve behind reverse proxy (NGINX) and HTTPS.
- Use persistent storage for vector DB and logs.
- Monitor model memory and CPU; use GPU for large models.
- Backup vector DB periodically.

Scaling notes
- Detector can be deployed on edge nodes; backend horizontally scalable with stateless API and shared vector DB (or managed service).