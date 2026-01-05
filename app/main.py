import os
import io
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from typing import Optional

from app.detector import Detector
from app.llm_rag import RAGStore, Summarizer

DATA_DIR = os.environ.get('CMP_DATA_DIR', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title='CopMap Prototype - AI-Driven Patrol & Bandobast Automation')

# simple CPU default; users can set CMP_DEVICE env var
detector = Detector(device='cpu', score_thresh=0.6)
rag = RAGStore(storage_dir=os.path.join(DATA_DIR, 'rag_store'))
summarizer = Summarizer(rag)

# Request/Response models
class EventPayload(BaseModel):
    text: str
    metadata: Optional[dict] = None
    camera_id: Optional[str] = None
    timestamp: Optional[str] = None


class AlertResponse(BaseModel):
    type: str
    severity: str
    message: str
    timestamp: str


@app.get('/')
async def health_check():
    """Health check endpoint"""
    return JSONResponse({'status': 'ok', 'service': 'CopMap AI Patrol System'})


@app.post('/detect')
async def detect(image: UploadFile = File(...), camera_id: str = 'default'):
    """
    Detect objects in an image using Faster R-CNN.
    Returns detections, person count, and crowd-based alerts.
    """
    if not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Expected an image file')
    
    contents = await image.read()
    pil = Image.open(io.BytesIO(contents)).convert('RGB')
    result = detector.detect_pil(pil)

    # alert heuristics
    alerts = []
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    if result['person_count'] >= 12:
        alerts.append({
            'type': 'crowd_density',
            'severity': 'high',
            'message': f"{result['person_count']} persons detected - HIGH crowd density",
            'timestamp': timestamp,
            'recommended_action': 'increase_monitoring'
        })
    elif result['person_count'] >= 6:
        alerts.append({
            'type': 'crowd_density',
            'severity': 'medium',
            'message': f"{result['person_count']} persons detected - MEDIUM crowd density",
            'timestamp': timestamp,
            'recommended_action': 'monitor'
        })

    # ingest event into RAG store
    short_text = f"Detection[{camera_id}]: {result['person_count']} persons; objects={[d['label'] for d in result['detections'][:3]]}"
    rag.ingest(short_text, metadata={'source': image.filename, 'camera_id': camera_id, 'timestamp': timestamp})

    return JSONResponse({
        'timestamp': timestamp,
        'camera_id': camera_id,
        'result': result,
        'alerts': alerts,
        'metadata': {'model': 'fasterrcnn_resnet50_fpn', 'confidence_threshold': 0.6}
    })


@app.post('/ingest')
async def ingest_event(payload: EventPayload):
    """
    Ingest a patrol event into the vector database for RAG retrieval.
    Used to log incident reports, observations, and summaries.
    """
    text = payload.text
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail='`text` required and non-empty')
    
    metadata = payload.metadata or {}
    metadata['camera_id'] = payload.camera_id or 'manual_input'
    metadata['timestamp'] = payload.timestamp or datetime.utcnow().isoformat() + 'Z'
    
    rag.ingest(text, metadata=metadata)
    
    return JSONResponse({
        'status': 'ok',
        'message': 'Event ingested into vector store',
        'timestamp': metadata['timestamp']
    })


@app.get('/summarize')
async def summarize(q: str = 'patrol events', top_k: int = 5):
    """
    Generate an LLM-based summary of patrol events using RAG retrieval.
    Queries similar events and generates insights, patterns, and recommendations.
    """
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail='top_k must be between 1 and 20')
    
    summary = summarizer.summarize(query=q, top_k=top_k)
    
    return JSONResponse({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'query': q,
        'summary': summary,
        'metadata': {
            'model': 'flan-t5-small',
            'retrieval_top_k': top_k,
            'cost_aware': True
        }
    })


@app.post('/suspicious')
async def flag_suspicious(payload: EventPayload):
    """
    Flag a potentially suspicious activity for investigation.
    Uses heuristics and context to suggest severity and recommended actions.
    """
    text = payload.text
    if not text:
        raise HTTPException(status_code=400, detail='`text` required')
    
    timestamp = payload.timestamp or datetime.utcnow().isoformat() + 'Z'
    
    # Simple heuristic flags for suspicious activity
    suspicious_keywords = ['loitering', 'weapon', 'suspicious', 'unauthorized', 'breach', 'theft', 'assault']
    high_severity_keywords = ['weapon', 'armed', 'explosion', 'active threat']
    
    text_lower = text.lower()
    has_suspicious = any(kw in text_lower for kw in suspicious_keywords)
    has_high_severity = any(kw in text_lower for kw in high_severity_keywords)
    
    severity = 'high' if has_high_severity else 'medium' if has_suspicious else 'low'
    
    # Retrieve similar historical incidents for context
    context_items = rag.search(text, top_k=3)
    context_insights = [{'text': item['text'], 'metadata': item['metadata']} for item in context_items]
    
    # Ingest this suspicious activity
    rag.ingest(f"[SUSPICIOUS] {text}", metadata={
        'type': 'suspicious_activity',
        'camera_id': payload.camera_id or 'unknown',
        'timestamp': timestamp,
        'severity': severity
    })
    
    return JSONResponse({
        'timestamp': timestamp,
        'camera_id': payload.camera_id or 'unknown',
        'alert_id': f"alert_{datetime.utcnow().timestamp()}",
        'alert_type': 'suspicious_activity',
        'severity': severity,
        'message': text,
        'heuristic_flags': ['keyword_match'] if has_suspicious else [],
        'historical_context': context_insights,
        'recommended_actions': [
            'Dispatch officer for verification',
            'Increase monitoring of area',
            'Check for past incidents',
            'Prepare incident report'
        ] if severity == 'high' else ['Monitor situation', 'Log observation']
    })


@app.get('/status')
async def system_status():
    """Check system health and statistics."""
    return JSONResponse({
        'status': 'operational',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'components': {
            'detector': 'ready',
            'rag_store': f'ready ({len(rag.docs)} documents indexed)',
            'summarizer': 'ready'
        },
        'models': {
            'detection': 'fasterrcnn_resnet50_fpn',
            'embedding': 'all-MiniLM-L6-v2',
            'summarization': 'flan-t5-small'
        }
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
