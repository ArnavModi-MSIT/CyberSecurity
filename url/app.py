from fastapi import FastAPI
from joblib import load
import numpy as np
import re
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client
from pathlib import Path
import os
from fastapi.middleware.cors import CORSMiddleware  # <-- Add this

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = None
scaler = None
encoder = None

# Supabase client initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class URLRequest(BaseModel):
    url: str

class FeedbackRequest(BaseModel):
    url: str
    type: str

def extract_features(url):
    url = str(url).lower()
    tokens = re.split(r'[/.-_]', url)
    valid_tokens = [t for t in tokens if t]
    avg_token_length = np.mean([len(t) for t in valid_tokens]) if valid_tokens else 0
    
    return {
        'url_length': len(url),
        'dots_count': url.count('.'),
        'digits_count': sum(map(str.isdigit, url)),
        'special_chars_count': len(re.findall(r'[^a-z0-9.-_/]', url)),
        'path_depth': url.count('/'),
        'has_http': int(url.startswith('http://')),
        'has_www': int('www.' in url),
        'avg_token_length': avg_token_length,
        'tld': int(url.split('.')[-1] in {'com', 'net', 'org'}),
        'shortened': int('bit.ly' in url or 'goo.gl' in url),
        'hex_chars': int(bool(re.search(r'%[0-9a-f]{2}', url)))
    }

MODEL_PATH = Path(__file__).parent / "phishing_detector.joblib"

@app.on_event("startup")
async def load_model():
    global model, scaler, encoder, supabase
    try:
        # Load ML model
        data = load(MODEL_PATH)
        model = data['model']
        scaler = data['scaler']
        encoder = data['encoder']
        print(f"Supabase Connected: {supabase is not None}")
        print(f"Supabase URL: {SUPABASE_URL[:15]}...")  # Log partial URL
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
        print(f"Supabase Init Error: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "ML Service API"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(url_request: URLRequest):
    try:
        # Check database first
        existing = supabase.table("urls")\
            .select("type")\
            .eq("url", url_request.url)\
            .execute()
        
        # Add error logging
        print(f"Supabase response: {existing}")
        
        if existing.data:
            return {
                "url": url_request.url,
                "prediction": "good" if existing.data[0]['type'].lower() in ['good', 'benign'] else "bad",
                "source": "database",
                "confidence": 1.0
            }

        # If URL not found in database, use ML model
        # Extract features
        features = extract_features(url_request.url)
        features_df = pd.DataFrame([features])
        
        # Transform features - Use only the features that were part of the training
        cat_features = encoder.transform(features_df[['has_http', 'has_www']])
        num_features = scaler.transform(features_df[['url_length', 'dots_count', 'digits_count', 
                                                    'special_chars_count', 'path_depth', 'avg_token_length']])
        
        # Combine features
        X = np.hstack([num_features, cat_features])
        
        # Predict
        proba = model.predict_proba(X)[0]
        
        result = {
            "url": url_request.url,
            "prediction": "good" if proba[0] > 0.7 else "bad",
            "source": "model",
            "confidence": float(max(proba))
        }

        return result
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        # Insert or update the URL in Supabase
        response = supabase.table("urls").upsert({
            "url": feedback.url,
            "type": feedback.type
        }).execute()
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/test-supabase")
async def test_db():
    try:
        test = supabase.table("urls").select("count").execute()
        return {"status": "connected", "data": test.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}