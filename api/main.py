from fastapi import FastAPI, Request, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from joblib import load
import numpy as np
import re
from pydantic import BaseModel
import pandas as pd
from supabase import create_client, Client
from groq_chatbot import GroqChatbot
from pathlib import Path
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model = None
scaler = None
encoder = None
chatbot = None

# Supabase client initialization
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class URLRequest(BaseModel):
    url: str

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
    global model, scaler, encoder, chatbot
    try:
        # Load ML model first
        data = load(MODEL_PATH)
        model = data['model']
        scaler = data['scaler']
        encoder = data['encoder']
        
        # Initialize chatbot client with error handling
        try:
            chatbot = GroqChatbot()
            print("Chatbot initialized successfully")
        except Exception as e:
            print(f"Chatbot initialization failed: {str(e)}")
            chatbot = None
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(url_request: URLRequest):
    try:
        # Check database first
        existing = supabase.table("urls").select("type").eq("url", url_request.url).execute()
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

        # No database insertion

        return result
        
    except Exception as e:
        return {"error": str(e)}

class ChatRequest(BaseModel):
    message: str

SYSTEM_PROMPT = """You are a cybersecurity assistant specializing in phishing detection. 
    Help users analyze and identify potential threats. Be concise and technical when needed.Dont answer any question that is not related to cyber security.Only answer in text format."""
    
@app.post("/chat")
async def chat_response(chat_request: ChatRequest):
    if not chatbot:
        return {"error": "Chat service unavailable"}
    
    try:
        response = chatbot.client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chat_request.message}
            ],
            model=chatbot.model,
            temperature=0.7,
            max_tokens=1024
        )
        return {
            "question": chat_request.message,
            "answer": response.choices[0].message.content
        }
    except Exception as e:
        return {"error": str(e)}