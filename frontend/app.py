from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import httpx
import os
from groq_chatbot import GroqChatbot

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize chatbot
chatbot = None

# ML Service URL
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")

class URLRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    message: str

class FeedbackRequest(BaseModel):
    url: str
    type: str

@app.on_event("startup")
async def startup_event():
    global chatbot
    try:
        chatbot = GroqChatbot()
        print("Chatbot initialized successfully")
    except Exception as e:
        print(f"Chatbot initialization failed: {str(e)}")
        chatbot = None

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_url(url_request: URLRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/predict",
                json={"url": url_request.url}
            )
            return response.json()
    except Exception as e:
        return {"error": str(e)}

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

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ML_SERVICE_URL}/feedback",
                json={"url": feedback.url, "type": feedback.type}
            )
            return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}