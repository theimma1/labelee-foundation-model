# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import random
import uuid

app = FastAPI(title="Labelee Personalization API")

# --- Data Models ---
class PredictionRequest(BaseModel):
    image_url: str
    text_prompt: str

class PredictionResponse(BaseModel):
    score: float

class PersonalizationJobRequest(BaseModel):
    user_model_name: str
    dataset_url: str

class PersonalizationJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Labelee API"}

@app.post("/api/v1/predict/{model_id}", response_model=PredictionResponse)
def predict(model_id: str, request: PredictionRequest):
    """MOCKED: This simulates making a prediction."""
    print(f"Received prediction request for model {model_id}")

    # --- NEW: Add input validation ---
    if not request.image_url or not request.text_prompt:
        raise HTTPException(status_code=400, detail="Image URL and Text Prompt cannot be empty.")

    time.sleep(0.5)
    return PredictionResponse(score=random.uniform(0.75, 0.95))

@app.post("/api/v1/jobs", response_model=PersonalizationJobResponse)
def create_personalization_job(request: PersonalizationJobRequest):
    """MOCKED: This simulates starting a new fine-tuning job."""
    print(f"Received request to start fine-tuning job: {request.user_model_name}")
    
    # Generate a fake job ID
    job_id = str(uuid.uuid4())
    
    return PersonalizationJobResponse(
        job_id=job_id,
        status="QUEUED",
        message="Personalization job has been successfully queued."
    )