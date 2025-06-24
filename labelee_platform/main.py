from fastapi import FastAPI
from pydantic import BaseModel
import time
import random

app = FastAPI(title="Labelee Personalization API")

# Define the data structures for requests/responses
class PredictionRequest(BaseModel):
    image_url: str
    text_prompt: str
    task: str

class PredictionResponse(BaseModel):
    score: float

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    personalized_model_id: str | None = None

# --- MOCKED API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Labelee API"}

@app.post("/api/v1/predict/{model_id}", response_model=PredictionResponse)
def predict(model_id: str, request: PredictionRequest):
    """MOCKED: This simulates making a prediction."""
    print(f"Received prediction request for model {model_id}")
    # In the future, this will call the real model. For now, return a fake score.
    time.sleep(0.5) # Simulate work
    return PredictionResponse(score=random.random())

@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """MOCKED: This simulates checking a fine-tuning job."""
    print(f"Checking status for job {job_id}")
    # In the future, this will check a real database.
    return JobStatusResponse(
        job_id=job_id,
        status="COMPLETED",
        personalized_model_id=f"model-for-{job_id}"
    )