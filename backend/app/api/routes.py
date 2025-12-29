import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.predictor import train_model

router = APIRouter()

class TrainingRequest(BaseModel):
  learning_rate: Optional[float] = 0.01
  n_iterations: Optional[int] = 10000

@router.post("/train")
def train_endpoint(request: TrainingRequest = None):
  try:
    csv_path = os.path.join(os.path.dirname(__file__),'..', 'data','housing_data.csv')
    csv_path = os.path.abspath(csv_path)

    #check if file exists
    if not os.path.exists(csv_path):
      raise HTTPException(status_code=404, detail="Dataset file not found")

    learning_rate = request.learning_rate if request else 0.01
    n_iterations = request.n_iterations if request else 10000

    result = train_model(
      csv_path=csv_path,
      learning_rate=learning_rate,
      n_iterations=n_iterations
    )

    if result['status'] == 'error':
      raise HTTPException(status_code=404, detail=result['message'])
    return result



  except HTTPException:
      raise
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")