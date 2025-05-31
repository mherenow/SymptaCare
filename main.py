from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from symptom_extractor import LlamaSymptomExtractor
from diagnosis_engine import DiagnosisEngine
import uvicorn

app = FastAPI(title="SymptaCare API", description="API for symptom extraction and diagnosis")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize the SymptomExtractor and DiagnosisEngine
symptom_extractor = LlamaSymptomExtractor()
diagnosis_engine = DiagnosisEngine()

# Define the request model
class DiagnosisRequest(BaseModel):
    age: int
    gender: str
    input: str
    

@app.post('/diagnose')
async def diagnose(request: DiagnosisRequest):
    """
    API endpoint to diagnose based on user input.
    Expects JSON input with 'age', 'gender', and 'input'.
    """
    try:
        # Extract symptoms using SymptomExtractor
        symptoms = symptom_extractor.extract_symptoms(request.input)

        # If no symptoms are extracted, return an appropriate response
        if not symptoms:
            return {
                "message": "No symptoms could be identified from the input.",
                "suggestion": "Please provide more detailed information about your symptoms."
            }

        # Get diagnosis using DiagnosisEngine
        diagnosis = diagnosis_engine.get_diagnosis(symptoms, request.age, request.gender)

        # Return the diagnosis
        return diagnosis

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)