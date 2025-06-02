import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from diagnosis_engine import DiagnosisEngine
from symptom_extractor import LlamaSymptomExtractor

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

# Log API key status (safely)
api_key_status = "Present" if os.getenv("NVIDIA_API_KEY") else "Missing"
print(f"[DIAGNOSTIC] NVIDIA_API_KEY status: {api_key_status}")

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
    print(f"[DIAGNOSTIC] Received request - Age: {request.age}, Gender: {request.gender}")
    print(f"[DIAGNOSTIC] User input: {request.input[:100]}...")  # Show first 100 chars only
    
    try:
        # Extract symptoms using SymptomExtractor
        symptoms = symptom_extractor.extract_symptoms(request.input)

        # If no symptoms are extracted, return an appropriate response
        if not symptoms:
            print("[DIAGNOSTIC] No symptoms extracted from input")
            return {
                "message": "No symptoms could be identified from the input.",
                "suggestion": "Please provide more detailed information about your symptoms."
            }

        print(f"[DIAGNOSTIC] Extracted symptoms: {symptoms}")

        # Get diagnosis using DiagnosisEngine
        diagnosis = diagnosis_engine.get_diagnosis(symptoms, request.age, request.gender)
        
        # Log diagnosis summary
        top_condition = diagnosis["conditions"][0]["condition"] if diagnosis["conditions"] else "None"
        print(f"[DIAGNOSTIC] Top condition: {top_condition}, Severity: {diagnosis['severity']}")

        # Return the diagnosis
        return diagnosis

    except Exception as e:
        print(f"[DIAGNOSTIC] Error during diagnosis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)