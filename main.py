from flask import Flask, request, jsonify
from symptom_extractor import SymptomExtractor
from diagnosis_engine import DiagnosisEngine

app = Flask(__name__)

# Initialize the SymptomExtractor and DiagnosisEngine
symptom_extractor = SymptomExtractor()
diagnosis_engine = DiagnosisEngine()

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """
    API endpoint to diagnose based on user input.
    Expects JSON input with 'age', 'gender', and 'input'.
    """
    try:
        # Parse the JSON request
        data = request.get_json()
        age = data.get('age')
        gender = data.get('gender')
        user_input = data.get('input')

        # Validate input
        if not age or not gender or not user_input:
            return jsonify({"error": "Missing required fields: 'age', 'gender', or 'input'"}), 400

        # Extract symptoms using SymptomExtractor
        symptoms = symptom_extractor.extract_symptoms(user_input)

        # If no symptoms are extracted, return an appropriate response
        if not symptoms:
            return jsonify({
                "message": "No symptoms could be identified from the input.",
                "suggestion": "Please provide more detailed information about your symptoms."
            }), 200

        # Get diagnosis using DiagnosisEngine
        diagnosis = diagnosis_engine.get_diagnosis(symptoms, age, gender)

        # Return the diagnosis as JSON
        return jsonify(diagnosis), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)