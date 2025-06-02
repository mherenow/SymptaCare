import json
import logging
import math
import os
from collections import defaultdict
from fuzzywuzzy import fuzz, process

#Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output with simple format
        logging.FileHandler("diagnosis_engine.log", mode='a')  # File output with detailed format
    ]
)

# Create separate formatter for file handler with detailed format
file_handler = logging.FileHandler("diagnosis_engine.log", mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('DiagnosisEngine')
logger.handlers.clear()  # Clear default handlers
logger.addHandler(logging.StreamHandler())  # Simple console output
logger.addHandler(file_handler)  # Detailed file output
logger.setLevel(logging.INFO)

class DiagnosisEngine:
    """Engine for providing diagnosis based on extracted symptoms, age, and sex using local logic only"""
    def __init__(self):
        # No OpenEMR API setup
        # Cache for conditions and their symptoms
        self.condition_cache = {}
        # Load common condition database
        self.condition_db = self._load_condition_database()
        # Severity levels for triage
        self.severity_levels = {
            "emergency": ["heart attack", "stroke", "severe bleeding", "breathing difficulty", 
                         "chest pain", "severe head injury", "seizure", "unconsciousness",
                         "anaphylaxis", "severe burn", "poisoning", "suicide attempt",
                         "lack of responsiveness", "severe yellowing of skin", "severe unusual drowsiness",
                         "severe liver failure", "acute liver failure", "severe liver damage",
                         "severe skin turning yellow", "severe yellowish skin", "severe yellowish eyes",
                         "severe yellow tint to skin", "severe yellow tint to eyes"],
            "severe": ["broken bone", "deep wound", "high fever", "severe pain", 
                      "dehydration", "infection", "moderate burn", "concussion",
                      "severe vomiting", "severe diarrhea", "asthma attack", "allergic reaction",
                      "moderate yellowing of skin", "moderate unusual drowsiness",
                      "liver disease", "hepatitis", "liver damage", "jaundice",
                      "skin turning yellow", "skin taking on yellow shade", "skin becoming yellow",
                      "yellow tint to skin", "yellowish skin", "yellowish discoloration",
                      "yellowish tint", "yellowish hue", "yellowish cast",
                      "eyes turning yellow", "eyes taking on yellow shade", "eyes becoming yellow",
                      "yellow tint to eyes", "yellowish eyes", "yellowish discoloration of eyes",
                      "yellowish tint in eyes", "yellowish hue in eyes", "yellowish cast in eyes"],
            "moderate": ["common cold", "earache", "sore throat", "mild fever",
                        "mild pain", "mild nausea", "mild diarrhea", "cough",
                        "mild headache", "mild rash", "mild allergies",
                        "mild yellowing of skin", "mild unusual drowsiness"],
            "mild": ["minor ache", "minor burn", "minor cut", "insect bite",
                    "mild skin irritation", "mild congestion", "mild soreness"]
        }
        
        # Age-specific condition weights
        self.age_weights = {
            # Children (0-12)
            "child": {
                "common cold": 1.2, "ear infection": 1.3, "strep throat": 1.2,
                "chickenpox": 1.4, "hand foot mouth disease": 1.3, "roseola": 1.2,
                "measles": 1.1, "mumps": 1.1, "whooping cough": 1.3, "croup": 1.4,
                "asthma": 1.2, "bronchiolitis": 1.4, "rsv": 1.4, "pneumonia": 1.2,
                "food allergies": 1.3, "eczema": 1.2, "conjunctivitis": 1.1
            },
            # Teenagers (13-19)
            "teen": {
                "acne": 1.4, "mononucleosis": 1.3, "sports injuries": 1.4,
                "anxiety": 1.2, "depression": 1.2, "eating disorders": 1.1,
                "migraine": 1.1, "appendicitis": 1.1, "scoliosis": 1.1,
                "strep throat": 1.1, "influenza": 1.0, "allergies": 1.0
            },
            # Adults (20-64)
            "adult": {
                "hypertension": 1.1, "diabetes": 1.1, "depression": 1.2,
                "anxiety": 1.2, "back pain": 1.3, "osteoarthritis": 1.1,
                "gerd": 1.1, "migraine": 1.1, "urinary tract infection": 1.1,
                "irritable bowel syndrome": 1.1, "asthma": 1.0
            },
            # Seniors (65+)
            "senior": {
                "hypertension": 1.3, "coronary artery disease": 1.3, "arthritis": 1.3,
                "osteoporosis": 1.3, "diabetes": 1.2, "alzheimer's": 1.2,
                "parkinson's": 1.2, "copd": 1.2, "cataracts": 1.2, "glaucoma": 1.2,
                "hearing loss": 1.2, "atrial fibrillation": 1.2, "chronic kidney disease": 1.2,
                "fall injuries": 1.3, "pneumonia": 1.2, "urinary incontinence": 1.1
            }
        }
        
        # Sex-specific condition weights
        self.sex_weights = {
            "male": {
                "prostate issues": 1.5, "testicular cancer": 1.5, "erectile dysfunction": 1.4,
                "heart disease": 1.2, "sleep apnea": 1.1, "gout": 1.2,
                "kidney stones": 1.1, "color blindness": 1.1, "hemophilia": 1.2,
                "baldness": 1.3
            },
            "female": {
                "breast cancer": 1.3, "ovarian cancer": 1.5, "cervical cancer": 1.5,
                "endometriosis": 1.5, "pcos": 1.4, "menstrual disorders": 1.4,
                "pregnancy complications": 1.5, "osteoporosis": 1.2, "thyroid disorders": 1.2,
                "fibromyalgia": 1.2, "lupus": 1.2, "migraine": 1.1, "depression": 1.1
            }
        }
        
        # Common home remedy recommendations for minor conditions
        self.home_remedies = {
            "common cold": [
                "Rest and stay hydrated",
                "Use a humidifier or take steamy showers",
                "Try saline nasal drops or sprays",
                "Use over-the-counter pain relievers like acetaminophen or ibuprofen",
                "Gargle with salt water for sore throat"
            ],
            "headache": [
                "Rest in a quiet, dark room",
                "Apply a cold or warm compress to your head",
                "Stay hydrated",
                "Consider over-the-counter pain relievers like acetaminophen or ibuprofen"
            ],
            "indigestion": [
                "Eat smaller meals and avoid fatty foods",
                "Avoid lying down right after eating",
                "Try over-the-counter antacids",
                "Drink ginger or chamomile tea",
                "Avoid trigger foods like spicy foods, citrus, and caffeine"
            ],
            "mild fever": [
                "Rest and stay hydrated",
                "Take a lukewarm bath",
                "Use a light blanket if you have chills",
                "Consider over-the-counter fever reducers like acetaminophen or ibuprofen"
            ],
            "sore throat": [
                "Gargle with salt water",
                "Drink warm fluids like tea with honey",
                "Use throat lozenges or hard candies",
                "Run a humidifier",
                "Consider over-the-counter pain relievers like acetaminophen or ibuprofen"
            ],
            "minor cough": [
                "Stay hydrated",
                "Use honey (for adults and children over 1 year)",
                "Try throat lozenges or hard candies",
                "Use a humidifier",
                "Consider over-the-counter cough suppressants for dry coughs"
            ],
            "mild allergies": [
                "Avoid known allergens",
                "Use over-the-counter antihistamines",
                "Try nasal irrigation with saline",
                "Use air purifiers in your home",
                "Keep windows closed during high pollen seasons"
            ],
            "minor burn": [
                "Run cool (not cold) water over the burn for 10-15 minutes",
                "Apply aloe vera gel",
                "Take over-the-counter pain relievers like acetaminophen or ibuprofen",
                "Don't break blisters",
                "Cover with a clean, non-stick bandage"
            ],
            "insect bite": [
                "Wash the area with soap and water",
                "Apply a cold compress",
                "Use anti-itch creams like calamine lotion",
                "Consider taking an oral antihistamine",
                "Apply a baking soda paste to the bite"
            ],
            "minor cut": [
                "Clean the wound with soap and water",
                "Apply gentle pressure to stop bleeding",
                "Apply an antibiotic ointment",
                "Cover with a clean bandage",
                "Change the bandage daily"
            ]
        }
    
    def _load_condition_database(self):
        """Load comprehensive database of common conditions and their associated symptoms"""
        return {
            # Respiratory conditions
            "common cold": ["cough", "congestion", "runny nose", "sore throat", "sneezing", "mild fever"],
            "influenza": ["fever", "chills", "body aches", "fatigue", "cough", "sore throat", "congestion", "headache"],
            "covid-19": ["fever", "cough", "shortness of breath", "fatigue", "body aches", "loss of taste", "loss of smell", "sore throat"],
            "pneumonia": ["cough", "fever", "chills", "shortness of breath", "chest pain", "fatigue", "phlegm"],
            "bronchitis": ["cough", "phlegm", "fatigue", "shortness of breath", "chest discomfort", "mild fever"],
            "asthma": ["shortness of breath", "wheezing", "cough", "chest tightness"],
            "sinusitis": ["facial pain", "congestion", "runny nose", "reduced sense of smell", "headache", "cough"],
            
            # Cardiovascular conditions
            "hypertension": ["headache", "shortness of breath", "chest pain", "dizziness", "blurred vision"],
            "heart attack": ["chest pain", "shortness of breath", "pain in arms", "cold sweat", "nausea", "dizziness"],
            "stroke": ["sudden numbness", "confusion", "trouble speaking", "trouble walking", "severe headache", "dizziness", "lack of responsiveness"],
            "heart failure": ["shortness of breath", "fatigue", "swelling", "rapid heartbeat", "persistent cough", "unusual drowsiness"],
            
            # Gastrointestinal conditions
            "gastroenteritis": ["diarrhea", "nausea", "vomiting", "abdominal cramps", "mild fever", "headache"],
            "irritable bowel syndrome": ["abdominal pain", "bloating", "gas", "diarrhea", "constipation", "mucus in stool"],
            "peptic ulcer": ["abdominal pain", "bloating", "heartburn", "nausea", "weight loss"],
            "gastroesophageal reflux disease": ["heartburn", "chest pain", "regurgitation", "difficulty swallowing", "chronic cough"],
            "food poisoning": ["nausea", "vomiting", "diarrhea", "abdominal cramps", "fever", "headache"],
            "gallstones": ["upper abdominal pain", "nausea", "vomiting", "back pain", "digestive problems"],
            "celiac disease": ["diarrhea", "bloating", "gas", "fatigue", "weight loss", "anemia"],
            "liver disease": ["yellowing of skin", "abdominal pain", "fatigue", "nausea", "vomiting", "unusual drowsiness", "lack of responsiveness",
                            "skin turning yellow", "skin taking on yellow shade", "skin becoming yellow",
                            "yellow tint to skin", "yellowish skin", "yellowish discoloration",
                            "yellowish tint", "yellowish hue", "yellowish cast",
                            "eyes turning yellow", "eyes taking on yellow shade", "eyes becoming yellow",
                            "yellow tint to eyes", "yellowish eyes", "yellowish discoloration of eyes",
                            "yellowish tint in eyes", "yellowish hue in eyes", "yellowish cast in eyes"],
            "hepatitis": ["yellowing of skin", "abdominal pain", "fatigue", "nausea", "vomiting", "unusual drowsiness",
                         "skin turning yellow", "skin taking on yellow shade", "skin becoming yellow",
                         "yellow tint to skin", "yellowish skin", "yellowish discoloration",
                         "yellowish tint", "yellowish hue", "yellowish cast",
                         "eyes turning yellow", "eyes taking on yellow shade", "eyes becoming yellow",
                         "yellow tint to eyes", "yellowish eyes", "yellowish discoloration of eyes",
                         "yellowish tint in eyes", "yellowish hue in eyes", "yellowish cast in eyes"],
            
            # Neurological conditions
            "migraine": ["severe headache", "nausea", "vomiting", "light sensitivity", "sound sensitivity", "vision changes"],
            "tension headache": ["dull headache", "pressure sensation", "tenderness", "tightness"],
            "epilepsy": ["seizures", "confusion", "staring spells", "jerky movements", "loss of consciousness", "lack of responsiveness"],
            "multiple sclerosis": ["fatigue", "vision problems", "numbness", "tingling", "weakness", "balance problems"],
            "parkinson's disease": ["tremor", "stiffness", "slow movement", "balance problems", "speech changes"],
            "brain injury": ["headache", "confusion", "dizziness", "nausea", "vomiting", "unusual drowsiness", "lack of responsiveness"],
            "meningitis": ["severe headache", "fever", "stiff neck", "confusion", "unusual drowsiness", "lack of responsiveness"],
            "encephalitis": ["fever", "headache", "confusion", "seizures", "unusual drowsiness", "lack of responsiveness"],
            
            # Musculoskeletal conditions
            "arthritis": ["joint pain", "stiffness", "swelling", "reduced range of motion", "redness"],
            "osteoporosis": ["back pain", "stooped posture", "height loss", "bone fracture"],
            "fibromyalgia": ["widespread pain", "fatigue", "sleep problems", "cognitive difficulties", "headache"],
            "carpal tunnel syndrome": ["numbness", "tingling", "weakness", "pain in hands and fingers"],
            "tendinitis": ["pain", "tenderness", "mild swelling", "restricted movement"],
            
            # Dermatological conditions
            "eczema": ["dry skin", "itching", "red rash", "swelling", "crusting"],
            "psoriasis": ["red patches", "silver scales", "dry skin", "itching", "burning", "soreness"],
            "acne": ["pimples", "whiteheads", "blackheads", "red spots", "oily skin"],
            "rosacea": ["facial redness", "swollen bumps", "eye problems", "enlarged nose"],
            "contact dermatitis": ["red rash", "itching", "burning", "swelling", "blisters"],
            "jaundice": ["yellowing of skin", "yellowing of eyes", "dark urine", "pale stools", "itching", "fatigue"],
            
            # Mental health conditions
            "depression": ["sadness", "loss of interest", "sleep changes", "fatigue", "difficulty concentrating", "suicidal thoughts"],
            "anxiety disorder": ["excessive worry", "restlessness", "fatigue", "difficulty concentrating", "irritability", "muscle tension", "sleep problems"],
            "bipolar disorder": ["mood swings", "high energy", "low energy", "sleep problems", "poor judgment"],
            "schizophrenia": ["hallucinations", "delusions", "thought disorders", "lack of motivation", "social withdrawal"],
            "post-traumatic stress disorder": ["flashbacks", "nightmares", "severe anxiety", "uncontrollable thoughts", "mood changes"],
            
            # Endocrine conditions
            "diabetes": ["increased thirst", "frequent urination", "extreme hunger", "unexplained weight loss", "fatigue", "blurred vision"],
            "hypothyroidism": ["fatigue", "weight gain", "cold sensitivity", "constipation", "dry skin", "depression", "unusual drowsiness"],
            "hyperthyroidism": ["weight loss", "rapid heartbeat", "increased appetite", "anxiety", "tremor", "sweating"],
            "adrenal insufficiency": ["fatigue", "weight loss", "abdominal pain", "nausea", "vomiting", "low blood pressure", "unusual drowsiness"],
            
            # Infectious diseases
            "tuberculosis": ["cough", "chest pain", "bloody sputum", "fatigue", "fever", "night sweats", "weight loss"],
            "malaria": ["fever", "chills", "headache", "muscle aches", "fatigue", "nausea", "vomiting", "unusual drowsiness"],
            "lyme disease": ["rash", "fever", "chills", "fatigue", "body aches", "headache", "joint pain"],
            "mono": ["fatigue", "sore throat", "fever", "swollen lymph nodes", "headache", "rash"],
            
            # Minor conditions
            "minor headache": ["mild head pain", "pressure", "tension"],
            "indigestion": ["bloating", "nausea", "upper abdominal pain", "heartburn"],
            "mild fever": ["elevated temperature", "chills", "sweating", "headache", "muscle aches"],
            "insect bite": ["redness", "swelling", "itching", "localized pain"],
            "minor cut": ["skin break", "bleeding", "localized pain"],
            "minor burn": ["redness", "pain", "swelling", "blisters"],
            "sunburn": ["red skin", "pain", "swelling", "blisters", "peeling skin"],
            "minor cough": ["throat irritation", "chest discomfort"]
        }
    
    def _get_age_group(self, age):
        """Determine age group based on age"""
        if 0 <= age <= 12:
            return "child"
        elif 13 <= age <= 19:
            return "teen"
        elif 20 <= age <= 55:
            return "adult"
        else:
            return "senior"
    
    def _get_diagnosis_conditions(self, symptoms, age, sex):
        """Get diagnosis conditions using local symptom-to-condition matching"""
        logger.info("Using local diagnosis logic")
        
        # Calculate condition scores based on symptoms
        condition_scores = {}
        for condition, condition_symptoms in self.condition_db.items():
            score = 0
            matched_symptoms = []
            
            # For each reported symptom, check how well it matches the condition's symptoms
            for symptom in symptoms:
                best_match = None
                best_score = 0
                
                for condition_symptom in condition_symptoms:
                    # Calculate multiple similarity metrics for more robust matching
                    token_sort_score = fuzz.token_sort_ratio(symptom, condition_symptom)
                    token_set_score = fuzz.token_set_ratio(symptom, condition_symptom) 
                    
                    # Take the average of the two scores
                    similarity = (token_sort_score + token_set_score) / 2
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = condition_symptom
                
                # If we have a good match, add to the score
                if best_score > 70:  # Threshold for considering it a match
                    match_value = best_score / 100.0  # Convert to 0-1 scale
                    score += match_value
                    matched_symptoms.append((best_match, best_score))
            
            # Only include conditions that match at least one symptom well
            if matched_symptoms:
                # Calculate what percentage of the condition's typical symptoms were reported
                coverage = len(matched_symptoms) / len(condition_symptoms)
                
                # Weight by both match quality and coverage
                final_score = score * math.sqrt(coverage) * len(matched_symptoms)
                
                # Apply age and sex specific weights
                age_group = self._get_age_group(age)
                
                # Apply age weights if applicable
                if condition.lower() in self.age_weights.get(age_group, {}):
                    final_score *= self.age_weights[age_group][condition.lower()]
                
                # Apply sex weights if applicable
                if sex.lower() in self.sex_weights and condition.lower() in self.sex_weights[sex.lower()]:
                    final_score *= self.sex_weights[sex.lower()][condition.lower()]
                
                condition_scores[condition] = {
                    "score": final_score,
                    "matched_symptoms": matched_symptoms,
                    "confidence": min(final_score / len(condition_symptoms) * 100, 95)  # Cap at 95%
                }
        
        # Sort by score
        sorted_conditions = sorted(condition_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        # Take top results
        return [
            {
                "condition": condition,
                "confidence": round(data["confidence"], 1),
                "matched_symptoms": [match[0] for match in data["matched_symptoms"]]
            }
            for condition, data in sorted_conditions[:5]  # Return top 5 possibilities
        ]
    
    def _assess_severity(self, conditions, symptoms):
        """Assess the severity of the conditions and symptoms"""
        # Check for emergency symptoms first
        for symptom in symptoms:
            # Check for emergency-level jaundice symptoms
            if any(emergency_symptom in symptom.lower() for emergency_symptom in [
                "severe yellowing", "severe yellowish", "severe yellow tint",
                "severe skin turning yellow", "severe eyes turning yellow"
            ]):
                return "emergency", "Seek immediate medical attention - severe jaundice symptoms detected"
            
            # Check other emergency symptoms
            for emergency_symptom in self.severity_levels["emergency"]:
                if fuzz.token_sort_ratio(symptom, emergency_symptom) > 85:
                    return "emergency", "Seek immediate medical attention"
        
        # Check if any of the top conditions are emergencies
        for condition in conditions:
            condition_name = condition["condition"].lower()
            # Check emergency conditions
            for emergency_condition in self.severity_levels["emergency"]:
                if fuzz.token_sort_ratio(condition_name, emergency_condition) > 85:
                    return "emergency", "Seek immediate medical attention"
            # Check severe conditions
            for severe_condition in self.severity_levels["severe"]:
                if fuzz.token_sort_ratio(condition_name, severe_condition) > 85:
                    return "severe", "Seek medical attention within 24 hours"
        
        # Check for moderate symptoms/conditions regardless of confidence
        for symptom in symptoms:
            # Check for moderate-level jaundice symptoms
            if any(moderate_symptom in symptom.lower() for moderate_symptom in [
                "mild yellowing", "mild yellowish", "mild yellow tint",
                "mild skin turning yellow", "mild eyes turning yellow"
            ]):
                return "moderate", "Consider consulting a healthcare provider - mild jaundice symptoms detected"
            
            # Check other moderate symptoms
            for moderate_condition in self.severity_levels["moderate"]:
                if fuzz.token_sort_ratio(symptom, moderate_condition) > 85:
                    return "moderate", "Consider consulting a healthcare provider"
        
        # Check for severe symptoms that aren't emergencies
        for symptom in symptoms:
            # Check for severe-level jaundice symptoms
            if any(severe_symptom in symptom.lower() for severe_symptom in [
                "yellowing", "yellowish", "yellow tint",
                "skin turning yellow", "eyes turning yellow"
            ]):
                return "severe", "Seek medical attention within 24 hours - jaundice symptoms detected"
            
            # Check other severe symptoms
            for severe_condition in self.severity_levels["severe"]:
                if fuzz.token_sort_ratio(symptom, severe_condition) > 85:
                    return "severe", "Seek medical attention within 24 hours"
        
        # Default to caution
        for condition in conditions:
            condition_name = condition["condition"].lower()
            for moderate_condition in self.severity_levels["moderate"]:
                if fuzz.token_sort_ratio(condition_name, moderate_condition) > 85:
                    return "moderate", "Consider consulting a healthcare provider"
        # Check for mild symptoms/conditions regardless of confidence
        for symptom in symptoms:
            for mild_condition in self.severity_levels["mild"]:
                if fuzz.token_sort_ratio(symptom, mild_condition) > 85:
                    return "mild", "Self-care may be appropriate"
        for condition in conditions:
            condition_name = condition["condition"].lower()
            for mild_condition in self.severity_levels["mild"]:
                if fuzz.token_sort_ratio(condition_name, mild_condition) > 85:
                    return "mild", "Self-care may be appropriate"
        # Default to caution
        return "moderate", "Consider consulting a healthcare provider"
    
    def _get_home_remedies(self, condition):
        """Get home remedies for a given condition"""
        # Find the closest matching condition in our home remedies database
        best_match = process.extractOne(
            condition.lower(),
            list(self.home_remedies.keys()),
            scorer=fuzz.token_set_ratio,
            score_cutoff=75
        )
        
        if best_match:
            return self.home_remedies[best_match[0]]
        return []
    
    def _format_diagnosis_info(self, diagnosis_data, symptoms, age, sex):
        """Format diagnosis information into a user-friendly response"""
        conditions = diagnosis_data["conditions"]
        severity, severity_advice = self._assess_severity(conditions, symptoms)
        
        if not conditions:
            response = {
                "conditions": [],
                "severity": "unknown",
                "advice": "If symptoms persist, please consult with a healthcare provider.",
                "home_remedies": []
            }
        else:
            top_condition = conditions[0]["condition"]
            
            # Always provide home remedies if available for the most likely condition
            home_remedies = self._get_home_remedies(top_condition)
            
            response = {
                "conditions": conditions,
                "severity": severity,
                "advice": severity_advice,
                "home_remedies": home_remedies
            }
            # Add disclaimer
            response["disclaimer"] = "This is not a definitive medical diagnosis. Always consult with a healthcare professional for proper evaluation."
        
        return response
    
    def get_diagnosis(self, symptoms, age, sex):
        """Get diagnosis based on symptoms, age, and sex using local logic"""
        logger.info(f"Processing diagnosis request for age={age}, sex={sex}, symptoms={symptoms}")
        if not symptoms:
            return {
                "conditions": [],
                "severity": "unknown",
                "advice": "If you're experiencing symptoms, please provide details for better guidance.",
                "home_remedies": [],
                "disclaimer": "This is not a definitive medical diagnosis. Always consult with a healthcare professional for proper evaluation."
            }
        
        # Use local diagnosis logic as the main method
        conditions = self._get_diagnosis_conditions(symptoms, age, sex)
        diagnosis_data = {
            "conditions": conditions
        }
        return self._format_diagnosis_info(diagnosis_data, symptoms, age, sex)

# Test cases for demonstration
if __name__ == "__main__":
    engine = DiagnosisEngine()
    test_cases = [
        # Mild cases for home remedy
        {
            "symptoms": ["sore throat", "mild fever"],
            "age": 25,
            "sex": "female"
        },
        {
            "symptoms": ["minor cut", "localized pain"],
            "age": 30,
            "sex": "male"
        },
        {
            "symptoms": ["insect bite", "itching"],
            "age": 12,
            "sex": "male"
        },
        {
            "symptoms": ["minor burn", "redness"],
            "age": 40,
            "sex": "female"
        },
        # Existing moderate/severe cases
        {
            "symptoms": ["cough", "fever", "fatigue", "sore throat", "congestion"],
            "age": 35,
            "sex": "male"
        },
        {
            "symptoms": ["headache", "nausea", "sensitivity to light", "blurred vision"],
            "age": 28,
            "sex": "female"
        },
        {
            "symptoms": ["chest pain", "shortness of breath", "sweating", "nausea"],
            "age": 62,
            "sex": "male"
        },
        {
            "symptoms": ["rash", "itching", "redness", "swelling"],
            "age": 8,
            "sex": "female"
        },
        {
            "symptoms": ["abdominal pain", "bloating", "diarrhea", "gas"],
            "age": 42,
            "sex": "female"
        },
        {
            "symptoms": ["chest pain"],
            "age": 68,
            "sex": "male"
        },
        {
            "symptoms": ["severe wound on knee"],
            "age": 21,
            "sex": "male"
        }
    
    ]
    for i, case in enumerate(test_cases):
        result = engine.get_diagnosis(case["symptoms"], case["age"], case["sex"])
        most_likely = result["conditions"][0]["condition"] if result["conditions"] else None
        other_conditions = [c["condition"] for c in result["conditions"][1:]] if len(result["conditions"]) > 1 else []
        print({
            "test_case": i+1,
            "most_likely_condition": most_likely,
            "other_possible_conditions": other_conditions,
            "severity": result["severity"],
            "home_remedies": result["home_remedies"]
        })
