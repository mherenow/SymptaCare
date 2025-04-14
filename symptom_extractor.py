import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language

class SymptomExtractor:
    """
    A simplified class for extracting health symptoms from text using NLP techniques.
    Only returns the extracted symptoms without additional metadata.
    """
    
    def __init__(self):
        # Load SpaCy model - medium-sized English model with word vectors
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # If model isn't installed, provide instructions
            print("Please install the required SpaCy model by running:")
            print("python -m spacy download en_core_web_md")
            raise
            
        # Initialize matcher components
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Set up pattern matching
        self._setup_patterns()
        
        # Add custom component to the pipeline
        if not self.nlp.has_pipe("symptom_extractor"):
            self.nlp.add_pipe("symptom_extractor", last=True)
        
        # Common symptom list - this could be expanded significantly
        self.common_symptoms = [
            "fever", "cough", "headache", "sore throat", "shortness of breath", 
            "fatigue", "nausea", "vomiting", "diarrhea", "rash", "chest pain",
            "stomach ache", "back pain", "joint pain", "muscle pain", "dizziness",
            "congestion", "runny nose", "chills", "sweating", "loss of appetite",
            "insomnia", "anxiety", "depression", "irritability", "confusion",
            "swelling", "redness", "itching", "blurred vision", "difficulty swallowing",
            "sneezing", "wheezing"
        ]
        
        # Add symptoms to the phrase matcher
        symptom_patterns = [self.nlp.make_doc(symptom) for symptom in self.common_symptoms]
        self.phrase_matcher.add("COMMON_SYMPTOMS", symptom_patterns)
    
    def _setup_patterns(self):
        """Set up pattern matching for symptoms"""
        
        # Pattern 1: "I have/feel/am experiencing [symptom]"
        have_pattern = [
            [{"LOWER": {"IN": ["i", "ive", "i've"]}}, 
             {"LOWER": {"IN": ["have", "had", "experiencing", "experience", "experienced"]}},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ"]}},  # Optional determiner or adjective
             {"POS": {"IN": ["NOUN", "ADJ"]}}],
             
            [{"LOWER": {"IN": ["i", "ive", "i've"]}}, 
             {"LOWER": {"IN": ["feel", "felt", "experiencing", "experience", "experienced"]}},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ"]}},  # Optional determiner or adjective
             {"POS": {"IN": ["NOUN", "ADJ"]}}],
             
            [{"LOWER": {"IN": ["i", "ive", "i've"]}}, 
             {"LOWER": {"IN": ["am", "was", "been"]}},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ"]}},  # Optional determiner or adjective
             {"POS": {"IN": ["NOUN", "ADJ"]}}]
        ]
        
        # Pattern 2: "My [body part] is [condition]"
        body_part_pattern = [
            [{"LOWER": "my"}, 
             {"POS": "NOUN"},  # Body part
             {"LOWER": {"IN": ["is", "feels", "hurts", "aches"]}},
             {"OP": "?", "POS": "ADV"},  # Optional adverb (really, very)
             {"POS": "ADJ"}]  # Condition (sore, painful, itchy)
        ]
        
        # Add patterns to matcher
        self.matcher.add("HAVE_SYMPTOM", have_pattern)
        self.matcher.add("BODY_PART_CONDITION", body_part_pattern)
    
    def extract_symptoms(self, text):
        """
        Extract only symptom texts from the input text
        
        Args:
            text (str): User input text
        
        Returns:
            list: List of extracted symptom strings
        """
        doc = self.nlp(text)
        symptoms_data = []
        
        # Use the phrase matcher to find common symptoms
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            symptoms_data.append(span.text)
        
        # Use pattern matcher for more complex symptom patterns
        pattern_matches = self.matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            symptom_text = self._extract_symptom_from_span(span)
            if symptom_text:
                symptoms_data.append(symptom_text)
        
        # Add any symptoms found through custom NER component
        for ent in doc.ents:
            if ent.label_ == "SYMPTOM":
                symptoms_data.append(ent.text)
        
        # Deduplicate symptoms and return only unique symptom strings
        unique_symptoms = list(set([s.lower() for s in symptoms_data]))
        return unique_symptoms
    
    def _extract_symptom_from_span(self, span):
        """Extract the actual symptom from a matched span"""
        
        # Logic depends on the pattern matched
        if span[0].text.lower() in ["i", "ive", "i've"]:
            # For "I have/feel/am" patterns, the symptom is likely at the end
            return span[-1].text
            
        elif span[0].text.lower() == "my":
            # For "My [body part] is [condition]" patterns, combine body part and condition
            body_part = span[1].text
            condition = span[-1].text
            return f"{condition} {body_part}"
        
        return None

@Language.factory("symptom_extractor")
def create_symptom_component(nlp, name):
    """Custom component to identify symptoms through custom rules"""
    return SymptomExtractorComponent()

class SymptomExtractorComponent:
    def __init__(self):
        # Additional terms that might indicate symptoms
        self.symptom_indicators = ["pain", "ache", "discomfort", "sore", "hurt", 
                                   "swollen", "inflamed", "irritated"]
    
    def __call__(self, doc):
        """Process the document and add entities"""
        new_ents = list(doc.ents)
        
        # Check for specific symptom patterns
        for i, token in enumerate(doc):
            # Check for symptoms with body parts
            if token.text.lower() in self.symptom_indicators:
                # Look for body parts after symptom indicators
                if i < len(doc) - 1 and doc[i+1].pos_ == "NOUN":
                    start = i
                    end = i + 2
                    symptom_span = Span(doc, start, end, label="SYMPTOM")
                    new_ents.append(symptom_span)
        
        # Set the entities
        doc.ents = new_ents
        return doc

def test_symptom_extractor():
    """Test function to demonstrate the symptom extractor's capabilities"""
    extractor = SymptomExtractor()
    
    # Test cases
    test_texts = [
        "I've been having a headache for the past two days.",
        "My throat is really sore and I have a fever.",
        "I feel dizzy when I stand up and I've been coughing a lot.",
        "I'm experiencing chest pain and shortness of breath.",
        "My back hurts and I have a runny nose.",
        "I think I might have the flu because I have body aches and chills."
    ]
    
    for text in test_texts:
        print(f"\nInput: {text}")
        symptoms = extractor.extract_symptoms(text)
        print("Symptoms found:", ", ".join(symptoms))
    
    return extractor

if __name__ == "__main__":
    # Run the test function
    extractor = test_symptom_extractor()