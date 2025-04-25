import spacy
import requests
import json
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
from fuzzywuzzy import fuzz, process
import re
from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess
import nltk
from nltk.corpus import wordnet

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading Wordnet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class SymptomExtractor:
    """Symptom extractor using multiple NLP techniques and medical knowledge bases"""
    def __init__(self, use_umls_api=False, umls_api_key = None):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Please install the required SpaCy model by running:")
            print("python -m spacy download en_core_web_md")
            raise

        # Intitalize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")

        # Add custom component to pipeline
        if not self.nlp.has_pipe("symptom_extractor"):
            self.nlp.add_pipe("symptom_extractor", last=True)

        # Load Symptom Database
        self.symptom_db = self._load_symptom_database()

        # Create symptom patterns
        self._setup_patterns()

        # Process symptoms for phrase matcher
        symptom_matcher = [self.nlp(symptom) for symptom in self.symptom_db]
        self.phrase_matcher.add("SYMPTOM", symptom_matcher)

        # Build lemma lookup for symptom verification
        self.symptom_lemmas = set()
        for symptom in self.symptom_db['common_symptoms']:
            doc = self.nlp(symptom)
            for token in doc:
                self.symptom_lemmas.add(token.lemma_.lower())

        # (Optional) UMLS API setup
        self.use_umls_api = use_umls_api
        self.umls_api_key = umls_api_key
        if use_umls_api and not umls_api_key:
            print("UMLS API key is required for UMLS API usage.")

        # Load symptom synonyms
        self.symptom_synonyms = self._build_symptom_synonyms()

    def _load_symptom_database(self):
        """Load comprehensive symptom database with categories"""

        # Core common symptoms
        common_symptoms = [
            # Respiratory symptoms
            "cough", "shortness of breath", "wheezing", "sore throat", "runny nose", 
            "stuffy nose", "sneeze", "congestion", "chest tightness", "difficulty breathing",
            "labored breathing", "rapid breathing", "shallow breathing", "coughing up blood",
            "phlegm", "mucus", "postnasal drip", "hoarseness", "voice changes",
            
            # Pain symptoms
            "headache", "chest pain", "back pain", "stomach pain", "abdominal pain", 
            "joint pain", "muscle pain", "neck pain", "ear pain", "throat pain",
            "shoulder pain", "knee pain", "toothache", "body ache", "eye pain",
            "migraine", "tension headache", "cluster headache", "radiating pain",
            "localized pain", "generalized pain", "sharp pain", "dull pain",
            
            # Digestive symptoms
            "nausea", "vomit", "diarrhea", "constipation", "bloating", "indigestion",
            "heartburn", "stomach ache", "loss of appetite", "increased appetite",
            "difficulty swallowing", "gas", "acid reflux", "belching", "burping",
            "abdominal cramps", "stomach cramps", "food intolerance", "food sensitivity",
            
            # General symptoms
            "fever", "chills", "fatigue", "weakness", "dizziness", "lightheaded",
            "fainting", "weight loss", "weight gain", "night sweats", "sweating",
            "dehydration", "thirst", "swelling", "inflammation", "malaise",
            "lethargy", "exhaustion", "general discomfort", "body temperature changes",
            
            # Skin symptoms
            "rash", "itching", "hives", "dry skin", "bruising", "bleeding",
            "redness", "swelling", "lump", "blister", "wound", "sore",
            "skin discoloration", "skin lesions", "skin peeling", "skin cracking",
            "skin irritation", "skin sensitivity", "skin dryness", "skin oiliness",
            
            # Neurological symptoms
            "confusion", "memory loss", "trouble speaking", "seizure", "tremor",
            "numbness", "tingling", "paralysis", "weakness", "coordination problems",
            "difficulty walking", "blurred vision", "double vision", "vision loss",
            "balance problems", "vertigo", "dizziness", "lightheadedness",
            "cognitive impairment", "mental fog", "brain fog",
            
            # Psychological symptoms
            "anxiety", "depression", "stress", "insomnia", "trouble sleeping",
            "irritability", "mood swings", "hallucinations", "paranoia", "panic attack",
            "difficulty concentrating", "brain fog", "confusion", "disorientation",
            "emotional instability", "mood changes", "personality changes",
            "behavioral changes", "social withdrawal",
            
            # Urinary/reproductive symptoms
            "frequent urination", "painful urination", "blood in urine", "urinary incontinence",
            "discharge", "irregular periods", "missed period", "heavy menstruation",
            "erectile dysfunction", "low libido", "itching", "burning sensation",
            "urinary urgency", "urinary frequency", "urinary hesitancy", "urinary retention",
            "sexual dysfunction", "menstrual cramps", "pelvic pain"
        ]
        
        # Symptom modifiers and descriptors
        symptom_modifiers = [
            "severe", "mild", "moderate", "intermittent", "constant", "chronic", 
            "acute", "sharp", "dull", "throbbing", "stabbing", "burning",
            "intense", "unbearable", "occasional", "frequent", "persistent",
            "worsening", "improving", "stable", "fluctuating", "progressive",
            "sudden", "gradual", "recurring", "episodic", "continuous"
        ]
        
        # Body parts
        body_parts = [
            "head", "neck", "chest", "back", "stomach", "abdomen", "arm", "leg", 
            "foot", "hand", "eye", "ear", "nose", "throat", "shoulder", "elbow", 
            "wrist", "hip", "knee", "ankle", "toe", "finger", "joint", "muscle",
            "skin", "face", "forehead", "temple", "jaw", "mouth", "tongue", "gum",
            "tooth", "lymph node", "groin", "pelvis", "rib", "spine", "tailbone",
            "brain", "heart", "lungs", "liver", "kidney", "bladder", "intestines",
            "stomach", "pancreas", "spleen", "gallbladder", "thyroid", "adrenal"
        ]
        
        # Symptom patterns with [modifier] + [symptom] + [body part]
        advanced_patterns = [
            "[MODIFIER] [SYMPTOM]",
            "[SYMPTOM] in [BODY_PART]",
            "[MODIFIER] [SYMPTOM] in [BODY_PART]",
            "[BODY_PART] [SYMPTOM]"
        ]
        
        return {
            "common_symptoms": common_symptoms,
            "symptom_modifiers": symptom_modifiers,
            "body_parts": body_parts,
            "advanced_patterns": advanced_patterns
        }
    
    def _build_symptom_synonyms(self):
        """Build a dictionary of symptom synonyms using WordNet"""
        synonym_dict = {}
        for synonym in self.symptom_db['common_symptoms']:
            # Split multi word symptoms
            words = synonym.split()
            synonyms = set()

            for word in words:
                # Get synonyms from WordNet
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        # Add synonyms
                        synonym = lemma.name().replace("_", " ")
                        synonyms.add(synonym)

            # Add the original symptom to the synonyms
            synonyms.add(synonym)
            synonym_dict[synonym] = list(synonyms)

        return synonym_dict
    
    def _setup_patterns(self):
        """Setup patterns for matching symptom"""

        # Pattern 1: "I have/feel/am experiencing [symptom]"
        have_pattern = [
            [{"LOWER": {"IN": ["i", "ive", "i've", "im", "i'm"]}}, 
             {"LEMMA": {"IN": ["have", "experience", "feel", "be", "get", "develop", "suffer"]}},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ", "ADV"]}},
             {"POS": {"IN": ["NOUN", "ADJ", "VERB"]}}],
             
            [{"LOWER": {"IN": ["i", "ive", "i've", "im", "i'm"]}}, 
             {"LEMMA": {"IN": ["feel", "experience", "have"]}},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ", "ADV"]}},
             {"POS": {"IN": ["NOUN", "ADJ", "VERB"]}}],
             
            [{"LOWER": {"IN": ["i", "ive", "i've", "im", "i'm"]}}, 
             {"LEMMA": {"IN": ["be", "feel", "have"]}},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ", "ADV"]}},
             {"POS": {"IN": ["NOUN", "ADJ", "VERB"]}}]
        ]
        
        # Pattern 2: "My [body part] is [condition]"
        body_part_pattern = [
            [{"LOWER": {"IN": ["my", "the", "this"]}}, 
             {"POS": "NOUN"},
             {"LEMMA": {"IN": ["be", "feel", "hurt", "ache", "pain", "bother"]}},
             {"OP": "?", "POS": "ADV"},
             {"POS": {"IN": ["ADJ", "VERB", "NOUN"]}}],
             
            [{"LOWER": {"IN": ["my", "the", "this"]}}, 
             {"POS": "NOUN"},
             {"LEMMA": {"IN": ["be", "feel"]}},
             {"OP": "?", "POS": "ADV"},
             {"POS": "ADJ"},
             {"LOWER": "in"},
             {"POS": "NOUN"}]
        ]
        
        # Pattern 3: "I'm [verbing]"
        verb_pattern = [
            [{"LOWER": {"IN": ["i", "im", "i'm", "ive", "i've"]}},
             {"POS": "AUX"},
             {"POS": "VERB", "OP": "+"}],
             
            [{"LOWER": {"IN": ["i", "im", "i'm", "ive", "i've"]}},
             {"POS": "VERB", "OP": "+"}]
        ]
        
        # Pattern 4: Description with time - "been [symptom] for [time]"
        time_pattern = [
            [{"LEMMA": "be"}, 
             {"POS": "VERB", "OP": "+"},
             {"LOWER": "for"},
             {"OP": "+", "POS": {"IN": ["NUM", "ADJ", "NOUN"]}}],
             
            [{"LEMMA": "have"}, 
             {"POS": "VERB", "OP": "+"},
             {"LOWER": "since"},
             {"OP": "+", "POS": {"IN": ["NUM", "ADJ", "NOUN"]}}]
        ]
        
        # Pattern 5: "There is [symptom]"
        existential_pattern = [
            [{"LOWER": {"IN": ["there", "here"]}},
             {"LEMMA": "be"},
             {"OP": "?", "POS": {"IN": ["DET", "ADJ"]}},
             {"POS": {"IN": ["NOUN", "ADJ"]}}]
        ]
        
        # Add patterns to matcher
        self.matcher.add("HAVE_SYMPTOM", have_pattern)
        self.matcher.add("BODY_PART_CONDITION", body_part_pattern)
        self.matcher.add("VERB_SYMPTOM", verb_pattern)
        self.matcher.add("TIME_SYMPTOM", time_pattern)
        self.matcher.add("EXISTENTIAL_SYMPTOM", existential_pattern)

    def extract_symptoms(self, text):
        """Extract symptoms from user text"""
        doc = self.nlp(text)
        symptoms_data = []

        # Phrase matching for direct symptom mentions
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            # Check context around the match
            context_start = max(0, start - 3)
            context_end = min(len(doc), end + 3)
            context = doc[context_start:context_end]
            
            # Skip if the match is negated
            if not self._is_negated(context):
                symptoms_data.append(span.text)

        # Pattern matching for symptom contexts
        pattern_matches = self.matcher(doc)
        for match_id, start, end in pattern_matches:
            span = doc[start:end]
            # Check context around the match
            context_start = max(0, start - 3)
            context_end = min(len(doc), end + 3)
            context = doc[context_start:context_end]
            
            # Skip if the match is negated
            if not self._is_negated(context):
                symptom_text = self._extract_symptom_from_span(span)
                if symptom_text:
                    symptoms_data.append(symptom_text)

        # Verb form detection
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in self.symptom_lemmas:
                # Check context around the verb
                start = max(0, token.i - 3)
                end = min(len(doc), token.i + 3)
                context = doc[start:end]
                
                # Skip if the verb is negated
                if not self._is_negated(context):
                    symptoms_data.append(token.lemma_)
        
        # NER component detection
        for ent in doc.ents:
            if ent.label_ == "SYMPTOM":
                # Check context around the entity
                start = max(0, ent.start - 3)
                end = min(len(doc), ent.end + 3)
                context = doc[start:end]
                
                # Skip if the entity is negated
                if not self._is_negated(context):
                    symptoms_data.append(ent.text)
        
        # Improved fuzzy matching for symptom approximations
        chunks = []
        for chunk in doc.noun_chunks:
            chunks.append(chunk.text)
        for token in doc:
            if token.pos_ in ["NOUN", "ADJ"]:
                chunks.append(token.text)
        
        for chunk in chunks:
            # Check context around the chunk
            start = max(0, doc.text.find(chunk) - 20)
            end = min(len(doc.text), doc.text.find(chunk) + len(chunk) + 20)
            context = doc.text[start:end]
            
            # Skip if the chunk is negated
            if not self._is_negated(self.nlp(context)):
                # Check if this chunk might be a symptom using fuzzy matching
                best_match = process.extractOne(
                    chunk, 
                    self.symptom_db['common_symptoms'], 
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=85  # Relaxed threshold
                )
                if best_match:
                    symptoms_data.append(best_match[0])
        
        # Deduplicate and clean up symptoms
        return self._deduplicate_symptoms(symptoms_data)
    
    def _extract_symptom_from_span(self, span):
        """Extract symptom from a matched span"""
        string_id = self.nlp.vocab.strings[span.label]

        if string_id == "HAVE_SYMPTOM":
            # Make sure the extracted symptom is in our database or a common symptom
            symptom_text = span[-1].text.lower()
            
            # Check if it's directly in our symptom database
            for db_symptom in self.symptom_db['common_symptoms']:
                if fuzz.ratio(symptom_text, db_symptom) > 85:
                    return db_symptom
            
            # If not directly found, return only if it's a strong match
            return span[-1].text if span[-1].lemma_ in self.symptom_lemmas else None
        
        elif string_id == "BODY_PART_CONDITION":
            body_part = span[1].text
            condition = span[-1].text
            
            # Only return if the condition is in our modifiers or symptom lemmas
            if (condition.lower() in self.symptom_db['symptom_modifiers'] or 
                self.nlp(condition)[0].lemma_ in self.symptom_lemmas):
                return f"{condition} in {body_part}"
            return None
        
        elif string_id == "VERB_SYMPTOM":
            for token in span:
                if token.pos_ == "VERB" and token.lemma_ in self.symptom_lemmas:
                    return token.lemma_
            
        elif string_id == "TIME_SYMPTOM":
            for token in span:
                if token.pos_ == "VERB" and token.lemma_ in self.symptom_lemmas:
                    return token.lemma_
        
        return None

    def _is_valid_symptom_context(self, context):
        """Check if the context is valid for a symptom mention"""
        # Check for common symptom-related verbs and patterns
        symptom_verbs = {"have", "feel", "experience", "suffer", "get", "develop", "been", "am", "is", "are", "was", "were"}
        symptom_patterns = {"i", "my", "me", "i'm", "i've", "im", "ive", "been", "having", "feeling", "experiencing"}
        
        # Check if any token in context is a symptom-related verb
        has_symptom_verb = any(token.lemma_ in symptom_verbs for token in context)
        
        # Check if the context starts with a valid pattern
        has_valid_pattern = any(token.text.lower() in symptom_patterns for token in context[:3])
        
        # Check for body parts or symptom-related nouns
        has_body_part = any(token.text.lower() in self.symptom_db['body_parts'] for token in context)
        
        # Check for symptom-related adjectives or nouns
        has_symptom_word = any(token.lemma_ in self.symptom_lemmas for token in context if token.pos_ in ["ADJ", "NOUN"])
        
        # Check for common symptom phrases
        has_symptom_phrase = any(
            token.text.lower() in {"pain", "ache", "sore", "hurt", "swollen", "inflamed", "irritated", "itchy", "burning", "tender", "stiff"}
            for token in context
        )
        
        return (has_symptom_verb or has_valid_pattern) and (has_body_part or has_symptom_word or has_symptom_phrase)

    def _is_valid_symptom(self, symptom):
        """Validate if a potential symptom is actually a symptom"""
        # Check if it's directly in our symptom database
        if symptom.lower() in [s.lower() for s in self.symptom_db['common_symptoms']]:
            return True
            
        # Check if it's a body part (alone it's not a symptom)
        if symptom.lower() in [bp.lower() for bp in self.symptom_db['body_parts']]:
            return False
            
        # Check if it's a meaningful symptom word
        doc = self.nlp(symptom)
        if any(token.lemma_ in self.symptom_lemmas for token in doc):
            return True
            
        # Check for common symptom words
        common_symptom_words = {
            "pain", "ache", "sore", "hurt", "swollen", "inflamed", "irritated", 
            "itchy", "burning", "tender", "stiff", "numb", "weak", "dizzy", 
            "nauseous", "vomit", "cough", "sneeze", "fever", "chill", "headache",
            "migraine", "rash", "tired", "exhausted", "anxious", "irritable"
        }
        
        if any(word in symptom.lower() for word in common_symptom_words):
            return True
            
        return False

    def _deduplicate_symptoms(self, symptoms_data):
        """Deduplicate symptoms using lemmatization and fuzzy matching"""
        if not symptoms_data:
            return []

        # First normalize all symptoms to lowercase
        normalized_symptoms = [s.lower().strip() for s in symptoms_data if s and s.strip()]
        
        # Filter out invalid symptoms
        filtered_symptoms = []
        for symptom in normalized_symptoms:
            if self._is_valid_symptom(symptom):
                filtered_symptoms.append(symptom)
        
        # Now deduplicate what's left with improved grouping
        unique_symptoms = []
        grouped_symptoms = {}
        
        # Group similar symptoms with enhanced similarity detection
        for symptom in filtered_symptoms:
            matched = False
            
            # Check if this symptom belongs to an existing group
            for group_key in grouped_symptoms:
                # Calculate multiple similarity metrics
                token_sort_similarity = fuzz.token_sort_ratio(symptom, group_key)
                token_set_similarity = fuzz.token_set_ratio(symptom, group_key)
                semantic_similarity = self._get_semantic_similarity(symptom, group_key)
                
                # Weighted similarity score
                similarity = (token_sort_similarity * 0.4 + 
                            token_set_similarity * 0.4 + 
                            semantic_similarity * 0.2)
                
                if similarity > 85:  # Relaxed threshold
                    grouped_symptoms[group_key].append(symptom)
                    matched = True
                    break
            
            # If not matched to any group, create a new group
            if not matched:
                grouped_symptoms[symptom] = [symptom]
        
        # Choose the best representative from each group
        for group_key, group_items in grouped_symptoms.items():
            # Find the symptom that best matches our database
            best_symptom = None
            best_score = 0
            
            for symptom in group_items:
                # Check if this symptom is directly in our database
                direct_match = process.extractOne(
                    symptom,
                    self.symptom_db['common_symptoms'],
                    scorer=fuzz.ratio,
                    score_cutoff=85  # Relaxed threshold
                )
                
                if direct_match:
                    best_symptom = direct_match[0]
                    break
                
                # Enhanced matching for non-direct matches
                for db_symptom in self.symptom_db['common_symptoms']:
                    # Calculate multiple similarity metrics
                    token_sort_score = fuzz.token_sort_ratio(symptom, db_symptom)
                    token_set_score = fuzz.token_set_ratio(symptom, db_symptom)
                    semantic_score = self._get_semantic_similarity(symptom, db_symptom)
                    
                    # Weighted score
                    score = (token_sort_score * 0.4 + 
                            token_set_score * 0.4 + 
                            semantic_score * 0.2)
                    
                    if score > best_score:
                        best_score = score
                        best_symptom = db_symptom if score > 85 else symptom
            
            if best_symptom:
                unique_symptoms.append(best_symptom)
        
        return unique_symptoms

    def _get_semantic_similarity(self, word1, word2):
        """Calculate semantic similarity between two words using WordNet"""
        max_similarity = 0.0
        
        # Get synsets for both words
        synsets1 = wordnet.synsets(word1)
        synsets2 = wordnet.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0.0
        
        # Calculate maximum similarity between any pair of synsets
        for syn1 in synsets1:
            for syn2 in synsets2:
                try:
                    similarity = syn1.path_similarity(syn2)
                    if similarity and similarity > max_similarity:
                        max_similarity = similarity
                except:
                    continue
        
        # Convert similarity to percentage
        return max_similarity * 100
    
    def _query_umls_api(self, text):
        """Query the UMLS API for medical concepts"""
        symptoms = []
        
        try:
            """
            auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
            search_endpoint = "https://uts-ws.nlm.nih.gov/rest/search/current"
            
            # Get auth token
            auth_params = {
                "apiKey": self.umls_api_key
            }
            auth_response = requests.post(auth_endpoint, data=auth_params)
            tgt = auth_response.text
            
            # Generate service ticket
            service_ticket_endpoint = f"{tgt}/ticket"
            ticket_params = {
                "service": "http://umlsks.nlm.nih.gov"
            }
            ticket_response = requests.post(service_ticket_endpoint, data=ticket_params)
            st = ticket_response.text
            
            # Search for terms
            search_params = {
                "string": text,
                "searchType": "exact",
                "ticket": st,
                "sabs": "SNOMEDCT_US",  # Use SNOMED CT terminology
                "returnIdType": "concept"
            }
            
            search_response = requests.get(search_endpoint, params=search_params)
            results = search_response.json()
            
            # Extract symptoms from results
            if 'result' in results and 'results' in results['result']:
                for result in results['result']['results']:
                    if 'ui' in result and 'name' in result:
                        # Filter for symptom semantic types
                        if self._is_symptom_concept(result):
                            symptoms.append(result['name'])
            """
            pass
            
        except Exception as e:
            print(f"UMLS API query failed: {e}")
        
        return symptoms
    
    def _is_symptom_concept(self, concept):
        """Check if a UMLS concept is a symptom (placeholder)"""
        return True
    
    def _is_negated(self, context):
        """Check if a symptom mention is negated in its context"""
        negation_indicators = [
            "no", "not", "never", "none", "neither", "nor", "without",
            "don't", "doesn't", "didn't", "won't", "wouldn't", "can't",
            "couldn't", "shouldn't", "haven't", "hasn't", "hadn't"
        ]
        
        for token in context:
            if token.text.lower() in negation_indicators:
                return True
            if token.dep_ == "neg":
                return True
        
        return False

@Language.factory("symptom_extractor")
def create_symptom_component(nlp, name):
    """Custom component to identify symptoms through custom rules"""
    return SymptomExtractorComponent()

class SymptomExtractorComponent:
    def __init__(self):
        self.symptom_indicators = [
            "pain", "ache", "discomfort", "sore", "hurt", "swollen", 
            "inflamed", "irritated", "itchy", "burning", "tender", "stiff",
            "numb", "tight", "weak", "sick", "ill"
        ]
    
    def __call__(self, doc):
        """Process the document and add symptom entities"""
        new_ents = list(doc.ents)
        
        # Check for specific symptom patterns
        for i, token in enumerate(doc):
            # Check for symptoms with body parts
            if token.lemma_.lower() in self.symptom_indicators:
                # Look for body parts after symptom indicators
                if i < len(doc) - 1 and doc[i+1].pos_ == "NOUN":
                    start = i
                    end = i + 2
                    symptom_span = Span(doc, start, end, label="SYMPTOM")
                    new_ents.append(symptom_span)
        
        doc.ents = new_ents
        return doc

def test_symptom_extractor():
    """Test function to demonstrate the symptom extractor's capabilities"""
    # Initialize extractor without UMLS API
    extractor = SymptomExtractor(use_umls_api=False)
    
    # Test cases with various phrasings
    test_texts = [
        "I've been having a headache for the past two days.",
        "My throat is really sore and I have a fever.",
        "I feel dizzy when I stand up and I've been coughing a lot.",
        "I'm experiencing chest pain and shortness of breath.",
        "My back hurts and I have a runny nose.",
        "I think I might have the flu because I have body aches and chills.",
        "I've been coughing and sneezing since yesterday.",
        "My stomach has been upset and I feel nauseated.",
        "I'm having trouble sleeping and feel anxious all the time.",
        "My knee is swollen and it's painful when I walk.",
        "I have a strange rash on my arm that's really itchy.",
        "I've had these migraines for about a week now.",
        "Got a stuffy nose, and I'm feeling really tired.",
        "My eyes are watery and I keep sneezing.",
        "Been throwing up since last night and can't keep anything down.",
        "I have trouble sleeping, feeling exhausted, anxious, and irritable."        
    ]
    
    for text in test_texts:
        print(f"\nInput: {text}")
        symptoms = extractor.extract_symptoms(text)
        print("Symptoms found:", ", ".join(symptoms) if symptoms else "None")
    
    return extractor

if __name__ == "__main__":
    # Run the test function
    extractor = test_symptom_extractor()