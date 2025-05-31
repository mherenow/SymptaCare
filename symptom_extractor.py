import spacy
import requests
import json
from dotenv import load_dotenv
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
from spacy.language import Language
import os
import re
import nltk
from nltk.corpus import wordnet
import logging
from typing import List, Dict, Set, Optional, Tuple
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',  # Simplified format
    handlers=[
        logging.StreamHandler(),  # Add console handler
        logging.FileHandler('symptom_extractor.log')  # Keep file logging
    ]
)
logger = logging.getLogger('SymptomExtractor')

# Set logging level for non-API operations to WARNING to reduce noise
logging.getLogger('SymptomExtractor').setLevel(logging.WARNING)

load_dotenv()

# Download NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading Wordnet...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

print("[PRINT DIAGNOSTIC] symptom_extractor.py loaded from", __file__)

class LlamaSymptomExtractor:
    """Enhanced symptom extractor using Llama 3.3 AI model and UMLS API"""
    
    def __init__(self, use_umls_api=True, nvidia_api_key=None, umls_api_key=None):
        # Load API keys
        self.nvidia_api_key = nvidia_api_key or os.getenv("NVIDIA_API_KEY")
        self.umls_api_key = umls_api_key or os.getenv("UMLS_API_KEY")
        
        if not self.nvidia_api_key:
            raise ValueError("NVIDIA_API_KEY is required for Llama 3.3 integration")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            os.system("python -m spacy download en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")
        
        # UMLS API setup
        self.use_umls_api = use_umls_api and bool(self.umls_api_key)
        if use_umls_api and not self.umls_api_key:
            logger.warning("UMLS API key not found. UMLS integration disabled.")
            self.use_umls_api = False
        
        # Initialize matchers for basic pattern detection
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        
        # Load symptom database
        self.symptom_db = self._load_symptom_database()
        
        # Setup basic patterns for fallback
        self._setup_basic_patterns()
        
        # Llama API configuration
        self.llama_api_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        self.llama_headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_interval = 1.0  # Minimum seconds between API calls
        
        logger.info("LlamaSymptomExtractor initialized successfully")

    def _load_symptom_database(self) -> Dict:
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
            "nausea", "vomiting", "diarrhea", "constipation", "bloating", "indigestion",
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
            "burn", "minor burn", "small burn", "first degree burn", "second degree burn",
            "third degree burn", "chemical burn", "thermal burn", "sunburn",
            
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
            "vaginal discharge", "penile discharge", "irregular periods", "missed period", 
            "heavy menstruation", "erectile dysfunction", "low libido", "genital itching", 
            "burning sensation", "urinary urgency", "urinary frequency", "urinary hesitancy", 
            "urinary retention", "sexual dysfunction", "menstrual cramps", "pelvic pain"
        ]
        
        # Symptom modifiers and descriptors
        symptom_modifiers = [
            "severe", "mild", "moderate", "intermittent", "constant", "chronic", 
            "acute", "sharp", "dull", "throbbing", "stabbing", "burning",
            "intense", "unbearable", "occasional", "frequent", "persistent",
            "worsening", "improving", "stable", "fluctuating", "progressive",
            "sudden", "gradual", "recurring", "episodic", "continuous",
            "minor", "small", "slight", "tiny", "minimal", "superficial",
            "first degree", "second degree", "third degree", "chemical", "thermal"
        ]
        
        # Create modified symptoms by combining modifiers with base symptoms
        modified_symptoms = []
        for modifier in symptom_modifiers:
            for symptom in common_symptoms:
                modified_symptoms.append(f"{modifier} {symptom}")
        
        # Add modified symptoms to the common symptoms list
        common_symptoms.extend(modified_symptoms)
        
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
        
        return {
            "common_symptoms": common_symptoms,
            "symptom_modifiers": symptom_modifiers,
            "body_parts": body_parts
        }

    def _setup_basic_patterns(self):
        """Setup basic spaCy patterns for fallback extraction"""
        # Pattern for "I have/feel/experience [symptom]"
        have_patterns = [
            [{"LOWER": {"IN": ["i", "i've", "i'm", "ive", "im"]}}, 
             {"LEMMA": {"IN": ["have", "experience", "feel", "be", "get", "develop", "suffer"]}},
             {"OP": "*", "POS": {"IN": ["DET", "ADJ", "ADV"]}},
             {"POS": {"IN": ["NOUN", "ADJ"]}, "OP": "+"}]
        ]
        # Pattern for "My [body part] [condition]"
        body_part_patterns = [
            [{"LOWER": {"IN": ["my", "the"]}}, 
             {"POS": "NOUN"},
             {"LEMMA": {"IN": ["be", "feel", "hurt", "ache", "pain"]}},
             {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "+"}]
        ]
        self.matcher.add("SYMPTOM_HAVE", have_patterns)
        self.matcher.add("SYMPTOM_BODY_PART", body_part_patterns)
        # Add phrase matcher for multi-word symptoms
        multi_word_symptoms = [symptom for symptom in self.symptom_db['common_symptoms'] if ' ' in symptom]
        patterns = [self.nlp(symptom) for symptom in multi_word_symptoms]
        self.phrase_matcher.add("MULTIWORD_SYMPTOM", patterns)

    def _rate_limit_api_call(self):
        """Implement rate limiting for API calls"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_api_interval:
            sleep_time = self.min_api_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()

    def _query_llama_for_symptoms(self, text: str) -> List[str]:
        """Query Llama 3.3 model to extract symptoms from text"""
        
        print("[PRINT DIAGNOSTIC] _query_llama_for_symptoms called")
        self._rate_limit_api_call()
        
        # Create a comprehensive prompt for symptom extraction
        prompt = f"""You are a medical AI assistant specialized in extracting symptoms from patient descriptions. 

Your task is to identify and extract ALL symptoms mentioned in the following text. 

Rules:
1. Extract symptoms as they appear in medical terminology
2. ALWAYS include severity modifiers when present (e.g., "mild fever", "severe pain")
3. Include body location if mentioned (e.g., "chest pain" not just "pain")
4. Normalize similar terms (e.g., "throwing up" → "vomiting")
5. Only extract actual symptoms, not causes or diagnoses
6. Ignore negated symptoms (e.g., "no fever" should not extract "fever")
7. Return symptoms as a JSON list of strings
8. IMPORTANT: Preserve all modifiers and descriptors exactly as they appear in the text

Symptom Normalization Rules:
1. For minor/small/tiny/slight symptoms:
   - Convert "small burn" to "minor burn"
   - Convert "tiny cut" to "minor cut"
   - Convert "slight pain" to "minor pain"
   - IMPORTANT: For minor symptoms, DO NOT include location information
   - Example: "small burn on wrist" → "minor burn"
   - Example: "tiny cut on finger" → "minor cut"
   - Example: "slight pain in back" → "minor pain"

2. For severity levels:
   - mild → mild
   - moderate → moderate
   - severe/intense/extreme → severe
   - acute → acute
   - chronic → chronic

3. For location information:
   - Only include location for non-minor symptoms
   - Keep prepositions (on, in, at, of) for non-minor symptoms
   - Keep body parts after prepositions for non-minor symptoms
   - Example: "severe burn on wrist" → "severe burn on wrist"
   - Example: "acute pain in chest" → "acute pain in chest"

4. For degree-based severity:
   - Keep "first degree", "second degree", "third degree" as is
   - Example: "first degree burn" → "first degree burn"

Common symptom categories to look for:
- Pain (headache, chest pain, back pain, etc.)
- Respiratory (cough, shortness of breath, wheezing, etc.)
- Digestive (nausea, vomiting, diarrhea, constipation, etc.)
- General (fever, fatigue, dizziness, weakness, etc.)
- Skin (rash, itching, swelling, burns, cuts, etc.)
- Neurological (confusion, memory loss, numbness, etc.)
- Psychological (anxiety, depression, insomnia, etc.)

Common severity modifiers:
- mild, moderate, severe
- acute, chronic
- intermittent, constant
- sharp, dull, throbbing
- intense, unbearable
- minor, small, tiny, slight

Patient text: "{text}"

Extract symptoms and return as JSON array of strings. Remember to normalize symptoms according to the rules above:"""

        try:
            payload = {
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant that extracts symptoms from patient descriptions. Always respond with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 1000,
                "top_p": 0.9
            }
            logger.info(f"[DIAGNOSTIC] Querying Llama for: {text[:50]}...")
            response = requests.post(
                self.llama_api_url, 
                headers=self.llama_headers, 
                json=payload,
                timeout=120
            )
            logger.info(f"[DIAGNOSTIC] Llama API response status: {response.status_code}")
            logger.info(f"[DIAGNOSTIC] Llama API response content: {response.text}")
            print(f"[PRINT DIAGNOSTIC] Llama API response status: {response.status_code}")
            print(f"[PRINT DIAGNOSTIC] Llama API response content: {response.text}")
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                print(f"[PRINT DIAGNOSTIC] Llama API parsed content: {content}")
                logger.info(f"[DIAGNOSTIC] Llama API parsed content: {content}")
                try:
                    json_start = content.find('[')
                    json_end = content.rfind(']') + 1
                    if json_start != -1 and json_end != 0:
                        json_content = content[json_start:json_end]
                        symptoms = json.loads(json_content)
                        logger.info(f"[DIAGNOSTIC] Parsed symptoms: {symptoms}")
                        print(f"[PRINT DIAGNOSTIC] Parsed symptoms: {symptoms}")
                        return symptoms
                    else:
                        logger.warning("[DIAGNOSTIC] No symptoms found in response content")
                        print("[PRINT DIAGNOSTIC] No symptoms found in response content")
                        return []
                except json.JSONDecodeError as e:
                    logger.error(f"[DIAGNOSTIC] JSON parse error: {e}")
                    print(f"[PRINT DIAGNOSTIC] JSON parse error: {e}")
                    return []
            else:
                logger.error(f"[DIAGNOSTIC] API error: {response.status_code}")
                print(f"[PRINT DIAGNOSTIC] API error: {response.status_code}")
                return []
        except requests.exceptions.Timeout:
            logger.error("[DIAGNOSTIC] API timeout")
            print("[PRINT DIAGNOSTIC] API timeout")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"[DIAGNOSTIC] Request failed: {e}")
            print(f"[PRINT DIAGNOSTIC] Request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"[DIAGNOSTIC] Unexpected error: {e}")
            print(f"[PRINT DIAGNOSTIC] Unexpected error: {e}")
            return []

    def _query_umls_api(self, symptoms: List[str]) -> List[str]:
        """Query UMLS API to validate and expand symptom list"""
        if not self.use_umls_api or not symptoms:
            return symptoms
        
        validated_symptoms = []
        
        try:
            logger.info(f"Validating {len(symptoms)} symptoms")
            
            # Get authentication token with retry logic
            auth_endpoint = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
            auth_params = {"apikey": self.umls_api_key}
            
            # Add retry logic for authentication
            max_retries = 3
            auth_response = None
            for attempt in range(max_retries):
                try:
                    auth_response = requests.post(auth_endpoint, data=auth_params, timeout=30)
                    if auth_response.status_code == 201:
                        # Extract TGT URL from HTML response
                        html_content = auth_response.text
                        tgt_url_match = re.search(r'action="([^"]+)"', html_content)
                        if not tgt_url_match:
                            logger.error("Could not extract TGT URL from response")
                            continue
                            
                        tgt_url = tgt_url_match.group(1)
                        
                        # Generate service ticket
                        service = "http://umlsks.nlm.nih.gov"
                        ticket_params = {"service": service}
                        
                        ticket_response = requests.post(tgt_url, data=ticket_params, timeout=15)
                        if ticket_response.status_code != 200:
                            logger.error(f"Service ticket generation failed: {ticket_response.status_code}")
                            continue
                            
                        st = ticket_response.text
                        if not st:
                            logger.error("Empty service ticket received")
                            continue
                        
                        # Search for each symptom
                        search_endpoint = "https://uts-ws.nlm.nih.gov/rest/search/current"
                        
                        for symptom in symptoms:
                            try:
                                search_params = {
                                    "string": symptom,
                                    "searchType": "approximate",
                                    "ticket": st,
                                    "sabs": "SNOMEDCT_US,MSH",  # SNOMED CT and MeSH
                                    "returnIdType": "concept",
                                    "pageSize": "5"
                                }
                                
                                search_response = requests.get(search_endpoint, params=search_params, timeout=15)
                                
                                if search_response.status_code == 200:
                                    results = search_response.json()
                                    
                                    if 'result' in results and 'results' in results['result']:
                                        # Find the best match
                                        best_match = None
                                        best_score = 0
                                        
                                        for result in results['result']['results']:
                                            if self._is_symptom_concept(result):
                                                # Simple scoring based on name similarity
                                                score = self._calculate_similarity(symptom.lower(), result['name'].lower())
                                                if score > best_score:
                                                    best_score = score
                                                    best_match = result['name']
                                        
                                        if best_match and best_score > 0.7:
                                            validated_symptoms.append(best_match)
                                            logger.debug(f"UMLS validated: {symptom} -> {best_match}")
                                        else:
                                            validated_symptoms.append(symptom)  # Keep original if no good match
                                    else:
                                        validated_symptoms.append(symptom)  # Keep original if no results
                                else:
                                    validated_symptoms.append(symptom)  # Keep original if search fails
                                    
                            except Exception as e:
                                logger.warning(f"UMLS validation failed for '{symptom}': {e}")
                                validated_symptoms.append(symptom)  # Keep original on error
                                
                            # Brief pause between UMLS queries
                            time.sleep(0.2)  # Increased delay to avoid rate limiting
                        
                        # If we got here, we successfully processed all symptoms
                        break
                        
                    elif auth_response.status_code == 400:
                        logger.error("Invalid API key")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            return symptoms
                    else:
                        logger.error(f"Auth failed: {auth_response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            return symptoms
                except requests.exceptions.RequestException as e:
                    logger.error(f"Auth request failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return symptoms
                    
        except Exception as e:
            logger.error(f"UMLS API validation failed: {e}")
            return symptoms
        
        logger.info(f"UMLS validation completed: {len(validated_symptoms)} symptoms")
        return validated_symptoms

    def _is_symptom_concept(self, concept: Dict) -> bool:
        """Check if a UMLS concept represents a symptom"""
        # List of semantic types that typically represent symptoms
        symptom_semantic_types = {
            "T184",  # Sign or Symptom
            "T033",  # Finding
            "T046",  # Pathologic Function
            # Add more semantic types as needed
        }
        
        # Check semantic types if available
        if 'semanticTypes' in concept:
            for sem_type in concept['semanticTypes']:
                if sem_type.get('TUI') in symptom_semantic_types:
                    return True
        
        # If no semantic types, use keyword-based heuristics
        name = concept.get('name', '').lower()
        symptom_keywords = [
            'pain', 'ache', 'symptom', 'sign', 'complaint', 'discomfort',
            'disorder', 'syndrome', 'condition', 'manifestation'
        ]
        
        return any(keyword in name for keyword in symptom_keywords)

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple Jaccard similarity
        set1 = set(str1.split())
        set2 = set(str2.split())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0

    def _basic_pattern_extraction(self, text: str) -> List[str]:
        """Fallback extraction using spaCy patterns"""
        doc = self.nlp(text)
        symptoms = []
        # Use phrase matcher for multi-word symptoms
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            symptom = span.text.lower()
            if symptom in self.symptom_db['common_symptoms']:
                symptoms.append(symptom)
        # Use pattern matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            for i, token in enumerate(span):
                if token.pos_ == "ADJ" and i + 1 < len(span):
                    next_token = span[i + 1]
                    if next_token.pos_ in ["NOUN", "ADJ"]:
                        potential_symptom = f"{token.text.lower()} {next_token.text.lower()}"
                        if any(symptom == potential_symptom for symptom in self.symptom_db['common_symptoms']):
                            symptoms.append(potential_symptom)
                            continue
                        if any(symptom == next_token.text.lower() for symptom in self.symptom_db['common_symptoms']):
                            symptoms.append(potential_symptom)
                            continue
                if token.pos_ in ["NOUN", "ADJ"] and not self._is_negated(doc[max(0, start-3):min(len(doc), end+3)]):
                    if any(symptom == token.text.lower() for symptom in self.symptom_db['common_symptoms']):
                        symptoms.append(token.text.lower())
        return symptoms

    def _is_negated(self, context) -> bool:
        """Check if symptoms in context are negated"""
        negation_words = {"no", "not", "never", "without", "don't", "doesn't", "didn't"}
        
        for token in context:
            if token.text.lower() in negation_words or token.dep_ == "neg":
                return True
        return False

    def _clean_and_deduplicate(self, symptoms: List[str]) -> List[str]:
        """Clean and deduplicate the symptom list"""
        if not symptoms:
            return []
        
        # Clean symptoms
        cleaned = []
        for symptom in symptoms:
            if isinstance(symptom, str):
                # Remove extra whitespace and convert to lowercase
                clean_symptom = ' '.join(symptom.strip().lower().split())
                if clean_symptom and len(clean_symptom) > 2:  # Avoid single chars/empty
                    # Normalize symptom descriptions
                    words = clean_symptom.split()
                    
                    # Handle severity modifiers
                    severity_modifiers = {
                        'small': 'minor',
                        'tiny': 'minor',
                        'slight': 'minor',
                        'minimal': 'minor',
                        'superficial': 'minor',
                        'mild': 'mild',
                        'moderate': 'moderate',
                        'severe': 'severe',
                        'intense': 'severe',
                        'extreme': 'severe',
                        'acute': 'acute',
                        'chronic': 'chronic'
                    }
                    
                    # Handle location modifiers
                    location_modifiers = {
                        'on': 'on',
                        'in': 'in',
                        'at': 'at',
                        'of': 'of',
                        'the': 'the'
                    }
                    
                    # Normalize the symptom
                    normalized_words = []
                    i = 0
                    while i < len(words):
                        word = words[i]
                        
                        # Check for severity modifier
                        if word in severity_modifiers:
                            normalized_words.append(severity_modifiers[word])
                        # Check for location modifier
                        elif word in location_modifiers:
                            # Keep location modifiers and the following word (body part)
                            normalized_words.append(word)
                            if i + 1 < len(words):
                                normalized_words.append(words[i + 1])
                                i += 1
                        # Check for degree-based severity
                        elif word in ['first', 'second', 'third'] and i + 1 < len(words) and words[i + 1] == 'degree':
                            normalized_words.extend([word, 'degree'])
                            i += 1
                        # Keep other words as is
                        else:
                            normalized_words.append(word)
                        i += 1
                    
                    # Reconstruct the normalized symptom
                    clean_symptom = ' '.join(normalized_words)
                    
                    # Check if this is a modified symptom
                    if len(normalized_words) > 1:
                        # If it's a modified symptom, keep it as is
                        cleaned.append(clean_symptom)
                    else:
                        # For single words, check if they're in our symptom database
                        if any(s == clean_symptom for s in self.symptom_db['common_symptoms']):
                            cleaned.append(clean_symptom)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symptoms = []
        
        for symptom in cleaned:
            if symptom not in seen:
                seen.add(symptom)
                unique_symptoms.append(symptom)
        
        # Filter out body parts that aren't symptoms
        filtered_symptoms = []
        for symptom in unique_symptoms:
            # Skip if it's just a body part without a symptom descriptor
            if not any(bp == symptom for bp in self.symptom_db['body_parts']):
                filtered_symptoms.append(symptom)
        
        return filtered_symptoms

    def extract_symptoms(self, text: str) -> List[str]:
        """Main method to extract symptoms from text using Llama 3.3 and UMLS"""
        print("[PRINT DIAGNOSTIC] extract_symptoms called")
        logger.info("[DIAGNOSTIC] extract_symptoms called with text: %s", text)
        if not text or not text.strip():
            return []
        
        logger.info(f"Processing: {text[:50]}...")  # Truncate long text
        
        # First try Llama extraction
        try:
            llama_symptoms = self._query_llama_for_symptoms(text)
            logger.info(f"[DIAGNOSTIC] Llama symptoms: {llama_symptoms}")
            print(f"[PRINT DIAGNOSTIC] Llama symptoms: {llama_symptoms}")
            
            if llama_symptoms:
                logger.info("[DIAGNOSTIC] Using Llama symptoms")
                print("[PRINT DIAGNOSTIC] Using Llama symptoms")
                symptoms = llama_symptoms
            else:
                logger.warning("[DIAGNOSTIC] Llama returned no symptoms, using fallback extraction")
                print("[PRINT DIAGNOSTIC] Llama returned no symptoms, using fallback extraction")
                symptoms = self._basic_pattern_extraction(text)
                
        except Exception as e:
            logger.error(f"[DIAGNOSTIC] Llama extraction failed: {e}")
            print(f"[PRINT DIAGNOSTIC] Llama extraction failed: {e}")
            symptoms = self._basic_pattern_extraction(text)
        
        # Clean and deduplicate symptoms
        cleaned_symptoms = self._clean_and_deduplicate(symptoms)
        
        # Validate and enhance with UMLS if enabled
        if self.use_umls_api and cleaned_symptoms:
            try:
                validated_symptoms = self._query_umls_api(cleaned_symptoms)
                final_symptoms = self._clean_and_deduplicate(validated_symptoms)
            except Exception as e:
                logger.error(f"UMLS validation failed: {e}")
                final_symptoms = cleaned_symptoms
        else:
            final_symptoms = cleaned_symptoms
        
        logger.info(f"Found {len(final_symptoms)} symptoms: {', '.join(final_symptoms)}")
        return final_symptoms


def test_llama_symptom_extractor():
    """Test function for the Llama-based symptom extractor"""
    
    # Initialize extractor
    try:
        extractor = LlamaSymptomExtractor(use_umls_api=True)
    except ValueError as e:
        print(f"Error: {e}")
        print("Set NVIDIA_API_KEY in .env file")
        return
    
    # Test cases
    test_texts = [
        "I have a mild fever",
        "I've been having a severe headache for the past two days.",
        "My throat is really sore and I have a high fever of 101°F.",
        "I feel dizzy when I stand up and I've been coughing a lot with phlegm.",
        "I'm experiencing sharp chest pain and shortness of breath.",
        "My lower back hurts and I have a runny nose with congestion.",
        "I think I might have the flu because I have body aches and chills.",
        "I've been coughing and sneezing since yesterday, also feeling fatigued.",
        "My stomach has been upset and I feel nauseated, threw up this morning.",
        "I'm having trouble sleeping and feel anxious all the time, very irritable.",
        "My right knee is swollen and it's painful when I walk or bend it.",
        "I have a strange red rash on my arm that's really itchy and spreading.",
        "I've had these intense migraines for about a week now with light sensitivity.",
        "Got a stuffy nose, watery eyes, and I'm feeling really tired and weak.",
        "My eyes are watery, keep sneezing, and have postnasal drip.",
        "Been throwing up since last night and can't keep anything down, also diarrhea.",
        "I have difficulty sleeping, feeling exhausted, anxious, and very irritable lately.",
        "I feel a sharp pain in my chest."
    ]
    
    print("\nTesting Symptom Extractor")
    print("=" * 30)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text[:50]}...")
        print("-" * 30)
        
        try:
            symptoms = extractor.extract_symptoms(text)
            if symptoms:
                print(f"Symptoms: {', '.join(symptoms)}")
            else:
                print("No symptoms found")
        except Exception as e:
            print(f"Error: {e}")
    
    return extractor


if __name__ == "__main__":
    # Run the test
    test_llama_symptom_extractor()
