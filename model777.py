# required packages and libraries

!pip install Bio-Epidemiology-NER
!pip install pandas==1.5.3
import nltk
nltk.download('punkt_tab')
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import pandas as pd
import re
import time
from typing import List, Dict
import requests

# Main functions defined for working of the model
# Preprocessing Text 

def basic_cleaning(text):
    # Convert to lowercase, remove non-alphanumeric characters, and extra spaces
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text, stopwords):
    Split text into words and manually check for stopwords to simplify the loop
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def load_abbreviation_dict(csv_file):
    df = pd.read_csv(csv_file)
    abbrev_dict = pd.Series(df['full_form'].values, index=df['abbrevation']).to_dict()
    return abbrev_dict

def expand_abbreviations(text, abbrev_dict):
    # Split the text into words and expand abbreviations using the abbreviation dictionary
    words = text.split()
    expanded_text = [abbrev_dict.get(word, word) for word in words]
    return ' '.join(expanded_text)

def custom_preprocessing(text, abbrev_dict, stopwords, medical_terms):
     # Apply basic cleaning, expand abbreviations, and remove stopwords
    text = basic_cleaning(text)
    text = expand_abbreviations(text, abbrev_dict)
    text = remove_stopwords(text, stopwords)
    return text

# integrating UMLS API using specific API key

import requests

API_KEY = 'YOUR API KEY' 

# Function to get Ticket Granting Ticket (TGT)
def get_umls_auth_token(api_key):
    auth_endpoint = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = f'apikey={api_key}'

    response = requests.post(auth_endpoint, headers=headers, data=data)
    if response.status_code == 201:
        # Extract TGT from the Location header
        tgt = response.headers['location']
        return tgt
    else:
        raise Exception('Error retrieving UMLS authentication token: ' + response.text)

# Function to get a Service Ticket (ST) using TGT
def get_service_ticket(tgt):
    service = 'http://umlsks.nlm.nih.gov'
    response = requests.post(tgt, data={'service': service})
    if response.status_code == 200:
        return response.text
    else:
        raise Exception('Error retrieving UMLS service ticket: ' + response.text)
print(get_umls_auth_token(API_KEY))
get_service_ticket(get_umls_auth_token(API_KEY))  

# Function to search for the CUI of a medical term
def search_umls_cui(term, ticket):
    search_endpoint = 'https://uts-ws.nlm.nih.gov/rest/search/current'
    params = {
        'string': term,
        'ticket': ticket,
        'searchType': 'exact'
    }

    response = requests.get(search_endpoint, params=params)
    if response.status_code == 200:
        results = response.json()
        if results['result']['results']:
            # Extract the first CUI found
            return results['result']['results'][0]['ui']
        else:
            print("No CUI found for the term.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_umls_definitions(cui, ticket):
    definition_endpoint = f'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/definitions'
    params = {'ticket': ticket}

    response = requests.get(definition_endpoint, params=params)
    if response.status_code == 200:
        definitions = response.json()
        if definitions['result']:
            return [item['value'] for item in definitions['result']]
        else:
            print("No definitions found for the CUI.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def get_medical_term_definitions(term):
    try:
        # Get the authentication token (TGT)
        tgt = get_umls_auth_token(API_KEY)

        # Get a service ticket (ST)
        ticket = get_service_ticket(tgt)

        # Search for the CUI of the term
        cui = search_umls_cui(term, ticket)

        if cui:
            print(f"Found CUI: {cui}")

            # Get definitions using the CUI
            ticket = get_service_ticket(tgt)  # Refresh the ticket for each API call
            definitions = get_umls_definitions(cui, ticket)

            if definitions:
                print(f"Definitions for '{term}':")
                for idx, definition in enumerate(definitions, 1):
                    print(f"{idx}. {definition}")
            else:
                print(f"No definitions available for term '{term}'.")
        else:
            print(f"Could not find CUI for term '{term}'.")

    except Exception as e:
        print(f"Error: {str(e)}")


# main code working of the model

from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import pandas as pd
import re
import time
from typing import List, Dict

# Predefined Variables
API_KEY = "YOUR API KEY"  
abbrev_dict = {}  
ENGLISH_STOP_WORDS = set()
t = None  // Placeholder for any tokenizer/model you use in preprocessing

# Preprocessing Function
def custom_preprocessing(text: str, abbrev_dict: Dict, stop_words: set, model=None) -> str:
    text = text.lower()
    # Replace abbreviations
    for abbrev, full in abbrev_dict.items():
        text = text.replace(abbrev, full)
    # Remove stop words
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Function to Query UMLS for Definitions
def get_umls_definition(term: str, api_key: str, preferred_language='english') -> List[str]:
    try:
       # Get the authentication token (TGT)
        tgt = get_umls_auth_token(api_key)
        if not tgt:
            raise Exception("Failed to get authentication token.")

       # Get a service ticket (ST)
        ticket = get_service_ticket(tgt)
        if not ticket:
            raise Exception("Failed to get service ticket.")

        # Search for the CUI of the term
        cui = search_umls_cui(term, ticket)
        if not cui:
            raise Exception(f"Could not find CUI for term '{term}'.")

         # Get definitions using the CUI
        # Refresh the ticket for each API call
        ticket = get_service_ticket(tgt)  # Refresh ticket
        definitions = get_umls_definitions(cui, ticket)
        if not definitions:
            raise Exception(f"No definitions found for term '{term}'.")

        # Filter definitions based on preferred language
        if definitions:
            english_pattern = re.compile(r'^[a-zA-Z0-9\s.,\-\'()]+$')
            filtered_definitions = [d for d in definitions if english_pattern.match(d)] if preferred_language == 'english' else definitions

            # Adding a small delay between API calls to avoid rate limiting or other issues
            time.sleep(0.05)  # You can adjust this depending on the UMLS API rate limit

            return filtered_definitions
        return []

    except Exception as e:
        print(f"Error retrieving definition for term '{term}': {e}")
        return []

# Function to Split Combined Entities
def split_combined_entities(entities: List[str]) -> List[str]:
    split_entities = []   
    for entity in entities:
        # Split based on common delimiters and avoid splitting valid terms with internal spaces (e.g., "high blood pressure")
        terms = re.split(r',|\band\b|\s', entity)  # split by comma, 'and', or spaces
        terms = [term.strip() for term in terms if term.strip()]  # Clean up any empty strings
        
        # Append individual terms to the list
        split_entities.extend(terms)
    
    return split_entities

# Main Workflow
def process_medical_text(text: str, api_key: str) -> None:
    # Preprocess the text
    doc = custom_preprocessing(text, abbrev_dict, ENGLISH_STOP_WORDS, t)

    # Extract entities using the Bio_Epidemiology_NER model
    entities_df = ner_prediction(corpus=doc, compute='cpu')

    # Filter entities based on required tags (e.g., Disease_disorder, Medication, Sign_symptom)
    required_tags = ['Disease_disorder', 'Medication', 'Sign_symptom']
    filtered_df = entities_df[entities_df['entity_group'].isin(required_tags)].reset_index(drop=True)

    # Extract unique terms and split them if they are combined
    unique_terms = split_combined_entities(filtered_df['value'].unique())

    print("Detected Terms with Definitions:")
    # Query UMLS for definitions of each term
    for term in unique_terms:
        definitions = get_umls_definition(term, api_key)

        if definitions:
            print(f"\nTerm: {term}")
            for idx, definition in enumerate(definitions, 1):
                print(f"  {idx}. {definition}")
        else:
            print(f"\nTerm: {term}")
            print("  No definition found in UMLS.")

# wroking
text = """
sample report
"""
process_medical_text(text, API_KEY)

  
