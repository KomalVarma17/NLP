# -*- coding: utf-8 -*-
"""NLP project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HDfQtZ-3EP5n1m3ZfXe_C6k-0EdhfBP0
"""

import pandas as pd
df1 = pd.read_csv('/content/drive/MyDrive/NLP/cleaned_medical_terms.csv')
df2 = pd.read_csv('/content/drive/MyDrive/NLP/cleaned_healthcare_reports.csv')
print(df1.columns)
print(df2.columns)

!pip install Bio-Epidemiology-NER

from Bio_Epidemiology_NER.bio_recognizer import ner_prediction

!pip install pandas==1.5.3

import pandas as pd
print(pd.__version__)

from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import pandas as pd

# Define the document for testing
doc = """
The patient was diagnosed with hypertension and given amlodipine.
"""

# Step 1: Get NER predictions using the Bio_Epidemiology_NER model
df = ner_prediction(corpus=doc, compute='cpu')

# Step 2: Display the extracted entities (check if it's not empty)
print("Extracted Entities:")
print(df)

# Step 3: Save entities to a CSV file manually
csv_file = '/content/entities_output.csv'
df.to_csv(csv_file, index=False)

print(f"Entities saved to {csv_file}")

f = pd.read_csv('/content/entities_output.csv')
print(f.head(20))

unique_values = f['entity_group'].unique()

# Print the unique values
print(unique_values)

import pandas as pd

pd.read_csv('/content/drive/MyDrive/NLP/medical_terms.csv')

d = pd.read_csv('/content/drive/MyDrive/NLP/healthcare_reports.csv')

d.columns
d.head(20)

d.drop(['Unnamed: 0', 'medical_specialty', 'keywords', 'processed_transcription', 'tokenized_transcription', 'embedding' ], axis=1, inplace=True)

d.to_csv('/content/drive/MyDrive/NLP/Hhealthcare_reports.csv', index=False)

t = pd.read_csv('/content/drive/MyDrive/NLP/medical_terms.csv')
t.columns

a = pd.read_csv('/content/drive/MyDrive/NLP/medical_abbrevations.csv')
a.head(10)

import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def basic_cleaning(text):
    # Convert to lowercase, remove non-alphanumeric characters, and extra spaces
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text, stopwords):
    # Split text into words and manually check for stopwords to simplify the loop
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def load_abbreviation_dict(csv_file):
    # Read the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    # Convert the DataFrame into a dictionary (abbreviation -> full form)
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

abbrev_dict = load_abbreviation_dict('/content/drive/MyDrive/NLP/medical_abbrevations.csv')
text = "The patient was diagnosed with hypertension and given amlodipine."

cleaned_text = custom_preprocessing(text, abbrev_dict, ENGLISH_STOP_WORDS, t)

print(cleaned_text)

from Bio_Epidemiology_NER.bio_recognizer import ner_prediction
import pandas as pd

# Define the document for testing
doc = """
Medical Report:
Patient Name: John Doe
Date of Birth: 02/15/1975
Report Date: 11/10/2024
Chief Complaint:
The patient presents with a complaint of persistent chest pain, particularly after physical exertion. The pain radiates to the left arm and is associated with shortness of breath.
History of Present Illness:
John Doe is a 49-year-old male with a known history of hypertension and type 2 diabetes mellitus. He reports experiencing chest pain for the past two weeks, which has progressively worsened. The pain is described as a pressure-like sensation, occurring mainly during activities such as walking up stairs. The patient denies any nausea, vomiting, or diaphoresis. No history of similar episodes in the past.
Past Medical History:
Hypertension (diagnosed 5 years ago)
Type 2 Diabetes Mellitus (diagnosed 8 years ago)
Hyperlipidemia
No history of coronary artery disease or previous myocardial infarction
Medications:
Metformin 500 mg twice daily
Lisinopril 10 mg daily
Atorvastatin 20 mg at bedtime
Aspirin 81 mg daily
Allergies:
No known drug allergies.
Family History:
Father: Passed away at 60 due to a heart attack.
Mother: History of hypertension and stroke.
Social History:
Non-smoker
Drinks alcohol occasionally (1-2 drinks per week)
Sedentary lifestyle with minimal exercise
Physical Examination:
Vital Signs: Blood Pressure: 145/90 mmHg, Heart Rate: 82 bpm, Respiratory Rate: 18 breaths per minute, Temperature: 98.6°F, Oxygen Saturation: 98% on room air.
Cardiovascular: S1 and S2 heard, no murmurs, gallops, or rubs. No jugular venous distension.
Respiratory: Lungs clear to auscultation bilaterally, no wheezes or crackles.
Abdomen: Soft, non-tender, non-distended.
Extremities: No edema, pulses 2+ bilaterally.
Assessment and Plan:
Suspected Angina: Given the patient's history of hypertension, diabetes, and hyperlipidemia, his symptoms are concerning for angina. Recommend starting a beta-blocker and scheduling an exercise stress test for further evaluation.
Hypertension: Blood pressure remains elevated; increase Lisinopril to 20 mg daily.
Diabetes Management: Continue current regimen with Metformin; consider adding a GLP-1 receptor agonist for better glycemic control.
Lifestyle Modifications: Encourage a heart-healthy diet, regular exercise, and weight management.
Follow-up: Schedule a follow-up appointment in 2 weeks.
Laboratory Results (from prior visit):
HbA1c: 7.8%
LDL Cholesterol: 145 mg/dL
HDL Cholesterol: 40 mg/dL
Triglycerides: 180 mg/dL
Blood Urea Nitrogen (BUN): 14 mg/dL
Creatinine: 0.9 mg/dL
Impression:
The patient's symptoms and risk factors are suggestive of stable angina. Further evaluation with a stress test is necessary to rule out ischemic heart disease. Blood pressure and diabetes control need to be optimized.
"""

# Step 1: Get NER predictions using the Bio_Epidemiology_NER model
df = ner_prediction(corpus=doc, compute='cpu')

# Step 2: Display the extracted entities (check if it's not empty)
print("Extracted Entities:")
print(df)

# Step 3: Save entities to a CSV file manually
csv_file = '/content/entities_output.csv'
df.to_csv(csv_file, index=False)

print(f"Entities saved to {csv_file}")

entities_df = pd.read_csv('/content/entities_output.csv')

# Check the structure of the extracted entities
print(entities_df.head())

pd.read_csv('/content/drive/MyDrive/NLP/entities_df.csv')
entities_df.head(20)

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

# Load the Sentence-BERT model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load your medical terms CSV file
medical_terms_df = pd.read_csv('/content/drive/MyDrive/NLP/medical_terms.csv')

# Generate embeddings for medical terms only (ignore explanations)
medical_terms_df['embedding'] = medical_terms_df['Medical Term'].apply(
    lambda term: model.encode(term)
)

# Save embeddings to a `pkl` file
with open('/content/drive/MyDrive/NLP/medical_terms_embeddings.pkl', 'wb') as f:
    pickle.dump(medical_terms_df[['Medical Term', 'embedding']], f)

print("Embeddings saved to pkl file successfully.")

with open('/content/drive/MyDrive/NLP/medical_terms_embeddings.pkl', 'rb') as f:
        terms_embeddings = pickle.load(f)

print(terms_embeddings)

# Function to find the most similar medical term for a given entity using precomputed embeddings
def find_most_similar_term(entity, terms_embeddings, model):
    # Generate embedding for the entity
    entity_embedding = model.encode(entity)

    # Calculate cosine similarity between entity embedding and precomputed embeddings
    similarities = cosine_similarity(
        [entity_embedding],
        list(terms_embeddings['embedding'])
    )[0]

    # Find the index of the most similar medical term
    best_match_index = similarities.argmax()

    # Get the most similar medical term, explanation, and similarity score
    most_similar_term = terms_embeddings.iloc[best_match_index]
    return most_similar_term['Medical Term'], similarities[best_match_index]

entities_df['matched_term'] = entities_df['value'].apply(
    lambda x: find_most_similar_term(x, terms_embeddings, model)
)

# Save the output to a CSV
entities_df.to_csv('/content/drive/MyDrive/NLP/entities_df.csv', index=False)

pd.read_csv('/content/drive/MyDrive/NLP/entities_df.csv')

unique_values = entities_df['entity_group'].unique()

# Print the unique values
print(unique_values)

entities_df.head(50)

med = entities_df.loc[entities_df['entity_group'] == 'Biological_structure']

med.head(20)

