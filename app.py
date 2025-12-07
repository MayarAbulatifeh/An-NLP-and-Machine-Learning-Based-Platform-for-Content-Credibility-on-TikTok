# ===============================
#  FAKTZ MODEL API (Flask)
#  - Receives raw text
#  - Cleans internally
#  - Classifies using svm model
#  - Returns JSON (label + confidence)
# ===============================

from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy

# Download NLTK resources 
nltk.download("wordnet")
nltk.download("omw-1.4")

nlp = spacy.load("en_core_web_sm")



# --- Load model and vectorizer ---
model = joblib.load("faktz_final_model.pkl")
vectorizer = joblib.load("faktz_tfidf_vectorizer.pkl")

# --- Lemmatizer ---
lemmatizer = WordNetLemmatizer()

# --- Full Cleaning Function ---
def clean_and_lemmatize(text):
    text = text.lower()

    # Expand contractions
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        "n't": " not",
        "'re": " are",
        "'s": " is",
        "'d": " would",
        "'ll": " will",
        "'t": " not",
        "'ve": " have",
        "'m": " am"
    }
    for k, v in contractions.items():
        text = re.sub(k, v, text)

    # Remove URLs, hashtags, mentions, HTML
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Remove emojis/symbols
    emoji_pattern = re.compile(
        "[" 
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r"", text)

    # Remove punctuation/numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split()]

    # Fix spaces
    return " ".join(tokens).strip()


support_pattern = re.compile(
    r'\b(according to|study|research|scientists?|experts?|doctors?|university|journal|WHO|CDC|Harvard|Stanford|Reuters|CNN|BBC|data|report)\b',
    re.IGNORECASE
)

def extract_resource_name(text):
    """
    Extract all organization or known source names from the text.
    Returns them in one readable line.
    """
    doc = nlp(text)
    orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]

    if orgs:
        # Remove duplicates and join neatly
        unique_orgs = list(dict.fromkeys(orgs))
        return f"Resources: {', '.join(unique_orgs)}"

    # fallback regex detection
    match = re.findall(r'\b(according to|study|research|WHO|CDC|Harvard|Stanford|Reuters|BBC|CNN|University|Association|Journal)\b', text, re.IGNORECASE)
    if match:
        unique_matches = list(dict.fromkeys([m.title() for m in match]))
        return f"Resources: {', '.join(unique_matches)}"

    return "Resources: None detected"

# --- Flask App ---
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Validate input
    if not text or not isinstance(text, str):
        return jsonify({
            "success": False,
            "error": "Invalid or missing 'text' field."
        }), 400

    # Clean text internally
    cleaned = clean_and_lemmatize(text)

    # Vectorize
    X = vectorizer.transform([cleaned])

    # Transform input
    proba = model.predict_proba(X)[0]
    label = model.classes_[np.argmax(proba)]
    confidence = np.max(proba) * 100



    # Extract resource if Supported Claim
    resource = extract_resource_name(text) if label.lower() == "supported claim" else None

    return jsonify({
        "label": label,
        "confidence (%)": round(confidence, 2),
        "resource": resource
    }), 200

# --- Run locally ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
