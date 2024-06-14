import src.mytools as mt
import mlflow
import joblib

# Chargement du modèle SpaCy:
nlp = mt.load_spacy_model("en_core_web_sm")

# Chargement du vectorizer :
vectorizer_path = mlflow.artifacts.download_artifacts("runs:/cf44df76dc5649e2a3b389e6f0552647/tfidf_vectorizer/vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# Chargement du binarizer :
binarizer_path = mlflow.artifacts.download_artifacts("runs:/59d7eb11b8f0468584fa2e59346d1d75/binarizer/binarizer.pkl")
binarizer = joblib.load(binarizer_path)

# Chargement du modèle de classification :
model = mlflow.sklearn.load_model("runs:/432b355cce7644d4a3fdd89fa5f29204/SGDClassifier")