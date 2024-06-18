import src.mytools as mt
import streamlit as st
import mlflow
import joblib
import spacy
import os

# Définir l'URI de suivi MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://ec2-52-209-5-80.eu-west-1.compute.amazonaws.com:5000"))

# Chargement du modèle SpaCy:
nlp = mt.load_spacy_model("en_core_web_sm")

# Chargement du vectorizer :
adresse = "s3://mlflow-cfikri/795809222013714058/045ef2a624f847be94de59dabbcc71a9/artifacts/tfidf_vectorizer/vectorizer.pkl"
vectorizer_path = mlflow.artifacts.download_artifacts(adresse)
vectorizer = joblib.load(vectorizer_path)

# Chargement du binarizer :
adresse = "s3://mlflow-cfikri/795809222013714058/02eed1cf1390491e8e525f9ac2e6e17b/artifacts/binarizer/binarizer.pkl"
binarizer_path = mlflow.artifacts.download_artifacts(adresse)
binarizer = joblib.load(binarizer_path)

# Chargement du modèle de classification :
adresse = "s3://mlflow-cfikri/795809222013714058/1acc9429a88f4555933b8fc8796ed48b/artifacts/SGDClassifier"
model = mlflow.sklearn.load_model(adresse)

# Titre de l'interface Streamlit :
st.title('Classification de questions')

# Champ de saisie du titre :
titre = st.text_input('TITRE :')

# Champ de saisie de la question :
question = st.text_input('Question :')

# Bouton de prédiction
if st.button('Suggérer des Tags'):
    if titre and question:
        texte = mt.process_text(nlp, titre + ' ' + question)
        texte = ' '.join(texte)
        tags = mt.predict_tags(vectorizer, binarizer, model, texte)
        tags = tags[0].tolist()
        st.write(f'Tags suggérés : {tags}')
    else:
        st.write('Veuillez saisir un titre ET une question.')