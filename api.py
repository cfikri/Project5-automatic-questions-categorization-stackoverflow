import src.mytools as mt
import streamlit as st
import mlflow
import joblib
import spacy

# Chargement du modèle linguistique de processing:
nlp = spacy.load("en_core_web_lg")

# Chargement du vectorizer :
vectorizer_path = mlflow.artifacts.download_artifacts("runs:/cf44df76dc5649e2a3b389e6f0552647/tfidf_vectorizer/vectorizer.pkl")
vectorizer = joblib.load(vectorizer_path)

# Chargement du binarizer :
binarizer_path = mlflow.artifacts.download_artifacts("runs:/59d7eb11b8f0468584fa2e59346d1d75/binarizer/binarizer.pkl")
binarizer = joblib.load(binarizer_path)

# Chargement du modèle de classification :
model = mlflow.sklearn.load_model("runs:/432b355cce7644d4a3fdd89fa5f29204/SGDClassifier")

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
        tags = predict_tags(texte)
        tags = tags[0].tolist()
        st.write(f'Tags suggérés : {tags}')
    else:
        st.write('Veuillez saisir un titre ET une question.')
