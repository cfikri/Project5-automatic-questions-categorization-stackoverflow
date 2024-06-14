import src.mytools as mt
import streamlit as st

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