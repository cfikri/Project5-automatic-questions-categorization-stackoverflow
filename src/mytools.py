import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
import re
import mlflow


# La fonction *valeurs_manquantes* ci-dessous permet de déterminer le nombre et le pourcentage de valeurs manquantes (à 0.1% près) de chaque features d'un dataset.

def valeurs_manquantes(DataFrame):
    effectif = DataFrame.isna().sum()
    taux = DataFrame.isna().mean().round(3)*100
    result = pd.DataFrame({'effectif' : effectif, 'taux' : taux})
    return result.loc[result.effectif !=0, :] 

# La fonction *stats* ci-dessous prend en argument un DataFrame et renvoie un tableau contenant les principaux indicateurs statistiques de ses variables (effectif, moyenne, écart-type, médiane, quartiles, min et max).

def stats(DataFrame):
    return DataFrame.describe().round(3).T

# La fonction *test_std* prend en arguments un DataFrame et un entier n, et un renvoie pour chaque variable le taux (à 0.01% près) de valeurs situées en dehors de l'intervalle [moyenne - n.ecart-type , moyenne + n.ecart-type]. Cette fonction permet donc de connaitre le taux d'outliers de chaque variable selon la méthode des sigmas.

def test_std(DataFrame,n):
    features = stats(DataFrame).index
    outliers = pd.DataFrame()
    for feature in features:
        mean = stats(DataFrame).loc[feature,'mean']
        std = stats(DataFrame).loc[feature,'std']
        condition = ((DataFrame[feature] > mean + n*std) | (DataFrame[feature] < mean - n*std))
        outliers[feature] = condition
    nbr_outliers = outliers.sum()
    taux_outliers = outliers.mean().round(4)*100
    return pd.DataFrame({'nbr_outliers' : nbr_outliers, 'taux_outliers' : taux_outliers})

# La fonction *test_interquartile* prend en argument un DataFrame et un renvoie pour chaque variable le taux (à 0.01% près) de valeurs situées en dehors de l'intervalle [median - 1.5.ecart-interquartile , median + 1.5.ecart-interquartile]. Cette fonction permet donc de connaitre le taux d'outliers de chaque variable selon la méthode interquartile.

def test_interquartile(DataFrame):
    features = stats(DataFrame).index
    outliers = pd.DataFrame()
    for feature in features:
        Q1 = stats(DataFrame).loc[feature,'25%']
        Q3 = stats(DataFrame).loc[feature,'75%']
        IQ = Q3 - Q1
        condition = ((DataFrame[feature] > Q3 + 1.5*IQ) | (DataFrame[feature] < Q1 - 1.5*IQ))
        outliers[feature] = condition
    nbr_outliers = outliers.sum()
    taux_outliers = outliers.mean().round(4)*100
    return pd.DataFrame({'nbr_outliers' : nbr_outliers, 'taux_outliers' : taux_outliers})

# Suppriment les outliers détectés avec la méthode du z-score ou la méthode interquartile.

def S_outliers_drop(DataFrame,n):
    features = stats(DataFrame).index
    result = DataFrame
    for feature in features:
        mean = stats(DataFrame).loc[feature,'mean']
        std = stats(DataFrame).loc[feature,'std']
        condition = ((DataFrame[feature] > mean + n*std) | (DataFrame[feature] < mean - n*std))
        result[feature].mask(condition == True, pd.NA, inplace = True)
    return result

def IQ_outliers_drop(DataFrame):
    features = stats(DataFrame).index
    result = DataFrame
    for feature in features:
        Q1 = stats(DataFrame).loc[feature,'25%']
        Q3 = stats(DataFrame).loc[feature,'75%']
        IQ = Q3 - Q1
        condition = ((DataFrame[feature] > Q3 + 1.5*IQ) | (DataFrame[feature] < Q1 - 1.5*IQ))
        result[feature].mask(condition == True, pd.NA, inplace = True)
    return result

# La fonction *stats_extend* ci-dessous vise à présenter sous-forme de tableau les principaux indicateurs statistiques d'une DataFrame :
# - Les indicateurs de tendance centrale : moyenne et médiane ;
# - Les indicateurs de dispersion : étendue ,écart-type, quartiles et écart-interquartile ;
# - Les indicateurs de forme : skewness (asymétrie) et kurtosis (aplatissement).

def stats_extend(DataFrame):
    result = stats(DataFrame)
    quantitatif = DataFrame.select_dtypes(include=['int', 'float'])
    result.rename(columns = {'25%':'Q1', '50%':'med', '75%':'Q3' }, inplace=True)
    del result['count']
    result['etendue'] = result['max'] - result['min']
    result['IQR'] = result['Q3'] - result['Q1']
    result['skew'] = quantitatif.skew()
    result['kurtosis'] = quantitatif.kurtosis()
    return result

# La fonction *variance* ci-dessous permet de calculer la variance d'un échantillon donné, c'est à dire la somme des carrés des écarts de ses valeurs à leur moyenne. Cette fonction prend en argument un array ou un DataFrame et renvoie un array ou une Series contenant la variance de chaque colonne du tableau.

def variance(donnees):
    return (donnees.std(ddof=0)**2)*len(donnees)

# La fonction **correlation_graph** ci-dessous affiche le cercle de correlation dans le plan factoriel choisi. Elle prend trois arguments :
# - *pca* : Il s'agit de l'ACP appliquée aux données scalées ;
# - *x_y* : Il s'agit des indices des composantes principales (plan factoriel) choisies ;
# - features : Il s'agit de la liste des noms des variables que l'on souhaite représenter.

def correlation_graph(pca, x_y,features):
    
    pcs = pca.components_
    scree = (pca.explained_variance_ratio_*100).round(1)
    
    # Extrait les indices des composantes principales retenues.
    x,y = x_y

    # Taille de l'image (en inches).
    fig, ax = plt.subplots(figsize=(12, 12))

    # Pour chacune de nos variables par un vecteur (une flèche) avec le nom de la variable à côté.
    for i in range(pcs.shape[1]):
        ax.arrow(0,0, pcs[x,i], pcs[y,i], head_width=0.04, head_length=0.06, width=0.01)
        plt.text(pcs[x,i] + 0.05, pcs[y,i] + 0.05, features[i])

    # Affichage des lignes horizontales et verticales.
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué.
    plt.xlabel(f'CP{x+1} ({scree[x]}%)')
    plt.ylabel(f'CP{y+1} ({scree[y]}%)')
    
    # Affichage du titre.
    plt.title(f'Cercle des corrélations dans le plan factoriel (CP{x+1},CP{y+1})')

    # Traçage du cercle unité.
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), color='red')

    # Réglage des axes et affichage de la figure.
    plt.axis('equal')
    plt.show()

# La fonction **log_mlflow_run** ci-dessous permet de standariser le tracking des données via `MLflow`. Elle permet de sauvegarder les modèles entraînés ainsi que leurs principales caractéristiques : *paramètres*, *métriques*, *tags* et éventuels *artifacts*.

def log_mlflow_run(run_name, parameters, metrics, model=None, model_name=None, tags=None, artifacts=None):
    """
    Fonction pour logger les paramètres, métriques, tags et artifacts dans MLflow.
    
    Arguments:
    - parameters : Dictionnaire des paramètres du run.
    - metrics : Dictionnaire des métriques à suivre.
    - model : Modèle à enregistrer.
    - model_name : nom du modèle à enregistrer.
    - tags : Dictionnaire des tags à ajouter à l'expérience.
    - artifacts : Dictionnaire des chemins des artifacts à ajouter à l'expérience.

    """
    # Vérifier s'il y a un run actif du même nom
    if mlflow.active_run():
        mlflow.end_run()
    
    # Démarre un nouveau run
    mlflow.start_run(run_name=run_name)
    
    # Ajout des paramètres
    for key, value in parameters.items():
        mlflow.log_param(key, value)
        
    # Ajout des métriques
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    
    try:
        # Ajout des tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        # Ajout des artifacts
        if artifacts:
            for key, value in artifacts.items():
                mlflow.log_artifact(value, key)

        # Enregistrement du modèle
        if model:
            mlflow.sklearn.log_model(model, model_name)
              
    except Exception as e:
        print(f"Erreur lors de la journalisation dans MLflow: {e}")
        
    finally:
        # Fin du run
        mlflow.end_run()

# La fonction **process_text** ci-dessous permet d'extraire les mots d'un texte en supprimant les caractères non-alphabétiques (sauf les +, - et _) à l'aide d'une expression régulière et les mots trop courants (stopwords). Elle possède trois arguments :
# - ***text*** : La chaine de caractères que l'on souhaite traiter ;
# - ***allowed_words*** (optionnel) : Liste de mots autorisés. Les mots qui ne sont pas dans cette liste sont exclus. Par défaut, allowed_words = None ;
# - ***unique*** (optionnel) : *True* ou *False*. Supprime les doublons. Par défaut, unique=False.

def process_text(nlp, text, allowed_words=None, unique=False):

    # Suppression des caractères qui ne sont pas des lettres ou un des symboles (+, -, _) :
    text = re.sub(r'[^a-zA-Z\s\+\-_]', ' ', text)

    # Application du modèle linguistique au texte :
    doc = nlp(text)
    
    # Lemmatisation des mots, exclusion des stopwords, des mots de longueur inférieure à 3 et éventuellement, des mots non-autorisés :
    if allowed_words:
        words = [token.lemma_.lower() for token in doc\
                 if not token.is_stop and len(token.text) >= 3 and token.lemma_.lower() in allowed_words]
    else:
        words = [token.lemma_.lower() for token in doc if not token.is_stop and len(token.text) >= 3]

    # Suppression des doublons.
    if unique == True:
        words = list(set(words))
    
    return words

# La fonction **get_topics_key_words** ci-dessous permet d'enregistrer les mots clés de chaque topics dans un dictionnaire. Elle possède trois arguments :
# - *model* : Modèle entraîné pour générer les topics ;
# - *features_names* : Liste des mots formant les topics ;
# - *nb_key_words* : Nombre de mots les plus fréquents que l'on souhaite afficher, pour chaque topic.

def get_topics_key_words(model, feature_names, nb_key_words):
    topics = {}
    for topic_id, topic in enumerate(model.components_):
        topics[f'Topic {topic_id}'] = " ".join([feature_names[i] for i in topic.argsort()[-nb_key_words:]])
    return topics

# La fonction **tag_coverage** prends en arguments les tags générés ou prédits et les tags réels (sous forme de Series Pandas) et calcule le taux de couverture des tags réels par les tags générés ou prédits.

def coverage_score(tags_pred, tags_cible):
    coverages = []
    for pred, cible in zip(tags_pred.str.split(), tags_cible.str.split()):
        common_tags = set(pred).intersection(cible)
        coverages.append(len(common_tags) / len(cible) if len(cible) > 0 else 0)
    return sum(coverages) / len(coverages)

# Calculer la similarité cosinus entre deux vecteurs :
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Calcul la distance relative entre deux vecteurs :
def relative_distance(x, y):
    return np.linalg.norm(x - y)/np.linalg.norm(y)

# Calculer la distance moyenne entre les lignes de deux DataFrames :
def average_distance(df1, df2, metric=cosine_similarity):
    distances = []
    for index in df1.index:
            distances.append(metric(df1.loc[index,:], df2.loc[index,:]))
    return np.mean(distances)