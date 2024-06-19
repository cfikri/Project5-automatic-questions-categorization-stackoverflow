import pandas as pd
import src.mytools as mt
import spacy
import mlflow
import pytest
from unittest.mock import MagicMock

### TEST DE LA FONCTION coverage_score() ###

def test_coverage_score_same_tags():
    tags_pred = pd.Series(['avec', 'jean', 'sans', 'partir']) 
    expected = 1.
    assert mt.coverage_score(tags_pred, tags_pred) == expected

def test_coverage_score_no_common_tags():
    tags_pred = pd.Series(['avec', 'jean', 'sans', 'partir'])
    tags = pd.Series(['ave', 'jan', 'ans', 'prtir'])
    expected = 0.
    assert mt.coverage_score(tags_pred, tags) == expected

def test_coverage_score_half_common_tags():
    tags_pred = pd.Series(['avec', 'test', 'unitaire', 'projet', 'bien', 'construit'])
    tags = pd.Series(['sans', 'test', 'unitaire', 'projet', 'mal', 'fait' ])
    expected = 0.5
    assert mt.coverage_score(tags_pred, tags) == expected

### TEST DE LA FONCTION predict_tags() ###

def test_predict_tags():
    # Créer des mocks pour vectorizer, binarizer et model
    vectorizer_mock = MagicMock()
    binarizer_mock = MagicMock()
    model_mock = MagicMock()

    # Configurer le comportement des mocks
    vectorizer_mock.transform.return_value = "mocked_vector"
    model_mock.predict.return_value = "mocked_prediction"
    binarizer_mock.inverse_transform.return_value = ["tag1", "tag2"]

    # Appeler la fonction predict_tags avec les mocks
    tags = mt.predict_tags(vectorizer_mock, binarizer_mock, model_mock, "Example text")

    # Vérifier que les mocks ont été appelés avec les bons arguments
    vectorizer_mock.transform.assert_called_once_with(["Example text"])
    model_mock.predict.assert_called_once_with("mocked_vector")
    binarizer_mock.inverse_transform.assert_called_once_with("mocked_prediction")

    # Vérifier que la fonction retourne les tags attendus
    assert tags == ["tag1", "tag2"]

### TEST DE LA FONCTION process_text() ###

nlp = spacy.load("en_core_web_sm")

def test_characters_lemmatization():
    text = "Hello, world! This is a test with numbers 123 and symbols *&^%$#@!"
    result = mt.process_text(nlp, text)
    expected = ["hello", "world", "test", "number", "symbol"]
    assert result == expected

def test_allowed_words():
    text = "The quick brown fox jumps over the lazy dog."
    allowed_words = {"quick", "fox", "dog", "wine"}
    result = mt.process_text(nlp, text, allowed_words)
    expected = ["quick", "fox", "dog"]
    assert result == expected

def test_unique_words():
    text = "Hello hello world world"
    result = mt.process_text(nlp, text, unique=True)
    expected = ["hello", "world"]
    assert set(result) == set(expected)

def test_all_features():
    text = "The quick quick brown fox jumps over the lazy dog dog."
    allowed_words = {"quick", "brown", "fox", "jump", "wine"}
    result = mt.process_text(nlp, text, allowed_words, unique=True)
    expected = ["quick", "brown", "fox", "jump"]
    assert set(result) == set(expected)