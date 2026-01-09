import logging
from typing import List

try:
    from spacy.tokens import Span, Token
except Exception:
    Span = Token = None


def load_spacy():
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_lg")
        except Exception:
            nlp = spacy.load("en_core_web_sm")

        logging.info("Loaded spaCy model.")
        return nlp

    except Exception as e:
        logging.error(f"Could not load spaCy model: {e}")
        return None


def is_content_word_and_non_stopword(nlp, token: Token) -> bool:
    return (
        token.pos_ in {"NOUN", "PROPN", "ADJ"}
        and token.lemma_ not in nlp.Defaults.stop_words
    )


def get_content_words_from_sent(nlp, sent: Span) -> List[Token]:
    return [token for token in sent if is_content_word_and_non_stopword(nlp, token)]


def get_concept_lemmas(nlp, concept: str) -> List[str]:
    doc = nlp(concept)
    return [token.lemma_ for token in doc]


def has_noun(nlp, text: str) -> bool:
    doc = nlp(text)
    return any(token.pos_ in {"NOUN", "PROPN"} for token in doc)
