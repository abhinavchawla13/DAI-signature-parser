import numpy as np
from spacy.lang.en import English
import spacy
import re

def convert(content, threshold=.9):
    pos_tagger = English()  # part-of-speech tagger
    # original_email = _read_email(fname)
    original_email = content
    sentences = _corpus2sentences(original_email)  # convert to sentences

    # iterate through sentence, write to a new file if not signature block
    # fn = fname.split(".")
    # new_fname = fn[0] + "_clean." + fn[1]
    return _generate_text(sentences)

def _read_email(fname):
    with open(fname, 'r', encoding="utf8") as email:
        text = email.read()
    return text

def _read_salutation(fname):
    return [line.lower().rstrip('\n') for line in open(fname)]


def _corpus2sentences(corpus):
    """split corpus into a list of sentences.
    """
    return corpus.strip().split('\n')

def _generate_text(sentences, threshold=0.9):
    """iterate through sentences. if the sentence is not a signature block, 
    write to file.

    if probability(signature block) > threshold, then it is a signature block.

    Parameters
    ----------
    sentence : str
        Represents line in email block.
    POS_parser: obj
        Spacy English object used to tag parts-of-speech. Will explore using
        other POS taggers like NLTK's.
    threshold: float
        Lower thresholds will result in more false positives.
    """
    tagger = spacy.load('en_core_web_sm')
    salutations = _read_salutation('salutations.txt')
    clean_signature = []

    last_possible_sentence_line = 0
    for i in range(len(sentences) - 1, 0, -1):
        # 1. probability of verbs/adj < threshold
        # 2. length greater than 4
        doc = tagger(sentences[i])
        if (_prob_block(doc) < threshold and len(doc) > 5):
            last_possible_sentence_line = i
            break
        if (sentences[i].lower().rstrip('\n,. ') in salutations):
            last_possible_sentence_line = i
            break

    
    for j in range(last_possible_sentence_line + 1, len(sentences)):
            if sentences[j]:
                split_sentences = sentences[j].split("|")
                for line in split_sentences:
                    if (line and not re.match(".*(.png|.jpg|.jpeg)( |\n)*", line)):
                        clean_signature.append(line.strip() + "\n")

    return clean_signature

def _prob_block(doc):
    """Calculate probability that a sentence is an email block.
    
    https://spacy.io/usage/linguistic-features

    Parameters
    ----------
    sentence : str
        Line in email block.

    Returns
    -------
    probability(signature block | line)
    """
    
    verb_count = np.sum([(token.pos_ != "VERB" and token.pos_ != "ADP") for token in doc])
    if (len(doc) > 0):
        return float(verb_count) / len(doc)
    return 1