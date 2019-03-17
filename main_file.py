import spacy
from spacy import displacy
import glob
spacy.prefer_gpu()
nlp = spacy.load('en')

# https://nlpforhackers.io/complete-guide-to-spacy/

text = """London is the capital and most populous city of England and 
the United Kingdom.  Standing on the River Thames in the south east 
of the island of Great Britain, London has been a major settlement 
for two millennia. It was founded by the Romans, who named it Londinium.
"""

def get_docs():
    files = glob.glob("/data/*.txt")
    text_list = []
    for file in files:
        f = open(file, "r")
        text_list.append(f.read())
    return text_list

def run_nlp_pipeline(text):
    doc = nlp(text)
    return doc



def get_ner():
    # text_list = get_docs()
    # for text in text_list:
    doc = run_nlp_pipeline(text)

    for entity in doc.ents:
        print(entity.text)
        print(entity.label)
    for token in doc:
        print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

    displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})

get_ner()