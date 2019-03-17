import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import glob
import json
import urllib
spacy.prefer_gpu()
nlp = spacy.load('en')

# https://nlpforhackers.io/complete-guide-to-spacy/

# text = """London is the capital and most populous city of England and
# the United Kingdom.  Standing on the River Thames in the south east
# of the island of Great Britain, London has been a major settlement
# for two millennia. It was founded by the Romans, who named it Londinium.
# """
# for token in doc:
#     print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

# displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})


def get_docs():
    files = glob.glob("./data/*.csv")
    print(files)
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file))
    return df_list

def run_nlp_pipeline(text):
    doc = nlp(text)
    return doc



def get_entity():
    df_list = get_docs()

    for index_df, df in enumerate(df_list):
        df['Entity'] = np.nan
        for index, row in df.iterrows():
            text = row['Summary']
            doc = run_nlp_pipeline(str(text))
            entities = set(entity.text for entity in doc.ents)
            # for entity in doc.ents:
            #     entities.add(entity.text)
            df.loc[index, 'Entity'] = str(entities)
        save_pkl(df, index_df)
        print(df.head())


    google_KG_API_call(doc.ents[1].text)

def google_KG_API_call(query):
    print("query: ", query)
    api_key = "AIzaSyAYo4v0dIpFEECne8X_WAoaR0uwdrBWBl0"
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'query': query,
        'limit': 10,
        'indent': True,
        'key': api_key
    }
    url = service_url + '?' + urllib.parse.urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read())
    for element in response['itemListElement']:
        print(element['result']['name'] + ' (' + str(element['resultScore']) + ')')

def save_pkl(df, file_name = 'untitled'):
    df.to_pickle("./pickel_files/{}.pkl".format(file_name))

def get_df_from_pkl(file_name = 'untitled'):
    return pd.read_pickle("./pickel_files/{}.pkl".format(file_name))

get_entity()