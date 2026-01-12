import json
from nltk.tokenize import sent_tokenize

with open("/data/bartelsdp/BioGenExtension/src/data/pmid_data.json") as file:
    abstracts = json.load(file)

for key in abstracts.keys():
    if abstracts[key]["abstract"]:
        abstracts[key]["sentences"] = sent_tokenize(abstracts[key]["abstract"])

with open("data/abstract_sentences.json", 'w') as file:
    json.dump(abstracts, file, indent=4)