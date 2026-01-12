print("Loading libraries . . .")

import sys
import json
from sentence_transformers import SentenceTransformer, util

print("Done")

class EmbeddingSimilarity:
    def __init__(self, model='all-MiniLM-L6-v2'):
 
        self.model = SentenceTransformer(model)
 
    def compute_cosine_similarity(self, answer_sent_list, doc_sentence_list):
        embeddings1 = self.model.encode(answer_sent_list, convert_to_tensor=True)
        embeddings2 = self.model.encode(doc_sentence_list, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
 
        return cosine_scores

with open("/data/bartelsdp/BioGenExtension/src/data/sentencewise_nuggets2.json") as file:
    questions = json.load(file)

with open("/data/bartelsdp/BioGenExtension/src/data/abstract_sentences.json") as file:
    document_sentences = json.load(file)

es = EmbeddingSimilarity()

for key in questions.keys():
    question = questions[key]
    for question_sentence in question:
        question_sentence["maxSimSentences"] = {}
        
        for pmid in question_sentence["pmids"]:
            if pmid in document_sentences:
                cosine_scores = es.compute_cosine_similarity([question_sentence["sentence"]], document_sentences[pmid]["sentences"]).cpu().tolist()
                max_index = cosine_scores[0].index(max(cosine_scores[0]))
                max_score = cosine_scores[0][max_index]
                most_similar_sentence = document_sentences[pmid]["sentences"][max_index]

                question_sentence["maxSimSentences"][pmid] = {
                    "maxSimSentence": most_similar_sentence,
                    "cosineScore": max_score
                }

            else:
                question_sentence["maxSimSentences"][pmid] = None
        
    with open("data/maxSimSentences.json", 'w') as file:
        json.dump(questions, file, indent=4)
    print(key)

with open("data/maxSimSentences.json", 'w') as file:
    json.dump(questions, file, indent=4)

