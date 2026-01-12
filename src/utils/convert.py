import json
import jsonlines
import re

with open("data/medaesqa_v1.json") as file:
    data = json.load(file)

with open("data/abstracts.json") as file:
    abstracts = json.load(file)

for question in data:
    for i in range(30):
        for sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
            if sentence["citation_assessment"]:
                pmids = set()

                for citation in sentence["citation_assessment"]:
                    pmid = citation["cited_pmid"]

                    if pmid not in pmids:
                        pmids.add(pmid)

                        if abstracts[pmid]["abstract"]:
                            document = abstracts[pmid]["title"] + ' ' + abstracts[pmid]["abstract"]

                        else:
                            document = abstracts[pmid]["title"]

                        if citation["evidence_relation"] == "supporting":
                            label = "CORRECT"

                        else:
                            label = "INCORRECT"
                        
                        obj = {
                            "id": question["question_id"] + ' ' + "M" + str(i + 1) + ' ' + sentence["answer_sentence_id"] + ' ' + pmid,
                            "text": document,
                            "claim": re.sub("\[.*?\]", "", sentence["answer_sentence"]),
                            "label": label
                            }

                        with jsonlines.open('data-dev.jsonl', mode='a') as writer:
                            writer.write(obj)
                        