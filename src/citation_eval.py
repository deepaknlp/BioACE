import sys
import os
import json
import re

def run_nugget_evaluation(data, method):
    for question in data:
        for i in range(30):
            for answer_sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
                if answer_sentence["citation_assessment"]:
                    pmids = []

                    for citation in answer_sentence["citation_assessment"]:
                        pmid = citation["cited_pmid"]

                        if pmid not in pmids:
                            pmids.append(pmid)

                            if pmid in abstracts:
                                document = '\n'.join(abstracts[pmid]["nuggets"])
                                sentence = '\n'.join(answer_sentence["nuggets_declarative"])
                                answer_sentence[pmid] = method.process_data(document, sentence)

                            else:
                                print("PMID " + pmid + " not found")

        with open("data/run6/" + sys.argv[1] + ".json", 'w') as file:
            json.dump(data, file, indent=4)

    return data

def run_evaluation(data, method):
    out = []

    for question in data:
        eval = {}
        eval["question_id"] = question["question_id"]
        eval["machine_generated_answers"] = {}

        for i in range(30):
            eval["machine_generated_answers"]["M" + str(i + 1)] = {}
            eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"] = []
            
            for sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
                eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"].append({})
                eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"][-1]["answer_sentence_id"] = sentence["answer_sentence_id"]
                eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"][-1]["citation_assessment"] = []

                if sentence["citation_assessment"]:
                    pmids = []

                    for citation in sentence["citation_assessment"]:
                        pmid = citation["cited_pmid"]

                        if citation["evidence_relation"] == "invalid citation":
                            continue

                        if pmid not in pmids:
                            eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"][-1]["citation_assessment"].append({})
                            eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"][-1]["citation_assessment"][-1]["cited_pmid"] = pmid
                            pmids.append(pmid)

                            print(question["question_id"], i + 1, sentence["answer_sentence_id"], pmid)

                            if abstracts[pmid]["abstract"]:
                                document = abstracts[pmid]["title"] + ' ' + abstracts[pmid]["abstract"]

                            else:
                                document = abstracts[pmid]["title"]

                            eval["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"][-1]["citation_assessment"][-1]["evidence_relation"] = method.process_data(document, re.sub("\[.*?\]", "", sentence["answer_sentence"]))

        out.append(eval)
        with open("data/run5/" + sys.argv[1] + ".json", 'w') as file:
            json.dump(out, file, indent=4)

    return out

os.environ["HF_HOME"] = "/data/DIR_NLP_Models/huggingface"

with open("data/biogen_nuggets.json") as file:
    data = json.load(file)

with open("data/pmid_nuggets_2.json") as file:
    abstracts = json.load(file)

if sys.argv[1] == "alignscore":
    from lib.alignscore import AlignScorer
    model = AlignScorer()

elif sys.argv[1] == "attrscore_alpaca":
    from lib.attrscore_alpaca import AttrScore_Alpaca
    model = AttrScore_Alpaca()

elif sys.argv[1] == "attrscore_flan_t5":
    from lib.attrscore_flan_t5 import AttrScore_FLAN_T5
    model = AttrScore_FLAN_T5()

elif sys.argv[1] == "flan_t5":
    from lib.flan_t5 import FLAN_T5
    model = FLAN_T5()

elif sys.argv[1] == "flan_ul2":
    from lib.flan_ul2 import FLAN_UL2
    model = FLAN_UL2()

elif sys.argv[1] == "clinicalmosaic":
    from lib.clinicalmosaic import ClinicalMosaic
    model = ClinicalMosaic()

elif sys.argv[1] == "gpt_4o":
    pass

elif sys.argv[1] == "llama_2":
    pass

elif sys.argv[1] == "llama_3":
    from lib.llama_3 import Llama_3
    model = Llama_3()

elif sys.argv[1] == "summac":
    from lib.summac import SummaCScore
    model = SummaCScore()

elif sys.argv[1] == "t5_xxl_true":
    from lib.t5_xxl_true import T5_XXL_TRUE
    model = T5_XXL_TRUE()

# sentence = data[0]["machine_generated_answers"]["M1"]["answer_sentences"][1]["answer_sentence"]
# pmid = data[0]["machine_generated_answers"]["M1"]["answer_sentences"][1]["citation_assessment"][0]["cited_pmid"]
# document = abstracts[pmid]["title"] + ' ' + abstracts[pmid]["abstract"]

# print(model.process_data(document, sentence))

# evals = run_evaluation(data, model)
evals = run_nugget_evaluation(data, model)

with open("data/run6/" + sys.argv[1] + ".json", 'w') as file:
    json.dump(evals, file, indent=4)