import sys
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

with open("/data/bartelsdp/BioGenExtension/src/data/medesqa_v3.json") as file:
    data = json.load(file)

with open("/data/bartelsdp/BioGenExtension/src/data/run2/" + sys.argv[1]) as file:
    responses = json.load(file)

y_true = []
y_pred = []

def eval(gold_label, generated_label):
    if gold_label != "invalid citation":
        if generated_label.lower().startswith("attributable"):
            y_pred.append(0)

        elif generated_label.lower().startswith("not attributable"):
            y_pred.append(1)
            
        else:
            return

        if gold_label == "supporting":
            y_true.append(0)

        elif gold_label == "neutral" or gold_label == "not relevant":
            y_true.append(1)

        elif gold_label == "contradicting":
            y_true.append(1)

def eval_ternary(gold_label, generated_label):
    if gold_label != "invalid citation":
        if generated_label.lower().startswith("support"):
            y_pred.append(0)

        elif generated_label.lower().startswith("neutral"):
            y_pred.append(1)

        elif generated_label.lower().startswith("contradict"):
            y_pred.append(2)

        else:
            return

        if gold_label == "supporting":
            y_true.append(0)

        elif gold_label == "neutral" or gold_label == "not relevant":
            y_true.append(1)

        elif gold_label == "contradicting":
            y_true.append(2)

def eval_attrscore_flan_t5(gold_label, generated_label):
    if gold_label != "invalid citation":
        if generated_label == "Attributable":
            y_pred.append(0)

        elif generated_label == "Extrapolatory":
            y_pred.append(1)

        elif generated_label == "Contradictory":
            y_pred.append(1)
            
        else:
            return

        if gold_label == "supporting":
            y_true.append(0)

        elif gold_label == "neutral" or gold_label == "not relevant":
            y_true.append(1)

        elif gold_label == "contradicting":
            y_true.append(1)

def eval_attrscore_alpaca(gold_label, generated_label):
    if gold_label != "invalid citation":
        if generated_label.lower().endswith("### response:attributable"):
            y_pred.append(0)

        elif generated_label.lower().endswith("### response:extrapolatory"):
            y_pred.append(1)

        elif generated_label.lower().endswith("### response:contradictory"):
            y_pred.append(1)
            
        else:
            return

        if gold_label == "supporting":
            y_true.append(0)

        elif gold_label == "neutral" or gold_label == "not relevant":
            y_true.append(1)

        elif gold_label == "contradicting":
            y_true.append(1)

def eval_autoais(gold_label, generated_label):
    generated_label = generated_label.strip()

    if gold_label != "invalid citation":
        if generated_label == "1":
            y_pred.append(0)

        else:
            y_pred.append(1)

        if gold_label == "supporting":
            y_true.append(0)

        elif gold_label == "neutral" or gold_label == "not relevant":
            y_true.append(1)

        elif gold_label == "contradicting":
            y_true.append(1)

def eval_threshold(gold_label, generated_label, threshold):
    if gold_label != "invalid citation":
        if generated_label >= threshold:
            y_pred.append(0)

        else:
            y_pred.append(1)

        if gold_label == "supporting":
            y_true.append(0)

        elif gold_label == "neutral" or gold_label == "not relevant":
            y_true.append(1)

        elif gold_label == "contradicting":
            y_true.append(1)

def run_eval():
    for i in range(len(responses)):
        if int(responses[i]["question_id"]) <= 155:
            for j in range(30):
                for k in range(len(data[i]["machine_generated_answers"]["M" + str(j + 1)]["answer_sentences"])):
                    if data[i]["machine_generated_answers"]["M" + str(j + 1)]["answer_sentences"][k]["citation_assessment"]:
                        gold_labels = {}
                        for l in range(len(data[i]["machine_generated_answers"]["M" + str(j + 1)]["answer_sentences"][k]["citation_assessment"])):
                            citation_assessment = data[i]["machine_generated_answers"]["M" + str(j + 1)]["answer_sentences"][k]["citation_assessment"][l]

                            if citation_assessment["cited_pmid"] in gold_labels:
                                if gold_labels[citation_assessment["cited_pmid"]] != "supporting" and citation_assessment["evidence_relation"] == "supporting":
                                    gold_labels[citation_assessment["cited_pmid"]] = "supporting"

                            else:
                                gold_labels[citation_assessment["cited_pmid"]] = citation_assessment["evidence_relation"]

                        for l in range(len(responses[i]["machine_generated_answers"]["M" + str(j + 1)]["answer_sentences"][k]["citation_assessment"])):
                            citation_assessment = responses[i]["machine_generated_answers"]["M" + str(j + 1)]["answer_sentences"][k]["citation_assessment"][l]
                            
                            if citation_assessment["cited_pmid"] in gold_labels:
                                gold_label = gold_labels[citation_assessment["cited_pmid"]]

                            else:
                                print("[ERROR] Unlabeled PMID ", citation_assessment["cited_pmid"])
                                continue

                            if sys.argv[1] == "llama_3.json":
                                if type(citation_assessment["evidence_relation"]) is dict:
                                    eval(gold_label, citation_assessment["evidence_relation"]["content"])

                                else:
                                    eval(gold_label, citation_assessment["evidence_relation"])

                            elif sys.argv[1] == "attrscore_flan_t5.json":
                                eval_attrscore_flan_t5(gold_label, citation_assessment["evidence_relation"])

                            elif sys.argv[1] == "flan_t5.json":
                                eval(gold_label, citation_assessment["evidence_relation"])

                            elif sys.argv[1] == "flan_ul2.json":
                                eval(gold_label, citation_assessment["evidence_relation"])

                            elif sys.argv[1] == "attrscore_alpaca.json":
                                eval_attrscore_alpaca(gold_label, citation_assessment["evidence_relation"])

                            elif sys.argv[1] == "t5_xxl_true.json":
                                eval_autoais(gold_label, citation_assessment["evidence_relation"][0])

                            elif sys.argv[1] == "summac.json":
                                eval_threshold(gold_label, citation_assessment["evidence_relation"][1], 0)

                            elif sys.argv[1] == "alignscore.json":
                                eval_threshold(gold_label, citation_assessment["evidence_relation"], 0.5)

                            else:
                                print("[ERROR] Unkown model ", sys.argv[1].split(".json")[0])
                                sys.exit()

def run_eval_nuggets():
    for question in responses:
        for i in range(30):
            for answer_sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
                gold_labels = {}

                if answer_sentence["citation_assessment"]:
                    for citation_assessment in answer_sentence["citation_assessment"]:
                        if citation_assessment["cited_pmid"] in gold_labels:
                            if gold_labels[citation_assessment["cited_pmid"]] != "supporting" and citation_assessment["evidence_relation"] == "supporting":
                                gold_labels[citation_assessment["cited_pmid"]] = "supporting"

                        else:
                            gold_labels[citation_assessment["cited_pmid"]] = citation_assessment["evidence_relation"]

                    for pmid, gold_label in gold_labels.items():
                        generated_label = "not attributable"
                        
                        if pmid in answer_sentence:
                            num_nugs = len(answer_sentence["nuggets_declarative"])
                            num_labels = len(answer_sentence[pmid])
                            chunk_size = int(num_labels / num_nugs)
                            num_attr = 0

                            for j in range(num_nugs):
                                for gen_label in answer_sentence[pmid][j * chunk_size:j * chunk_size + chunk_size]:
                                    if gen_label.lower().startswith("attributable"):
                                        num_attr += 1
                                        break

                            if num_attr >= 0.1 * num_nugs:
                                generated_label = "attributable"
                                
                            eval(gold_label, generated_label)

                        # else:
                            # print("[ERROR] Missing PMID ", pmid)




run_eval()
# run_eval_nuggets()

"""
x = ["Supporting", "Neutral"]
y1 = [truePositive, trueNegative]
y2 = [falseNegative, falsePositive]

plt.bar(x, y1)
plt.bar(x, y2, bottom=y1)
plt.title(sys.argv[1][:-5])
plt.legend(["Correct", "Incorrect"])
plt.savefig(sys.argv[1][:-5] + ".png")"


groundTruth = [1] * scores["supporting"] + [0] * scores["not_supporting"]
predicted = [1] * scores["truePositive"] + [0] * scores["falseNegative"] + [1] * scores["falsePositive"] + [0] * scores["trueNegative"]

print("accuracy: ", (scores["truePositive"] + scores["trueNegative"]) / (scores["supporting"] + scores["not_supporting"]))
print("precision: ", scores["truePositive"] / (scores["truePositive"] + scores["falsePositive"]))
print("recall: ", scores["truePositive"] / (scores["truePositive"] + scores["falseNegative"]))
print("f1: ", (2 * scores["truePositive"]) / ((2 * scores["truePositive"]) + scores["falsePositive"] + scores["falseNegative"]))
print("pearsonr: ", pearsonr(predicted, groundTruth))
"""

print(classification_report(y_true, y_pred, target_names=["attributable", "not attributable"]))
# print(classification_report(y_true, y_pred, target_names=["support", "neutral", "contradict"]))