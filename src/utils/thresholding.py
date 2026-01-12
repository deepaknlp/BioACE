import sys
import json
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fvcore.nn import FlopCountAnalysis

import optuna

with open("/data/bartelsdp/BioGenExtension/src/data/medesqa_v3.json") as file:
    data = json.load(file)

with open("/data/bartelsdp/BioGenExtension/src/data/run2/" + sys.argv[1]) as file:
    responses = json.load(file)

y_true = []
y_pred = []

def objective(trial):
    t = trial.suggest_float("threshold", 0.0, 1.0)

    f1 = f1_score(y_true, np.array(y_pred) >= t)
    acc = accuracy_score(y_true, np.array(y_pred) >= t)

    return f1, acc

def eval_threshold(gold_label, generated_label):
    if gold_label != "invalid citation":
        y_pred.append(generated_label)

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

                            if sys.argv[1] == "summac.json":
                                eval_threshold(gold_label, citation_assessment["evidence_relation"][1])

                            elif sys.argv[1] == "alignscore.json":
                                eval_threshold(gold_label, citation_assessment["evidence_relation"])

                            else:
                                print("[ERROR] Unkown model ", sys.argv[1].split(".json")[0])
                                sys.exit()

run_eval()

thresholds = np.linspace(0, 1, 11)
f1_scores = [f1_score(y_true, y_pred >= t) for t in thresholds]
accuracy_scores = [accuracy_score(y_true, y_pred >= t) for t in thresholds]
print(thresholds[np.argmax(accuracy_scores)])

study = optuna.create_study(directions=["maximize", "minimize"])
study.optimize(objective, n_trials=100)

for trial in study.best_trials:
    f1, acc = trial.values
    print(trial.params["threshold"])