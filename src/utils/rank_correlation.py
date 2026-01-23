import json
from scipy.stats import kendalltau, spearmanr

with open("/data/bartelsdp/BioGenExtension/src/data/results/answer_nuggets-document_nuggets/llama3.json") as file:
    data = json.load(file)

stats = [
    {
        "num_coverage": 0,
        "num_sentences": 0,
        "num_citations": 0,
        "num_support": 0,
        "num_contradict": 0,
    }
    for _ in range(30)
]


for question in data:
    if int(question["question_id"]) <= 155:
        for i in range(30):
            for answer_sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
                stats[i]["num_sentences"] += 1

                if "labels" in answer_sentence:
                    coverage = False

                    for citation in answer_sentence["labels"].values():
                        stats[i]["num_citations"] += 1

                        if citation.lower().startswith("support"):
                            stats[i]["num_support"] += 1
                            coverage = True

                        elif citation.lower().startswith("contradict"):
                            stats[i]["num_contradict"] += 1

                if coverage:
                    stats[i]["num_coverage"] += 1

    else:
        break

metrics = [
    {
        "citation_coverage": 0,
        "citation_support_rate": 0,
        "citation_contradict_rate": 0,
    }
    for _ in range(30)
]

for i in range(30):
    if stats[i]["num_sentences"] == 0:
        metrics[i]["citation_coverage"] = 0

    else:
        metrics[i]["citation_coverage"] = stats[i]["num_coverage"] / stats[i]["num_sentences"]

    if stats[i]["num_citations"] == 0:
        metrics[i]["citation_support_rate"] = 0

    else:
        metrics[i]["citation_support_rate"] = stats[i]["num_support"] / stats[i]["num_citations"]

    if stats[i]["num_citations"] == 0:
        metrics[i]["citation_contradict_rate"] = 0

    else:
        metrics[i]["citation_contradict_rate"] = stats[i]["num_contradict"] / stats[i]["num_citations"]

for i in range(30):
    print(metrics[i]["citation_support_rate"])

fields = [
    "citation_coverage",
    "citation_support_rate",
    "citation_contradict_rate",
]

rankings = {}

for field in fields:
    rankings[field + "_ranking"] = sorted(
        range(30),
        key=lambda i: metrics[i][field],
        reverse=True
    )

print(rankings["citation_coverage_ranking"])

medaesqa_citation_coverage_ranking = [0, 10, 3, 19, 16, 23, 28, 11, 20, 17, 25, 2, 15, 5, 18, 8, 1, 7, 9, 13, 12, 22, 6, 4, 29, 14, 21, 27, 24, 26]
medaesqa_citation_support_rate_ranking = [25, 5, 18, 10, 15, 0, 6, 3, 19, 11, 1, 28, 13, 2, 17, 8, 20, 23, 9, 7, 16, 12, 4, 29, 14, 21, 22, 27, 24, 26]
medaesqa_citation_contradict_rate_ranking = [25, 5, 18, 10, 15, 0, 6, 3, 19, 11, 1, 28, 13, 2, 17, 8, 20, 23, 9, 7, 16, 12, 4, 29, 14, 21, 22, 27, 24, 26]

# Kendall's tau
tau, tau_p = kendalltau(rankings["citation_coverage_ranking"], medaesqa_citation_coverage_ranking)

# Spearman's rho
rho, rho_p = spearmanr(rankings["citation_coverage_ranking"], medaesqa_citation_coverage_ranking)

print("Citation Coverage")
print("Spearman's rho:", rho)
print("Kendall's tau:", tau)
print()

# Kendall's tau
tau, tau_p = kendalltau(rankings["citation_support_rate_ranking"], medaesqa_citation_support_rate_ranking)

# Spearman's rho
rho, rho_p = spearmanr(rankings["citation_support_rate_ranking"], medaesqa_citation_support_rate_ranking)

print("Citation Support Rate")
print("Spearman's rho:", rho)
print("Kendall's tau:", tau)
print()

# Kendall's tau
tau, tau_p = kendalltau(rankings["citation_contradict_rate_ranking"], medaesqa_citation_contradict_rate_ranking)

# Spearman's rho
rho, rho_p = spearmanr(rankings["citation_contradict_rate_ranking"], medaesqa_citation_contradict_rate_ranking)

print("Citation Contradict Rate")
print("Spearman's rho:", rho)
print("Kendall's tau:", tau)

