import sys
import os
import json
import torch
import transformers

def run_document_evaluation(data):
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
                                answer_sentence[pmid] = process_data(document, sentence)

                            else:
                                print("PMID " + pmid + " not found")

        with open("data/run6/" + sys.argv[1] + ".json", 'w') as file:
            json.dump(data, file, indent=4)

def run_maxSimSentence_evaluation(data):
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
                                answer_sentence[pmid] = process_data(document, sentence)

                            else:
                                print("PMID " + pmid + " not found")

        with open("data/run6/" + sys.argv[1] + ".json", 'w') as file:
            json.dump(data, file, indent=4)

def run_nuggets_evaluation(data):
    messages = []
    prompt = "For the following lists of answer and document nuggets, select one of the following labels:\n\nSupports: There is at least one document nugget that supports/agrees with at least answer nugget and there are no document nuggets that contradict any answer nuggets.\nContradicts: There is at least one document nugget that disagrees with an answer nugget or states its opposite.\nNeutral: The document nuggets are topically relevant, but lack any information to support or contradict any of the answer nuggets.\nNot relevant: The document nuggets are not relevant to the answer nuggets.\nThe response should only include the label.\n\nAnswer Nuggets:\n[sentence]\n\nDocument Nuggets:\n[document]"

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

                                user = prompt.replace("[document]", document).replace("[sentence]", sentence)

                                messages.append([
                                    {"role": "user", "content": user},
                                ])

                            else:
                                print("PMID " + pmid + " not found")

    outputs = process_data(messages)
    i = 0

    for question in data:
        for i in range(30):
            for answer_sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
                if answer_sentence["citation_assessment"]:
                    pmids = []
                    answer_sentence["labels"] = {}

                    for citation in answer_sentence["citation_assessment"]:
                        pmid = citation["cited_pmid"]

                        if pmid not in pmids:
                            pmids.append(pmid)

                            if pmid in abstracts:
                                answer_sentence["labels"][pmid] = outputs[i][0]["generated_text"][-1]["content"]
                                i += 1

                            else:
                                print("PMID " + pmid + " not found")

    return data

def process_data(messages):
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    pipeline.tokenizer.padding = True
    pipeline.tokenizer.padding_side = 'left'

    outputs = pipeline(
        messages,
        max_new_tokens=16,
        batch_size=16,
    )

    return outputs

with open("data/biogen_nuggets.json") as file:
    data = json.load(file)

with open("data/pmid_nuggets_2.json") as file:
    abstracts = json.load(file)

evals = run_nuggets_evaluation(data)

with open("data/results/answer_nuggets-document_nuggets/llama3.json", 'w') as file:
    json.dump(evals, file, indent=4)
