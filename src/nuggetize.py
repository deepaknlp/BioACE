import json
import transformers
import torch
import os

os.environ["HF_HOME"] = "/data/DIR_NLP_Models/huggingface"

with open("data/pubmed_data.json") as file:
    data = json.load(file)

with open("data/prompts_biogen_extension.json") as file:
    prompts = json.load(file)

prompt = prompts[5]
model_id = "meta-llama/Llama-3.3-70B-Instruct"
messages = []

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

for abstract in data.values():
    if abstract["abstract"] != "":
        a = abstract["title"] + ' ' + abstract["abstract"]

    else:
        a = abstract["title"]

    user = prompt["user"]
    user = user.replace("Abstract: a", "Abstract: " + a)

    messages.append([
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": user},
    ])

pipeline.tokenizer.padding = True
pipeline.tokenizer.padding_side = 'left'

outputs = pipeline(
    messages,
    max_new_tokens=1024,
    batch_size=8,
)

for i, abstract in enumerate(data.values()):
    abstract["nuggets"] = outputs[i][0]["generated_text"][-1]["content"].split('\n')

with open("data/pubmed_data_nuggets_complete.json", "w") as file:
    json.dump(data, file, indent=4)
