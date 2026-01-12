import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import argparse
import torch
from transformers import TrainerCallback
from peft import LoraConfig, PeftModel

name = "google/flan-t5-xxl"
model = AutoModelForSeq2SeqLM.from_pretrained(name, device_map="balanced", load_in_4bit=True)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(name)

print(model.hf_device_map)

num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
    allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved(i) / 1024**3    # Convert bytes to GB
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")

prompt = "### Instruction:\nPlease solely verify whether the reference can support the claim. Options: 'support', 'contradict', or 'neutral'.\n###Input:\nClaim: [sentence]\n\nReference: [document]\n\n### Output:"

def parse_args():
    # First-stage parser to extract model_name
    partial_parser = argparse.ArgumentParser(add_help=False)
    partial_parser.add_argument("--model_name", type=str, default="google/flan-ul2")
    partial_args, _ = partial_parser.parse_known_args()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="/data/guptadk/Documents/text-generation-webui/user_data/models")
    # parser.add_argument("--train_path", type=str, default="/data/guptadk/Documents/BioGen/data/Evaluations/biogen_nuggets.json")
    # parser.add_argument("--question_path", type=str, default="../data/BioGenData/BioGen2024topics-json.txt")
    # parser.add_argument("--test_path", type=str, default="/data/guptadk/Documents/BioGen/data/Evaluations/biogen_nuggets.json")
    parser.add_argument("--model_name", type=str, default=partial_args.model_name)
    parser.add_argument("--output_dir", type=str, default=f"../data/finetuned_models/{partial_args.model_name.split('/', 1)[1]}")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--do_inference", action="store_true", help="Run inference after training")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-5)
    return parser.parse_args()

def preprocess(examples, max_length):
    inputs = [
        prompt.replace("[document]", doc).replace("[sentence]", sent)
        for doc, sent in zip(examples["document"], examples["sentence"])
    ]

    tokenized_inputs = tokenizer(
        inputs,
        padding="max_length",    # pad all to max_length
        truncation=True,         # cut off if longer than max_length
        max_length=max_length    # set desired max length
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        label = tokenizer(
            examples["label"],
            padding="max_length",    # pad all to max_length
            truncation=True,         # cut off if longer than max_length
            max_length=max_length    # set desired max length
        )

    tokenized_inputs["labels"] = label["input_ids"]
    return tokenized_inputs

def load_data(max_length):
    dataset = {
        "documents": [],
        "sentences": [],
        "labels": [],
    }

    with open("../data/medesqa_v3.json") as file:
        data = json.load(file)

    with open("../data/pmid_data.json") as file:
        abstracts = json.load(file)

    for question in data:
        for i in range(30):
            for answer_sentence in question["machine_generated_answers"]["M" + str(i + 1)]["answer_sentences"]:
                if answer_sentence["citation_assessment"]:
                    for citation in answer_sentence["citation_assessment"]:
                        if citation["evidence_relation"] == "supporting" or citation["evidence_relation"] == "contradicting" or citation["evidence_relation"] == "neutral":
                            pmid = citation["cited_pmid"]

                            if pmid in abstracts:
                                dataset["labels"].append(citation["evidence_relation"])
                                dataset["sentences"].append(citation["evidence_support"])

                                if abstracts[pmid]["abstract"]:
                                    document = abstracts[pmid]["title"] + ' ' + abstracts[pmid]["abstract"]

                                else:
                                    document = abstracts[pmid]["title"]

                                dataset["documents"].append(document)

                            else:
                                print("[ERROR] PMID " + pmid + " not found")

    # Convert to list of dicts (each dict is one example)
    examples = [
        {"document": d, "sentence": s, "label": l}
        for d, s, l in zip(dataset["documents"], dataset["sentences"], dataset["labels"])
    ]

    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_list(examples)

    return hf_dataset.map(preprocess, fn_kwargs={"max_length": max_length}, batched=True)

class DeviceCheckCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        print(model.hf_device_map)

        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"GPU {i} - {torch.cuda.get_device_name(i)}")
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert bytes to GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3    # Convert bytes to GB
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")

        return control

def main():
    args = parse_args()
    tokenized_data = load_data(args.max_seq_length)
    tokenized_data = tokenized_data.remove_columns(["document", "sentence", "label"])

    print("Data format: ")
    print(tokenized_data[0])
    print(type(tokenized_data[0]['labels']))                # Should be list
    print(tokenized_data[0]['labels'])                      # Should be a flat list of ints
    print([type(i) for i in tokenized_data[0]['labels']])   # Should all be <class 'int'>

    lora_config = LoraConfig(
            r=16,
            target_modules=["q","k","v","o"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
    )

    model.add_adapter(lora_config)

    tokenized_dataset = {
        "train": tokenized_data.select(range(45, 65)),
        "eval": tokenized_data.select(range(40, 45)),
    }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        # predict_with_generate=True,
        bf16=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=10,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        gradient_checkpointing=True,
        prediction_loss_only=True,  # avoid keeping logits in memory
        logging_strategy="steps",
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        # compute_metrics=compute_metrics,
    )

    trainer.add_callback(DeviceCheckCallback())
    trainer.train()

if __name__ == "__main__":
    main()