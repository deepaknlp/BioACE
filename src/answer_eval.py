import os
import torch
import joblib
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import argparse
import random
from prettytable import PrettyTable
import re
from sklearn.mixture import BayesianGaussianMixture
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, AutoModel
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
import numpy as np
from transformers import  AutoTokenizer

from pyserini.search.lucene import LuceneSearcher

seed_value=42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_th_config(seed_value)
def load_json(path_from_load):
    with open(path_from_load, 'r') as rfile:
        data = json.load(rfile)
    return data


def save_json(data, path_to_save):
    with open(path_to_save, 'w') as wfile:
        json.dump(data, wfile, indent=4)
    print(f"Saved json file as {path_to_save}")
def extract_label(output, labels):
    cleaned = output.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)

    normalized_labels = [label.lower() for label in labels]

    for i, norm_label in enumerate(normalized_labels):
        if re.search(rf"\b{norm_label}\b", cleaned):
            return norm_label

    for i, label in enumerate(labels):
        if label.lower() in output.lower():
            return label.lower()

    print(f"[Unclear Output] {output}")
    return "unclear"

def extract_nuggets(output):
    cleaned = output.strip().lower()
    return cleaned.split('\n')


def predict_label(tokenizer, model, answer, evidence_list, max_length=512, batch_size=16):

    modified_texts = [f"ANSWER: {answer} EVIDENCE: {ev}" for ev in evidence_list]
    ds = Dataset.from_dict({"text": modified_texts})
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding=True, max_length=max_length)
    ds = ds.map(tok, batched=True)
    ds = ds.remove_columns("text")
    data_collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            logits = model(**batch).logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            results.extend(batch_preds)
    return results



class SentenceAlignerEM:
    def __init__(self, path_to_gmm_params, optimal_threshold,  embedding_model='all-MiniLM-L6-v2'):

        self.path_to_save_gmm_params = path_to_gmm_params
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimal_threshold= optimal_threshold
        if 'sup-simcse-roberta-large' in embedding_model:
            model_name = "princeton-nlp/sup-simcse-roberta-large"
            self.sim_cse_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sim_cse_model = AutoModel.from_pretrained(model_name)
            self.sim_cse_model.to(self.device)
        else:
            self.sim_cse_model=None
            self.model = SentenceTransformer(embedding_model)
            if self.device == 'cuda':
                self.model.to(self.device)
        self.gmm_cls = BayesianGaussianMixture
    def get_simcse_embeddings(self, sentences):
        inputs = self.sim_cse_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = self.sim_cse_model(**inputs)
            sentence_embeddings = outputs.pooler_output
        return sentence_embeddings.cpu().numpy()
    def compute_similarity_matrix(self, ground_truth, predictions):
        if self.sim_cse_model is None:
            gt_embeddings = self.model.encode(ground_truth, device=self.device)
            pred_embeddings = self.model.encode(predictions, device=self.device)
        else:
            gt_embeddings = self.get_simcse_embeddings(ground_truth)
            pred_embeddings = self.get_simcse_embeddings(predictions)
        return cosine_similarity(pred_embeddings, gt_embeddings)



    def align_sentences(
            self,
            fitted_gmm,
            scores_matrix,
            ground_truth,
            predictions
    ):

        flat_sim = scores_matrix.flatten().reshape(-1, 1)
        responsibilities = fitted_gmm.predict_proba(flat_sim)
        match_component = np.argmax(fitted_gmm.means_)
        match_probs = responsibilities[:, match_component].reshape(scores_matrix.shape)
        assigned_rows = set()
        assigned_cols = set()
        assignments = {}
        assignment_with_scores=[]

        sorted_pairs = sorted(
            ((i, j, match_probs[i][j]) for i in range(match_probs.shape[0]) for j in range(match_probs.shape[1])),
            key=lambda x: x[2],
            reverse=True
        )

        for i, j, prob in sorted_pairs:
            if prob >= self.optimal_threshold and i not in assigned_rows and j not in assigned_cols:
                assignments[predictions[i]] = ground_truth[j]
                assigned_rows.add(i)
                assigned_cols.add(j)
                assignment_with_scores.append((ground_truth[j], predictions[i], float(prob)))

        return assignments,assignment_with_scores

    def evaluate_alignment(self, alignment, ground_truth, predictions):
        y_pred = [1 if pred in alignment and alignment[pred] in ground_truth else 0 for pred in predictions]
        y_true = [1] * len(ground_truth)

        precision = sum(y_pred) / len(y_pred) if y_pred else 0.0
        recall = sum(y_pred) / len(y_true) if y_true else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1_score


    def get_alignment_performance(self, data_items, pt_nuggets):
        question2precision_score = {}
        question2recall_score={}

        if gt_nuggets is None:
            print("Precision/recall can not be computed without ground-truth nuggets!")
            return
        similarities = []
        all_gt = []
        all_preds = []
        question_ids=[]
        for data_item in data_items:
            question_id = data_item['metadata']['topic_id']
            answer_sent_list=[]
            for response in data_item["responses"]:
                answer_sentence = response['text']
                answer_sent_list.append(answer_sentence)
            if question_id not in gt_nuggets or question_id not in pt_nuggets:
                print(f"Skipping question id: {question_id} as it does not have ground truth or predicted nuggets!")
                continue
            ground_truth_nuggets=gt_nuggets[question_id]
            predicted_nuggets=pt_nuggets[question_id]

            if len(ground_truth_nuggets) == 0 or len(predicted_nuggets) == 0:
                continue
            # print("Ground truth nuggets:", ground_truth_nuggets)
            # print("Predicted nuggets:", predicted_nuggets)
            # print(f"Question id: {question_id}")

            scores = self.compute_similarity_matrix(ground_truth_nuggets, predicted_nuggets)
            similarities.append(scores)
            all_gt.append(ground_truth_nuggets)
            all_preds.append(predicted_nuggets)
            question_ids.append(question_id)


        gmm_data = []
        for scores_matrix in similarities:
            gmm_data.extend(scores_matrix.flatten())
        print(f"Loading GMM params from file: {self.path_to_save_gmm_params}")
        fitted_gmm = joblib.load(self.path_to_save_gmm_params)

        for qid, scores_matrix, gt, preds in zip(question_ids, similarities, all_gt, all_preds):
            alignment, _ = self.align_sentences(fitted_gmm, scores_matrix, gt, preds)
            precision, recall, f1 = self.evaluate_alignment(alignment, gt, preds)

            question2precision_score[qid]=precision
            question2recall_score[qid]=recall

        return question2precision_score, question2recall_score

def retrieve_top_pmids_with_contents(query, top_k=20):
    hits = lucene_bm25_searcher.search(query, k=top_k)
    pmids = []
    for hit in hits:
        try:
            pmid = int(hit.docid)
            pmids.append((pmid, hit.lucene_document.get('raw')))
        except ValueError:
            continue
    return pmids


class BioACEEvaluator:
    SUPPORTED_METRICS = {"correctness", "completeness", "precision", "recall"}

    def __init__(
        self,
        metrics,
        completeness_model_name,
        completeness_model_dir,
        nuggets_generation_model_name,
        correctness_model_dir,
        path_to_gmm_config,
        optimal_threshold,
        lm_max_seq_length=4096
    ):
        self.metrics = set(metrics)
        invalid = self.metrics - self.SUPPORTED_METRICS
        if invalid:
            raise ValueError(f"Unsupported metrics: {invalid}")


        self.lm_max_seq_length = lm_max_seq_length
        self.completeness_model_dir = completeness_model_dir
        self.completeness_model_name = completeness_model_name
        self.nuggets_generation_model_name = nuggets_generation_model_name
        self.correctness_model_dir = correctness_model_dir
        self.path_to_gmm_config = path_to_gmm_config
        self.optimal_threshold = optimal_threshold

        self.completeness_model = None
        self.completeness_tokenizer = None
        self.correctness_model = None
        self.correctness_tokenizer = None
        self.nuggets_generation_model = None
        self.nuggets_generation_tokenizer = None
        self.precision_recall_model = None

        self._load_required_models()

    def _load_required_models(self):
        if "completeness" in self.metrics:
            self.completeness_model, self.completeness_tokenizer = self.load_completeness_model()

        if "correctness" in self.metrics:
            self.correctness_model, self.correctness_tokenizer = self.load_correctness_model()

        if "precision" in self.metrics or "recall" in self.metrics:
            if self.completeness_model_name == self.nuggets_generation_model_name:
                self.completeness_model, self.completeness_tokenizer = self.load_completeness_model()
                self.nuggets_generation_model = self.completeness_model
                self.nuggets_generation_tokenizer = self.completeness_tokenizer
            else:
                self.nuggets_generation_model, self.nuggets_generation_tokenizer = (
                    self.load_nuggets_generation_model()
                )

            self.precision_recall_model = SentenceAlignerEM(
                self.path_to_gmm_config,
                self.optimal_threshold
            )



    def load_completeness_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(self.completeness_model_dir, self.completeness_model_name),
            max_seq_length=self.lm_max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        model.eval()
        return model, tokenizer

    def load_nuggets_generation_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=os.path.join(self.completeness_model_dir, self.nuggets_generation_model_name),
            max_seq_length=self.lm_max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        model.eval()
        return model, tokenizer
    def load_correctness_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.correctness_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.correctness_model_dir, num_labels=2)
        model.to(device)
        model.eval()
        return model, tokenizer

    def generate_nuggets(self, data_items):
        sys_prompt = """You are NuggetExtractLLM, an AI assistant specialized in extracting information nuggets
                    from a given answer. A nugget is an atomic fact.:"""

        user_prompt= """Generate all the information nuggets that are required to completely answer the query given below.
                    Each nugget must contain one, and only one, fact. A nugget must be as concise and as specific as possible.
                    A nugget cannot contain a list, each element in a list must be its own nugget. Each nugget must directly
                    answer the query. The list of nuggets must not contain redundant information. Return a list of nuggets
                    such that each nugget is on a new line. Do not number or bullet the list. Do not include anything in your
                    response except for the list of nuggets. Here is an example of the output format:
                    nugget1
                    nugget2
                    . . .
                    """

        question2nuggets={}
        for data_item in data_items:
            question = data_item['metadata']['question']
            question_id = data_item['metadata']['topic_id']
            answer = data_item['metadata']['answer']
            batch_prompts = []
            if answer.strip()=="":
                question2nuggets[question_id]=[]
                continue

            instruction = user_prompt + "\n" + "Question: " + question + "\n" + "Answer: " + answer
            prompt_text = self.nuggets_generation_tokenizer.apply_chat_template([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": instruction}
            ], tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt_text)

            batch_inputs = self.nuggets_generation_tokenizer(batch_prompts, return_tensors="pt", padding=True,
                                                       truncation=True).to(self.nuggets_generation_model.device)
            input_ids = batch_inputs["input_ids"]
            with torch.no_grad():
                outputs = self.completeness_model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=100
                )
            decoded_nuggets_list=[]
            for output_ids, prompt_ids in zip(outputs, input_ids):
                generated_ids = output_ids[len(prompt_ids):]
                gen_output = self.nuggets_generation_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                # print(gen_output)
                decoded = extract_nuggets(gen_output)
                decoded_nuggets_list.append(decoded)

            assert len(decoded_nuggets_list)==len(batch_prompts)
            question2nuggets[question_id] = decoded_nuggets_list[0]


        return question2nuggets

    def evaluate_completeness(self, data_items):
        prompt = """You are an expert annotator. Given a question and an answer sentence, your task is to assign a single label from the following list: ['Required', 'Unnecessary', 'Borderline', 'Inappropriate']. 
            The label definition are as follows: 
            Required: The answer sentence is necessary to have in the generated answer for completeness of the answers.
            Unnecessary: The answer sentence is not required to have included in the generated answer. An answer sentence may be unnecessary for several reasons:
            (a) If including it would cause information overload if it is added to the answer;
            (b) If it is trivial, e.g., stating that many treatment options exist.
            (c) If it consists entirely of a recommendation to see a health professional.
            (d) If it is not relevant to the answer, e.g., describing the causes of a disease when the question is about treatments,
            Borderline: If an answer sentence is relevant, possibly even “good to know,” but not required, the answer sentence may be marked borderline.
            Inappropriate: The assertion may harm the patient, e.g., if according to the answer, physical therapy reduces the pain level, but the patient experiences more pain due to hip mobilization, the patient may start doubting they are receiving adequate treatment.
            Do not generate anything else. 
            Respond ONLY with the label no explanation."
            """
        question2comp_score = {}

        for data_item in data_items:
            question = data_item['metadata']['question']
            question_id = data_item['metadata']['topic_id']

            batch_prompts=[]
            answer_label_predictions=[]
            if len(data_item["responses"]) == 0:
                question2comp_score[question_id]=0.0
                continue
            for response in data_item["responses"]:
                answer_sentence = response['text']
                # print(f"Answer sentence: {answer_sentence}")

                instruction = "Question: " + question + "\n" + "Answer Sentence: " + answer_sentence

                prompt_text = self.completeness_tokenizer.apply_chat_template([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": instruction}
                ], tokenize=False, add_generation_prompt=True)
                batch_prompts.append(prompt_text)
            # print(f"Batch Prompts: {batch_prompts}")
            batch_inputs = self.completeness_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(self.completeness_model.device)
            input_ids = batch_inputs["input_ids"]
            with torch.no_grad():
                outputs = self.completeness_model.generate(
                    input_ids=batch_inputs["input_ids"],
                    attention_mask=batch_inputs["attention_mask"],
                    max_new_tokens=100
                )
            required_count=0
            for output_ids, prompt_ids in zip(outputs, input_ids):
                generated_ids = output_ids[len(prompt_ids):]
                gen_output = self.completeness_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                decoded = extract_label(gen_output,labels = ['required', 'unnecessary', 'borderline', 'inappropriate'])
                answer_label_predictions.append(decoded)
                if decoded.lower() == 'required' :
                    required_count+=1

            total_answer_sentences = len(data_item["responses"])
            acc = required_count / total_answer_sentences if total_answer_sentences > 0 else 0
            question2comp_score[question_id] = acc



            data_item['completeness_prediction']=answer_label_predictions
        return question2comp_score

    def evaluate_correctness(self, data_items):
        question2correctness_score={}
        for data_item in data_items:
            question= data_item['metadata']['question']
            question_id= data_item['metadata']['topic_id']

            answer_sentences_mapping = {}
            for response in data_item["responses"]:
                answer_sentence= response['text']
                hits = retrieve_top_pmids_with_contents(question, top_k=20)
                doc_sents = [json.loads(x[1])['contents'] for x in hits]

                if answer_sentence not in answer_sentences_mapping:
                    preds = predict_label(self.correctness_tokenizer, self.correctness_model, answer_sentence, doc_sents)

                    if any(p == 1 for p in preds):
                        answer_sentences_mapping[answer_sentence] = 1
                    else:
                        answer_sentences_mapping[answer_sentence] = 0
                elif answer_sentence in answer_sentences_mapping and answer_sentences_mapping[answer_sentence] == 0:
                    preds = predict_label(self.correctness_tokenizer, self.correctness_model, answer_sentence, doc_sents)
                    if any(p == 1 for p in preds):
                        answer_sentences_mapping[answer_sentence] = 1
                    else:
                        answer_sentences_mapping[answer_sentence] = 0

            total_answer_sentences = len( data_item["responses"])
            answer_correct = sum(answer_sentences_mapping.values())
            acc = answer_correct / total_answer_sentences if total_answer_sentences > 0 else 0
            question2correctness_score[question_id] = acc

        return question2correctness_score



    def evaluate_precision_recall(self, data_items, pt_nuggets):
        question2precision_score, question2recall_score =self.precision_recall_model.get_alignment_performance(data_items, pt_nuggets)
        return question2precision_score, question2recall_score

    def evaluate_all(
            self,
            data_items,
            verbose=True,
            enabled_metrics=("correctness", "completeness", "precision", "recall"),
            export_results_path="evaluation_results.json"
    ):

        valid_metrics = {"correctness", "completeness", "precision", "recall"}
        enabled_metrics = set(enabled_metrics)

        invalid = enabled_metrics - valid_metrics
        if invalid:
            raise ValueError(f"Invalid metric(s) requested: {invalid}")

        if "precision" in enabled_metrics or "recall" in enabled_metrics:
            if not hasattr(self, "precision_recall_model"):
                raise RuntimeError(
                    "Precision/Recall requested but precision_recall_model is not initialized"
                )


        table = PrettyTable()
        table.field_names = ["QID"] + [m.capitalize() for m in enabled_metrics]


        results = {}

        if "precision" in enabled_metrics or "recall" in enabled_metrics:
            pt_nuggets = self.generate_nuggets(data_items)

        if "correctness" in enabled_metrics:
            question2correctness_score = self.evaluate_correctness(data_items)
        else:
            question2correctness_score = {}

        if "completeness" in enabled_metrics:
            question2comp_score = self.evaluate_completeness(data_items)
        else:
            question2comp_score = {}

        if "precision" in enabled_metrics or "recall" in enabled_metrics:
            question2precision_score, question2recall_score = self.evaluate_precision_recall(
                data_items, pt_nuggets
            )
        else:
            question2precision_score, question2recall_score = {}, {}


        total_questions = len(data_items)
        overall_scores = {m: [] for m in enabled_metrics}
        per_question_results = {}

        for item in data_items:
            qid = item["metadata"].get("topic_id")

            if qid is None:
                print("Skipping item with missing qid")
                continue

            row = {"QID": qid}
            per_question_results[qid] = {}

            if "correctness" in enabled_metrics:
                score = question2correctness_score.get(qid, 0.0)
                row["Correctness"] = f"{score:.4f}"
                per_question_results[qid]["correctness"] = score
                overall_scores["correctness"].append(score)

            if "completeness" in enabled_metrics:
                score = question2comp_score.get(qid, 0.0)
                row["Completeness"] = f"{score:.4f}"
                per_question_results[qid]["completeness"] = score
                overall_scores["completeness"].append(score)

            if "precision" in enabled_metrics:
                score = question2precision_score.get(qid, 0.0)
                row["Precision"] = f"{score:.4f}"
                per_question_results[qid]["precision"] = score
                overall_scores["precision"].append(score)

            if "recall" in enabled_metrics:
                score = question2recall_score.get(qid, 0.0)
                row["Recall"] = f"{score:.4f}"
                per_question_results[qid]["recall"] = score
                overall_scores["recall"].append(score)

            if verbose:
                table.add_row([row[col] for col in table.field_names])


        if verbose:
            print("\nEvaluation Results")
            print(table)


        overall_summary = {}
        print("\n===== OVERALL SCORES =====")
        print(f"Total questions : {total_questions}")

        for metric, values in overall_scores.items():
            avg = sum(values) / total_questions if values else 0.0
            overall_summary[metric] = avg
            print(f"Overall {metric.capitalize()} Score: {avg:.4f}")


        export_payload = {
            "enabled_metrics": sorted(enabled_metrics),
            "total_questions": total_questions,
            "per_question": per_question_results,
            "overall": overall_summary,
        }

        save_json(export_payload, export_results_path)

        print(f"\nJSON results written to: {export_results_path}")
        print("Evaluation completed.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["correctness", "completeness", "precision", "recall"],
        choices=["correctness", "completeness", "precision", "recall"],
        help="Metrics to evaluate"
    )
    parser.add_argument("--dataset_path", type=str,
                        default="../resources/data/task_b_baseline_output.json")
    parser.add_argument("--lucene_indexer_path", type=str,
                        default='../resources/data/indexes/pubmed_baseline_collection_jsonl')
    parser.add_argument("--gt_nugget_path", type=str,
                        default="../resources/data/task_b_gt_nuggets_first2.json")
    parser.add_argument("--completeness_model_name", type=str, default='completeness_model')
    parser.add_argument("--nuggets_generation_model_name", type=str, default='completeness_model')
    parser.add_argument("--completeness_model_dir", type=str,
                        default="../resources/models")
    parser.add_argument("--correctness_model_dir", type=str,
                        default="../resources/models/correctness_model")
    parser.add_argument("--path_to_gmm_config", type=str,
                        default="../resources/models/precision_recall_model/gmm_params.pkl")
    parser.add_argument("--optimal_threshold", type=float, default=0.6267)

    args = parser.parse_args()


    dataset = load_json(args.dataset_path)
    lucene_bm25_searcher = LuceneSearcher(args.lucene_indexer_path)
    gt_nuggets = load_json(args.gt_nugget_path)

    evaluator = BioACEEvaluator(
        metrics=args.metrics,
        completeness_model_name=args.completeness_model_name,
        nuggets_generation_model_name=args.nuggets_generation_model_name,
        completeness_model_dir=args.completeness_model_dir,
        correctness_model_dir=args.correctness_model_dir,
        path_to_gmm_config=args.path_to_gmm_config,
        optimal_threshold=args.optimal_threshold
    )

    evaluator.evaluate_all(dataset, enabled_metrics=args.metrics,
                           export_results_path=args.dataset_path.replace(".json", "") + "-results(" + "-".join(
                               args.metrics) + ").json")


