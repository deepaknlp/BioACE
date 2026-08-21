from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are NuggetExtractLLM, an AI assistant specialized in extracting information nuggets from a given abstract. "
    "A nugget is an atomic fact."
)

DEFAULT_USER_TEMPLATE = (
    "List all of the information nuggets in the abstract given below. "
    "Each nugget must contain one, and only one, fact from the abstract. "
    "A nugget must be as concise and as specific as possible. Each element in a list must be its own nugget. "
    "The list of nuggets must not contain redundant information. Each nugget should be a semantic triple in the form Predicate (subject, object). "
    "A predicate must be one word. Return a list of nuggets such that each nugget is on a new line. "
    "Do not number or bullet the list. Do not include anything in your response except for the list of nuggets. "
    "Here is an example of the output format:\n"
    "Predicate (subject, object)\n"
    "Predicate (subject, object)\n"
    "…\n"
    "Here is an example abstract: During infections, a battle for iron takes place between the human host and the invading pathogens. "
    "Lymphocytes need iron to mount an effective cellular and humoral response. "
    "Viruses depend on iron to replicate within living host cells. "
    "During the acute phase of infection, blood levels of iron decrease. "
    "Ferritin levels are high. "
    "Elevated serum ferritin is associated with increased mortality. "
    "As a major iron storage protein, ferritin is essential to iron homeostasis and is involved in a wide range of physiologic and pathologic processes. "
    "The inflammation cascade and poor prognosis of COVID-19 may be attributed to high ferritin levels. "
    "Iron depletion therapy was proposed as a novel therapeutic approach in the COVID-19 pandemic.\n"
    "This is the list of nuggets that should be extracted from this abstract:\n"
    "Compete (Lymphocytes and viruses, iron)\n"
    "Need (lymphocytes, iron for cellular response)\n"
    "Need (lymphocytes, iron for humoral response)\n"
    "Need (Viruses, iron to replicate)\n"
    "Decrease (Infection, iron levels in the blood)\n"
    "Decrease (Infection, ferritin levels in the blood)\n"
    "Associate (High ferritin, increased mortality)\n"
    "Need (Iron homeostasis, ferritin)\n"
    "Involve (Physiologic processes, ferritin)\n"
    "Involve (Pathologic processes, ferritin)\n"
    "Indicate (High ferritin, response to inflammation)\n"
    "Associate (High ferritin levels, poor outcomes)\n"
    "Manage (Iron depletion therapy, viral activity in COVID-19)\n"
    "Manage (Iron depletion therapy, fibrotic activity in COVID-19)\n"
    "Abstract: {context}."
)


@dataclass
class NuggetInput:
    item_id: str
    text: str
    question: str = ""
    answer: str = ""
    title: str = ""
    abstract: str = ""
    source: Any = None
    responses: list[dict[str, Any]] = field(default_factory=list)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
        file.write("\n")
    print(f"Saved json file as {output_path}")


def first_present(mapping: dict[str, Any], fields: list[str]) -> Any:
    for field_name in fields:
        value = mapping.get(field_name)
        if value not in (None, ""):
            return value
    return ""


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def get_text_from_mapping(mapping: dict[str, Any], fields: list[str]) -> str:
    return stringify(first_present(mapping, fields))


def get_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    for metadata_key in ("metadata", "meta_data"):
        metadata = record.get(metadata_key)
        if isinstance(metadata, dict):
            return metadata
    return {}


def get_record_id(record: dict[str, Any], id_fields: list[str], default: str) -> str:
    id_field_candidates = ["topic_id", "question_id", "qa_id", "id"] + id_fields
    metadata = get_record_metadata(record)
    return (
        get_text_from_mapping(metadata, id_field_candidates)
        or get_text_from_mapping(record, id_field_candidates)
        or default
    )


def get_record_question(record: dict[str, Any], question_fields: list[str]) -> str:
    question_field_candidates = ["question", "query"] + question_fields
    metadata = get_record_metadata(record)
    return (
        get_text_from_mapping(metadata, question_field_candidates)
        or get_text_from_mapping(record, question_field_candidates)
    )


def normalize_responses(raw_responses: Any, fallback_answer: str = "") -> list[dict[str, str]]:
    responses: list[dict[str, str]] = []

    if isinstance(raw_responses, list):
        for response in raw_responses:
            if isinstance(response, str):
                text = response.strip()
            elif isinstance(response, dict):
                text = get_text_from_mapping(
                    response,
                    ["text", "answer_sentence", "sentence", "content", "response"],
                )
            else:
                text = stringify(response)

            if text:
                responses.append({"text": text})

    if not responses and fallback_answer:
        responses = [{"text": sentence} for sentence in split_answer_sentences(fallback_answer)]

    return responses


def split_answer_sentences(answer: str) -> list[str]:
    answer = stringify(answer)
    if not answer:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer) if part.strip()]
    return sentences or [answer]


def natural_sort_key(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z_]+)(\d+)", value)
    if not match:
        return value, -1
    return match.group(1), int(match.group(2))


def infer_input_format(data: Any) -> str:
    if isinstance(data, dict):
        values = list(data.values())
        if values and all(is_nugget_list(value) for value in values):
            return "nugget-map"
        if values and all(isinstance(value, dict) for value in values):
            if any("title" in value or "abstract" in value for value in values):
                return "abstract-map"
            if any(isinstance(value.get("answer"), list) for value in values):
                return "task-a"
            if any("metadata" in value or "meta_data" in value or "responses" in value for value in values):
                return "answer-eval"
        return "flat"

    if isinstance(data, list):
        first = next((item for item in data if isinstance(item, dict)), None)
        if first is None:
            return "flat"
        if "machine_generated_answers" in first:
            return "medaesqa"
        if isinstance(first.get("answer"), list):
            return "task-a"
        if "metadata" in first or "meta_data" in first or "responses" in first:
            return "answer-eval"
        return "flat"

    raise ValueError("Input JSON must be either a list or an object.")


def is_nugget_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def extract_items(
    data: Any,
    input_format: str,
    answer_key: str | None,
    id_fields: list[str],
    question_fields: list[str],
    answer_fields: list[str],
    title_fields: list[str],
    abstract_fields: list[str],
) -> list[NuggetInput]:
    if input_format == "auto":
        input_format = infer_input_format(data)

    if input_format == "nugget-map":
        if not isinstance(data, dict):
            raise ValueError("nugget-map input must be a JSON object.")
        return [
            NuggetInput(item_id=stringify(item_id), text="\n".join(nuggets), source=nuggets)
            for item_id, nuggets in data.items()
            if is_nugget_list(nuggets)
        ]

    if input_format == "abstract-map":
        if not isinstance(data, dict):
            raise ValueError("abstract-map input must be a JSON object keyed by item id.")
        return [
            abstract_record_to_item(item_id, record, title_fields, abstract_fields)
            for item_id, record in data.items()
        ]

    records = dict_records_to_list(data)

    if input_format == "answer-eval":
        return [
            answer_eval_record_to_item(index, record, id_fields, question_fields, answer_fields)
            for index, record in enumerate(records)
        ]

    if input_format == "task-a":
        return [
            task_a_record_to_item(index, record, id_fields, question_fields)
            for index, record in enumerate(records)
        ]

    if input_format == "medaesqa":
        return [
            medaesqa_record_to_item(index, record, answer_key, id_fields, question_fields)
            for index, record in enumerate(records)
        ]

    if input_format == "flat":
        return [
            flat_record_to_item(index, record, id_fields, question_fields, answer_fields)
            for index, record in enumerate(records)
        ]

    raise ValueError(f"Unsupported input format: {input_format}")


def dict_records_to_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = []
        for key, value in data.items():
            if isinstance(value, dict):
                record = dict(value)
                record.setdefault("id", key)
                records.append(record)
            else:
                records.append({"id": key, "answer": value})
    else:
        raise ValueError("Expected a list or object of records.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every input record must be a JSON object.")
    return records


def answer_eval_record_to_item(
    index: int,
    record: dict[str, Any],
    id_fields: list[str],
    question_fields: list[str],
    answer_fields: list[str],
) -> NuggetInput:
    metadata = get_record_metadata(record)
    item_id = get_record_id(record, id_fields, str(index))
    question = get_record_question(record, question_fields)
    answer = get_text_from_mapping(metadata, ["answer"] + answer_fields)
    answer = answer or get_text_from_mapping(record, ["answer"] + answer_fields)
    responses = normalize_responses(record.get("responses"), answer)
    if not answer:
        answer = " ".join(response["text"] for response in responses)
    return NuggetInput(
        item_id=item_id or str(index),
        text=answer,
        question=question,
        answer=answer,
        source=record,
        responses=responses,
    )


def task_a_record_to_item(
    index: int,
    record: dict[str, Any],
    id_fields: list[str],
    question_fields: list[str],
) -> NuggetInput:
    item_id = get_record_id(record, id_fields, str(index))
    question = get_record_question(record, question_fields)
    answer_entries = record.get("answer", [])
    responses = normalize_responses(answer_entries)
    answer = " ".join(response["text"] for response in responses)
    return NuggetInput(
        item_id=item_id,
        text=answer,
        question=question,
        answer=answer,
        source=record,
        responses=responses,
    )


def medaesqa_record_to_item(
    index: int,
    record: dict[str, Any],
    answer_key: str | None,
    id_fields: list[str],
    question_fields: list[str],
) -> NuggetInput:
    item_id = get_record_id(record, id_fields, str(index))
    question = get_record_question(record, question_fields)
    answers = record.get("machine_generated_answers")

    if not isinstance(answers, dict) or not answers:
        return NuggetInput(item_id=item_id, text="", question=question, source=record)

    selected_key = answer_key
    if selected_key is None:
        selected_key = sorted(answers, key=natural_sort_key)[0]
    if selected_key not in answers:
        raise ValueError(f"Answer key {selected_key!r} not found for item {item_id}.")

    selected_answer = answers[selected_key]
    if not isinstance(selected_answer, dict):
        selected_answer = {"answer": selected_answer}

    answer = get_text_from_mapping(selected_answer, ["answer", "text", "content"])
    responses = normalize_responses(selected_answer.get("answer_sentences"), answer)
    if not answer:
        answer = " ".join(response["text"] for response in responses)

    return NuggetInput(
        item_id=item_id,
        text=answer,
        question=question,
        answer=answer,
        source=record,
        responses=responses,
    )


def flat_record_to_item(
    index: int,
    record: dict[str, Any],
    id_fields: list[str],
    question_fields: list[str],
    answer_fields: list[str],
) -> NuggetInput:
    item_id = get_record_id(record, id_fields, str(index))
    question = get_record_question(record, question_fields)
    answer = get_text_from_mapping(record, ["answer", "response", "text", "content"] + answer_fields)
    responses = normalize_responses(record.get("responses"), answer)
    return NuggetInput(
        item_id=item_id,
        text=answer,
        question=question,
        answer=answer,
        source=record,
        responses=responses,
    )


def abstract_record_to_item(
    item_id: str,
    record: Any,
    title_fields: list[str],
    abstract_fields: list[str],
) -> NuggetInput:
    if not isinstance(record, dict):
        text = stringify(record)
        return NuggetInput(item_id=stringify(item_id), text=text, answer=text, source=record)

    title = get_text_from_mapping(record, ["title"] + title_fields)
    abstract = get_text_from_mapping(record, ["abstract"] + abstract_fields)
    text = " ".join(part for part in [title, abstract] if part)
    return NuggetInput(
        item_id=stringify(item_id),
        text=text,
        answer=text,
        title=title,
        abstract=abstract,
        source=record,
    )


def load_prompt(prompt_file: str | None, prompt_index: int) -> tuple[str, str]:
    if not prompt_file:
        return DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_TEMPLATE

    prompt_data = load_json(prompt_file)
    if isinstance(prompt_data, list):
        try:
            prompt = prompt_data[prompt_index]
        except IndexError as exc:
            raise ValueError(f"Prompt index {prompt_index} is out of range.") from exc
    elif isinstance(prompt_data, dict):
        prompt = prompt_data
    else:
        raise ValueError("Prompt file must contain either an object or a list of objects.")

    if not isinstance(prompt, dict):
        raise ValueError("Selected prompt must be a JSON object.")

    system_prompt = stringify(prompt.get("system")) or DEFAULT_SYSTEM_PROMPT
    user_template = stringify(prompt.get("user")) or DEFAULT_USER_TEMPLATE
    return system_prompt, user_template


def build_prompt_context(item: NuggetInput) -> str:
    if item.question:
        return f"Question: {item.question}\nAnswer: {item.text}"
    if item.title or item.abstract:
        return f"Title: {item.title}\nAbstract: {item.abstract}".strip()
    return f"Text: {item.text}"


def render_user_prompt(template: str, item: NuggetInput) -> str:
    values = {
        "id": item.item_id,
        "text": item.text,
        "question": item.question,
        "answer": item.answer or item.text,
        "title": item.title,
        "abstract": item.abstract,
        "context": build_prompt_context(item),
    }

    try:
        rendered = template.format(**values)
    except (KeyError, ValueError):
        rendered = template

    rendered = rendered.replace("Abstract: a", f"Abstract: {item.text}")
    rendered = rendered.replace("[document]", item.text)
    rendered = rendered.replace("[text]", item.text)
    rendered = rendered.replace("[answer]", item.answer or item.text)
    rendered = rendered.replace("[question]", item.question)

    if rendered == template and item.text not in rendered:
        rendered = f"{rendered.rstrip()}\n\n{build_prompt_context(item)}"

    return rendered


def make_messages(items: list[NuggetInput], system_prompt: str, user_template: str) -> list[list[dict[str, str]]]:
    messages = []
    for item in items:
        messages.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_user_prompt(user_template, item)},
            ]
        )
    return messages


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    if dtype_name == "none":
        return None
    if dtype_name == "auto":
        return "auto"
    return getattr(torch_module, dtype_name)


def load_generation_pipeline(args: argparse.Namespace) -> Any:
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    import torch
    import transformers

    model_kwargs: dict[str, Any] = {}
    torch_dtype = resolve_torch_dtype(torch, args.torch_dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    generator = transformers.pipeline(
        "text-generation",
        model=args.model,
        model_kwargs=model_kwargs,
        device_map=args.device_map,
    )

    tokenizer = generator.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding = True
    tokenizer.padding_side = args.padding_side
    return generator


def get_generated_content(output: Any) -> str:
    first = output[0] if isinstance(output, list) and output else output
    if isinstance(first, dict):
        generated_text = first.get("generated_text", first.get("text", ""))
    else:
        generated_text = first

    if isinstance(generated_text, list) and generated_text:
        last_message = generated_text[-1]
        if isinstance(last_message, dict):
            return stringify(last_message.get("content"))
        return stringify(last_message)

    return stringify(generated_text)


def split_nuggets(output: str) -> list[str]:
    nuggets = []
    for line in output.splitlines():
        line = line.strip().strip("`")
        line = re.sub(r"^\s*(?:[-*]|\d+[\).:]|[A-Za-z][\).:])\s*", "", line)
        if line and line.lower() not in {"nuggets:", "nugget list:"}:
            nuggets.append(line)
    return nuggets


def generate_nuggets(
    items: list[NuggetInput],
    generator: Any,
    system_prompt: str,
    user_template: str,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, list[str]]:
    item_id_to_nuggets: dict[str, list[str]] = {
        item.item_id: split_nuggets(item.text) for item in items if is_nugget_list(item.source)
    }

    pending = [item for item in items if item.item_id not in item_id_to_nuggets]
    for item in pending:
        if not item.text.strip():
            item_id_to_nuggets[item.item_id] = []

    pending = [item for item in pending if item.item_id not in item_id_to_nuggets]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        messages = make_messages(batch, system_prompt, user_template)
        outputs = generator(messages, max_new_tokens=max_new_tokens, batch_size=batch_size)
        for item, output in zip(batch, outputs):
            item_id_to_nuggets[item.item_id] = split_nuggets(get_generated_content(output))
        print(f"Nuggetized {min(start + batch_size, len(pending))}/{len(pending)} records")

    return item_id_to_nuggets


def seed_nuggets_from_items(items: list[NuggetInput]) -> dict[str, list[str]]:
    return {
        item.item_id: split_nuggets(item.text)
        for item in items
        if is_nugget_list(item.source)
    }


def items_requiring_generation(items: list[NuggetInput]) -> list[NuggetInput]:
    seeded = seed_nuggets_from_items(items)
    return [item for item in items if item.item_id not in seeded and item.text.strip()]


def build_answer_eval_nugget_output(
    items: list[NuggetInput], item_id_to_nuggets: dict[str, list[str]]
) -> dict[str, list[str]]:
    return {item.item_id: item_id_to_nuggets.get(item.item_id, []) for item in items}


def build_answer_eval_dataset_output(items: list[NuggetInput]) -> list[dict[str, Any]]:
    output = []
    for item in items:
        answer = item.answer or item.text
        responses = item.responses or normalize_responses(None, answer)
        output.append(
            {
                "metadata": {
                    "topic_id": item.item_id,
                    "question": item.question,
                    "answer": answer,
                },
                "responses": responses,
            }
        )
    return output


def build_augmented_output(data: Any, items: list[NuggetInput], item_id_to_nuggets: dict[str, list[str]], field_name: str) -> Any:
    if isinstance(data, dict):
        output = dict(data)
        for item in items:
            nuggets = item_id_to_nuggets.get(item.item_id, [])
            existing = output.get(item.item_id)
            if isinstance(existing, dict):
                existing = dict(existing)
                existing[field_name] = nuggets
                output[item.item_id] = existing
            else:
                output[item.item_id] = {field_name: nuggets, "text": item.text}
        return output

    output = []
    for item in items:
        if isinstance(item.source, dict):
            record = dict(item.source)
        else:
            record = {"id": item.item_id, "text": item.text}
        record[field_name] = item_id_to_nuggets.get(item.item_id, [])
        output.append(record)
    return output


def parse_csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate nugget JSON from several BioACE-style input formats. "
            "The default output is compatible with answer_eval.py --gt_nugget_path."
        )
    )
    parser.add_argument("input_path", nargs="?", help="Path to the input JSON file.")
    parser.add_argument("output_path", nargs="?", help="Path where the output JSON should be written.")
    parser.add_argument("--input", dest="input_path_flag", help="Path to the input JSON file.")
    parser.add_argument("--output", dest="output_path_flag", help="Path where the output JSON should be written.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "answer-eval", "task-a", "medaesqa", "abstract-map", "nugget-map", "flat"],
        default="auto",
        help="Input schema to read. Use auto unless field names are ambiguous.",
    )
    parser.add_argument(
        "--output-format",
        choices=["answer-eval-nuggets", "answer-eval-dataset", "augmented"],
        default="answer-eval-nuggets",
        help=(
            "answer-eval-nuggets writes {topic_id: [nuggets]} for answer_eval.py --gt_nugget_path; "
            "answer-eval-dataset writes records for answer_eval.py --dataset_path; "
            "augmented preserves the source and adds a nuggets field."
        ),
    )
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="Text-generation model id or path.")
    parser.add_argument("--hf-home", help="Optional Hugging Face cache directory.")
    parser.add_argument("--prompt-file", help="Optional JSON prompt file with system/user fields.")
    parser.add_argument("--prompt-index", type=int, default=0, help="Prompt index when --prompt-file is a list.")
    parser.add_argument("--batch-size", type=int, default=8, help="Generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens per generated nugget list.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map value.")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bfloat16", "float16", "float32", "none"],
        default="bfloat16",
        help="Torch dtype passed to the model.",
    )
    parser.add_argument("--padding-side", choices=["left", "right"], default="left", help="Tokenizer padding side.")
    parser.add_argument("--answer-key", help="Machine-generated answer key to use for medaesqa input, for example M1.")
    parser.add_argument("--id-fields", default="", help="Comma-separated extra id field names to try.")
    parser.add_argument("--question-fields", default="", help="Comma-separated extra question field names to try.")
    parser.add_argument("--answer-fields", default="", help="Comma-separated extra answer/text field names to try.")
    parser.add_argument("--title-fields", default="", help="Comma-separated extra title field names to try.")
    parser.add_argument("--abstract-fields", default="", help="Comma-separated extra abstract field names to try.")
    parser.add_argument("--nuggets-field", default="nuggets", help="Field name used by --output-format augmented.")
    parser.add_argument("--limit", type=int, help="Process only the first N normalized records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path_flag or args.input_path
    output_path = args.output_path_flag or args.output_path

    if not input_path or not output_path:
        print("Both input and output paths are required.", file=sys.stderr)
        sys.exit(2)

    data = load_json(input_path)
    items = extract_items(
        data=data,
        input_format=args.input_format,
        answer_key=args.answer_key,
        id_fields=parse_csv_list(args.id_fields),
        question_fields=parse_csv_list(args.question_fields),
        answer_fields=parse_csv_list(args.answer_fields),
        title_fields=parse_csv_list(args.title_fields),
        abstract_fields=parse_csv_list(args.abstract_fields),
    )

    if args.limit is not None:
        items = items[: args.limit]

    if args.output_format == "answer-eval-dataset":
        save_json(build_answer_eval_dataset_output(items), output_path)
        return

    if items_requiring_generation(items):
        system_prompt, user_template = load_prompt(args.prompt_file, args.prompt_index)
        generator = load_generation_pipeline(args)
        item_id_to_nuggets = generate_nuggets(
            items=items,
            generator=generator,
            system_prompt=system_prompt,
            user_template=user_template,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        item_id_to_nuggets = seed_nuggets_from_items(items)

    if args.output_format == "answer-eval-nuggets":
        output = build_answer_eval_nugget_output(items, item_id_to_nuggets)
    else:
        output = build_augmented_output(data, items, item_id_to_nuggets, args.nuggets_field)

    save_json(output, output_path)


if __name__ == "__main__":
    main()
