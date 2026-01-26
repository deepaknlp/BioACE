import argparse
import json
import sys
from scipy.stats import kendalltau, spearmanr


def compute_stats_medaesqa(data):
    stats = [
        {
            "num_coverage": 0,
            "num_sentences": 0,
            "num_citations": 0,
            "num_support": 0,
            "num_contradict": 0,
            "citation_coverage": 0.0,
            "citation_support_rate": 0.0,
            "citation_contradict_rate": 0.0,
        }
        for _ in range(30)
    ]

    for question in data:
        if int(question["question_id"]) <= 155:
            for i in range(30):
                key = "M" + str(i + 1)
                m = question.get("machine_generated_answers", {}).get(key, {})
                answer_sentences = m.get("answer_sentences", [])

                for answer_sentence in answer_sentences:
                    stats[i]["num_sentences"] += 1
                    coverage = False

                    # labels might not be present
                    if "labels" in answer_sentence and isinstance(answer_sentence["labels"], dict):
                        for citation in answer_sentence["labels"].values():
                            stats[i]["num_citations"] += 1
                            lower = str(citation).lower()

                            if lower.startswith("support"):
                                stats[i]["num_support"] += 1
                                coverage = True
                            elif lower.startswith("contradict"):
                                stats[i]["num_contradict"] += 1
                                coverage = True

                    if coverage:
                        stats[i]["num_coverage"] += 1
        else:
            break

    # compute the rate fields
    for i in range(30):
        s = stats[i]

        s["citation_coverage"] = (
            s["num_coverage"] / s["num_sentences"]
            if s["num_sentences"] > 0
            else 0.0
        )

        if s["num_citations"] == 0:
            s["citation_support_rate"] = 0.0
            s["citation_contradict_rate"] = 0.0
        else:
            s["citation_support_rate"] = s["num_support"] / s["num_citations"]
            s["citation_contradict_rate"] = s["num_contradict"] / s["num_citations"]

    return stats


def compute_stats_task_a(data):
    auto_stats = None
    stats = {
        "num_coverage": 0,
        "num_sentences": 0,
        "num_citations": 0,
        "num_support": 0,
        "num_contradict": 0,
        "citation_coverage": 0.0,
        "citation_support_rate": 0.0,
        "citation_contradict_rate": 0.0,
    }

    for question in data:
        stats["num_sentences"] += len(question["answer"])

        for answer in question["answer"]:
            stats["num_citations"] += len(answer["supported_citations"])
            stats["num_support"] += len(answer["supported_citations"])

            stats["num_citations"] += len(answer["contradicted_citations"])
            stats["num_contradict"] += len(answer["contradicted_citations"])

            if len(answer["supported_citations"]) > 0:
                stats["num_coverage"] += 1

    stats["citation_coverage"] = (
        stats["num_coverage"] / stats["num_sentences"]
        if stats["num_sentences"] > 0
        else 0.0
    )

    if stats["num_citations"] == 0:
        stats["citation_support_rate"] = 0.0
        stats["citation_contradict_rate"] = 0.0

    else:
        stats["citation_support_rate"] = (
            stats["num_support"] / stats["num_citations"]
        )

        stats["citation_contradict_rate"] = (
            stats["num_contradict"] / stats["num_citations"]
        )

    # If automatic labels exist

    if "supported_citations_labels" in data[0]["answer"][0]:
        auto_stats = {
            "num_coverage": 0,
            "num_sentences": 0,
            "num_citations": 0,
            "num_support": 0,
            "num_contradict": 0,
            "citation_coverage": 0.0,
            "citation_support_rate": 0.0,
            "citation_contradict_rate": 0.0,
        }

        for question in data:
            auto_stats["num_sentences"] += len(question["answer"])

            for answer in question["answer"]:
                covered = False

                for val in answer["supported_citations_labels"].values():
                    auto_stats["num_citations"] += 1

                    if val.lower() == "supports":
                        if not covered:
                            covered = True

                        auto_stats["num_support"] += 1

                    elif val.lower() == "contradicts":
                        auto_stats["num_contradict"] += 1

                for val in answer["contradicted_citations_labels"].values():
                    auto_stats["num_citations"] += 1

                    if val.lower() == "supports":
                        if not covered:
                            covered = True

                        auto_stats["num_support"] += 1

                    elif val.lower() == "contradicts":
                        auto_stats["num_contradict"] += 1

                if covered:
                    auto_stats["num_coverage"] += 1

        auto_stats["citation_coverage"] = (
            auto_stats["num_coverage"] / auto_stats["num_sentences"]
            if auto_stats["num_sentences"] > 0
            else 0.0
        )

        if auto_stats["num_citations"] == 0:
            auto_stats["citation_support_rate"] = 0.0
            auto_stats["citation_contradict_rate"] = 0.0

        else:
            auto_stats["citation_support_rate"] = (
                auto_stats["num_support"] / auto_stats["num_citations"]
            )

            auto_stats["citation_contradict_rate"] = (
                auto_stats["num_contradict"] / auto_stats["num_citations"]
            )
    
    return stats, auto_stats


def extract_metric_array(stats, metric):
    return [stats[i].get(metric, 0.0) for i in range(30)]


def compute_correlations(a, b):
    tau, tau_p = kendalltau(a, b)
    rho, rho_p = spearmanr(a, b)
    return (tau, tau_p), (rho, rho_p)


def pretty_print_stats(prefix, stats, _):
    print(f"--- {prefix} ---")
    print("Model\tCitation Coverage\tSupport Rate\tContradict Rate")

    for i in range(30):
        m = stats[i]
        print(
            f"M{i+1}\t{m['citation_coverage']:.4f}\t\t"
            f"{m['citation_support_rate']:.4f}\t\t"
            f"{m['citation_contradict_rate']:.4f}"
        )

    print()

def pretty_print_stats_task_a(prefix, stats, auto_stats):
    print(f"--- {prefix} ---")
    print(f"{'Citation Coverage':<20}{'Support Rate':<15}{'Contradict Rate':<18}")
    print(
        f"{stats['citation_coverage']:<20.4f}"
        f"{stats['citation_support_rate']:<15.4f}"
        f"{stats['citation_contradict_rate']:<18.4f}"
    )
    print()

    if auto_stats is not None:
        print(f"--- Automatic Labels ---")
        print(f"{'Citation Coverage':<20}{'Support Rate':<15}{'Contradict Rate':<18}")
        print(
            f"{auto_stats['citation_coverage']:<20.4f}"
            f"{auto_stats['citation_support_rate']:<15.4f}"
            f"{auto_stats['citation_contradict_rate']:<18.4f}"
        )
        print()


def load_json_or_exit(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute citation statistics"
    )

    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or two JSON files (run A [run B])",
    )

    parser.add_argument(
        "--format",
        choices=["medaesqa", "task_a"],
        default="medaesqa",
        help="Input JSON format (default: medaesqa)",
    )

    args = parser.parse_args()

    if args.format == "task_a":
        if len(args.files) != 1:
            parser.error("For --format task_a, please provide exactly 1 input file.")

    else:
        if not (1 <= len(args.files) <= 2):
            parser.error("Please provide at least 1 and at most 2 input files.")

    # select stats function
    if args.format == "medaesqa":
        compute_fn = compute_stats_medaesqa
        pretty_fn = pretty_print_stats

    else:
        compute_fn = compute_stats_task_a
        pretty_fn = pretty_print_stats_task_a

    data_a = load_json_or_exit(args.files[0])
    stats_a, auto_stats = compute_fn(data_a)
    pretty_fn("Run A", stats_a, auto_stats)

    if args.format == "task_a":
        sys.exit(0)

    if len(args.files) == 1:
        sys.exit(0)

    data_b = load_json_or_exit(args.files[1])
    stats_b = compute_fn(data_b)
    pretty_fn("Run B", stats_b)

    cov_a = extract_metric_array(stats_a, "citation_coverage")
    cov_b = extract_metric_array(stats_b, "citation_coverage")
    sup_a = extract_metric_array(stats_a, "citation_support_rate")
    sup_b = extract_metric_array(stats_b, "citation_support_rate")
    con_a = extract_metric_array(stats_a, "citation_contradict_rate")
    con_b = extract_metric_array(stats_b, "citation_contradict_rate")

    for name, (arr_a, arr_b) in [
        ("Citation Coverage", (cov_a, cov_b)),
        ("Citation Support Rate", (sup_a, sup_b)),
        ("Citation Contradict Rate", (con_a, con_b)),
    ]:
        (tau, tau_p), (rho, rho_p) = compute_correlations(arr_a, arr_b)

        print(name)
        if tau != tau or rho != rho:
            print("  Warning: correlation undefined.")
            print(f"  Kendall's tau: {tau}, p-value: {tau_p}")
            print(f"  Spearman's rho: {rho}, p-value: {rho_p}")
        else:
            print(f"  Kendall's tau: {tau:.6f}, p-value: {tau_p:.6g}")
            print(f"  Spearman's rho: {rho:.6f}, p-value: {rho_p:.6g}")
        print()


if __name__ == "__main__":
    main()
