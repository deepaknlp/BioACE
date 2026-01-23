import argparse
import json
import sys

from scipy.stats import kendalltau, spearmanr


def compute_stats(data):
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

                            if isinstance(citation, str):
                                lower = citation.lower()

                                if lower.startswith("support"):
                                    stats[i]["num_support"] += 1
                                    coverage = True

                                elif lower.startswith("contradict"):
                                    stats[i]["num_contradict"] += 1
                                    coverage = True

                            else:
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

        if s["num_sentences"] == 0:
            s["citation_coverage"] = 0.0

        else:
            s["citation_coverage"] = s["num_coverage"] / s["num_sentences"]

        if s["num_citations"] == 0:
            s["citation_support_rate"] = 0.0
            s["citation_contradict_rate"] = 0.0

        else:
            s["citation_support_rate"] = s["num_support"] / s["num_citations"]
            s["citation_contradict_rate"] = s["num_contradict"] / s["num_citations"]

    return stats


def extract_metric_array(stats, metric):
    """Return list of metric values in order M1..M30."""
    return [stats[i].get(metric, 0.0) for i in range(30)]


def compute_correlations(a, b):

    tau, tau_p = kendalltau(a, b)
    rho, rho_p = spearmanr(a, b)
    return (tau, tau_p), (rho, rho_p)


def pretty_print_stats(prefix, stats):
    print(f"--- {prefix} ---")
    print("Model\tCitation Coverage\tSupport Rate\tContradict Rate")
    for i in range(30):
        m = stats[i]
        print(
            f"M{i+1}\t{m['citation_coverage']:.4f}\t\t{m['citation_support_rate']:.4f}\t\t{m['citation_contradict_rate']:.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute citation coverage/support/contradict rates correlations."
    )
    parser.add_argument("file_a", help="Path to first JSON file (run A)")
    parser.add_argument("file_b", help="Path to second JSON file (run B)")
    args = parser.parse_args()

    # load files
    try:
        with open(args.file_a, "r", encoding="utf-8") as f:
            data_a = json.load(f)

    except Exception as e:
        print(f"Failed to load {args.file_a}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.file_b, "r", encoding="utf-8") as f:
            data_b = json.load(f)

    except Exception as e:
        print(f"Failed to load {args.file_b}: {e}", file=sys.stderr)
        sys.exit(1)

    stats_a = compute_stats(data_a)
    stats_b = compute_stats(data_b)

    pretty_print_stats("Run A", stats_a)
    pretty_print_stats("Run B", stats_b)

    # extract arrays for comparisons
    cov_a = extract_metric_array(stats_a, "citation_coverage")
    cov_b = extract_metric_array(stats_b, "citation_coverage")

    sup_a = extract_metric_array(stats_a, "citation_support_rate")
    sup_b = extract_metric_array(stats_b, "citation_support_rate")

    con_a = extract_metric_array(stats_a, "citation_contradict_rate")
    con_b = extract_metric_array(stats_b, "citation_contradict_rate")

    # compute & print correlations for each metric
    for name, (arr_a, arr_b) in [
        ("Citation Coverage", (cov_a, cov_b)),
        ("Citation Support Rate", (sup_a, sup_b)),
        ("Citation Contradict Rate", (con_a, con_b)),
    ]:
        (tau, tau_p), (rho, rho_p) = compute_correlations(arr_a, arr_b)

        print(name)
        if tau != tau or rho != rho:  # check for NaN
            print("  Warning: correlation undefined (constant or identical arrays).")
            print(f"  Kendall's tau: {tau}, p-value: {tau_p}")
            print(f"  Spearman's rho: {rho}, p-value: {rho_p}")
        else:
            print(f"  Kendall's tau: {tau:.6f}, p-value: {tau_p:.6g}")
            print(f"  Spearman's rho: {rho:.6f}, p-value: {rho_p:.6g}")
        print()

    # exit normally
    sys.exit(0)


if __name__ == "__main__":
    main()
