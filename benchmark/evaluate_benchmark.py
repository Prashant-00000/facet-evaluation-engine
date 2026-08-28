"""
Evaluate benchmark results against human-reviewed reference labels.

Usage:

    python benchmark/evaluate_benchmark.py --results benchmark/mock_results.json

or:

    python benchmark/evaluate_benchmark.py --results benchmark/benchmark_results.json
"""

import argparse
import json
from pathlib import Path


REFERENCE_FILE = Path(
    "benchmark/reference_labels.json"
)


def load_json(path):
    """Load JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():

    # --------------------------------------------------
    # Command-line argument
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Evaluate facet benchmark results."
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Path to benchmark results JSON.",
    )

    args = parser.parse_args()

    results_file = Path(args.results)

    # --------------------------------------------------
    # Load files
    # --------------------------------------------------

    if not results_file.exists():

        print(
            f"ERROR: Results file not found: "
            f"{results_file}"
        )

        return

    if not REFERENCE_FILE.exists():

        print(
            f"ERROR: Reference file not found: "
            f"{REFERENCE_FILE}"
        )

        return

    results = load_json(
        results_file
    )

    references = load_json(
        REFERENCE_FILE
    )

    # --------------------------------------------------
    # Index actual results by conversation ID
    # --------------------------------------------------

    result_by_id = {
        item["id"]: item
        for item in results
        if "id" in item
    }

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total_expected = 0
    correct_facets = 0
    correct_scores = 0

    total_abstentions = 0
    correct_abstentions = 0

    unexpected_facets = []

    missing_conversations = []

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    print("=" * 60)
    print("FACET BENCHMARK EVALUATION")
    print("=" * 60)

    print(
        f"\nResults file: {results_file}"
    )

    print(
        f"Reference file: {REFERENCE_FILE}"
    )

    print(
        f"\nReference conversations: "
        f"{len(references)}"
    )

    print(
        f"Result conversations: "
        f"{len(result_by_id)}"
    )

    # --------------------------------------------------
    # Evaluate every reference conversation
    # --------------------------------------------------

    for conversation_id, reference in references.items():

        print("\n" + "-" * 60)
        print(
            f"Conversation: "
            f"{conversation_id}"
        )

        if conversation_id not in result_by_id:

            print(
                "MISSING RESULT"
            )

            missing_conversations.append(
                conversation_id
            )

            continue

        actual = result_by_id[
            conversation_id
        ]

        actual_results = actual.get(
            "results",
            []
        )

        actual_by_facet = {
            item["facet"]: item
            for item in actual_results
            if "facet" in item
        }

        # --------------------------------------------------
        # Expected scored facets
        # --------------------------------------------------

        for expected in reference.get(
            "expected",
            []
        ):

            facet = expected[
                "facet"
            ]

            expected_score = expected.get(
                "score"
            )

            total_expected += 1

            actual_item = (
                actual_by_facet.get(
                    facet
                )
            )

            if actual_item is None:

                print(
                    f"MISS: {facet}"
                )

                continue

            actual_score = actual_item.get(
                "score"
            )

            if actual_score is not None:

                correct_facets += 1

            if actual_score == expected_score:

                correct_scores += 1

                print(
                    f"CORRECT: {facet} "
                    f"score={actual_score}"
                )

            else:

                print(
                    f"WRONG SCORE: {facet} "
                    f"expected={expected_score}, "
                    f"actual={actual_score}"
                )

        # --------------------------------------------------
        # Expected abstentions
        # --------------------------------------------------

        for facet in reference.get(
            "abstain",
            []
        ):

            total_abstentions += 1

            actual_item = (
                actual_by_facet.get(
                    facet
                )
            )

            # Facet returned but correctly abstained.
            if (
                actual_item is not None
                and actual_item.get("score")
                is None
            ):

                correct_abstentions += 1

                print(
                    f"CORRECT ABSTENTION: "
                    f"{facet}"
                )

            # Facet wasn't returned at all.
            # This is also safe abstention behaviour.
            elif actual_item is None:

                correct_abstentions += 1

                print(
                    f"FILTERED/ABSTAINED: "
                    f"{facet}"
                )

            else:

                print(
                    f"INCORRECTLY SCORED: "
                    f"{facet} → "
                    f"{actual_item.get('score')}"
                )

        # --------------------------------------------------
        # Detect unexpected scored facets
        # --------------------------------------------------

        expected_names = {
            item["facet"]
            for item in reference.get(
                "expected",
                []
            )
        }

        abstention_names = set(
            reference.get(
                "abstain",
                []
            )
        )

        allowed_names = (
            expected_names
            | abstention_names
        )

        for actual_item in actual_results:

            facet = actual_item.get(
                "facet"
            )

            score = actual_item.get(
                "score"
            )

            if (
                score is not None
                and facet not in allowed_names
            ):

                unexpected_facets.append(
                    {
                        "conversation":
                            conversation_id,
                        "facet":
                            facet,
                        "score":
                            score,
                    }
                )

                print(
                    f"UNEXPECTED SCORE: "
                    f"{facet} → {score}"
                )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"\nExpected scored facets: "
        f"{total_expected}"
    )

    print(
        f"Correctly detected: "
        f"{correct_facets}"
    )

    print(
        f"Correct scores: "
        f"{correct_scores}"
    )

    print(
        f"\nExpected abstentions: "
        f"{total_abstentions}"
    )

    print(
        f"Correct abstentions: "
        f"{correct_abstentions}"
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    if total_expected > 0:

        detection_rate = (
            correct_facets
            / total_expected
        )

        score_accuracy = (
            correct_scores
            / total_expected
        )

        print(
            f"\nFacet detection rate: "
            f"{detection_rate:.2%}"
        )

        print(
            f"Score agreement: "
            f"{score_accuracy:.2%}"
        )

    else:

        print(
            "\nFacet detection rate: N/A"
        )

        print(
            "Score agreement: N/A"
        )

    if total_abstentions > 0:

        abstention_accuracy = (
            correct_abstentions
            / total_abstentions
        )

        print(
            f"Abstention accuracy: "
            f"{abstention_accuracy:.2%}"
        )

    else:

        print(
            "Abstention accuracy: N/A"
        )

    print(
        f"\nUnexpected scored facets: "
        f"{len(unexpected_facets)}"
    )

    print(
        f"Missing conversations: "
        f"{len(missing_conversations)}"
    )

    if missing_conversations:

        for conversation_id in (
            missing_conversations
        ):

            print(
                f"- {conversation_id}"
            )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()