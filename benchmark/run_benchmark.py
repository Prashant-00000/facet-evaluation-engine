"""
Safe and resumable benchmark runner.

Default mode:
    Uses MockLLMClient and is safe for local testing.

Real mode:
    python benchmark/run_benchmark.py --real

Real mode uses Ollama and should only be run on a machine
with sufficient resources.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow direct execution from the repository root.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from src.pipeline import FacetEvaluationPipeline
from src.llm_client import OllamaClient, MockLLMClient
from src.scorer import FacetScorer


INPUT_FILE = Path("benchmark/conversations.json")

MOCK_OUTPUT_FILE = Path(
    "benchmark/mock_results.json"
)

REAL_OUTPUT_FILE = Path(
    "benchmark/benchmark_results.json"
)

MODEL_NAME = "qwen2.5:7b-instruct"

TOP_K = 10

PAUSE_SECONDS = 5


def load_conversations():
    """Load benchmark conversations."""

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_previous_results(output_file):
    """Load existing results so the benchmark can resume."""

    if not output_file.exists():
        return []

    try:
        with open(
            output_file,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

    except (
        json.JSONDecodeError,
        OSError,
    ):
        print(
            "Warning: Could not read previous results."
        )

    return []


def save_results(results, output_file):
    """Save results immediately."""

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


def create_pipeline(use_real):
    """Create either a mock or real pipeline."""

    if use_real:

        print(
            "Using REAL Ollama model."
        )

        client = OllamaClient(
            model=MODEL_NAME
        )

    else:

        print(
            "Using MOCK LLM."
        )

        client = MockLLMClient()

    scorer = FacetScorer(client)

    return FacetEvaluationPipeline(
        scorer=scorer
    )


def main():

    parser = argparse.ArgumentParser(
        description="Run facet benchmark."
    )

    parser.add_argument(
        "--real",
        action="store_true",
        help="Use the real Ollama model.",
    )

    args = parser.parse_args()

    use_real = args.real

    if use_real:

        output_file = REAL_OUTPUT_FILE

    else:

        output_file = MOCK_OUTPUT_FILE

    print("=" * 60)
    print("FACET BENCHMARK")
    print("=" * 60)

    print(
        f"\nMode: "
        f"{'REAL' if use_real else 'MOCK'}"
    )

    print(
        f"Model: "
        f"{MODEL_NAME if use_real else 'MockLLM'}"
    )

    print(
        f"Output: "
        f"{output_file}"
    )

    conversations = load_conversations()

    previous_results = load_previous_results(
        output_file
    )

    completed_ids = {
        item["id"]
        for item in previous_results
        if "id" in item
    }

    print(
        f"\nTotal conversations: "
        f"{len(conversations)}"
    )

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Remaining: "
        f"{len(conversations) - len(completed_ids)}"
    )

    # --------------------------------------------------
    # Create pipeline
    # --------------------------------------------------

    pipeline = create_pipeline(
        use_real
    )

    # --------------------------------------------------
    # Run benchmark
    # --------------------------------------------------

    for index, item in enumerate(
        conversations,
        start=1,
    ):

        conversation_id = item["id"]

        if conversation_id in completed_ids:

            print(
                f"\n[{index}/{len(conversations)}] "
                f"Skipping {conversation_id} "
                f"(already completed)"
            )

            continue

        print("\n" + "-" * 60)

        print(
            f"Conversation "
            f"{index}/{len(conversations)}"
        )

        print(
            f"ID: {conversation_id}"
        )

        print(
            f"\n{item['conversation']}"
        )

        try:

            result = pipeline.evaluate(
                item["conversation"],
                top_k=TOP_K,
            )

            output = {
                "id": conversation_id,
                "conversation": item[
                    "conversation"
                ],
                "description": item[
                    "description"
                ],
                "retrieved_facets": result[
                    "retrieved_facets"
                ],
                "results": result[
                    "results"
                ],
                "error": result[
                    "error"
                ],
            }

            # --------------------------------------------------
            # Save immediately.
            # --------------------------------------------------

            previous_results.append(
                output
            )

            save_results(
                previous_results,
                output_file,
            )

            completed_ids.add(
                conversation_id
            )

            print("\nResults:")

            if result["results"]:

                for score in result["results"]:

                    print(
                        f"- {score['facet']}: "
                        f"{score['score']} "
                        f"(confidence="
                        f"{score['confidence']})"
                    )

            else:

                print("(none)")

            print(
                f"\nError: "
                f"{result['error']}"
            )

            print(
                f"\nSaved to: "
                f"{output_file}"
            )

        except KeyboardInterrupt:

            print(
                "\n\nBenchmark interrupted."
            )

            print(
                "Completed results "
                "have already been saved."
            )

            break

        except Exception as exc:

            print(
                f"\nERROR processing "
                f"{conversation_id}:"
            )

            print(exc)

            error_output = {
                "id": conversation_id,
                "conversation": item[
                    "conversation"
                ],
                "description": item[
                    "description"
                ],
                "retrieved_facets": [],
                "results": [],
                "error": str(exc),
            }

            previous_results.append(
                error_output
            )

            save_results(
                previous_results,
                output_file,
            )

        # --------------------------------------------------
        # Small pause between conversations.
        # --------------------------------------------------

        if index < len(conversations):

            print(
                f"\nWaiting "
                f"{PAUSE_SECONDS} seconds..."
            )

            time.sleep(
                PAUSE_SECONDS
            )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BENCHMARK STATUS")
    print("=" * 60)

    print(
        f"\nSaved results: "
        f"{len(previous_results)}"
    )

    print(
        f"Total conversations: "
        f"{len(conversations)}"
    )

    if len(previous_results) >= len(
        conversations
    ):

        print(
            "\nBENCHMARK COMPLETE"
        )

    else:

        print(
            "\nBENCHMARK PARTIALLY COMPLETE"
        )

        print(
            "Run the same command again "
            "to resume."
        )

    print(
        f"\nResults file: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()