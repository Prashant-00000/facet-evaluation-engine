import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import MockLLMClient
from src.scorer import FacetScorer


class FakeClient:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        return self.response


FACETS = [
    {"raw_value": "Risktaking"},
    {"raw_value": "Compassion"},
]


def run_test(name, response):
    scorer = FacetScorer(FakeClient(response))

    result = scorer.score(
        "I enjoy taking risks.",
        FACETS,
    )

    print(f"\n{name}")
    print(result)


def main():

    # 1. Invalid JSON
    run_test(
        "TEST 1 - Invalid JSON",
        "this is not json",
    )

    # 2. Invalid score
    run_test(
        "TEST 2 - Invalid score",
        """
        {
            "results": [
                {
                    "facet": "Risktaking",
                    "score": 7,
                    "confidence": 0.9,
                    "evidence": "I enjoy taking risks.",
                    "abstention_reason": null
                }
            ]
        }
        """,
    )

    # 3. Invalid confidence
    run_test(
        "TEST 3 - Invalid confidence",
        """
        {
            "results": [
                {
                    "facet": "Risktaking",
                    "score": 5,
                    "confidence": 2,
                    "evidence": "I enjoy taking risks.",
                    "abstention_reason": null
                }
            ]
        }
        """,
    )

    # 4. Invented facet
    run_test(
        "TEST 4 - Invented facet",
        """
        {
            "results": [
                {
                    "facet": "IQ",
                    "score": 5,
                    "confidence": 0.9,
                    "evidence": "Some evidence.",
                    "abstention_reason": null
                }
            ]
        }
        """,
    )

    # 5. Missing evidence
    run_test(
        "TEST 5 - Missing evidence",
        """
        {
            "results": [
                {
                    "facet": "Risktaking",
                    "score": 5,
                    "confidence": 0.9,
                    "evidence": "",
                    "abstention_reason": null
                }
            ]
        }
        """,
    )

    # 6. Valid abstention
    run_test(
        "TEST 6 - Valid abstention",
        """
        {
            "results": [
                {
                    "facet": "Risktaking",
                    "score": null,
                    "confidence": 0.1,
                    "evidence": null,
                    "abstention_reason": "Insufficient conversational evidence."
                },
                {
                    "facet": "Compassion",
                    "score": null,
                    "confidence": 0.1,
                    "evidence": null,
                    "abstention_reason": "Insufficient conversational evidence."
                }
            ]
        }
        """,
    )


if __name__ == "__main__":
    main()