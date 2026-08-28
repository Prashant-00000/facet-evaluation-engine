import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import MockLLMClient
from src.scorer import FacetScorer


def main():
    client = MockLLMClient()
    scorer = FacetScorer(client)

    conversation = (
        "I enjoy taking risks and trying new experiences."
    )

    facets = [
        {
            "raw_value": "Risktaking",
        },
        {
            "raw_value": "Compassion",
        },
    ]

    result = scorer.score(
        conversation,
        facets,
    )

    print(result)


if __name__ == "__main__":
    main()