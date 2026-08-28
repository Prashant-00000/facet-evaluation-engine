"""
End-to-end facet evaluation pipeline.

Flow:

Conversation
    ↓
TF-IDF retrieval
    ↓
Observable candidate facets
    ↓
Batch LLM scoring
    ↓
Validated results
"""

import os

from src.retrieval import FacetRetriever
from src.scorer import FacetScorer
from src.llm_client import MockLLMClient, OllamaClient


class FacetEvaluationPipeline:
    """
    Orchestrates retrieval and LLM scoring.
    """

    def __init__(
        self,
        retriever=None,
        scorer=None,
        use_real_llm=False,
    ):
        self.retriever = retriever or FacetRetriever()

        if scorer is not None:
            self.scorer = scorer

        elif use_real_llm:
            model = os.getenv(
                "OLLAMA_MODEL",
                "qwen2.5:7b-instruct",
            )

            self.scorer = FacetScorer(
                OllamaClient(model=model)
            )

        else:
            self.scorer = FacetScorer(
                MockLLMClient()
            )

    def evaluate(self, conversation, top_k=15):
        """
        Retrieve relevant facets and score them.
        """

        # --------------------------------------------------
        # Step 1: Retrieve candidates
        # --------------------------------------------------

        retrieved = self.retriever.retrieve(
            conversation,
            top_k=top_k,
        )

        if retrieved.empty:
            return {
                "conversation": conversation,
                "retrieved_facets": [],
                "results": [],
                "error": "no_relevant_facets",
            }

        # --------------------------------------------------
        # Step 2: Convert candidates to dictionaries
        # --------------------------------------------------

        facets = retrieved.to_dict(
            orient="records"
        )

        # --------------------------------------------------
        # Step 3: One batch LLM call
        # --------------------------------------------------

        scoring_result = self.scorer.score(
            conversation,
            facets,
        )

        # --------------------------------------------------
        # Step 4: Return final pipeline result
        # --------------------------------------------------

        return {
            "conversation": conversation,

            "retrieved_facets": [
                {
                    "facet": facet["raw_value"],
                    "facet_type": facet["facet_type"],
                    "retrieval_score": facet["retrieval_score"],
                }
                for facet in facets
            ],

            "results": scoring_result["results"],

            "error": scoring_result["error"],
        }


def main():
    """
    Manual pipeline test.

    By default this uses the Mock LLM.

    Set USE_REAL_LLM=true to use Ollama.
    """

    conversation = (
        "I enjoy taking risks and trying new experiences. "
        "I usually prefer adventurous choices over safe ones."
    )

    use_real_llm = (
        os.getenv("USE_REAL_LLM", "false").lower()
        == "true"
    )

    pipeline = FacetEvaluationPipeline(
        use_real_llm=use_real_llm
    )

    result = pipeline.evaluate(
        conversation,
        top_k=10,
    )

    print("=" * 60)
    print("FACET EVALUATION PIPELINE")
    print("=" * 60)

    print("\nLLM mode:")
    print("Ollama" if use_real_llm else "Mock")

    print("\nConversation:")
    print(result["conversation"])

    print("\nRetrieved facets:")

    for facet in result["retrieved_facets"]:
        print(
            f"- {facet['facet']} "
            f"(score={facet['retrieval_score']:.3f})"
        )

    print("\nLLM results:")

    if result["results"]:
        for item in result["results"]:
            print(item)
    else:
        print("(none)")

    print("\nPipeline error:")
    print(result["error"])


if __name__ == "__main__":
    main()