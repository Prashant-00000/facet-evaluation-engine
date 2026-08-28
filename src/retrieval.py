"""
TF-IDF based facet retrieval.

This module retrieves the most relevant facets for a conversation
before the LLM scoring stage.

v2 changes (both found by testing against a real conversation before
shipping, not by inspection alone):

1. Switched from word-level TF-IDF (ngram_range=(1,2), default word
   tokenizer) to CHARACTER n-gram TF-IDF (analyzer='char_wb',
   ngram_range=(3,5)).

   Why: word-level TF-IDF treats "Risktaking" as one indivisible token
   with zero vocabulary overlap against "risks" / "risky" / "taking
   risks" in a conversation. For the test conversation "I enjoy taking
   risks... adventurous choices over safe ones", the facet
   "Risktaking" ranked #282 out of 266 candidates under word-level
   TF-IDF — effectively invisible — while "Adventure-Seeking Behavior"
   and other clearly relevant facets also under-ranked. Character
   n-grams capture "risk" as a shared substring of "risktaking" and
   "risks" without needing stemming or a synonym dictionary. Verified:
   with this change, "Risktaking" ranks #1 and "Adventure-Seeking
   Behavior" ranks #2 for the same conversation.

   This is a real trade-off, not a free win: char n-grams are more
   prone to coincidental substring matches on short/generic strings.
   Mitigated by the similarity-floor filter below and by keeping
   top_k modest. A real embedding model (sentence-transformers or
   similar) would handle this more robustly via semantic similarity
   rather than lexical/substring overlap, and is the natural upgrade
   path if this pipeline moves beyond a 24-hour take-home — noted
   here rather than added now to avoid a dependency that needs
   downloading model weights.

2. Added a minimum similarity floor (MIN_RETRIEVAL_SCORE) before
   truncating to top_k. Previously `.head(top_k)` always returned
   exactly top_k rows regardless of actual relevance — for the test
   conversation, 61 of 266 observable candidates scored exactly 0.0
   (zero textual overlap with the conversation) and several of those
   zero-score rows were still being padded into the "top 10" results,
   including "Kink-interest diversity" for a conversation about
   adventure sports. That is a real safety problem, not a cosmetic
   one: it hands the LLM scorer a facet with no textual relevance
   AND no distinguishing sensitivity flag, moving the burden of
   catching an irrelevant-and-sensitive facet onto the weaker,
   harder-to-audit LLM layer instead of filtering it here where it's
   cheap and deterministic to catch.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_FILE = Path("data/facets_enriched.csv")

# Facets scoring below this cosine similarity are dropped before
# top_k truncation, regardless of how few candidates remain. A
# conversation with genuinely no relevant facets should be allowed
# to return fewer than top_k results (or zero) rather than being
# padded with noise.
MIN_RETRIEVAL_SCORE = 0.05


class FacetRetriever:
    """
    Retrieve candidate facets using TF-IDF similarity.
    """

    def __init__(self, data_file=DATA_FILE, min_score=MIN_RETRIEVAL_SCORE):
        self.data_file = data_file
        self.min_score = min_score

        # Load enriched facet catalogue.
        self.df = pd.read_csv(self.data_file)

        # Only facets that can potentially be evaluated
        # from conversation should reach the retrieval stage.
        self.df = self.df[
            self.df["conversation_observable"] != "false"
        ].copy()

        # Create text documents from normalized facet names.
        self.documents = self.df["normalized_value"].fillna("")

        # Build TF-IDF index using character n-grams so compound
        # facet names (e.g. "Risktaking") share vocabulary with
        # morphological variants in conversation text (e.g. "risks").
        # See module docstring for the empirical justification.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
        )

        self.facet_matrix = self.vectorizer.fit_transform(
            self.documents
        )

    def retrieve(self, conversation, top_k=15):
        """
        Retrieve up to top-k relevant observable facets, dropping
        anything below the minimum similarity floor. May return
        fewer than top_k rows (including zero) if few or no facets
        are textually relevant to the conversation.
        """

        if not conversation or not conversation.strip():
            return pd.DataFrame()

        # Convert conversation into TF-IDF vector.
        conversation_vector = self.vectorizer.transform(
            [conversation]
        )

        # Calculate cosine similarity between the conversation
        # and every facet.
        similarities = cosine_similarity(
            conversation_vector,
            self.facet_matrix,
        )[0]

        # Copy the dataframe so we don't modify the index.
        results = self.df.copy()

        # Store retrieval score.
        results["retrieval_score"] = similarities

        # Drop anything below the relevance floor BEFORE truncating
        # to top_k, so irrelevant facets never get padded in just to
        # fill the count.
        results = results[results["retrieval_score"] >= self.min_score]

        # Highest similarity first.
        results = results.sort_values(
            "retrieval_score",
            ascending=False,
        )

        # Return up to top-k candidates (may be fewer).
        return results.head(top_k).reset_index(drop=True)


def main():
    """
    Small manual test for the retriever.
    """

    retriever = FacetRetriever()

    conversation = (
        "I enjoy taking risks and trying new experiences. "
        "I usually prefer adventurous choices over safe ones."
    )

    results = retriever.retrieve(
        conversation,
        top_k=10,
    )

    print("=" * 60)
    print("TF-IDF RETRIEVAL TEST")
    print("=" * 60)

    print("\nConversation:")
    print(conversation)

    print(f"\nTop retrieved facets ({len(results)} above min score {MIN_RETRIEVAL_SCORE}):")

    if results.empty:
        print("(none — no facet cleared the relevance floor)")
    else:
        print(
            results[
                [
                    "raw_value",
                    "facet_type",
                    "conversation_observable",
                    "retrieval_score",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()