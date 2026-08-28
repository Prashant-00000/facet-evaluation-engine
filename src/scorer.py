"""
Batch facet scoring with defensive JSON validation.

The scorer sends all retrieved candidates in ONE LLM call.
Facet metadata such as sensitivity and observability is included
in the prompt so the model knows how cautiously each facet must
be evaluated.
"""

import json

from src.llm_client import LLMClient


SYSTEM_INSTRUCTIONS = """
You are a careful facet-evaluation system.

Evaluate ONLY the candidate facets provided to you.

Use ONLY explicit evidence from the conversation.

CRITICAL EVIDENCE RULE:
Evaluate every facet independently.

A facet being retrieved does NOT mean that it should be scored.

For a facet to receive a numeric score, the conversation
must contain a specific statement that directly supports
THAT EXACT facet.

Topical similarity, related words, or general context are
NOT sufficient evidence.

Before assigning a score, ask:
"What exact statement in the conversation supports this
specific facet?"

If you cannot identify a direct supporting statement,
ABSTAIN.

Do NOT assign a low score merely because the facet seems
vaguely related to the conversation.

The evidence field must contain the specific statement
that supports the facet. Do not use evidence that merely
describes the same general topic.

TONE AND SARCASM:
Consider whether statements may be sarcastic, ironic,
exaggerated, or joking.

Do not automatically interpret exaggerated positive
language as genuine positive evidence.

Example:
"I absolutely love being late to everything. It's my
favorite hobby."

may be sarcastic and should not automatically support
high enthusiasm, high-spiritedness, or engagement.

When sarcasm or irony makes the intended meaning uncertain,
prefer abstention rather than scoring from the literal
wording alone.

GENERAL RULES:
- Do not invent facts.
- Do not diagnose medical conditions.
- Do not infer information that was not stated.
- Do not use outside knowledge to fill missing evidence.
- Do not treat a related topic as evidence for a facet.
- You may abstain when evidence is insufficient.

SCORING:

score represents the apparent level of the facet:

1 = very low
2 = low
3 = moderate
4 = high
5 = very high

confidence is SEPARATE from score.

confidence is a number from 0 to 1 representing how confident
you are that the conversation supports the decision.

IMPORTANT:
A high confidence does NOT mean a high score.
A low score can have high confidence if the evidence clearly
shows low expression of the facet.

ABSTENTION:

If there is insufficient evidence:
- score must be null
- confidence should be low
- evidence must be null
- provide an abstention_reason

SENSITIVE FACETS:

For facets marked as high sensitivity:
- Only use explicit, clearly volunteered evidence.
- Never infer the attribute.
- Never diagnose or speculate.
- If the evidence is not explicit, abstain.

CLINICAL FACETS:

For clinical symptoms:
- Do not diagnose the person.
- Only score when the person explicitly describes sufficient
  conversational evidence.
- If the conversation does not establish the symptom clearly,
  abstain.

Return ONLY valid JSON.

Required structure:

{
  "results": [
    {
      "facet": "exact facet name",
      "score": 1,
      "confidence": 0.0,
      "evidence": "short evidence from conversation",
      "abstention_reason": null
    }
  ]
}
"""


class FacetScorer:
    """Score a batch of retrieved facets using one LLM call."""

    def __init__(self, client: LLMClient):
        self.client = client

    def build_prompt(self, conversation, facets):
        """
        Build one prompt containing the conversation and all
        candidate facets together with their metadata.
        """

        facet_blocks = []

        for i, facet in enumerate(facets, start=1):

            facet_name = facet["raw_value"]

            facet_type = facet.get(
                "facet_type",
                "unknown",
            )

            observability = facet.get(
                "conversation_observable",
                "unknown",
            )

            sensitivity = facet.get(
                "sensitivity",
                "unknown",
            )

            abstention_reason = facet.get(
                "abstention_reason",
                "",
            )

            block = f"""
FACET {i}
Name: {facet_name}
Type: {facet_type}
Conversation observable: {observability}
Sensitivity: {sensitivity}
Abstention guidance: {abstention_reason or "None"}
"""

            facet_blocks.append(block.strip())

        facet_text = "\n\n".join(facet_blocks)

        return f"""
{SYSTEM_INSTRUCTIONS}

CONVERSATION:
{conversation}

CANDIDATE FACETS:

{facet_text}

Evaluate EVERY candidate exactly once.

Use the metadata as guidance.

A facet being topically related to the conversation is NOT
sufficient evidence to assign a score.

If there is insufficient evidence, abstain.

For an abstention:
- score = null
- evidence = null
- provide a concise abstention_reason

For a scored facet:
- score must be 1 through 5
- evidence must identify the conversational evidence
- confidence must be between 0 and 1

Use the exact facet names provided.

Return ONLY JSON.
"""

    def validate_result(self, result, allowed_facets):
        """
        Validate one model result.

        Returns a cleaned result if valid.
        Returns None if invalid.
        """

        if not isinstance(result, dict):
            return None

        required_fields = {
            "facet",
            "score",
            "confidence",
            "evidence",
            "abstention_reason",
        }

        if not required_fields.issubset(result.keys()):
            return None

        facet = result["facet"]
        score = result["score"]
        confidence = result["confidence"]

        # --------------------------------------------------
        # Validate facet
        # --------------------------------------------------

        if facet not in allowed_facets:
            return None

        # --------------------------------------------------
        # Validate confidence
        # --------------------------------------------------

        if isinstance(confidence, bool):
            return None

        if not isinstance(
            confidence,
            (int, float),
        ):
            return None

        if not 0 <= confidence <= 1:
            return None

        # --------------------------------------------------
        # Validate score
        # --------------------------------------------------

        if score is not None:

            if isinstance(score, bool):
                return None

            if not isinstance(score, int):
                return None

            if score not in {1, 2, 3, 4, 5}:
                return None

        # --------------------------------------------------
        # Validate abstention
        # --------------------------------------------------

        if score is None:

            # Abstention must have a reason.
            if not result.get(
                "abstention_reason"
            ):
                return None

            # Abstention should not contain fabricated evidence.
            if result.get("evidence") not in (
                None,
                "",
            ):
                return None

        # --------------------------------------------------
        # Validate scored result
        # --------------------------------------------------

        else:

            # A scored facet must contain evidence.
            if not result.get("evidence"):
                return None

            # A scored result should not simultaneously claim
            # that there is insufficient evidence.
            if result.get("abstention_reason") not in (
                None,
                "",
            ):
                return None

        return {
            "facet": facet,
            "score": score,
            "confidence": confidence,
            "evidence": result.get(
                "evidence"
            ),
            "abstention_reason": result.get(
                "abstention_reason"
            ),
        }

    def parse_response(
        self,
        response,
        allowed_facets,
    ):
        """
        Parse and validate the complete LLM response.

        The model must return each candidate exactly once.
        """

        try:
            data = json.loads(response)

        except json.JSONDecodeError:
            return {
                "results": [],
                "error": "invalid_json",
            }

        if not isinstance(data, dict):
            return {
                "results": [],
                "error": "response_not_object",
            }

        results = data.get("results")

        if not isinstance(results, list):
            return {
                "results": [],
                "error": "results_not_list",
            }

        validated = []
        seen_facets = set()

        for result in results:

            cleaned = self.validate_result(
                result,
                allowed_facets,
            )

            if cleaned is None:
                continue

            facet = cleaned["facet"]

            # Reject duplicate results.
            if facet in seen_facets:
                continue

            seen_facets.add(facet)
            validated.append(cleaned)

        # --------------------------------------------------
        # Check for missing candidates
        # --------------------------------------------------

        missing_facets = sorted(
            allowed_facets - seen_facets
        )

        error = None

        if missing_facets:
            error = {
                "type": "missing_facets",
                "facets": missing_facets,
            }

        return {
            "results": validated,
            "error": error,
        }

    def score(self, conversation, facets):
        """
        Score all candidate facets in ONE LLM call.
        """

        if not conversation or not conversation.strip():
            return {
                "results": [],
                "error": "empty_conversation",
            }

        if not facets:
            return {
                "results": [],
                "error": "no_facets",
            }

        allowed_facets = {
            facet["raw_value"]
            for facet in facets
        }

        prompt = self.build_prompt(
            conversation,
            facets,
        )

        try:
            response = self.client.generate(
                prompt
            )

        except Exception as exc:
            return {
                "results": [],
                "error": f"llm_error: {exc}",
            }

        return self.parse_response(
            response,
            allowed_facets,
        )