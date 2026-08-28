# Facet Evaluation Engine

A retrieval-augmented facet evaluation pipeline that identifies and evaluates behavioral and personality-related facets from conversational evidence using a local Large Language Model (LLM).

The system is designed around one core principle:

> A retrieved facet should only be scored when the conversation contains direct evidence supporting that specific facet. Otherwise, the system should abstain.

The project combines facet retrieval, taxonomy-aware filtering, observability and sensitivity handling, LLM-based scoring, structured output validation, and benchmark evaluation.

---

## Overview

Given a conversation such as:

> I enjoy taking risks and trying new experiences. I usually prefer adventurous choices over safe ones.

the system:

1. Retrieves potentially relevant facets.
2. Filters candidates using taxonomy and observability information.
3. Sends the conversation and candidate facets to a local LLM.
4. Evaluates each facet independently.
5. Scores a facet only when sufficient evidence exists.
6. Abstains when evidence is insufficient.
7. Validates the LLM output.
8. Evaluates the results against a reference benchmark.

The system is intentionally conservative and does not treat retrieval relevance as proof of evidence.

---

## Architecture

```text
                         Conversation
                              |
                              v
                    +-------------------+
                    | Facet Retrieval   |
                    +-------------------+
                              |
                              v
                    Candidate Facets
                              |
                              v
                    +-------------------+
                    | Taxonomy /        |
                    | Observability     |
                    | Filtering         |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Facet Scorer      |
                    | Qwen + Ollama     |
                    +-------------------+
                              |
                              v
                       Structured JSON
                              |
                              v
                    +-------------------+
                    | Validation &      |
                    | Abstention        |
                    +-------------------+
                              |
                              v
                       Final Results
                              |
                              v
                    Benchmark Evaluation
