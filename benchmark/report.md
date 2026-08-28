# Facet Evaluation Engine — Benchmark Report

## 1. Objective

The benchmark evaluates whether the facet evaluation pipeline can:

1. Retrieve relevant facets from a conversation.
2. Score facets when sufficient conversational evidence exists.
3. Abstain when evidence is insufficient.
4. Avoid unsupported medical or sensitive inferences.
5. Handle ambiguous, contradictory, and sarcastic statements.

The benchmark contains 10 manually designed conversations covering these cases.

---

## 2. Benchmark Conversations

| ID | Test case | Main purpose |
|---|---|---|
| `clear_risk` | Clear positive evidence | Test direct facet scoring |
| `planning` | Planning behavior | Test retrieval of organizational facets |
| `helping` | Helping others | Test interpersonal evidence |
| `insufficient` | Weak evidence | Test abstention |
| `medical_trap` | Sleep problems | Test medical abstention |
| `sensitive_trap` | Private relationships | Test sensitive-topic handling |
| `skill_evidence` | Python development | Test skill-related evidence |
| `behavioral_count_trap` | Coffee consumption | Test missing quantitative evidence |
| `sarcasm` | Sarcastic statement | Test unreliable literal interpretation |
| `contradiction` | Mixed risk preferences | Test contradictory evidence |

---

## 3. Retrieval

The retrieval stage uses character n-gram TF-IDF:

```text
analyzer = "char_wb"
ngram_range = (3, 5)