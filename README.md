# Facet Evaluation Engine

A retrieval-augmented facet evaluation pipeline that identifies and evaluates behavioral and personality-related facets from conversational evidence using a local Large Language Model (LLM).

The system is designed around one core principle:

> A retrieved facet should only be scored when the conversation contains direct evidence supporting that specific facet. Otherwise, the system should abstain.

---

## 1. Overview

Given a conversation such as:

> I enjoy taking risks and trying new experiences. I usually prefer adventurous choices over safe ones.

the system:

1. Retrieves potentially relevant facets.
2. Filters candidates using taxonomy and observability information.
3. Sends the conversation and candidate facets to an LLM.
4. Evaluates every facet independently.
5. Scores a facet only when sufficient evidence exists.
6. Abstains when evidence is insufficient.
7. Validates the LLM output.
8. Evaluates the results against reference labels.

The system intentionally separates:

```
Retrieved facet
      ≠
Supported facet
```

Retrieval identifies candidates; conversational evidence determines whether a facet should actually be scored.

---

## 2. Architecture

```
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
```

---

## 3. Core Design Principle — Evidence-First Scoring

A facet is scored only when the conversation contains a specific statement directly supporting that exact facet.

**Example — sufficient evidence:**

Conversation: `"I enjoy taking risks and trying new experiences."`
Facet: `Risktaking`

```json
{
  "score": 4,
  "evidence": "I enjoy taking risks and trying new experiences."
}
```

**Example — insufficient evidence:**

Facet: `Cooking and Culinary Arts`

```json
{
  "score": null,
  "evidence": null,
  "abstention_reason": "No explicit evidence of cooking or culinary arts."
}
```

The fact that a facet was retrieved does not mean that the facet should be scored. The scorer is explicitly instructed to ask:

> What exact statement in the conversation supports this specific facet?

If no direct supporting statement exists, the model abstains.

---

## 4. Structured Output

**Scored facet:**

```json
{
  "facet": "Risktaking",
  "score": 4,
  "confidence": 0.9,
  "evidence": "I enjoy taking risks and trying new experiences.",
  "abstention_reason": null
}
```

**Abstained facet:**

```json
{
  "facet": "Cooking and Culinary Arts",
  "score": null,
  "confidence": 0.2,
  "evidence": null,
  "abstention_reason": "No explicit evidence of cooking or culinary arts."
}
```

For abstentions, `evidence` stays `null` and the explanation lives only in `abstention_reason` — this keeps evidence and uncertainty clearly separated.

---

## 5. Sensitive Facet Handling

The taxonomy contains sensitive and clinical facets, including:

- Depression Symptoms
- Depression (DEP)
- Sleep Apnea
- Sleep-disorder diagnosis
- Other medical and biological facets

The system is designed **not** to diagnose or infer sensitive medical conditions from casual conversational evidence.

For example:

> "I've been feeling exhausted recently and I haven't been sleeping very well. I wake up several times during the night."

should **not** automatically result in `Sleep Apnea → scored`. Instead, the system abstains when the conversation does not provide appropriate evidence.

The benchmark includes a dedicated medical/sensitive trap to test this behavior.

---

## 6. Sarcasm and Tone Handling

The scoring prompt explicitly considers sarcasm, irony, exaggeration, joking, and ambiguous tone.

For example:

> "Oh yes, I absolutely love being late to everything. It's my favorite hobby."

should not automatically be interpreted as genuine positive evidence. When tone makes the intended meaning uncertain, the system prefers abstention over scoring from literal wording alone.

This fixed an earlier benchmark failure where sarcastic wording was interpreted literally.

---

## 7. Pluggable LLM Design

The LLM interface is separated from the scoring logic:

```
LLMClient
   |
   +-- MockLLMClient   (deterministic, offline, for testing)
   |
   +-- OllamaClient    (real local inference)
```

- **MockLLMClient** — used for unit/regression/pipeline testing without requiring a running LLM. Not used for the final benchmark.
- **OllamaClient** — communicates with the locally running `qwen2.5:7b-instruct`. Used for the real benchmark.

This separation makes the core scoring logic testable independently of the real model.

---

## 8. Output Validation

The validation layer checks:

- Valid JSON
- Correct response structure
- Required `results` field
- Allowed facet names
- Valid score values
- Valid confidence values
- Evidence requirements
- Abstention consistency
- Duplicate facets
- Missing candidate facets

A scored facet without evidence is rejected — this provides a deterministic safety layer around the LLM.

---

## 9. Setup

### Requirements

- Python 3.10+
- Ollama
- Qwen 2.5 7B Instruct
- macOS, Linux, or Windows

The final benchmark was run on: Apple M2, 8 GB RAM, Python 3.13.6, Ollama 0.33.1, `qwen2.5:7b-instruct`.

### Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

If `requirements.txt` is not present:

```bash
python3 -m pip install pandas numpy requests
```

---

## 10. Ollama Setup

```bash
ollama --version
ollama pull qwen2.5:7b-instruct
ollama list
```

Expected model in the list: `qwen2.5:7b-instruct`

Test the model:

```bash
ollama run qwen2.5:7b-instruct
```

Then type `hello`, and exit with `/bye`.

Ollama normally runs its local server automatically. If you run `ollama serve` and get `bind: address already in use`, an Ollama server is already running — this is expected and fine.

---

## 11. Running the Pipeline

**macOS / Linux:**

```bash
export USE_REAL_LLM=true
python3 -m src.pipeline
```

**Windows PowerShell:**

```powershell
$env:USE_REAL_LLM="true"
python -m src.pipeline
```

The pipeline displays: the input conversation, retrieved facets and their retrieval scores, and the LLM results (score, confidence, evidence, abstention reason) for each facet.

---

## 12. Mock Mode

The project includes a deterministic mock LLM client, used for unit tests, regression tests, local development, and testing without Ollama running. **The mock client is not used when reporting the final benchmark results** — the final benchmark uses the real Qwen model through Ollama.

---

## 13. Running Tests

**Scorer test:**

```bash
python3 src/test_scorer.py
```

Expected final output: `TEST SCORER: PASS`. Uses `MockLLMClient`.

**Validation tests:**

```bash
python3 src/test_validation.py
```

Covers: invalid JSON, invalid scores, invalid confidence, invented facets, missing evidence, valid abstention.

---

## 14. Benchmark

The project includes a 10-conversation benchmark testing different evidence conditions:

- Clear positive evidence
- Planning
- Helping behavior
- Insufficient evidence
- Medical/sensitive trap
- Sensitive/private topics
- Skill evidence
- Behavioral count trap
- Sarcasm
- Contradictory statements

---

## 15. Running the Real Qwen Benchmark

```bash
ollama list
export USE_REAL_LLM=true
python3 benchmark/run_benchmark.py --real
```

Processes all 10 conversations. Results are saved to `benchmark/benchmark_results.json`.

---

## 16. Evaluating Benchmark Results

```bash
python3 benchmark/evaluate_benchmark.py \
    --results benchmark/benchmark_results.json
```

Compares `benchmark_results.json` against `benchmark/reference_labels.json` and reports: facet detection rate, score agreement, abstention accuracy, unexpected scored facets, and missing conversations.

---

## 17. Benchmark Results

Final real-Qwen benchmark — Model: `qwen2.5:7b-instruct`, Conversations: 10

| Metric | Result |
|---|---|
| Facet Detection Rate | 100.00% |
| Score Agreement | 66.67% |
| Abstention Accuracy | 92.59% |
| Unexpected Scored Facets | 2 |
| Missing Conversations | 0 |

Detailed evaluation:

```
Expected scored facets: 3
Correctly detected: 3
Correct scores: 2

Expected abstentions: 27
Correct abstentions: 25

Facet detection rate: 100.00%
Score agreement: 66.67%
Abstention accuracy: 92.59%

Unexpected scored facets: 2
Missing conversations: 0
```

---

## 18. Before vs After

| Metric | Before | After |
|---|---|---|
| Abstention accuracy | 85.19% | 92.59% |
| Unexpected scored facets | 9 | 2 |

After strengthening facet-specific evidence requirements and adding explicit sarcasm/tone handling, the main improvement was a substantial reduction in unsupported facet scoring.

---

## 19. Benchmark Findings

### Medical Trap

Input: *"I've been feeling exhausted recently and I haven't been sleeping very well. I wake up several times during the night."*

The earlier system incorrectly assigned unsupported scores to unrelated facets such as Perseverance, Bravery, Stress Recovery Ability, General Mood and Attitude, Sleep-environment temperature, and Engagement. After strengthening the facet-specific evidence requirement, these unrelated facets were abstained, and the sensitive/clinical facets were handled conservatively. This was one of the most important improvements in the final system.

### Sarcasm

Input: *"Oh yes, I absolutely love being late to everything. It's my favorite hobby."*

The earlier system interpreted this literally and produced unsupported positive scores. After adding explicit sarcasm and tone instructions, the final benchmark correctly abstained on the tested facets.

### Contradictory Statements

Input: *"I usually avoid taking risks because I prefer safe choices. But sometimes I deliberately try something completely new when I get bored."*

This remains a known limitation. The model produced `Risktaking → 2`, `Safety compliance → 4`, `Boredom Susceptibility → 3`, while the reference expected different treatment. The model can prioritize one side of conflicting evidence rather than fully reconciling both signals.

### Planning

Input: *"Before I start a project, I make a detailed plan. I usually organize the tasks and decide what I need to finish first."*

The model assigned `Organized lifestyle → 4`, while the reference benchmark treats this differently. This may indicate a difference between observable evidence, the intended facet definition, and the reference label. The implementation follows the supplied reference labels for evaluation rather than silently modifying them.

---

## 20. Known Limitations

**20.1 Self-Contradictory Statements** — The model can prioritize one side of conflicting evidence instead of reconciling both statements, representing uncertainty, or abstaining. This is the main remaining reasoning limitation identified by the benchmark.

**20.2 Indirect Evidence** — The `clear_risk` conversation still produced `Safety compliance → 2`. The language about preferring adventurous over safe choices is semantically related to safety but not necessarily direct evidence of a stable Safety compliance trait. A smaller residual limitation.

**20.3 Reference Label Ambiguity** — Some reference labels may not perfectly align with observable evidence. The benchmark is treated as the evaluation target, and potential label/facet-definition mismatches are documented rather than silently changed.

**20.4 Small Local Model** — `qwen2.5:7b-instruct` is practical for local execution but has limitations in complex reasoning, contradiction handling, sarcasm interpretation, fine-grained semantic distinctions, and consistent instruction following.

---

## 21. Design Decisions

**Evidence Before Score** — The system prioritizes evidence over forced scoring: insufficient evidence → abstain → transparent result, rather than weak association → guess a score → unsupported inference.

**Facet-Specific Evidence** — The scorer evaluates each facet independently; retrieval relevance alone is insufficient. For every candidate, the model must determine whether the conversation contains direct evidence for that exact facet.

**Evidence Gate** — The validation layer rejects a numeric score with no evidence. This catches cases where the LLM assigns a score without supporting evidence. The gate doesn't by itself judge whether evidence is semantically strong enough — that's handled by the prompt-level facet-specific evidence instructions.

**Separate Evidence and Abstention Reason** — For an abstained facet, `evidence = null` and `abstention_reason` holds the explanation. This prevents confusing an explanation of uncertainty with actual conversational evidence.

**Pluggable LLM Client** — Separating the LLM interface from the scorer enables `MockLLMClient` for deterministic tests and `OllamaClient` for real inference, improving testability and reducing coupling.

**Deterministic Inference** — The Ollama client uses `temperature = 0` to reduce stochastic variation between benchmark runs and improve reproducibility.

---

## 22. Project Structure

```
facet-evaluation-engine/
│
├── benchmark/
│   ├── conversations.json
│   ├── reference_labels.json
│   ├── run_benchmark.py
│   ├── evaluate_benchmark.py
│   └── report.md
│
├── data/
│   ├── facets_raw.csv
│   └── facets_enriched.csv
│
├── src/
│   ├── llm_client.py
│   ├── pipeline.py
│   ├── retrieval.py
│   ├── scorer.py
│   ├── taxonomy.py
│   ├── facet_audit.py
│   ├── test_scorer.py
│   └── test_validation.py
│
├── README.md
├── PROMPT_LOG.md
├── DECISIONS.md
└── DEBUGGING.md
```

---

## 23. Documentation

- **PROMPT_LOG.md** — Prompt development, prompt changes, and reasoning behind the final scoring instructions.
- **DECISIONS.md** — Technical and design decisions, including evidence-first scoring, abstention behavior, validation, and known limitations.
- **DEBUGGING.md** — Debugging work, observed model failures, fixes, benchmark comparisons, and remaining limitations.
- **benchmark/report.md** — Benchmark methodology and evaluation report.

> **Before submitting:** confirm `PROMPT_LOG.md`, `DECISIONS.md`, and `DEBUGGING.md` actually exist in the repository. If any are missing, create them rather than leaving the README pointing to nonexistent files.

---

## 24. Reproducibility

```bash
git clone https://github.com/Prashant-00000/facet-evaluation-engine.git
cd facet-evaluation-engine

python3 -m pip install -r requirements.txt
# or, if requirements.txt is unavailable:
python3 -m pip install pandas numpy requests

ollama pull qwen2.5:7b-instruct
ollama list

export USE_REAL_LLM=true
python3 benchmark/run_benchmark.py --real

python3 benchmark/evaluate_benchmark.py \
    --results benchmark/benchmark_results.json
```

---

## 25. Git Workflow

```bash
git status
git diff
git add <files>
git commit -m "Describe change"
git push
```

Generated files such as `src/__pycache__/` and `benchmark/benchmark_results.json` should generally remain untracked unless explicitly required.

---

## 26. Final Project Status

The current implementation provides: facet retrieval, taxonomy normalization, observability filtering, sensitivity handling, local Qwen-based facet scoring, structured JSON output, evidence-based abstention, deterministic output validation, duplicate detection, missing facet detection, facet-specific evidence requirements, evidence validation, sarcasm-aware prompting, pluggable mock/real LLM clients, automated benchmark evaluation, and benchmark reporting.

**Final real-Qwen benchmark:**

```
Facet Detection Rate: 100.00%
Score Agreement:      66.67%
Abstention Accuracy:  92.59%
Unexpected Scores:    2
Missing Conversations: 0
```

The benchmark demonstrates a substantial improvement in conservative evidence handling:

- Abstention accuracy: 85.19% → 92.59%
- Unexpected scores: 9 → 2

The remaining limitations are primarily related to self-contradictory statements, indirect evidence, reference-label ambiguity, and the reasoning limitations of a small local LLM. These limitations are documented rather than hidden from the evaluation.

---

## 27. Repository

[https://github.com/Prashant-00000/facet-evaluation-engine](https://github.com/Prashant-00000/facet-evaluation-engine)
