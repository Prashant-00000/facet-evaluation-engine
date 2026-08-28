import pandas as pd

from src.taxonomy import (
    normalize_facet,
    extract_number,
    classify_facet,
    has_sensitive_override,
)

INPUT_FILE = "data/facets_raw.csv"
OUTPUT_FILE = "data/facets_enriched.csv"

def get_observability(facet_type):
    if facet_type in {"medical_biological","cognitive_ability_test","behavioral_log_count",
                       "biographical_external","spiritual_religious_practice",
                       "astrology_pseudoscience","malformed_header"}:
        return "false"
    if facet_type in {"clinical_symptom","skill_knowledge"}:
        return "conditional"
    return "true"

def get_sensitivity(facet_type, raw_value):
    """
    Sensitivity is derived from facet_type first, then escalated
    (never downgraded) if the raw facet content itself touches a
    sensitive topic regardless of category. See taxonomy.py's
    SENSITIVE_OVERRIDE_PHRASES for the rationale.
    """
    if facet_type in {"medical_biological","clinical_symptom"}:
        base = "high"
    elif facet_type in {"cognitive_ability_test","biographical_external",
                         "spiritual_religious_practice","astrology_pseudoscience"}:
        base = "medium"
    else:
        base = "low"

    if has_sensitive_override(raw_value):
        return "high"
    return base

def get_abstention_reason(facet_type, raw_value, sensitivity):
    reasons = {
        "medical_biological": "Requires medical, laboratory, genetic, or biological evidence.",
        "clinical_symptom": "Only score when the person explicitly provides sufficient conversational evidence; do not diagnose.",
        "cognitive_ability_test": "Requires an administered cognitive or ability assessment.",
        "behavioral_log_count": "Requires quantified behavioral or activity data.",
        "biographical_external": "Requires explicit biographical or external evidence.",
        "spiritual_religious_practice": "Requires explicit information about the relevant practice or external activity.",
        "astrology_pseudoscience": "Cannot be established from ordinary conversation.",
        "malformed_header": "Header-like or malformed entry; excluded from scoring.",
    }
    reason = reasons.get(facet_type, "")

    # Content-based sensitivity override: even facet_types that are
    # otherwise scoreable (e.g. personality_trait) need an explicit
    # caution note when the raw content is high-sensitivity, so a
    # downstream scorer doesn't treat e.g. "Kink-interest diversity"
    # the same as "Merriness".
    if has_sensitive_override(raw_value) and not reason:
        reason = ("Sensitive-topic facet; only score on explicit, clearly "
                   "volunteered evidence, never inferred or probed for.")

    return reason

def get_scoring_definition(facet_type):
    if facet_type in {"medical_biological","cognitive_ability_test","behavioral_log_count",
                       "biographical_external","spiritual_religious_practice",
                       "astrology_pseudoscience","malformed_header"}:
        return ""
    return "1=very low evidence, 2=low evidence, 3=moderate evidence, 4=strong evidence, 5=very strong evidence"

def main():
    df = pd.read_csv(INPUT_FILE)
    df["raw_value"] = df["Facets"]
    df["normalized_value"] = df["Facets"].apply(normalize_facet)
    df["extracted_id"] = df["Facets"].apply(extract_number)
    df["facet_type"] = df["Facets"].apply(classify_facet)
    df["conversation_observable"] = df["facet_type"].apply(get_observability)
    df["sensitivity"] = df.apply(lambda r: get_sensitivity(r["facet_type"], r["raw_value"]), axis=1)
    df["scoring_definition"] = df["facet_type"].apply(get_scoring_definition)
    df["abstention_reason"] = df.apply(
        lambda r: get_abstention_reason(r["facet_type"], r["raw_value"], r["sensitivity"]), axis=1
    )
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"rows: {len(df)}")
    print(df["sensitivity"].value_counts().to_string())

if __name__ == "__main__":
    main()