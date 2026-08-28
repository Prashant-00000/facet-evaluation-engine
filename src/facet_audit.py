import pandas as pd

from src.taxonomy import (
    normalize_facet,
    extract_number,
    classify_facet,
)


INPUT_FILE = "data/facets_raw.csv"
OUTPUT_FILE = "data/facets_enriched.csv"


def get_observability(facet_type):
    """
    Decide whether a facet can potentially be evaluated
    from conversational evidence.
    """

    if facet_type in {
        "medical_biological",
        "cognitive_ability_test",
        "behavioral_log_count",
        "biographical_external",
        "spiritual_religious_practice",
        "astrology_pseudoscience",
        "malformed_header",
    }:
        return "false"

    if facet_type in {
        "clinical_symptom",
        "skill_knowledge",
    }:
        return "conditional"

    return "true"


def get_sensitivity(facet_type):
    """
    Assign a preliminary sensitivity level.
    """

    if facet_type in {
        "medical_biological",
        "clinical_symptom",
    }:
        return "high"

    if facet_type in {
        "cognitive_ability_test",
        "biographical_external",
        "spiritual_religious_practice",
        "astrology_pseudoscience",
    }:
        return "medium"

    return "low"


def get_abstention_reason(facet_type):
    """
    Explain why a facet should not automatically be scored.
    """

    reasons = {
        "medical_biological":
            "Requires medical, laboratory, genetic, or biological evidence.",

        "clinical_symptom":
            "Only score when the person explicitly provides sufficient conversational evidence; do not diagnose.",

        "cognitive_ability_test":
            "Requires an administered cognitive or ability assessment.",

        "behavioral_log_count":
            "Requires quantified behavioral or activity data.",

        "biographical_external":
            "Requires explicit biographical or external evidence.",

        "spiritual_religious_practice":
            "Requires explicit information about the relevant practice or external activity.",

        "astrology_pseudoscience":
            "Cannot be established from ordinary conversation.",

        "malformed_header":
            "Header-like or malformed entry; excluded from scoring.",
    }

    return reasons.get(facet_type, "")


def get_scoring_definition(facet_type):
    """
    Provide a generic five-level scoring framework.

    Later we can make these definitions more facet-specific.
    """

    if facet_type in {
        "medical_biological",
        "cognitive_ability_test",
        "behavioral_log_count",
        "biographical_external",
        "spiritual_religious_practice",
        "astrology_pseudoscience",
        "malformed_header",
    }:
        return ""

    return (
        "1=very low evidence, "
        "2=low evidence, "
        "3=moderate evidence, "
        "4=strong evidence, "
        "5=very strong evidence"
    )


def main():
    # Read the original raw dataset.
    df = pd.read_csv(INPUT_FILE)

    # Preserve the original facet exactly.
    df["raw_value"] = df["Facets"]

    # Normalize the facet for processing/retrieval.
    df["normalized_value"] = df["Facets"].apply(normalize_facet)

    # Extract leading numeric IDs when present.
    df["extracted_id"] = df["Facets"].apply(extract_number)

    # Classify each facet.
    df["facet_type"] = df["Facets"].apply(classify_facet)

    # Determine observability.
    df["conversation_observable"] = df["facet_type"].apply(
        get_observability
    )

    # Determine sensitivity.
    df["sensitivity"] = df["facet_type"].apply(
        get_sensitivity
    )

    # Add scoring definition.
    df["scoring_definition"] = df["facet_type"].apply(
        get_scoring_definition
    )

    # Add abstention reason.
    df["abstention_reason"] = df["facet_type"].apply(
        get_abstention_reason
    )

    # Save the enriched dataset.
    df.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("FACET AUDIT COMPLETE")
    print("=" * 60)

    print(f"\nInput rows: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nFacet type counts:")
    print(df["facet_type"].value_counts().to_string())

    print("\nObservability counts:")
    print(df["conversation_observable"].value_counts().to_string())


if __name__ == "__main__":
    main()