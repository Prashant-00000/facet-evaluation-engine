"""
Deterministic taxonomy rules for classifying facets.

This module provides reproducible rules used by facet_audit.py.

Change log (v2 — fixes found by running v1 against all 399 raw facets
and sampling the fallback bucket, per the audit's own review principle):

- CLINICAL_PHRASES: added "sleep apnea" and broadened "depression" to a
  bare word match (was previously only "depression symptoms"/"depression:",
  which missed "Depression (DEP)"). Both "Sleep Apnea" and "Depression (DEP)"
  were previously falling through to the personality_trait default with no
  abstention path — exactly the "naive scorer hallucinates a clinical fact"
  failure mode this taxonomy exists to prevent.
- COGNITIVE_TEST_PHRASES: added 8 psychometric items that were leaking into
  personality_trait: analogies, sentence structure, logical sequence
  identification, comprehension of spoken information, judging consequences,
  estimating calculations, understanding mathematical concepts, comparing
  alphanumeric data. Also added economic reasoning and mindfulness-technique
  inventories, which are self-report test items, not personality traits.
- BEHAVIORAL_PREFERENCE_PHRASES: added generic "usage" and "duration"
  (verified against the full 399-row file first: "usage" only ever appears
  in genuinely behavioral facets — transport/lending/finance-app usage —
  and "duration" only in "Eye-contact duration"; neither collides with an
  unrelated facet).
- ASTROLOGY_PHRASES: added the specific phrase "aura-color perception"
  (NOT a bare "aura" keyword — "aura" is a substring of "restaurant", which
  would have silently mis-tagged "Preference for Home-Cooked vs Restaurant
  Meals" as astrology/pseudoscience; verified by grepping the raw file).
- Removed "commute time/day" from BEHAVIORAL_PREFERENCE_PHRASES: it was
  dead/unreachable, since the generic "time/day" phrase in
  BEHAVIORAL_LOG_PHRASES is checked first (step 8) and always matches
  before BEHAVIORAL_PREFERENCE_PHRASES (step 10) is reached. Same end
  category either way, so this is a no-op cleanup, not a behavior change.

Every addition below was blast-radius-checked against the full 399-row
raw CSV before inclusion (see facet_audit.py's test invocation / the
project's tests/test_audit.py for the reproducible check).
"""

import re


FACET_TYPES = {
    "personality_trait",
    "clinical_symptom",
    "medical_biological",
    "cognitive_ability_test",
    "behavioral_log_count",
    "behavioral_preference",
    "biographical_external",
    "spiritual_religious_practice",
    "astrology_pseudoscience",
    "skill_knowledge",
    "malformed_header",
}


# ============================================================
# MEDICAL / BIOLOGICAL
# ============================================================

MEDICAL_PHRASES = [
    "fsh level",
    "parathyroid-hormone level",
    "chromatin-accessibility score",
    "serotonin transporter availability",
    "metabolic rate",
    "immune-response age",
    "basophil count",
    "polygenic risk",
    "caffeine sensitivity gene",
]


# ============================================================
# CLINICAL
# ============================================================

CLINICAL_PHRASES = [
    "depression",  # broadened from "depression symptoms"/"depression:" —
                   # catches "Depression (DEP)" too. Verified: every
                   # occurrence of "depression" in the raw file is clinical.
    "sleep-disorder diagnosis",
    "sleep apnea",
    "burnout symptoms",
    "psychoticism",
    "hypomania",
    "hysteria",
    "chronic pain",
]


# ============================================================
# COGNITIVE / TEST-BASED
# ============================================================

COGNITIVE_TEST_PHRASES = [
    "statistical reasoning",
    "spatial perception",
    "memory for sounds",
    "critical reasoning",
    "auditory memory",
    "divided attention ability",
    "auditory memory recall",
    "sequential memory recall",
    "intelligence quotient",
    "cognitive measure",
    "working memory index",
    "rapid cognitive processing",
    "mental arithmetic speed",
    "decision-making speed",
    "faux pas recognition accuracy",
    "spelling accuracy",
    # --- added in v2, verified against full file ---
    "analogies",
    "sentence structure",
    "logical sequence identification",
    "comprehension of spoken information",
    "judging consequences",
    "estimating calculations",
    "understanding mathematical concepts",
    "comparing alphanumeric data",
    "economic reasoning",
    "mindfulness techniques",
]


# ============================================================
# QUANTIFIED BEHAVIOR / ACTIVITY
# ============================================================

BEHAVIORAL_LOG_PHRASES = [
    "participation count",
    "count",
    "frequency",
    "hours",
    "sessions",
    "visits/year",
    "years",
    "km/week",
    "mg/day",
    "days",
    "months",
    "ratio",
    "%",
    "time/day",
    "time/week",
    "consistency",
    "accuracy",
    "usage frequency",
    "endorsements",
    "ideas generated/day",
]


# ============================================================
# BIOGRAPHICAL / EXTERNAL
# ============================================================

BIOGRAPHICAL_PHRASES = [
    "nationality",
    "passport-stamps count",
    "childhood experiences",
    "cultural identity",
]


# ============================================================
# SPIRITUAL / RELIGIOUS
# ============================================================

SPIRITUAL_PHRASES = [
    "spiritual",
    "spirituality",
    "sufi",
    "dhikr",
    "i ching",
    "iching",
    "quran",
    "bahá",
    "baha",
    "reiki",
    "religious",
    "religion",
    "hindu",
    "islamic",
    "jewish",
    "sikh",
    "buddhist",
    "kabbalah",
    "gnostic",
    "new-age",
    "meditation",
    "mantra",
    "yoga",
    "zohar",
    "scripture",
    "sephira",
    "archon",
    "sukkot",
    "shabbat",
    "eightfold path",
    "vrata",
    "ridván",
    "ridvan",
]


# ============================================================
# ASTROLOGY / PSEUDOSCIENCE
# ============================================================

ASTROLOGY_PHRASES = [
    "astrology",
    "zodiac",
    "rising sign",
    "horoscope",
    "astrological",
    "aura-color perception",  # full phrase only — bare "aura" is a
                               # substring of "restaurant" and would
                               # mis-tag unrelated facets.
]


# ============================================================
# SKILLS / KNOWLEDGE
# ============================================================

SKILL_PHRASES = [
    "anatomy knowledge",
    "network basics",
    "alphabetical filing skills",
    "social interaction skills",
    "numeric filing skills",
    "material properties knowledge",
    "cooking and culinary arts",
    "language use",
    "dance-style mastery",
    "storytelling proficiency",
    "delegation skills",
    "delegation ability",
    "non-verbal communication skills",
    "troubleshooting technical issues",
    "computer skills",
]


# ============================================================
# BEHAVIORAL PREFERENCES / HISTORY
# ============================================================

BEHAVIORAL_PREFERENCE_PHRASES = [
    "dietary habits",
    "eating habits",
    "preference for",
    "snacking behavior",
    "drug-use history",
    "peer-to-peer lending usage",
    "gamified-finance-app usage",
    "open-source contributions",
    "use of nature as a stress reliever",
    "home-security-system presence",
    "time outdoors/day",
    "travel-companions diversity",
    # --- added in v2, blast-radius checked against full file ---
    "usage",
    "duration",
]


# ============================================================
# SENSITIVITY OVERRIDES
# ============================================================

SENSITIVE_OVERRIDE_PHRASES = [
    "kink",
    "physical-violence",
    "drug-use",
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_facet(value: str) -> str:
    """
    Normalize whitespace and casing while preserving meaning.
    """
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value.lower()


def extract_number(value: str):
    """
    Extract a leading numeric identifier.

    Example:
        '800. Sufi practice: ...'
        -> 800
    """
    match = re.match(r"^\s*(\d+)\.\s+", str(value))

    if match:
        return int(match.group(1))

    return None


def is_malformed_header(value: str) -> bool:
    """
    Detect entries that appear to be headers or section labels.
    """
    return str(value).strip().endswith(":")


def contains_phrase(text: str, phrases: list) -> bool:
    """
    Match complete words/phrases rather than arbitrary substrings.

    This prevents false matches such as:

        gene -> general
        ratio -> desperation
        aura -> restaurant
    """

    for phrase in phrases:
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"

        if re.search(pattern, text):
            return True

    return False


def has_sensitive_override(value: str) -> bool:
    """Detect sensitive topics that require high-sensitivity handling."""
    return contains_phrase(normalize_facet(value), SENSITIVE_OVERRIDE_PHRASES)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_facet(value: str) -> str:
    """
    Assign one taxonomy category to a facet.

    Specific categories are checked before the broad
    personality fallback.
    """

    normalized = normalize_facet(value)

    # 1. Header-like / malformed entries
    if is_malformed_header(value):
        return "malformed_header"

    # 2. Astrology / pseudoscience
    if contains_phrase(normalized, ASTROLOGY_PHRASES):
        return "astrology_pseudoscience"

    # 3. Clinical symptoms / diagnoses
    if contains_phrase(normalized, CLINICAL_PHRASES):
        return "clinical_symptom"

    # 4. Medical / biological measurements
    if contains_phrase(normalized, MEDICAL_PHRASES):
        return "medical_biological"

    # 5. Cognitive tests / measurable abilities
    if contains_phrase(normalized, COGNITIVE_TEST_PHRASES):
        return "cognitive_ability_test"

    # 6. Spiritual / religious practices
    if contains_phrase(normalized, SPIRITUAL_PHRASES):
        return "spiritual_religious_practice"

    # 7. Biographical / externally verifiable information
    if contains_phrase(normalized, BIOGRAPHICAL_PHRASES):
        return "biographical_external"

    # 8. Quantified behavior
    if contains_phrase(normalized, BEHAVIORAL_LOG_PHRASES):
        return "behavioral_log_count"

    # 9. Skills / knowledge
    if contains_phrase(normalized, SKILL_PHRASES):
        return "skill_knowledge"

    # 10. Behavioral preferences / history
    if contains_phrase(normalized, BEHAVIORAL_PREFERENCE_PHRASES):
        return "behavioral_preference"

    # 11. Default: personality / psychological trait
    return "personality_trait"