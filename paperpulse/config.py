"""
PaperPulse configuration.

Central place for user-facing settings that affect paper selection.
Edit INTEREST_PROFILE to tune which papers get selected for you.
"""

# ---------------------------------------------------------------------------
# Interest profile
# ---------------------------------------------------------------------------
# Describes the ideal reader of PaperPulse.
# Used by both the embedding pre-filter and the LLM scoring step.
# Edit this to adjust which papers get selected for you.

INTEREST_PROFILE = """
AI-powered clinical decision support across all medical specialties,
including tumor board decisions, diagnosis assistance, and outcome prediction.
NLP applied to clinical notes and medical documentation.
Health data infrastructure including EHRs, interoperability, and FHIR.
Digital therapeutics.
Real-world clinical applications and results.
Novel technical approaches enabling clinical AI.
"""